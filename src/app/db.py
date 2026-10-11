"""Database access: one shared async pool whose every connection understands pgvector's `vector` type, and the one
place a sequence of numbers becomes a vector parameter.
"""

import math
import numbers
from typing import Any

from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from app import safety
from app.config import settings

EMBEDDING_DIM = 384  # chunk.embedding is vector(384): BAAI/bge-small-en-v1.5
CONNECT_TIMEOUT_S = 5  # a connection attempt never hangs; 127.0.0.1 answers at once when Postgres is up
CHECKOUT_TIMEOUT_S = 30.0  # a caller never waits forever for a free connection
_CHECK = AsyncConnectionPool.check_connection  # bound at import so make_pool does not depend on the class at call time


class VectorError(ValueError):
    """A value that cannot be stored or compared as a vector."""


def to_vector(value: Any, *, dim: int = EMBEDDING_DIM, allow_zero: bool = False) -> Vector:
    """A validated `Vector` from a list, tuple, 1-D numpy array or Vector, ready to be a query or insert parameter.
    """
    if isinstance(value, Vector):
        items = value.to_list()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif hasattr(value, "ndim") and hasattr(value, "tolist"):  # a numpy array or scalar
        if value.ndim != 1:
            raise VectorError(f"expected a flat vector, got a {value.ndim}-D array")
        items = value.tolist()
    else:
        raise VectorError(f"cannot make a vector from {type(value).__name__}")
    if len(items) != dim:
        raise VectorError(f"expected {dim} numbers, got {len(items)}")
    for x in items:
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            raise VectorError(f"every coordinate must be a number, found {type(x).__name__}")
    vector = Vector([float(x) for x in items])
    stored = vector.to_list()  # as float32, which is how Postgres will hold it
    if not all(math.isfinite(x) for x in stored):
        raise VectorError("every coordinate must be finite (no NaN, inf, or value beyond float32 range)")
    if not allow_zero and not any(stored):
        raise VectorError("the zero vector has no direction, so its cosine distance is undefined")
    return vector


async def _configure(conn: AsyncConnection) -> None:
    await register_vector_async(conn)
    # The rollback is readable and never writable, whatever the caller meant.
    database = (await (await conn.execute("SELECT current_database()")).fetchone())[0]
    if database in safety.PROTECTED:
        await conn.execute("SET default_transaction_read_only = on")
    # register_vector_async runs a catalog query, which opens a transaction on a non-autocommit connection.
    await conn.commit()


def make_pool(dsn: str | None = None, *, min_size: int = 1, max_size: int = 10, timeout: float = CHECKOUT_TIMEOUT_S,
              kwargs: dict[str, Any] | None = None, **options: Any) -> AsyncConnectionPool:
    """A closed pool: every connection is vector-aware (`configure`), is verified when borrowed, so one killed on the
    server is replaced instead of handed out broken (`check`), and no wait or connection attempt is unbounded.
    """
    return AsyncConnectionPool(
        dsn or safety.working_dsn(settings.pg_dsn),
        min_size=min_size,
        max_size=max_size,
        timeout=timeout,
        configure=_configure,
        check=_CHECK,
        kwargs={"connect_timeout": CONNECT_TIMEOUT_S, **(kwargs or {})},
        # open=False: the pool must be opened from a running event loop, e.g. in the FastAPI lifespan via `await
        # pool.open` / `await pool.close`.
        open=False,
        **options,
    )


pool = make_pool()
