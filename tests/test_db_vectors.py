"""The database pool and pgvector adaptation, proved before any search SQL exists."""

import asyncio
import math
from contextlib import AsyncExitStack

import numpy as np
import psycopg
import pytest
from dbhelpers import connect, require
from pgvector import Vector
from psycopg_pool import AsyncConnectionPool, PoolTimeout

import app.db as db
from app.config import settings
from app.db import EMBEDDING_DIM, VectorError, make_pool, to_vector
from ingest.runtime import run

DIM = EMBEDDING_DIM


def basis(i: int, scale: float = 1.0) -> list[float]:
    """A vector along axis i: cosine distances between basis vectors are known exactly (0, 1 or 2)."""
    return [scale if j == i else 0.0 for j in range(DIM)]


# --------------------------------------------------------------------------- to_vector: what may become a query or
# stored vector (no database) ---------------------------------------------------------------------------

FORMS = {
    "list": lambda v: list(v),
    "tuple": lambda v: tuple(v),
    "numpy float32": lambda v: np.asarray(v, dtype=np.float32),
    "numpy float64": lambda v: np.asarray(v, dtype=np.float64),
    "list of numpy float32 scalars": lambda v: [np.float32(x) for x in v],
    "list of ints": lambda v: [int(x) for x in v],
    "pgvector Vector": lambda v: Vector(list(v)),
}


@pytest.mark.parametrize("form", FORMS)
def test_every_shape_an_embedding_can_take_becomes_the_same_vector(form):
    original = basis(7, 3.0)
    out = to_vector(FORMS[form](original))
    assert isinstance(out, Vector) and out.dimensions() == DIM
    assert out.to_list() == original  # 3.0 and 0.0 are exact in float32


def test_values_are_rounded_to_float32_exactly_as_the_database_stores_them():
    out = to_vector([0.1] * DIM)
    assert out.to_list()[0] == float(np.float32(0.1)) != 0.1


def test_the_service_json_round_trip_is_lossless():
    v = np.random.default_rng(0).standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    wire = [float(x) for x in v]  # what the GPU service's JSON parses to: float32 values held in Python floats
    assert np.array_equal(to_vector(wire).to_numpy(), v)


def test_order_is_preserved():
    values = [float(i) for i in range(1, DIM + 1)]
    assert to_vector(values).to_list() == values


@pytest.mark.parametrize("bad,why", [
    (None, "None"),
    ("0.1,0.2", "a string"),
    (b"\x00" * DIM, "bytes"),
    ({"v": 1}, "a dict"),
    (5.0, "a bare number"),
    (iter(basis(0)), "a generator"),
    ([basis(0)], "a nested list"),
    (np.zeros((2, DIM)), "a 2-D array"),
    (np.zeros((DIM, 1)), "a column array"),
])
def test_things_that_are_not_a_flat_sequence_of_numbers_are_refused(bad, why):
    with pytest.raises(VectorError):
        to_vector(bad)


def test_a_two_dimensional_array_is_refused_by_name_not_by_accident():
    with pytest.raises(VectorError, match="2-D"):
        to_vector(np.zeros((2, DIM)))
    with pytest.raises(VectorError, match="1-D|flat"):
        to_vector(np.zeros((DIM, 1)))


@pytest.mark.parametrize("element", ["1.5", True, None, b"1", [1.0], complex(1, 2)])
def test_elements_that_only_look_like_numbers_are_refused(element):
    values = basis(0)
    values[5] = element
    with pytest.raises(VectorError, match="number"):
        to_vector(values)


@pytest.mark.parametrize("size", [0, 1, DIM - 1, DIM + 1, 2 * DIM])
def test_the_wrong_size_is_refused_before_the_database_sees_it(size):
    with pytest.raises(VectorError, match=str(DIM)):
        to_vector([1.0] * size)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_refused(bad):
    values = basis(0)
    values[100] = bad
    with pytest.raises(VectorError, match="finite"):
        to_vector(values)


def test_a_value_too_big_for_float32_is_refused_because_it_would_silently_become_inf():
    values = basis(0)
    values[3] = 1e39
    assert Vector(list(values)).to_list()[3] == math.inf  # what pgvector's own constructor does with it
    with pytest.raises(VectorError, match="finite"):
        to_vector(values)


def test_the_zero_vector_is_refused_because_its_cosine_distance_is_undefined():
    with pytest.raises(VectorError, match="zero"):
        to_vector([0.0] * DIM)
    assert to_vector([0.0] * DIM, allow_zero=True).dimensions() == DIM


def test_a_different_size_can_be_asked_for():
    assert to_vector([1.0, 2.0, 3.0], dim=3).to_list() == [1.0, 2.0, 3.0]
    with pytest.raises(VectorError, match="3"):
        to_vector([1.0] * DIM, dim=3)


def test_a_vector_that_already_has_the_wrong_size_is_refused_too():
    with pytest.raises(VectorError):
        to_vector(Vector([1.0, 2.0]))


def test_vector_error_is_a_value_error_so_callers_can_catch_either():
    assert issubclass(VectorError, ValueError)


def test_the_module_never_imports_numpy_itself():
    # numpy is a test-only dependency; the runtime path must work in the numpy-free environment CI and production use
    import inspect

    assert "import numpy" not in inspect.getsource(db) and "from numpy" not in inspect.getsource(db)


# --------------------------------------------------------------------------- the pool's wiring (no database)
# ---------------------------------------------------------------------------


class RecordingConnection:
    def __init__(self, database: str = "hansard_v2"):
        self.calls: list[str] = []
        self.database = database

    async def execute(self, sql):
        self.calls.append(sql)
        database = self.database

        class Cursor:
            async def fetchone(self):
                return (database,)

        return Cursor()

    async def commit(self):
        self.calls.append("commit")


def _configure_calls(monkeypatch, database):
    conn = RecordingConnection(database)

    async def fake_register(connection):
        assert connection is conn
        conn.calls.append("register_vector_async")

    monkeypatch.setattr(db, "register_vector_async", fake_register)
    run(db._configure(conn))
    return conn.calls


def test_configure_registers_the_vector_type_and_then_ends_the_transaction(monkeypatch):
    # registration runs a catalog query that opens a transaction; the pool discards a connection left in one
    assert _configure_calls(monkeypatch, "hansard_v2") == [
        "register_vector_async", "SELECT current_database()", "commit"]


def test_configure_makes_a_connection_to_the_rollback_read_only_before_ending_the_transaction(monkeypatch):
    """The SET is session-level, so it must come before the commit and outlive it."""
    assert _configure_calls(monkeypatch, "hansard") == [
        "register_vector_async", "SELECT current_database()", "SET default_transaction_read_only = on", "commit"]


class RecordingPool:
    instances: list["RecordingPool"] = []

    def __init__(self, conninfo, **kwargs):
        self.conninfo, self.kwargs = conninfo, kwargs
        RecordingPool.instances.append(self)


def test_make_pool_uses_register_vector_in_configure_and_checks_connections_on_borrow(monkeypatch):
    real_check = AsyncConnectionPool.check_connection
    monkeypatch.setattr(db, "AsyncConnectionPool", RecordingPool)
    RecordingPool.instances.clear()
    make_pool("postgresql://example/db", min_size=2, max_size=5)
    (built,) = RecordingPool.instances
    assert built.conninfo == "postgresql://example/db"
    assert built.kwargs["configure"] is db._configure
    assert built.kwargs["check"] is real_check
    assert built.kwargs["open"] is False  # opened later, from a running event loop
    assert (built.kwargs["min_size"], built.kwargs["max_size"]) == (2, 5)
    assert 0 < built.kwargs["timeout"] <= 60  # a checkout never waits forever
    assert built.kwargs["kwargs"]["connect_timeout"] > 0  # nor does a connection attempt


def test_the_default_dsn_is_the_configured_one_unless_that_is_the_rollback(monkeypatch):
    """PG_DSN's host and credentials always; its database unless that is protected."""
    from app import safety

    monkeypatch.setattr(db, "AsyncConnectionPool", RecordingPool)
    RecordingPool.instances.clear()
    make_pool()
    assert RecordingPool.instances[0].conninfo == safety.working_dsn(settings.pg_dsn)


def test_a_caller_can_add_connection_arguments_without_losing_the_defaults(monkeypatch):
    monkeypatch.setattr(db, "AsyncConnectionPool", RecordingPool)
    RecordingPool.instances.clear()
    make_pool("postgresql://x/y", kwargs={"application_name": "probe"})
    extra = RecordingPool.instances[0].kwargs["kwargs"]
    assert extra["application_name"] == "probe" and "connect_timeout" in extra


def test_the_shared_pool_is_a_real_pool_that_is_not_open_at_import():
    assert isinstance(db.pool, AsyncConnectionPool) and db.pool.closed


# --------------------------------------------------------------------------- against Postgres
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live():
    conn = connect()
    if conn is None:
        pytest.skip("Postgres is not reachable")
    if not conn.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector'").fetchone()[0]:
        pytest.skip("the database has no pgvector extension")
    yield conn
    conn.close()


def with_pool(scenario, **pool_kwargs):
    """Run `scenario(pool)` on a SelectorEventLoop with a fresh, opened pool that is always closed again."""

    async def main():
        p = make_pool(**pool_kwargs)
        await p.open(wait=True, timeout=10)
        try:
            return await scenario(p)
        finally:
            await p.close()

    return run(main())


def test_every_connection_the_pool_hands_out_understands_vector_and_is_left_idle(live):
    async def scenario(p):
        async with AsyncExitStack() as stack:
            conns = [await stack.enter_async_context(p.connection()) for _ in range(3)]  # three at once: three physical connections
            assert len({c.info.backend_pid for c in conns}) == 3
            values = []
            for c in conns:
                assert c.info.transaction_status == psycopg.pq.TransactionStatus.IDLE  # configure left no transaction open
                values.append((await (await c.execute("SELECT '[1,2,3]'::vector")).fetchone())[0])
            return values

    values = with_pool(scenario, min_size=1, max_size=3)
    assert all(isinstance(v, Vector) and v.to_list() == [1.0, 2.0, 3.0] for v in values)  # read back as Vector, not numpy


def test_the_shared_module_pool_registers_vector_too(live):
    async def main():
        await db.pool.open(wait=True, timeout=10)
        try:
            async with db.pool.connection() as conn:
                return (await (await conn.execute("SELECT '[4,5]'::vector")).fetchone())[0]
        finally:
            await db.pool.close()

    assert run(main()) == Vector([4.0, 5.0])


PROBE = f"CREATE TEMP TABLE probe (id int PRIMARY KEY, v vector({DIM})) ON COMMIT DROP"


async def distances(conn, stored, query_param, cast=""):
    """Insert `stored` (a param the adapter must handle) as id 1 and return its `<=>` distance to `query_param`."""
    async with conn.transaction():
        await conn.execute(PROBE)
        await conn.execute("INSERT INTO probe VALUES (1, %s)", (stored,))
        row = await (await conn.execute(f"SELECT v <=> %s{cast} FROM probe", (query_param,))).fetchone()
        return row[0]


@pytest.mark.parametrize("form", FORMS)
def test_every_form_can_be_inserted_and_compared_with_the_cosine_operator(live, form):
    async def scenario(p):
        out = {}
        async with p.connection() as conn:
            stored = to_vector(FORMS[form](basis(3)))
            out["same"] = await distances(conn, stored, to_vector(FORMS[form](basis(3))))
            out["orthogonal"] = await distances(conn, stored, to_vector(FORMS[form](basis(4))))
            out["opposite"] = await distances(conn, stored, to_vector(FORMS[form](basis(3, -1.0))))
            out["scaled"] = await distances(conn, stored, to_vector(FORMS[form](basis(3, 250.0))))
        return out

    got = with_pool(scenario)
    assert got["same"] == pytest.approx(0.0, abs=1e-6)
    assert got["orthogonal"] == pytest.approx(1.0)
    assert got["opposite"] == pytest.approx(2.0)
    assert got["scaled"] == pytest.approx(0.0, abs=1e-6)  # cosine ignores length


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_numpy_array_needs_no_helper_at_all(live, dtype):
    async def scenario(p):
        async with p.connection() as conn:
            return await distances(conn, np.asarray(basis(9), dtype=dtype), np.asarray(basis(9), dtype=dtype))

    assert with_pool(scenario) == pytest.approx(0.0, abs=1e-6)


def test_a_bare_list_is_the_trap_it_adapts_as_double_precision_array_which_the_operator_rejects(live):
    async def scenario(p):
        async with p.connection() as conn:
            async with conn.transaction():
                await conn.execute(PROBE)
                await conn.execute("INSERT INTO probe VALUES (1, %s::vector)", (basis(3),))
            try:
                async with conn.transaction():
                    await conn.execute(PROBE)
                    await conn.execute("INSERT INTO probe VALUES (1, %s::vector)", (basis(3),))
                    await conn.execute("SELECT v <=> %s FROM probe", (basis(3),))  # no cast: what a naive caller writes
            except psycopg.errors.UndefinedFunction as exc:
                return str(exc)

    message = with_pool(scenario)
    assert message and "vector <=> double precision[]" in message


def test_an_explicit_cast_also_makes_a_list_work_but_to_vector_is_preferred_because_it_validates(live):
    async def scenario(p):
        async with p.connection() as conn:
            return await distances(conn, basis(3), basis(3), cast="::vector")

    assert with_pool(scenario) == pytest.approx(0.0, abs=1e-6)


def test_postgres_itself_rejects_the_bad_vectors_that_to_vector_refuses_earlier(live):
    async def scenario(p):
        errors = {}
        async with p.connection() as conn:
            for name, sql in {
                "size": "SELECT '[1,2]'::vector <=> '[1,2,3]'::vector",
                "nan": "SELECT '[NaN,1]'::vector",
                "inf": "SELECT '[Infinity,1]'::vector",
            }.items():
                try:
                    await conn.execute(sql)
                except psycopg.errors.Error as exc:
                    errors[name] = str(exc)
                await conn.rollback()
            zero = (await (await conn.execute("SELECT '[0,0]'::vector <=> '[1,0]'::vector")).fetchone())[0]
        return errors, zero

    errors, zero = with_pool(scenario)
    assert "different vector dimensions" in errors["size"]
    assert "NaN not allowed" in errors["nan"] and "infinite" in errors["inf"]
    assert math.isnan(zero)  # no error at all: the zero vector just yields NaN, which is why to_vector refuses it


def test_a_stored_embedding_can_be_used_to_search_and_finds_itself_first(live):
    require(live, "SELECT count(*) FROM chunk WHERE embedding IS NOT NULL", "no chunk has been embedded yet")

    async def scenario(p):
        async with p.connection() as conn:
            stored = (await (await conn.execute("SELECT id, embedding FROM chunk WHERE embedding IS NOT NULL ORDER BY id LIMIT 1")).fetchone())
            chunk_id, vector = stored
            assert isinstance(vector, Vector)
            hits = {}
            for name, param in {"Vector": to_vector(vector), "numpy": vector.to_numpy(), "list": to_vector(vector.to_list())}.items():
                rows = await (await conn.execute("SELECT id, embedding <=> %s FROM chunk WHERE embedding IS NOT NULL ORDER BY embedding <=> %s LIMIT 3", (param, param))).fetchall()
                hits[name] = rows
            return chunk_id, hits

    chunk_id, hits = with_pool(scenario)
    for name, rows in hits.items():
        assert rows[0][0] == chunk_id and rows[0][1] == pytest.approx(0.0, abs=1e-5), name
        assert [r[1] for r in rows] == sorted(r[1] for r in rows)  # nearest first


def test_set_local_is_honoured_even_before_pgvector_is_loaded_in_the_session_and_reverts_at_commit(live):
    # pgvector's hnsw.* settings only exist once its library is loaded (any vector operation does that), so a SHOW on a
    # fresh connection fails.
    async def scenario(p):
        async with p.connection() as conn:
            fresh = conn.info.backend_pid
            try:
                await conn.execute("SHOW hnsw.ef_search")
                unknown_before = False
            except psycopg.errors.UndefinedObject:
                unknown_before = True
            await conn.rollback()
            async with conn.transaction():
                await conn.execute("SET LOCAL hnsw.ef_search = 100")
                await conn.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                await conn.execute("SELECT '[1,0]'::vector <=> '[1,0]'::vector")  # loads the library inside the transaction
                inside = ((await (await conn.execute("SHOW hnsw.ef_search")).fetchone())[0],
                          (await (await conn.execute("SHOW hnsw.iterative_scan")).fetchone())[0])
            after = (await (await conn.execute("SHOW hnsw.ef_search")).fetchone())[0]
        return fresh, unknown_before, inside, after

    _, unknown_before, inside, after = with_pool(scenario, min_size=1, max_size=1)
    assert unknown_before and inside == ("100", "strict_order") and after == "40"


def test_set_local_reverts_at_commit_and_at_rollback_but_a_plain_set_leaks_into_the_next_borrower(live):
    async def scenario(p):
        seen = {}
        async with p.connection() as conn:
            pid = conn.info.backend_pid
            await conn.execute("SELECT '[1]'::vector")  # load pgvector so its settings can be read
            seen["default"] = (await (await conn.execute("SHOW hnsw.ef_search")).fetchone())[0]
            await conn.rollback()  # a statement outside conn.transaction opens an implicit transaction: end it, or the
            # block below is only a savepoint inside it (see the next test)
            async with conn.transaction():
                await conn.execute("SET LOCAL hnsw.ef_search = 100")
                await conn.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                seen["inside"] = ((await (await conn.execute("SHOW hnsw.ef_search")).fetchone())[0],
                                  (await (await conn.execute("SHOW hnsw.iterative_scan")).fetchone())[0])
            seen["after_commit"] = (await (await conn.execute("SHOW hnsw.ef_search")).fetchone())[0]
            await conn.rollback()
            try:
                async with conn.transaction():
                    await conn.execute("SET LOCAL hnsw.ef_search = 77")
                    raise RuntimeError("the search failed half way")
            except RuntimeError:
                pass
            seen["after_rollback"] = (await (await conn.execute("SHOW hnsw.ef_search")).fetchone())[0]
            await conn.rollback()
            await conn.execute("SET hnsw.ef_search = 55")  # the mistake: a session-level SET
            await conn.commit()
        async with p.connection() as again:  # the pool has only one connection, so this is the same session
            seen["same_connection"] = again.info.backend_pid == pid
            seen["leaked"] = (await (await again.execute("SHOW hnsw.ef_search")).fetchone())[0]
            await again.execute("RESET hnsw.ef_search")
        return seen

    seen = with_pool(scenario, min_size=1, max_size=1)
    assert seen["default"] == "40" and seen["inside"] == ("100", "strict_order")
    assert seen["after_commit"] == "40" and seen["after_rollback"] == "40"
    assert seen["same_connection"] and seen["leaked"] == "55"  # why the search code must only ever use SET LOCAL


def test_inside_an_already_open_transaction_conn_transaction_is_only_a_savepoint_so_set_local_outlives_it(live):
    # The rule this pins for the search code: open the transaction on a freshly borrowed (idle) connection, before any
    # other statement, and check it.
    async def scenario(p):
        async with p.connection() as conn:
            idle_on_checkout = conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
            await conn.execute("SELECT '[1]'::vector")  # any statement: now an implicit transaction is open
            async with conn.transaction():  # so this is a savepoint
                await conn.execute("SET LOCAL hnsw.ef_search = 100")
            still_set = (await (await conn.execute("SHOW hnsw.ef_search")).fetchone())[0]
            await conn.commit()
            cleared = (await (await conn.execute("SHOW hnsw.ef_search")).fetchone())[0]
            await conn.rollback()
        return idle_on_checkout, still_set, cleared

    idle_on_checkout, still_set, cleared = with_pool(scenario, min_size=1, max_size=1)
    assert idle_on_checkout and still_set == "100" and cleared == "40"


def test_a_connection_killed_on_the_server_is_replaced_transparently_on_the_next_borrow(live):
    async def scenario(p):
        async with p.connection() as conn:
            first_pid = conn.info.backend_pid
        with connect() as killer:
            killer.execute("SELECT pg_terminate_backend(%s)", (first_pid,))
        await asyncio.sleep(0.2)
        async with p.connection() as conn:
            value = (await (await conn.execute("SELECT '[1,2]'::vector")).fetchone())[0]
            return first_pid, conn.info.backend_pid, value

    first, second, value = with_pool(scenario, min_size=1, max_size=1)
    assert first != second and value == Vector([1.0, 2.0])  # a working, vector-aware replacement


def test_an_exhausted_pool_times_out_instead_of_hanging(live):
    async def scenario(p):
        async with p.connection():
            started = asyncio.get_running_loop().time()
            with pytest.raises(PoolTimeout):
                async with p.connection():
                    pass
            return asyncio.get_running_loop().time() - started

    waited = with_pool(scenario, min_size=1, max_size=1, timeout=0.3)
    assert 0.25 <= waited < 3


def test_the_pool_survives_many_concurrent_borrowers_and_returns_every_connection(live):
    async def one(p, i):
        async with p.connection() as conn:
            async with conn.transaction():
                await conn.execute(PROBE)
                await conn.execute("INSERT INTO probe VALUES (1, %s)", (to_vector(basis(i % DIM)),))
                return (await (await conn.execute("SELECT v <=> %s FROM probe", (to_vector(basis(i % DIM)),))).fetchone())[0]

    async def scenario(p):
        results = await asyncio.gather(*[one(p, i) for i in range(60)])
        stats = p.get_stats()
        return results, stats

    results, stats = with_pool(scenario, min_size=2, max_size=5)
    assert all(r == pytest.approx(0.0, abs=1e-6) for r in results)
    assert stats["requests_num"] == 60 and stats["requests_queued"] > 0  # more borrowers than connections: they queued
    assert stats["pool_size"] <= 5 and stats["requests_waiting"] == 0
    assert stats["pool_available"] == stats["pool_size"]  # every connection came back
