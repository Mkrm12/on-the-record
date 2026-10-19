import asyncio
from collections.abc import Coroutine
from typing import Any


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    """asyncio.run with a loop psycopg's async mode can actually use."""
    return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
