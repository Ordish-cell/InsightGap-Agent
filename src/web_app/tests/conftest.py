"""Shared pytest fixtures — Windows-compatible async event loop."""
from __future__ import annotations

import asyncio
import selectors
import sys

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use SelectorEventLoop on Windows for psycopg async compatibility."""
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()
