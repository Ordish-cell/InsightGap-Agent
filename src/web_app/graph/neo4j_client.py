import logging
from contextlib import contextmanager
from typing import Any, Iterator

from src.web_app.core.config import settings

logger = logging.getLogger(__name__)


class Neo4jUnavailable(RuntimeError):
    """Raised when Neo4j is disabled, unconfigured, or the driver is missing."""


class Neo4jClient:
    def __init__(self, driver: Any | None = None, database: str | None = None):
        self._driver = driver
        self._driver_init_attempted = driver is not None
        self.database = database or settings.neo4j_database

    @property
    def enabled(self) -> bool:
        return bool(settings.enable_neo4j and settings.neo4j_uri)

    def get_driver(self) -> Any:
        if self._driver is not None:
            return self._driver
        if self._driver_init_attempted:
            raise Neo4jUnavailable("Neo4j driver initialization already failed")
        self._driver_init_attempted = True
        if not self.enabled:
            raise Neo4jUnavailable("Neo4j is disabled or NEO4J_URI is not configured")
        if not settings.neo4j_username or not settings.neo4j_password:
            raise Neo4jUnavailable("NEO4J_USERNAME/NEO4J_PASSWORD are required")
        try:
            from neo4j import GraphDatabase
        except Exception as exc:  # pragma: no cover - covered through monkeypatch-friendly path
            raise Neo4jUnavailable("neo4j Python driver is not installed") from exc

        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_lifetime=settings.neo4j_max_connection_lifetime,
            max_connection_pool_size=settings.neo4j_max_connection_pool_size,
            connection_timeout=settings.neo4j_connection_timeout,
        )
        return self._driver

    @contextmanager
    def session(self) -> Iterator[Any]:
        driver = self.get_driver()
        session_kwargs = {"database": self.database} if self.database else {}
        with driver.session(**session_kwargs) as session:
            yield session

    def run_write(self, query: str, **params: Any) -> list[Any]:
        with self.session() as session:
            result = session.run(query, **params)
            return list(result)

    def run_read(self, query: str, **params: Any) -> list[Any]:
        with self.session() as session:
            result = session.run(query, **params)
            return list(result)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


neo4j_client = Neo4jClient()
