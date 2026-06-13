from __future__ import annotations

from src.web_app.graph.neo4j_client import Neo4jClient


CONSTRAINTS: tuple[str, ...] = (
    "CREATE CONSTRAINT user_user_id IF NOT EXISTS FOR (n:User) REQUIRE n.user_id IS UNIQUE",
    "CREATE CONSTRAINT user_memory_key IF NOT EXISTS FOR (n:UserMemory) REQUIRE (n.user_id, n.memory_id) IS UNIQUE",
    "CREATE CONSTRAINT memory_topic_key IF NOT EXISTS FOR (n:MemoryTopic) REQUIRE (n.user_id, n.key) IS UNIQUE",
    "CREATE CONSTRAINT memory_goal_key IF NOT EXISTS FOR (n:MemoryGoal) REQUIRE (n.user_id, n.key) IS UNIQUE",
    "CREATE CONSTRAINT memory_preference_key IF NOT EXISTS FOR (n:MemoryPreference) REQUIRE (n.user_id, n.key) IS UNIQUE",
    "CREATE CONSTRAINT memory_boundary_key IF NOT EXISTS FOR (n:MemoryBoundary) REQUIRE (n.user_id, n.key) IS UNIQUE",
    "CREATE CONSTRAINT project_key IF NOT EXISTS FOR (n:Project) REQUIRE n.project_key IS UNIQUE",
    "CREATE CONSTRAINT project_module_key IF NOT EXISTS FOR (n:ProjectModule) REQUIRE (n.project_key, n.key) IS UNIQUE",
    "CREATE CONSTRAINT project_service_key IF NOT EXISTS FOR (n:ProjectService) REQUIRE (n.project_key, n.key) IS UNIQUE",
    "CREATE CONSTRAINT project_repository_key IF NOT EXISTS FOR (n:ProjectRepository) REQUIRE (n.project_key, n.key) IS UNIQUE",
    "CREATE CONSTRAINT project_api_endpoint_key IF NOT EXISTS FOR (n:ProjectAPIEndpoint) REQUIRE (n.project_key, n.key) IS UNIQUE",
    "CREATE CONSTRAINT project_config_key IF NOT EXISTS FOR (n:ProjectConfigKey) REQUIRE (n.project_key, n.key) IS UNIQUE",
    "CREATE CONSTRAINT project_qdrant_collection_key IF NOT EXISTS FOR (n:ProjectQdrantCollection) REQUIRE (n.project_key, n.name) IS UNIQUE",
    "CREATE CONSTRAINT project_technology_key IF NOT EXISTS FOR (n:ProjectTechnology) REQUIRE (n.project_key, n.key) IS UNIQUE",
)


def ensure_constraints(client: Neo4jClient) -> None:
    for query in CONSTRAINTS:
        client.run_write(query)

