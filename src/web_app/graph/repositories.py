from __future__ import annotations

from typing import Any

from src.web_app.graph.neo4j_client import Neo4jClient, neo4j_client


class GraphRepository:
    def __init__(self, client: Neo4jClient | None = None):
        self.client = client or neo4j_client

    def upsert_memory_projection(
        self,
        *,
        user_id: int,
        memory: dict[str, Any],
        topics: list[str],
        goals: list[str],
        preferences: list[str],
        boundaries: list[str],
        project_key: str,
    ) -> None:
        query = """
        MERGE (u:User {user_id: $user_id})
          ON CREATE SET u.source = 'memory_projector', u.scope = 'user', u.created_at = datetime()
        SET u.updated_at = datetime()

        MERGE (m:UserMemory {user_id: $user_id, memory_id: $memory_id})
        SET m += $memory_props,
            m.updated_at = datetime()
        MERGE (u)-[:HAS_MEMORY]->(m)

        FOREACH (topic_key IN $topics |
          MERGE (t:MemoryTopic {user_id: $user_id, key: topic_key})
          ON CREATE SET t.source = 'memory_projector', t.scope = 'user', t.created_at = datetime()
          SET t.updated_at = datetime()
          MERGE (m)-[:MENTIONS]->(t)
          MERGE (u)-[:INTERESTED_IN]->(t)
        )

        FOREACH (goal_key IN $goals |
          MERGE (g:MemoryGoal {user_id: $user_id, key: goal_key})
          ON CREATE SET g.source = 'memory_projector', g.scope = 'user', g.created_at = datetime()
          SET g.updated_at = datetime()
          MERGE (m)-[:SUPPORTS]->(g)
          MERGE (u)-[:HAS_GOAL]->(g)
        )

        FOREACH (pref_key IN $preferences |
          MERGE (p:MemoryPreference {user_id: $user_id, key: pref_key})
          ON CREATE SET p.source = 'memory_projector', p.scope = 'user', p.created_at = datetime()
          SET p.updated_at = datetime()
          MERGE (m)-[:SUPPORTS]->(p)
          MERGE (u)-[:PREFERS]->(p)
        )

        FOREACH (boundary_key IN $boundaries |
          MERGE (b:MemoryBoundary {user_id: $user_id, key: boundary_key})
          ON CREATE SET b.source = 'memory_projector', b.scope = 'user', b.created_at = datetime()
          SET b.updated_at = datetime()
          MERGE (m)-[:SUPPORTS]->(b)
          MERGE (u)-[:HAS_BOUNDARY]->(b)
        )

        WITH u
        MERGE (project:Project {project_key: $project_key})
          ON CREATE SET project.source = 'project_projector', project.scope = 'project', project.created_at = datetime()
        SET project.updated_at = datetime()
        MERGE (u)-[:WORKS_ON]->(project)
        """
        self.client.run_write(
            query,
            user_id=user_id,
            memory_id=str(memory["memory_id"]),
            memory_props=memory,
            topics=topics,
            goals=goals,
            preferences=preferences,
            boundaries=boundaries,
            project_key=project_key,
        )

    def mark_memory_status(self, *, user_id: int, memory_id: int | str, status: str, reason: str = "") -> None:
        query = """
        MATCH (m:UserMemory {user_id: $user_id, memory_id: $memory_id})
        SET m.status = $status,
            m.status_reason = $reason,
            m.updated_at = datetime()
        """
        self.client.run_write(query, user_id=user_id, memory_id=str(memory_id), status=status, reason=reason)

    def get_user_memory_context(self, *, user_id: int, terms: list[str], limit: int = 8) -> list[dict[str, Any]]:
        query = """
        MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:UserMemory)
        WHERE coalesce(m.status, 'active') = 'active'
        OPTIONAL MATCH (m)-[:MENTIONS|SUPPORTS]->(target)
        WITH m, target,
             CASE
               WHEN size($terms) = 0 THEN 1
               WHEN any(term IN $terms WHERE toLower(coalesce(m.preview, '')) CONTAINS term) THEN 3
               WHEN any(term IN $terms WHERE toLower(coalesce(target.key, '')) CONTAINS term) THEN 2
               ELSE 0
             END AS relevance
        WHERE relevance > 0
        RETURN m.memory_id AS memory_id,
               m.memory_type AS memory_type,
               m.category AS category,
               m.importance AS importance,
               m.preview AS preview,
               CASE WHEN target IS NULL THEN [] ELSE labels(target) END AS target_labels,
               CASE WHEN target IS NULL THEN '' ELSE target.key END AS target_key,
               relevance AS relevance
        ORDER BY relevance DESC, coalesce(m.importance, 0) DESC
        LIMIT $limit
        """
        rows = self.client.run_read(query, user_id=user_id, terms=terms, limit=limit)
        return [_record_to_dict(row) for row in rows]

    def upsert_project_graph(self, graph: dict[str, Any]) -> None:
        self.client.run_write(
            """
        MERGE (p:Project {project_key: $project_key})
          ON CREATE SET p.created_at = datetime()
        SET p.name = $project_name,
            p.source = 'project_projector',
            p.scope = 'project',
            p.updated_at = datetime()
            """,
            project_key=graph["project_key"],
            project_name=graph.get("project_name", graph["project_key"]),
        )
        self._upsert_project_items(
            project_key=graph["project_key"],
            label="ProjectModule",
            relation="HAS_MODULE",
            items=graph.get("modules", []),
        )
        self._upsert_project_items(
            project_key=graph["project_key"],
            label="ProjectService",
            relation="HAS_SERVICE",
            items=graph.get("services", []),
        )
        self._upsert_project_items(
            project_key=graph["project_key"],
            label="ProjectRepository",
            relation="HAS_REPOSITORY",
            items=graph.get("repositories", []),
        )
        self._upsert_project_items(
            project_key=graph["project_key"],
            label="ProjectAPIEndpoint",
            relation="EXPOSES",
            items=graph.get("api_endpoints", []),
        )
        self._upsert_project_items(
            project_key=graph["project_key"],
            label="ProjectConfigKey",
            relation="USES_CONFIG",
            items=graph.get("config_keys", []),
        )
        self._upsert_project_items(
            project_key=graph["project_key"],
            label="ProjectTechnology",
            relation="USES_TECH",
            items=graph.get("technologies", []),
        )
        self._upsert_qdrant_collections(
            project_key=graph["project_key"],
            items=graph.get("qdrant_collections", []),
        )
        self.upsert_project_edges(project_key=graph["project_key"], edges=graph.get("edges", []))

    def _upsert_project_items(self, *, project_key: str, label: str, relation: str, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        query = f"""
        MATCH (p:Project {{project_key: $project_key}})
        UNWIND $items AS item
        MERGE (n:{label} {{project_key: $project_key, key: item.key}})
        SET n += item.props, n.updated_at = datetime()
        MERGE (p)-[:{relation}]->(n)
        """
        self.client.run_write(query, project_key=project_key, items=items)

    def _upsert_qdrant_collections(self, *, project_key: str, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        query = """
        MATCH (p:Project {project_key: $project_key})
        UNWIND $items AS item
        MERGE (n:ProjectQdrantCollection {project_key: $project_key, name: item.name})
        SET n += item.props, n.updated_at = datetime()
        MERGE (p)-[:USES_VECTOR_COLLECTION]->(n)
        """
        self.client.run_write(query, project_key=project_key, items=items)

    def upsert_project_edges(self, *, project_key: str, edges: list[dict[str, Any]]) -> None:
        if not edges:
            return
        query = """
        UNWIND $edges AS edge
        MATCH (a {project_key: $project_key, key: edge.from_key})
        MATCH (b {project_key: $project_key, key: edge.to_key})
        WHERE edge.from_label IN labels(a) AND edge.to_label IN labels(b)
        MERGE (a)-[r:PROJECT_RELATION {relation: edge.relation}]->(b)
        SET r.source = 'project_projector',
            r.scope = 'project',
            r.updated_at = datetime()
        """
        self.client.run_write(query, project_key=project_key, edges=edges)

    def get_project_context(self, *, project_key: str, terms: list[str], limit: int = 8) -> list[dict[str, Any]]:
        query = """
        MATCH (p:Project {project_key: $project_key})--(n)
        WHERE size($terms) = 0
           OR any(term IN $terms WHERE toLower(coalesce(n.key, '')) CONTAINS term)
           OR any(term IN $terms WHERE toLower(coalesce(n.name, '')) CONTAINS term)
           OR any(term IN $terms WHERE toLower(coalesce(n.path, '')) CONTAINS term)
        RETURN labels(n) AS labels,
               coalesce(n.key, n.name) AS key,
               n.name AS name,
               n.path AS path,
               properties(n).description AS description
        LIMIT $limit
        """
        rows = self.client.run_read(query, project_key=project_key, terms=terms, limit=limit)
        return [_record_to_dict(row) for row in rows]


def _record_to_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "data"):
        return dict(record.data())
    if isinstance(record, dict):
        return dict(record)
    try:
        return dict(record)
    except Exception:
        return {"value": record}
