from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_app.core.config import settings
from src.web_app.graph.neo4j_client import Neo4jUnavailable
from src.web_app.graph.project_projector import ProjectGraphBuilder
from src.web_app.graph.repositories import GraphRepository
from src.web_app.graph.schema import ensure_constraints

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("sync_project_graph")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or sync the Agent OS project knowledge graph.")
    parser.add_argument("--apply", action="store_true", help="Write graph nodes/edges to Neo4j.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the graph summary and sample nodes.")
    parser.add_argument("--json", action="store_true", help="Print full extracted graph as JSON.")
    parser.add_argument("--project-key", default=settings.neo4j_project_key)
    args = parser.parse_args()

    result = ProjectGraphBuilder(root=ROOT, project_key=args.project_key).build()
    print("project_graph.summary")
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    if args.json:
        print(json.dumps(result.graph, ensure_ascii=False, indent=2))

    if not args.apply:
        print("dry_run=true; pass --apply to write Neo4j.")
        return 0

    if not settings.enable_neo4j:
        print("Neo4j is disabled. Set ENABLE_NEO4J=true before --apply.", file=sys.stderr)
        return 2
    try:
        repo = GraphRepository()
        ensure_constraints(repo.client)
        repo.upsert_project_graph(result.graph)
        print("project_graph.synced=true")
        return 0
    except Neo4jUnavailable as exc:
        logger.error("project_graph.neo4j_unavailable: %s", exc)
        return 2
    except Exception:
        logger.exception("project_graph.sync_failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

