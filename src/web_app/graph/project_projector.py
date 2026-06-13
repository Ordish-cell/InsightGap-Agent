from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.web_app.core.config import settings

PROJECT_SCAN_DIRS = (
    "src/web_app/services",
    "src/web_app/rag",
    "src/web_app/api/v1",
    "src/web_app/db/repositories",
    "src/web_app/agent/runtime",
)
TECH_KEYS = (
    "fastapi", "langgraph", "langchain", "postgresql", "qdrant", "redis",
    "dashscope", "neo4j", "bm25", "rag",
)
ROUTER_RE = re.compile(r"@router\.(get|post|put|patch|delete)\(([^)]*)\)")
SETTINGS_RE = re.compile(r"settings\.([a-zA-Z_][a-zA-Z0-9_]*)")
COLLECTION_RE = re.compile(r"(agent_os_documents_v\d+|agent_os_documents|memory_vectors|hello_agents_vectors)")


@dataclass
class ProjectGraphBuildResult:
    project_key: str
    graph: dict[str, Any]
    summary: dict[str, int]


class ProjectGraphBuilder:
    def __init__(self, root: Path | str | None = None, project_key: str | None = None):
        self.root = Path(root or ".").resolve()
        self.project_key = project_key or settings.neo4j_project_key

    def build(self) -> ProjectGraphBuildResult:
        modules: dict[str, dict[str, Any]] = {}
        services: dict[str, dict[str, Any]] = {}
        repositories: dict[str, dict[str, Any]] = {}
        api_endpoints: dict[str, dict[str, Any]] = {}
        config_keys: dict[str, dict[str, Any]] = {}
        qdrant_collections: dict[str, dict[str, Any]] = {}
        technologies: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        for rel_dir in PROJECT_SCAN_DIRS:
            scan_dir = self.root / rel_dir
            if not scan_dir.exists():
                continue
            for file_path in sorted(scan_dir.rglob("*.py")):
                rel_path = file_path.relative_to(self.root).as_posix()
                module_key = rel_path.removesuffix(".py").replace("/", ".")
                modules[module_key] = _node(module_key, "ProjectModule", project_key=self.project_key, path=rel_path, name=module_key)
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                parsed_classes = _class_names(text)
                for class_name in parsed_classes:
                    if class_name.endswith("Repository"):
                        repositories[class_name] = _node(class_name, "ProjectRepository", project_key=self.project_key, path=rel_path, name=class_name)
                        edges.append(_edge("ProjectModule", module_key, "ProjectRepository", class_name, "DECLARES"))
                    elif class_name.endswith("Service") or "Service" in class_name:
                        services[class_name] = _node(class_name, "ProjectService", project_key=self.project_key, path=rel_path, name=class_name)
                        edges.append(_edge("ProjectModule", module_key, "ProjectService", class_name, "DECLARES"))
                for method, endpoint in _router_endpoints(text):
                    endpoint_key = f"{method.upper()} {endpoint}"
                    api_endpoints[endpoint_key] = _node(endpoint_key, "ProjectAPIEndpoint", project_key=self.project_key, path=rel_path, name=endpoint_key)
                    edges.append(_edge("ProjectModule", module_key, "ProjectAPIEndpoint", endpoint_key, "EXPOSES"))
                for key in sorted(set(SETTINGS_RE.findall(text))):
                    config_keys[key] = _node(key, "ProjectConfigKey", project_key=self.project_key, path=rel_path, name=key)
                    edges.append(_edge("ProjectModule", module_key, "ProjectConfigKey", key, "USES_CONFIG"))
                for name in sorted(set(COLLECTION_RE.findall(text))):
                    qdrant_collections[name] = {
                        "name": name,
                        "props": {
                            "name": name,
                            "key": name,
                            "source": "project_projector",
                            "scope": "project",
                            "project_key": self.project_key,
                        },
                    }
                    edges.append(_edge("ProjectModule", module_key, "ProjectQdrantCollection", name, "USES_VECTOR_COLLECTION"))
                lower = text.lower()
                for tech in TECH_KEYS:
                    if tech in lower:
                        technologies[tech] = _node(tech, "ProjectTechnology", project_key=self.project_key, name=tech)
                        edges.append(_edge("ProjectModule", module_key, "ProjectTechnology", tech, "MENTIONS_TECH"))

        env_keys = _safe_env_keys(self.root / ".env")
        for key in env_keys:
            config_keys[key] = _node(key, "ProjectConfigKey", project_key=self.project_key, path=".env", name=key)

        for name in _configured_qdrant_collections():
            qdrant_collections[name] = {
                "name": name,
                "props": {
                    "name": name,
                    "key": name,
                    "source": "project_projector",
                    "scope": "project",
                    "project_key": self.project_key,
                },
            }

        graph = {
            "project_key": self.project_key,
            "project_name": "Agent OS",
            "modules": list(modules.values()),
            "services": list(services.values()),
            "repositories": list(repositories.values()),
            "api_endpoints": list(api_endpoints.values()),
            "config_keys": list(config_keys.values()),
            "qdrant_collections": list(qdrant_collections.values()),
            "technologies": list(technologies.values()),
            "edges": edges,
        }
        summary = {
            "modules": len(modules),
            "services": len(services),
            "repositories": len(repositories),
            "api_endpoints": len(api_endpoints),
            "config_keys": len(config_keys),
            "qdrant_collections": len(qdrant_collections),
            "technologies": len(technologies),
            "edges": len(edges),
        }
        return ProjectGraphBuildResult(project_key=self.project_key, graph=graph, summary=summary)


def _node(key: str, label: str, project_key: str, **props: Any) -> dict[str, Any]:
    base = {
        "key": key,
        "label": label,
        "props": {
            "key": key,
            "source": "project_projector",
            "scope": "project",
            "project_key": project_key,
        },
    }
    base["props"].update({k: v for k, v in props.items() if v not in (None, "")})
    return base


def _edge(from_label: str, from_key: str, to_label: str, to_key: str, relation: str) -> dict[str, str]:
    return {
        "from_label": from_label,
        "from_key": from_key,
        "to_label": to_label,
        "to_key": to_key,
        "relation": relation,
    }


def _class_names(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def _router_endpoints(text: str) -> list[tuple[str, str]]:
    endpoints: list[tuple[str, str]] = []
    for match in ROUTER_RE.finditer(text):
        method = match.group(1)
        args = match.group(2)
        endpoint_match = re.search(r"[\"']([^\"']+)[\"']", args)
        if endpoint_match:
            endpoints.append((method, endpoint_match.group(1)))
    return endpoints


def _safe_env_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key and not _looks_secret_key(key):
            keys.append(key)
    return sorted(set(keys))


def _configured_qdrant_collections() -> list[str]:
    values = [
        settings.qdrant_collection,
        settings.qdrant_hybrid_collection,
        settings.memory_qdrant_collection,
    ]
    return sorted({str(value) for value in values if value})


def _looks_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in ("password", "secret", "token", "api_key", "apikey", "key"))
