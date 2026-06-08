"""Local file tools with workspace sandboxing.

All file operations are restricted to LOCAL_TOOLS_WORKSPACE_DIR.
Sensitive files (.env, keys, etc.) are blocked at the resolve level.
Delete operations require explicit L4 approval or are blocked entirely.
"""

import os
import re
from pathlib import Path
from typing import Any


# ── Sensitive file patterns that must never be read or written ──
_BLOCKED_PATTERNS: list[str] = [
    ".env",
    ".env.local",
    ".env.*",
    ".git",
    "id_rsa",
    "id_rsa.pub",
    "id_ed25519",
    "id_ed25519.pub",
    "*.pem",
    "*.key",
    "*secret*",
    "*token*",
    "*password*",
    "*credential*",
    "*.pfx",
    "*.p12",
    "*.jks",
    "authorized_keys",
    "known_hosts",
]

_BLOCKED_REGEX = re.compile(
    "|".join(
        pattern.replace(".", r"\.").replace("*", ".*")
        for pattern in _BLOCKED_PATTERNS
    ),
    re.IGNORECASE,
)


def _get_workspace_dir() -> Path:
    """Resolve the sandboxed workspace directory."""
    raw = os.getenv("LOCAL_TOOLS_WORKSPACE_DIR", "./agent_workspace")
    return Path(raw).resolve()


def _get_max_read_chars() -> int:
    return int(os.getenv("LOCAL_TOOLS_MAX_READ_CHARS", "8000"))


def _get_max_write_chars() -> int:
    return int(os.getenv("LOCAL_TOOLS_MAX_WRITE_CHARS", "20000"))


def _allow_delete() -> bool:
    return os.getenv("LOCAL_TOOLS_ALLOW_DELETE", "false").lower() == "true"


def _ensure_workspace() -> Path:
    ws = _get_workspace_dir()
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _resolve_safe_path(relative_path: str) -> Path:
    """Resolve a path relative to workspace, rejecting escapes."""
    ws = _ensure_workspace()
    resolved = (ws / relative_path).resolve()
    if not str(resolved).startswith(str(ws)):
        raise PermissionError(
            f"Path escapes workspace: {relative_path} -> {resolved}"
        )
    return resolved


def _is_sensitive(filename: str) -> bool:
    return bool(_BLOCKED_REGEX.search(Path(filename).name))


# ── Tool implementations ──────────────────────────────────────


def local_file_list(path: str = ".") -> dict[str, Any]:
    """List files in a workspace subdirectory. L1 risk."""
    ws = _ensure_workspace()
    target = (ws / path).resolve() if path else ws
    if not str(target).startswith(str(ws)):
        raise PermissionError(f"Path escapes workspace: {path}")
    if not target.exists():
        return {"path": str(target.relative_to(ws)), "files": [], "error": "directory_not_found"}
    entries: list[dict[str, Any]] = []
    try:
        for entry in sorted(target.iterdir()):
            name = entry.name
            if _is_sensitive(name):
                entries.append({"name": name, "type": "dir" if entry.is_dir() else "file", "size": 0, "blocked": True, "reason": "sensitive_file_hidden"})
                continue
            entries.append({
                "name": name,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else 0,
            })
    except PermissionError as exc:
        return {"path": str(target.relative_to(ws)), "files": [], "error": str(exc)}
    return {"path": str(target.relative_to(ws)), "files": entries}


def local_file_read(path: str, max_chars: int | None = None) -> dict[str, Any]:
    """Read a file within the workspace. L1/L2 risk."""
    limit = max_chars or _get_max_read_chars()
    safe = _resolve_safe_path(path)
    if _is_sensitive(safe.name):
        return {"path": path, "error": "sensitive_file_blocked", "content": ""}
    if not safe.exists():
        return {"path": path, "error": "file_not_found", "content": ""}
    if not safe.is_file():
        return {"path": path, "error": "not_a_file", "content": ""}
    content = safe.read_text(encoding="utf-8", errors="replace")
    truncated = len(content) > limit
    return {
        "path": str(safe.relative_to(_ensure_workspace())),
        "content": content[:limit],
        "total_chars": len(content),
        "truncated": truncated,
    }


def local_file_write(path: str, content: str, mode: str = "create_or_overwrite") -> dict[str, Any]:
    """Write a file within the workspace. L3 risk – requires approval."""
    max_chars = _get_max_write_chars()
    if len(content) > max_chars:
        return {"path": path, "error": f"content too large, max {max_chars} chars", "written": False}
    safe = _resolve_safe_path(path)
    if _is_sensitive(safe.name):
        return {"path": path, "error": "sensitive_file_blocked", "written": False}

    existed = safe.exists()
    safe.parent.mkdir(parents=True, exist_ok=True)

    if mode == "append":
        with open(safe, "a", encoding="utf-8") as f:
            f.write(content)
    else:
        safe.write_text(content, encoding="utf-8")

    return {
        "path": str(safe.relative_to(_ensure_workspace())),
        "written": True,
        "chars_written": len(content),
        "existed_before": existed,
        "mode": mode,
    }


def local_file_append(path: str, content: str) -> dict[str, Any]:
    """Append to a file. L3 risk – requires approval."""
    return local_file_write(path, content, mode="append")


def local_file_delete(path: str) -> dict[str, Any]:
    """Delete a file. L4 risk – blocked by default."""
    if not _allow_delete():
        return {"path": path, "error": "delete_disabled_by_config", "deleted": False,
                "message": "LOCAL_TOOLS_ALLOW_DELETE=false, deletion blocked."}
    safe = _resolve_safe_path(path)
    if _is_sensitive(safe.name):
        return {"path": path, "error": "sensitive_file_blocked", "deleted": False}
    if not safe.exists():
        return {"path": path, "error": "file_not_found", "deleted": False}
    safe.unlink()
    return {"path": str(safe.relative_to(_ensure_workspace())), "deleted": True}
