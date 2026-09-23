"""Structural error diagnostics without prompts, provider bodies or credentials."""
from pathlib import Path
import traceback

from .native_turn import NativeProtocolError
from .errors import ProviderStreamError


def error_diagnostics(exc: BaseException) -> dict[str, str]:
    # Unknown exception messages can contain a complete request/response. Keep
    # their type and stack location, never their message or frame locals.
    chain = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        detail = str(exc) if isinstance(exc, (NativeProtocolError, ProviderStreamError)) else ""
        if isinstance(exc, ValueError) and str(exc) == "No generation chunks were returned":
            detail = "No generation chunks were returned"
        frames = traceback.extract_tb(exc.__traceback__)
        stack = " > ".join(f"{Path(f.filename).name}:{f.name}:{f.lineno}" for f in frames[-8:])
        chain.append(f"{type(exc).__name__}: {detail} [{stack}]")
        exc = exc.__cause__
    return {"error_diagnostic": " <- ".join(chain)[:2000]}
