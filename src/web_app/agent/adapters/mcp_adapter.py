from typing import Any


class MockMCPAdapter:
    def call_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"tool_name": tool_name, "payload": payload, "status": "mock"}
