from __future__ import annotations

import sys
import types

from src.web_app.agent.runtime.node_groups import agent_nodes as _agent_nodes
from src.web_app.agent.runtime.node_groups import base as _base
from src.web_app.agent.runtime.node_groups import eval_final_nodes as _eval_final_nodes
from src.web_app.agent.runtime.node_groups import final_helpers as _final_helpers
from src.web_app.agent.runtime.node_groups import legacy_nodes as _legacy_nodes
from src.web_app.agent.runtime.node_groups import read_nodes as _read_nodes
from src.web_app.agent.runtime.node_groups import setup_nodes as _setup_nodes
from src.web_app.agent.runtime.node_groups import tool_helpers as _tool_helpers
from src.web_app.agent.runtime.node_groups.agent_nodes import AgentNodesMixin
from src.web_app.agent.runtime.node_groups.base import *  # noqa: F401,F403 - legacy re-export surface
from src.web_app.agent.runtime.node_groups.base import BaseNodesMixin
from src.web_app.agent.runtime.node_groups.eval_final_nodes import EvalFinalNodesMixin
from src.web_app.agent.runtime.node_groups.legacy_nodes import LegacyNodesMixin
from src.web_app.agent.runtime.node_groups.read_nodes import ReadNodesMixin
from src.web_app.agent.runtime.node_groups.setup_nodes import SetupNodesMixin
from src.web_app.agent.runtime.llm_supervisor import llm_supervisor_route_node
from src.web_app.agent.runtime.schemas import StateDelta
from src.web_app.agent.runtime.state_delta import apply_state_delta, record_node_result
from src.web_app.agent.runtime.supervisor import observe_supervisor_state


class RuntimeNodes(
    SetupNodesMixin,
    ReadNodesMixin,
    AgentNodesMixin,
    EvalFinalNodesMixin,
    LegacyNodesMixin,
    BaseNodesMixin,
):
    """Compatibility facade for LangGraph runtime nodes."""

    async def supervisor_observer(self, state):
        observation = observe_supervisor_state(state)
        delta = StateDelta(
            updates={
                "supervisor_decision": observation["supervisor_decision"],
                "supervisor_warnings": observation["supervisor_warnings"],
                "supervisor_trace": observation["supervisor_trace"],
            },
            warnings=list(observation.get("supervisor_warnings") or []),
            metadata={"source": "supervisor_observer"},
        )
        apply_state_delta(state, delta)
        record_node_result(
            state,
            node="supervisor_observer",
            delta=delta,
            summary="Observed supervisor runtime state.",
        )
        return state

    async def llm_supervisor_route(self, state, config=None):
        for key in (
            "explicit_route",
            "explicit_agent",
            "selected_agent",
            "selected_action",
            "user_clicked_action",
            "action_id",
            "target_agent",
            "workflow",
            "agent_llm_supervisor_enabled",
            "agent_llm_supervisor_mode",
            "agent_llm_supervisor_temperature",
            "agent_llm_supervisor_timeout_seconds",
        ):
            if key in self.payload and key not in state:
                state[key] = self.payload[key]
        merged_config = {"configurable": dict(self.payload)}
        if isinstance(config, dict) and isinstance(config.get("configurable"), dict):
            merged_config["configurable"].update(config["configurable"])
        result = await llm_supervisor_route_node(state, config=merged_config)
        record_node_result(
            result,
            node="llm_supervisor_route",
            delta=StateDelta(
                updates={
                    "route_plan": result.get("route_plan"),
                    "llm_supervisor_decision": result.get("llm_supervisor_decision"),
                    "llm_supervisor_trace": result.get("llm_supervisor_trace", []),
                    "llm_supervisor_validation_errors": result.get("llm_supervisor_validation_errors", []),
                },
                warnings=list(result.get("llm_supervisor_validation_errors", []) or []),
                metadata={"source": "llm_supervisor_route"},
            ),
            summary="Applied LLM supervisor route decision when enabled.",
        )
        return result


_PROPAGATE_MODULES = (
    _base,
    _setup_nodes,
    _read_nodes,
    _agent_nodes,
    _eval_final_nodes,
    _legacy_nodes,
    _tool_helpers,
    _final_helpers,
)


class _RuntimeNodesCompatModule(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for module in _PROPAGATE_MODULES:
            if hasattr(module, name):
                setattr(module, name, value)


sys.modules[__name__].__class__ = _RuntimeNodesCompatModule
