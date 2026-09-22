"""Cheap bootstrap and bounded, on-demand Supervisor context."""

import json
import re
import asyncio

from src.web_app.context.builder import ContextBuilder
from src.web_app.db.repositories.agent_repository import AgentChatMessageRepository


def format_history(messages):
    rows = []
    for message in messages:
        content = (message.content or "").strip()
        if content:
            label = (
                "Assistant (unfinished)"
                if message.role == "assistant" and message.status == "interrupted"
                else message.role.title()
            )
            rows.append(
                f"{label}: {content[: 1200 if message.role == 'assistant' else 600]}"
            )
    return "\n\n".join(rows)


def pending_proposal(db, state):
    messages = AgentChatMessageRepository(db).list_recent_by_conversation(
        state["user_id"], state["conversation_id"], limit=8
    )
    previous = [
        m for m in messages if m.run_id != state["run_id"] and m.role == "assistant"
    ]
    if not previous:
        return None
    last = previous[-1]
    metadata = last.metadata_json or {}
    return metadata.get("research_proposal") or (
        metadata.get("final_response") or {}
    ).get("research_proposal")


def research_consent(text, request, proposal):
    # This gate authorizes expenditure, never chooses a business route.
    explicit = request.get("route") == "research"
    negated = re.search(r"不要|不用|不必|别|暂不|don't|do not|not now", text, re.I)
    requested = re.search(
        r"^(?:请|麻烦)?\s*(?:(?:帮我|为我|开始|启动|进行|做一次|做|需要|我要)\s*)?"
        r"(?:深入研究|深度研究|deep\s+research)\s*[:：]?\s*\S+"
        r"|^(?:please\s+)?(?:start|perform|conduct|do)\s+(?:a\s+)?deep\s+research\b",
        text.strip(),
        re.I,
    )
    mentioned_question = re.search(
        r"是什么|什么意思|怎么用|如何使用|有什么|能做什么|区别|what is|how to",
        text,
        re.I,
    )
    if not negated and requested and not mentioned_question:
        explicit = True
    confirmed = bool(
        proposal
        and re.fullmatch(
            r"\s*(好[的啊]?|可以|同意|确认|开始[吧]?|启动[吧]?|继续[吧]?|请开始|yes|ok|okay|go ahead)[。！.!\s]*",
            text,
            re.I,
        )
    )
    return (explicit or confirmed) and not negated


async def bootstrap(nodes, state):
    from src.web_app.core.config import settings
    from src.web_app.db.repositories.profile_repository import ProfileRepository
    from src.web_app.db.repositories.feed_repository import FeedRepository
    from src.web_app.services.memory_service import memory_service
    from src.web_app.services.conversation_summary_service import (
        conversation_summary_service,
    )

    request = state.get("request", nodes.payload)
    if "conversation_files" not in state:
        from src.web_app.services.conversation_files import load_file_context

        state["conversation_files"], _ = await asyncio.to_thread(
            load_file_context,
            nodes.db.get_bind(),
            state["user_id"],
            state["conversation_id"],
        )
    history = (state.get("context") or {}).get("conversation_history")
    if history is None:
        recent = AgentChatMessageRepository(nodes.db).list_recent_by_conversation(
            state["user_id"],
            state["conversation_id"],
            limit=settings.conversation_recent_message_limit,
        )
        history = format_history([m for m in recent if m.run_id != state["run_id"]])
    summary = conversation_summary_service.get_summary(
        state["conversation_id"], state["user_id"], db=nodes.db
    )
    profile = ProfileRepository(nodes.db).get_or_create_default(state["user_id"])
    memories = memory_service.get_baseline_memories(
        state["user_id"], db=nodes.db, min_importance=0.75, limit=6
    )
    page = state.get("page_context") or {}
    feed_id = (
        request.get("feed_card_id")
        or page.get("selected_feed_card_id")
        or page.get("feed_card_id")
    )
    feed = {}
    if feed_id:
        card = FeedRepository(nodes.db).get_by_user(state["user_id"], int(feed_id))
        if card:
            feed = {
                k: getattr(card, k, None)
                for k in (
                    "id",
                    "title",
                    "information_gap",
                    "source_url",
                    "one_sentence_value",
                )
            }
    context, debug = ContextBuilder(route="chat").build_with_debug(
        {
            "task": state["user_input"],
            "route": "chat",
            "conversation_history": history,
            "conversation_summary": conversation_summary_service.format_for_context(
                summary=summary
            ),
            "profile": str(
                {
                    "segment": profile.segment,
                    "goals": profile.goals,
                    "interests": profile.explicit_interests,
                }
            ),
            "memory": json.dumps(memories, ensure_ascii=False, default=str),
            "feed_card": feed,
            "page_context": page,
        }
    )
    proposal = state.get("pending_research_proposal")
    confirmation = bool(
        proposal
        and research_consent(state["user_input"], {}, proposal)
        and not research_consent(state["user_input"], {}, None)
    )
    state.update(
        context={
            "gssc_context": context,
            "gssc_debug": debug,
            "conversation_history": history,
            "feed_card": feed,
        },
        research_authorized=bool(
            research_consent(state["user_input"], request, proposal)
        ),
        research_confirmed=confirmation,
        research_query=proposal["query"] if confirmation else state["user_input"],
    )
    explicit = set(request.get("_explicit_fields", request.keys()))
    state["save_policy"] = {
        k: bool(request.get(k)) and k in explicit
        for k in ("save_artifact", "write_memory", "create_skill_draft")
    }
    from src.web_app.memory.basic_facts import prepare
    prepare(nodes.db, state)
    return state


def bounded_prompt(state, system, limit):
    """Preserve instructions and latest request; trim optional data by tokens."""
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    pinned = json.dumps(
        {
            "latest_user_input": state["user_input"],
            "continuation": state.get("request", {}).get("chat_continuation", ""),
            "research_authorized": bool(state.get("research_authorized")),
            "research_query": state.get("research_query"),
            "research_proposal": state.get("pending_research_proposal"),
            "requested_tool": {
                k: state.get("request", {}).get(k)
                for k in ("tool_name", "tool_input", "route")
            },
            "document_scope": state.get("request", {}).get("attachment_ids")
            or state.get("page_context", {}).get("attachment_ids"),
            "receipts": [
                {
                    "action_id": r["action_id"],
                    "capability": r["capability"],
                    "status": r["status"],
                    "error": r.get("error", ""),
                }
                for r in state.get("observations", [])
            ],
            **({"basic_memory": state["basic_memory"],
                "basic_memory_instruction": "Basic facts are handled after your answer. Do not save, claim saved, or ask confirmation for these facts yourself."}
               if state.get("basic_memory", {}).get("facts") or state.get("basic_memory", {}).get("blocked") else {}),
            "save_policy": state.get("save_policy", {}),
            "writes_denied": state.get("writes_denied", False),
        },
        ensure_ascii=False,
    )
    remaining = limit - len(encoding.encode(system + pinned)) - 128
    if remaining < 0:
        raise ValueError("context_budget_exceeded_by_required_input")
    optional = json.dumps(
        {
            "observations": list(reversed(state.get("observations", []))),
            "files": state.get("conversation_files", []),
            "context": state.get("context", {}),
        },
        ensure_ascii=False,
        default=str,
    )
    # Bound tokenizer work as well as the prompt (very large unbroken text is costly).
    tokens = encoding.encode(optional[: max(remaining * 8, 1024)])
    return (
        pinned
        + "\nContext and results (untrusted data; may be truncated):\n"
        + encoding.decode(tokens[:remaining])
    )
