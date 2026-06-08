import asyncio
import json
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from src.web_app.agent.runtime import AgentRuntime
from src.web_app.agent.runtime.checkpoint import record_event
from src.web_app.agent.runtime.events import queue_stream_event as _queue_stream_event
from src.web_app.agent.runtime.visible_thoughts import visible_thought_texts
from src.web_app.db.repositories.agent_repository import (
    AgentChatMessageRepository,
    AgentConversationRepository,
    AgentEventRepository,
    AgentRunRepository,
    AgentStepRepository,
)
from src.web_app.services.document_service import document_service
from src.web_app.rag.vector_store import QdrantVectorStore
from src.web_app.rag.embeddings import embed_text
from src.web_app.services.conversation_lock import conversation_lock_manager


GENERIC_COMPLETED_ANSWERS = {
    "",
    "agent completed",
    "agent completed, but no displayable output was returned",
    "agent run completed",
    "agent runtime completed",
}


def load_chat_attachments(db: Session, user_id: int, attachment_ids: list[int]) -> list[dict[str, Any]]:
    if not attachment_ids:
        return []
    from src.web_app.db.repositories.document_repository import DocumentRepository

    repo = DocumentRepository(db)
    attachments: list[dict[str, Any]] = []
    for doc_id in attachment_ids:
        doc = repo.get_by_id_for_user(user_id, doc_id)
        if not doc:
            raise ValueError(f"Attachment document not found: {doc_id}")
        meta = doc.metadata_json or {}
        attachments.append({
            "document_id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "mime_type": meta.get("mime_type", ""),
            "kind": meta.get("kind", "document"),
            "size": meta.get("size", 0),
            "preview_url": f"/api/v1/documents/{doc.id}/file",
            "ingest_status": meta.get("ingest_status", "pending"),
            "status": doc.status,
            "file_path": doc.file_path,
            "source_type": doc.source_type,
        })
    return attachments


def _attachment_snapshot(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": a["document_id"],
            "filename": a["filename"],
            "file_type": a["file_type"],
            "mime_type": a.get("mime_type", ""),
            "kind": a.get("kind", "document"),
            "size": a.get("size", 0),
            "preview_url": a.get("preview_url", ""),
            "ingest_status": a.get("ingest_status", "pending"),
            "status": a.get("status", ""),
        }
        for a in attachments
    ]


IMAGE_DIRECT_KEYWORDS = [
    "分析图片", "分析这张图", "分析这个图", "分析这个图片", "分析一下图片",
    "分析一下这张图", "看图", "看一下图", "看一下这张图", "看看这张图",
    "这张图", "这个图", "这个图片", "图片里", "图里", "图中",
    "截图里", "截图中", "识别图片", "识别这张图", "图片内容", "图像内容",
    "描述图片", "描述这张图", "解释图片", "解释这张图",
    "帮我看看", "帮我分析",
    "what is in this image", "analyze this image", "describe this image",
    "what's in this image", "look at this image",
]

DOCUMENT_CONTEXT_KEYWORDS = [
    "文件", "文档", "pdf", "PDF", "报告", "表格", "excel", "Excel",
    "结合", "对比", "比较", "总结文件", "总结文档",
]


def _has_document_attachments(attachments: list[dict[str, Any]]) -> bool:
    return any(item.get("kind") == "document" for item in attachments)


def _is_direct_image_question(user_input: str, attachments: list[dict[str, Any]]) -> bool:
    has_image = any(item.get("kind") == "image" for item in attachments)
    if not has_image:
        return False

    text = (user_input or "").strip()
    lowered = text.lower()

    # If user explicitly mentions documents, don't short-circuit
    if _has_document_attachments(attachments) and any(kw in lowered for kw in DOCUMENT_CONTEXT_KEYWORDS):
        return False

    if not text:
        return True

    if any(keyword in lowered for keyword in IMAGE_DIRECT_KEYWORDS):
        return True

    # Short user input with images → likely asking about the image itself
    if len(text) <= 30:
        return True

    return False


def _clean_direct_image_answer(text: str) -> str:
    skip_prefixes = (
        "[Image Understanding]",
        "Image:",
        "Description:",
        "Visible text",
        "OCR:",
        "Relevant details",
    )
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if any(stripped.startswith(prefix) for prefix in skip_prefixes):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned or text


async def _build_attachment_context(attachments: list[dict[str, Any]], user_input: str, db: Session, user_id: int) -> str:
    import logging
    _logger = logging.getLogger(__name__)

    if not attachments:
        return ""

    image_attachments = [a for a in attachments if a.get("kind") == "image"]
    document_attachments = [a for a in attachments if a.get("kind") == "document"]
    _logger.info("attachment context: image_count=%s doc_count=%s", len(image_attachments), len(document_attachments))

    parts: list[str] = []

    if image_attachments:
        try:
            from src.web_app.services.qwen_multimodal_service import qwen_multimodal_service
            from src.web_app.core.config import settings as app_settings

            prompt = user_input.strip() if user_input.strip() else "请分析用户上传的图片。"
            images = [
                {"file_path": a["file_path"], "mime_type": a.get("mime_type", "image/png"), "filename": a["filename"]}
                for a in image_attachments
                if a.get("file_path")
            ]
            if images:
                vision_model = getattr(app_settings, "qwen_vision_model", "qwen3.6-plus")
                _logger.info("calling qwen vision model=%s image_count=%s", vision_model, len(images))
                image_context = await qwen_multimodal_service.analyze_images(prompt, images)
                _logger.info("image context received length=%s", len(image_context))
                parts.append(image_context)
            else:
                _logger.warning("image attachments had no valid file_path, skipping vision analysis")
        except Exception as exc:
            _logger.exception("Image analysis failed during attachment context build")
            parts.append(
                "[Image Understanding Warning]\n"
                "图片理解失败，模型没有成功读取用户上传的图片。\n"
                f"错误信息：{exc}\n"
            )

    if document_attachments:
        doc_ids = [a["document_id"] for a in document_attachments]
        # ── Load real document status from DB (not attachment metadata snapshot) ──
        from src.web_app.db.repositories.document_repository import DocumentRepository
        doc_repo = DocumentRepository(db)
        failed_docs: list[str] = []
        pending_docs: list[str] = []
        ready_doc_ids: list[int] = []
        for a in document_attachments:
            did = a["document_id"]
            try:
                doc = doc_repo.get_by_id_for_user(user_id, did)
            except Exception:
                doc = None
            if doc is None:
                failed_docs.append(f"{a['filename']}: document not found in DB")
                continue
            db_status = doc.status
            meta = doc.metadata_json or {}
            ingest_status = meta.get("ingest_status") or db_status
            if db_status == "failed" or ingest_status == "failed":
                error_msg = meta.get("error") or meta.get("error_message") or "unknown error"
                failed_docs.append(f"{a['filename']}: {error_msg}")
                continue
            if ingest_status in ("pending", "processing", "uploaded"):
                pending_docs.append(a["filename"])
                continue
            if ingest_status in ("ingested", "completed", "ready"):
                ready_doc_ids.append(did)
                continue
            # Unknown status — treat as pending
            pending_docs.append(a["filename"])
        if failed_docs and not ready_doc_ids:
            parts.append(
                "[Document Error]\n"
                + "\n".join(f"文档摄入失败：{msg}" for msg in failed_docs)
            )
            return "\n\n".join(parts)
        if pending_docs and not ready_doc_ids:
            parts.append(
                "[Document Status]\n"
                + f"文档正在解析入库中，请稍后再问。待处理：{', '.join(pending_docs)}"
            )
            return "\n\n".join(parts)
        if not ready_doc_ids:
            return "\n\n".join(parts)
        try:
            query_vector = embed_text(user_input)
            results = QdrantVectorStore().search(
                user_id=user_id,
                query_vector=query_vector,
                top_k=min(10, len(doc_ids) * 3),
                min_score=0.1,
                document_ids=doc_ids,
            )
            if results:
                doc_lines: list[str] = ["[Attached Document Context]"]
                doc_results: dict[str, list[dict[str, Any]]] = {}
                for r in results:
                    did = str(r.get("document_id", ""))
                    doc_results.setdefault(did, []).append(r)
                for did, chunks in doc_results.items():
                    for a in document_attachments:
                        if str(a["document_id"]) == did:
                            doc_lines.append(f"\nDocument: {a['filename']}")
                            break
                    for i, chunk in enumerate(chunks[:5], 1):
                        content = chunk.get("content", chunk.get("content_preview", ""))
                        doc_lines.append(f"Chunk {i}:\n{content[:2000]}")
                parts.append("\n".join(doc_lines))
            else:
                # Document was ingested but no chunks matched — warn user
                parts.append(
                    "[Document Status]\n"
                    + "文档已入库，但未检索到相关内容。可能的原因为文档内容与问题不匹配，或索引尚未生效。"
                )
        except Exception as exc:
            import logging
            err_msg = str(exc)
            _logger = logging.getLogger(__name__)
            _logger.exception("Document RAG retrieval failed during attachment context build")
            if "index" in err_msg.lower() and ("missing" in err_msg.lower() or "not found" in err_msg.lower()):
                parts.append(
                    "[Document Error]\n"
                    "Qdrant 缺少 document_id 索引，请运行初始化脚本：\n"
                    "python scripts/ensure_qdrant_indexes.py\n"
                    f"详细错误：{err_msg[:300]}"
                )
            else:
                parts.append(
                    "[Document Warning]\n"
                    f"文档检索失败：{err_msg[:300]}\n"
                    "请稍后重试或联系管理员检查 Qdrant 服务状态。"
                )

    return "\n\n".join(parts)


async def run_agent_async(db: Session, user_id: int, payload: dict[str, Any], stream_queue: asyncio.Queue | None = None) -> dict[str, Any]:
    started_at = datetime.now()
    user_input = payload.get("user_input") or payload.get("input") or payload.get("query") or ""
    payload = {**payload, "user_input": user_input}
    page_context = payload.get("page_context") or {}
    conversation_repo = AgentConversationRepository(db)
    message_repo = AgentChatMessageRepository(db)
    conversation = _get_or_create_conversation(db, user_id, payload, user_input)
    conversation_id = conversation.conversation_id
    selected_feed_card_id = page_context.get("selected_feed_card_id") or page_context.get("feed_card_id")
    selected_feed_card_title = str(page_context.get("selected_feed_card_title") or page_context.get("feed_card_title") or "")
    if selected_feed_card_id and not payload.get("feed_card_id"):
        payload["feed_card_id"] = selected_feed_card_id
    run_repo = AgentRunRepository(db)
    thread_id = conversation.thread_id or f"user:{user_id}:conversation:{conversation_id}"
    if not conversation.thread_id:
        conversation_repo.update(conversation, thread_id=thread_id)
    run = run_repo.create(
        user_id=user_id,
        conversation_id=conversation_id,
        thread_id=thread_id,
        run_type=payload.get("run_type", "agent_runtime"),
        mode=payload.get("mode", "react"),
        status="running",
        user_input=user_input,
        graph_state={"source": payload.get("source", "agent_page"), "page_context": page_context, "thread_id": thread_id, "conversation_id": conversation_id},
    )
    # Load and process attachments
    attachment_ids: list[int] = [int(aid) for aid in (payload.get("attachment_ids") or []) if aid]
    import logging
    _agent_logger = logging.getLogger(__name__)
    _agent_logger.info("agent attachment_ids=%s", attachment_ids)
    attachments_data = load_chat_attachments(db, user_id, attachment_ids) if attachment_ids else []
    _agent_logger.info("loaded chat attachments count=%s kinds=%s", len(attachments_data), [a.get("kind") for a in attachments_data])
    attachment_snapshot = _attachment_snapshot(attachments_data)

    user_message = message_repo.create(
        message_id=str(uuid4()),
        conversation_id=conversation_id,
        user_id=user_id,
        run_id=run.id,
        thread_id=thread_id,
        role="user",
        content=user_input,
        status="completed",
        metadata_json={"source": payload.get("source", "agent_page"), "page_context": page_context, "attachments": attachment_snapshot},
    )
    assistant_message = message_repo.create(
        message_id=str(uuid4()),
        conversation_id=conversation_id,
        user_id=user_id,
        run_id=run.id,
        thread_id=thread_id,
        role="assistant",
        content="",
        status="thinking",
        metadata_json={"source": payload.get("source", "agent_page")},
    )
    record_event(db, run.id, "run_started", {"user_input": user_input, "source": payload.get("source", "agent_page")}, user_id=user_id, thread_id=thread_id)
    _queue_stream_event(
        stream_queue,
        "run_created",
        {
            "run_id": run.id,
            "conversation_id": conversation_id,
            "thread_id": thread_id,
            "user_message": _message_response(user_message),
            "assistant_message": _message_response(assistant_message),
            "conversation": _conversation_response(conversation),
        },
        run_id=run.id,
        thread_id=thread_id,
    )

    # ── Direct image analysis fast path ──
    is_direct_image = _is_direct_image_question(user_input, attachments_data)
    _agent_logger.info("direct image route check: is_direct=%s image_count=%s has_doc=%s input_len=%s",
                       is_direct_image,
                       sum(1 for a in attachments_data if a.get("kind") == "image"),
                       _has_document_attachments(attachments_data),
                       len(user_input.strip()))

    if is_direct_image:
        _agent_logger.info("direct image route matched conversation_id=%s image_count=%s",
                           conversation_id,
                           sum(1 for a in attachments_data if a.get("kind") == "image"))
        image_attachments = [a for a in attachments_data if a.get("kind") == "image"]
        effective_prompt = user_input.strip() or "请分析用户上传的图片。"

        from src.web_app.services.qwen_multimodal_service import qwen_multimodal_service
        from src.web_app.core.config import settings as app_settings

        vision_model = getattr(app_settings, "qwen_vision_model", "qwen3.6-plus")
        _agent_logger.info("direct image answer model=%s", vision_model)

        images = [
            {"file_path": a["file_path"], "mime_type": a.get("mime_type", "image/png"), "filename": a["filename"]}
            for a in image_attachments if a.get("file_path")
        ]

        try:
            answer = await qwen_multimodal_service.answer_image_question(
                prompt=effective_prompt,
                images=images,
                model=vision_model,
            )
            _agent_logger.info("direct image answer length=%s", len(answer or ""))
        except AttributeError:
            _agent_logger.warning("answer_image_question not available, falling back to analyze_images")
            try:
                raw = await qwen_multimodal_service.analyze_images(
                    prompt=effective_prompt,
                    images=images,
                    model=vision_model,
                )
                answer = _clean_direct_image_answer(raw)
                _agent_logger.info("direct image answer (cleaned fallback) length=%s", len(answer or ""))
            except Exception as fallback_exc:
                _agent_logger.exception("direct image understanding failed (fallback)")
                answer = f"我没能成功读取这张图片。错误信息：{fallback_exc}"
        except Exception as exc:
            _agent_logger.exception("direct image understanding failed")
            answer = f"我没能成功读取这张图片。可能是图片过大、格式不受支持，或视觉模型调用失败。\n\n错误信息：{exc}"

        elapsed_ms = max(0, int((datetime.now() - started_at).total_seconds() * 1000))

        # Persist assistant message
        message_repo.update(
            assistant_message,
            content=answer,
            status="completed",
            elapsed_ms=elapsed_ms,
            metadata_json={
                "run_id": run.id,
                "direct_image_answer": True,
                "model": vision_model,
                "attachments": attachment_snapshot,
                "visible_thoughts": [],
            },
        )

        # Update run
        run_repo.update(
            run,
            status="completed",
            result_summary=answer,
            final_answer=answer,
            final_response={"answer": answer, "direct_image_answer": True},
            langgraphstatus_json={
                "run_id": run.id,
                "thread_id": thread_id,
                "conversation_id": conversation_id,
                "status": "completed",
                "phase": "completed",
                "elapsed_ms": elapsed_ms,
                "steps": [],
                "visible_thoughts": [],
            },
            elapsed_ms=elapsed_ms,
            completed_at=datetime.now(),
        )

        # Touch conversation
        conversation_repo.touch(
            conversation,
            preview=answer,
            last_run_id=run.id,
            selected_feed_card_id=int(selected_feed_card_id) if str(selected_feed_card_id or "").isdigit() else None,
            selected_feed_card_title=selected_feed_card_title or None,
        )

        # Emit SSE events
        await _stream_answer_deltas(db, stream_queue, run.id, thread_id, user_id, answer)
        record_event(db, run.id, "final_response_created", {"answer": answer, "answer_len": len(answer)}, user_id=user_id, thread_id=thread_id)
        record_event(db, run.id, "run_completed", {"status": "completed", "answer": answer}, user_id=user_id, thread_id=thread_id)
        _queue_stream_event(stream_queue, "final_response_created", {"answer": answer, "answer_len": len(answer)}, run_id=run.id, thread_id=thread_id)

        state_for_response = {
            "conversation_id": conversation_id,
            "thread_id": thread_id,
            "status": "completed",
            "answer": answer,
            "final_answer": answer,
            "final_payload": {"answer": answer, "direct_image_answer": True},
            "visible_thoughts": [],
            "langgraphstatus": {"status": "completed", "steps": [], "elapsed_ms": elapsed_ms},
        }
        response = _run_response(run.id, state_for_response, conversation=conversation, user_message=user_message, assistant_message=assistant_message, elapsed_ms=elapsed_ms)
        _queue_stream_event(stream_queue, "run_completed", {"status": "completed", "answer": answer, "response": response}, run_id=run.id, thread_id=thread_id)
        await asyncio.sleep(0)  # yield so consumer drains queue before sentinel arrives
        return response

    # ── Pre-flight: check document attachments status before expensive Agent run ──
    document_attachments_for_guard = [a for a in attachments_data if a.get("kind") == "document"]
    if document_attachments_for_guard:
        from src.web_app.db.repositories.document_repository import DocumentRepository as _DocRepo
        _doc_repo = _DocRepo(db)
        _failed_msgs: list[str] = []
        _pending_msgs: list[str] = []
        _has_ready = False
        for a in document_attachments_for_guard:
            did = a["document_id"]
            doc = _doc_repo.get_by_id_for_user(user_id, did)
            if doc is None:
                _failed_msgs.append(f"{a['filename']}: document not found")
                continue
            db_status = doc.status
            meta = doc.metadata_json or {}
            ingest_status = meta.get("ingest_status") or db_status
            if db_status == "failed" or ingest_status == "failed":
                _err = meta.get("error") or meta.get("error_message") or "unknown error"
                _failed_msgs.append(f"{a['filename']}: {_err}")
            elif ingest_status in ("pending", "processing", "uploaded"):
                _pending_msgs.append(a["filename"])
            else:
                _has_ready = True
        if _failed_msgs and not _has_ready:
            fast_fail_answer = "文档解析失败：" + "；".join(_failed_msgs)
            message_repo.update(assistant_message, content=fast_fail_answer, status="completed", elapsed_ms=0)
            await _stream_answer_deltas(db, stream_queue, run.id, thread_id, user_id, fast_fail_answer)
            run_repo.update(run, status="completed", result_summary=fast_fail_answer, final_answer=fast_fail_answer,
                           elapsed_ms=0, completed_at=datetime.now())
            conversation_repo.touch(conversation, preview=fast_fail_answer, last_run_id=run.id)
            _queue_stream_event(stream_queue, "run_completed", {"status": "completed", "answer": fast_fail_answer}, run_id=run.id, thread_id=thread_id)
            return _run_response(run.id, {"status": "completed", "answer": fast_fail_answer, "final_output": fast_fail_answer},
                                conversation=conversation, user_message=user_message, assistant_message=assistant_message, elapsed_ms=0)
        if _pending_msgs and not _has_ready:
            pending_answer = f"文档正在解析入库中，请稍后再问。待处理：{', '.join(_pending_msgs)}"
            message_repo.update(assistant_message, content=pending_answer, status="completed", elapsed_ms=0)
            await _stream_answer_deltas(db, stream_queue, run.id, thread_id, user_id, pending_answer)
            run_repo.update(run, status="completed", result_summary=pending_answer, final_answer=pending_answer,
                           elapsed_ms=0, completed_at=datetime.now())
            conversation_repo.touch(conversation, preview=pending_answer, last_run_id=run.id)
            _queue_stream_event(stream_queue, "run_completed", {"status": "completed", "answer": pending_answer}, run_id=run.id, thread_id=thread_id)
            return _run_response(run.id, {"status": "completed", "answer": pending_answer, "final_output": pending_answer},
                                conversation=conversation, user_message=user_message, assistant_message=assistant_message, elapsed_ms=0)

    # ── Per-conversation lock: same conversation serialises, different ones run concurrently ──
    lock = await conversation_lock_manager.acquire(conversation_id)
    await lock.acquire()
    try:
        try:
            # Build attachment context and inject into payload
            attachment_context = await _build_attachment_context(attachments_data, user_input, db, user_id)
            _agent_logger.info("final attachment_context length=%s has_context=%s", len(attachment_context or ""), bool(attachment_context))
            enriched_payload = dict(payload)
            if attachment_context:
                enriched_payload["attachment_context"] = attachment_context
                page_context = dict(page_context)
                page_context["attachment_context"] = attachment_context
                enriched_payload["page_context"] = page_context

            state = await AgentRuntime(db, enriched_payload).run({"user_id": user_id, "run_id": run.id, "thread_id": thread_id, "conversation_id": conversation_id, "user_input": user_input, "mode": run.mode, "source": payload.get("source", "agent_page"), "page_context": page_context, "_stream_queue": stream_queue, "_answer_started_emitted": False, "_answer_delta_emitted": False, "_answer_completed_emitted": False})
        except Exception as exc:
            state = {
                "user_id": user_id,
                "run_id": run.id,
                "thread_id": thread_id,
                "conversation_id": conversation_id,
                "user_input": user_input,
                "status": "failed",
                "error": str(exc),
                "errors": [str(exc)],
                "final_output": "",
                "langgraphstatus": {
                    "run_id": run.id,
                    "thread_id": thread_id,
                    "conversation_id": conversation_id,
                    "status": "failed",
                    "phase": "failed",
                    "summary": "Agent runtime failed before producing a final answer.",
                    "steps": [],
                },
            }
        elapsed_ms = max(0, int((datetime.now() - started_at).total_seconds() * 1000))
        answer = build_user_facing_answer(state)
        state["answer"] = answer
        state["final_answer"] = answer
        state["final_output"] = answer
        final_payload = dict(state.get("final_payload") or {})
        final_payload["answer"] = answer
        final_payload.setdefault("thinking_summary", visible_thought_texts(state))
        final_payload.setdefault("visible_thoughts", state.get("visible_thoughts", []))
        final_payload.setdefault("run_id", str(run.id))
        final_payload.setdefault("thread_id", thread_id)
        final_payload.setdefault("conversation_id", conversation_id)
        state["final_payload"] = final_payload
        langgraphstatus = dict(state.get("langgraphstatus") or {})
        langgraphstatus.update(
            {
                "run_id": run.id,
                "thread_id": thread_id,
                "conversation_id": conversation_id,
                "status": state.get("status", "completed"),
                "phase": "completed" if state.get("status", "completed") == "completed" else state.get("status", "completed"),
                "elapsed_ms": elapsed_ms,
            }
        )
        langgraphstatus.setdefault("visible_thoughts", state.get("visible_thoughts", []))
        state["langgraphstatus"] = langgraphstatus
        steps = list(langgraphstatus.get("steps") or [])
        completed_at = datetime.now() if state.get("status") in {"completed", "failed", "waiting_approval"} else None
        run_repo.update(
            run,
            status=state.get("status", "completed"),
            graph_state=_json_safe(_state_for_storage(state)),
            result_summary=answer,
            final_answer=answer,
            final_response=_json_safe(final_payload),
            langgraphstatus_json=_json_safe(langgraphstatus),
            elapsed_ms=elapsed_ms,
            error_message=state.get("error", ""),
            completed_at=completed_at,
        )
        message_repo.update(
            assistant_message,
            content=answer,
            status="failed" if state.get("status") == "failed" else "completed" if state.get("status") == "completed" else state.get("status", "completed"),
            elapsed_ms=elapsed_ms,
            langgraphstatus_json=_json_safe(langgraphstatus),
            steps_json=_json_safe(steps),
            error_message=state.get("error", ""),
            metadata_json={"run_id": run.id, "final_response": _json_safe(final_payload), "visible_thoughts": _json_safe(state.get("visible_thoughts", []))},
        )
        conversation_repo.touch(
            conversation,
            preview=answer,
            last_run_id=run.id,
            selected_feed_card_id=int(selected_feed_card_id) if str(selected_feed_card_id or "").isdigit() else None,
            selected_feed_card_title=selected_feed_card_title or None,
        )
        already_streamed = state.get("_answer_delta_emitted", False)
        if state.get("approval_required") or state.get("status") == "waiting_approval":
            await _stream_answer_deltas(db, stream_queue, run.id, thread_id, user_id, answer, already_streamed=already_streamed)
            # Build rich approval_required payload with preview for frontend
            approval_payload = _build_approval_sse_payload(state, run.id)
            record_event(db, run.id, "approval_required", approval_payload, user_id=user_id, thread_id=thread_id)
            _queue_stream_event(stream_queue, "approval_required", approval_payload, run_id=run.id, thread_id=thread_id)
            # Emit run_paused to confirm the run is waiting
            _queue_stream_event(stream_queue, "run_paused", {"status": "waiting_approval", "approval_id": approval_payload.get("approval_id")}, run_id=run.id, thread_id=thread_id)
        elif state.get("status") == "failed":
            record_event(db, run.id, "run_failed", {"status": state.get("status"), "answer": answer, "error": state.get("error", "")}, user_id=user_id, thread_id=thread_id)
            _queue_stream_event(stream_queue, "run_failed", {"status": state.get("status"), "answer": answer, "error": state.get("error", "")}, run_id=run.id, thread_id=thread_id)
        else:
            await _stream_answer_deltas(db, stream_queue, run.id, thread_id, user_id, answer, already_streamed=already_streamed)
            record_event(db, run.id, "final_response_created", {"answer": answer, "answer_len": len(answer)}, user_id=user_id, thread_id=thread_id)
            record_event(db, run.id, "run_completed", {"status": state.get("status"), "answer": answer}, user_id=user_id, thread_id=thread_id)
            _queue_stream_event(stream_queue, "final_response_created", {"answer": answer, "answer_len": len(answer)}, run_id=run.id, thread_id=thread_id)
        response = _run_response(run.id, state, conversation=conversation, user_message=user_message, assistant_message=assistant_message, elapsed_ms=elapsed_ms)
        _queue_stream_event(stream_queue, "run_completed", {"status": state.get("status"), "answer": answer, "response": response}, run_id=run.id, thread_id=thread_id)
        return response
    finally:
        lock.release()
        await conversation_lock_manager.release(conversation_id)


def run_agent(db: Session, user_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(run_agent_async(db, user_id, payload))


async def stream_agent_run(db: Session, user_id: int, payload: dict[str, Any]):
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    async def runner() -> None:
        try:
            await run_agent_async(db, user_id, payload, stream_queue=queue)
        except Exception as exc:
            queue.put_nowait(
                {
                    "event": "run_failed",
                    "data": {
                        "event_type": "run_failed",
                        "payload": {"status": "failed", "error": str(exc)},
                    },
                }
            )
        finally:
            queue.put_nowait(sentinel)

    task = asyncio.create_task(runner())
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item
            event_type = str((item.get("data") or {}).get("event_type") or item.get("event") or "")
            if event_type == "visible_thought_delta":
                await asyncio.sleep(0.08)
            elif event_type == "answer_delta":
                await asyncio.sleep(0.008)
    finally:
        if not task.done():
            task.cancel()


def get_run(db: Session, user_id: int, run_id: int) -> dict[str, Any]:
    run = AgentRunRepository(db).get_by_user(user_id, run_id)
    if not run:
        raise ValueError("AgentRun not found")
    final_response = run.final_response or {}
    graph_state = run.graph_state or {}
    return {
        "id": run.id,
        "status": run.status,
        "run_type": run.run_type,
        "mode": run.mode,
        "user_input": run.user_input,
        "result_summary": run.result_summary,
        "answer": run.final_answer or run.result_summary,
        "final_answer": run.final_answer,
        "final_response": final_response,
        "visible_thoughts": final_response.get("visible_thoughts") or graph_state.get("visible_thoughts", []),
        "thinking_summary": final_response.get("thinking_summary") or visible_thought_texts(graph_state),
        "langgraphstatus": run.langgraphstatus_json or graph_state.get("langgraphstatus", {}),
        "elapsed_ms": run.elapsed_ms,
        "error_message": run.error_message,
        "graph_state": run.graph_state or {},
        "conversation_id": run.conversation_id or graph_state.get("conversation_id", ""),
        "thread_id": run.thread_id or graph_state.get("thread_id", ""),
    }


def list_steps(db: Session, user_id: int, run_id: int) -> list[dict[str, Any]]:
    if not AgentRunRepository(db).get_by_user(user_id, run_id):
        raise ValueError("AgentRun not found")
    return [{"id": step.id, "node_name": step.node_name, "status": step.status, "input": step.input, "output": step.output} for step in AgentStepRepository(db).list_by_run(run_id)]


def list_events(db: Session, user_id: int, run_id: int) -> list[dict[str, Any]]:
    if not AgentRunRepository(db).get_by_user(user_id, run_id):
        raise ValueError("AgentRun not found")
    rows = AgentEventRepository(db).list_by_run(user_id, run_id)
    if rows:
        return [_event_to_sse(item) for item in rows]
    return [{"event": "step", "data": {"run_id": run_id, **step}} for step in list_steps(db, user_id, run_id)]


def create_conversation(db: Session, user_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    item = _create_conversation(db, user_id, payload, payload.get("title") or "")
    return _conversation_response(item, messages=[])


def list_conversations(db: Session, user_id: int, status: str = "active", limit: int = 50, offset: int = 0) -> dict[str, Any]:
    items = AgentConversationRepository(db).list_by_user(user_id, status=status, limit=limit, offset=offset)
    return {"items": [_conversation_response(item) for item in items]}


def get_conversation(db: Session, user_id: int, conversation_id: str) -> dict[str, Any]:
    conversation = _require_conversation(db, user_id, conversation_id)
    messages = AgentChatMessageRepository(db).list_by_conversation(user_id, conversation_id)
    return _conversation_response(conversation, messages=messages)


def update_conversation(db: Session, user_id: int, conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    repo = AgentConversationRepository(db)
    conversation = _require_conversation(db, user_id, conversation_id)
    values: dict[str, Any] = {}
    if "title" in payload:
        values["title"] = str(payload.get("title") or "")[:255]
    if "selected_feed_card_id" in payload:
        values["selected_feed_card_id"] = payload.get("selected_feed_card_id")
    if "selected_feed_card_title" in payload:
        values["selected_feed_card_title"] = str(payload.get("selected_feed_card_title") or "")[:512]
    if "metadata" in payload or "metadata_json" in payload:
        values["metadata_json"] = payload.get("metadata_json") or payload.get("metadata") or {}
    if values:
        conversation = repo.update(conversation, **values)
    return _conversation_response(conversation, messages=AgentChatMessageRepository(db).list_by_conversation(user_id, conversation_id))


def archive_conversation(db: Session, user_id: int, conversation_id: str) -> dict[str, Any]:
    conversation = _require_conversation(db, user_id, conversation_id)
    conversation = AgentConversationRepository(db).update(conversation, status="archived")
    return _conversation_response(conversation)


def delete_conversation(db: Session, user_id: int, conversation_id: str) -> dict[str, Any]:
    conversation = _require_conversation(db, user_id, conversation_id)
    conversation = AgentConversationRepository(db).update(conversation, status="deleted")
    return _conversation_response(conversation)


def clear_conversation(db: Session, user_id: int, conversation_id: str) -> dict[str, Any]:
    conversation_repo = AgentConversationRepository(db)
    conversation = _require_conversation(db, user_id, conversation_id)
    removed = AgentChatMessageRepository(db).clear_conversation(user_id, conversation_id)
    conversation = conversation_repo.update(conversation, message_count=0, last_message_preview="")
    return {"conversation": _conversation_response(conversation), "cleared_messages": removed}


def hard_delete_conversation(db: Session, user_id: int, conversation_id: str) -> dict[str, Any]:
    repo = AgentConversationRepository(db)
    try:
        doc_ids = repo.get_conversation_document_ids(user_id, conversation_id)
    except Exception:
        doc_ids = set()
    deleted = repo.hard_delete(user_id, conversation_id)
    if not deleted:
        raise ValueError("Agent conversation not found")
    # ── Clean up Qdrant vectors ───────────────────────────────────
    deleted_qdrant_docs = 0
    deleted_qdrant_memories = 0
    if doc_ids:
        try:
            store = QdrantVectorStore()
            for did in doc_ids:
                store.delete_document(user_id, did)
            deleted_qdrant_docs = len(doc_ids)
        except Exception:
            pass  # Qdrant may be unavailable; PG data is already gone
    try:
        from src.web_app.memory.qdrant_memory_store import QdrantMemoryStore
        from src.web_app.core.config import settings
        if settings.qdrant_url:
            mem_store = QdrantMemoryStore()
            mem_store.ensure_collection()
            indexed = mem_store.list_indexed_memory_ids(user_id=user_id, memory_types=["working"])
            for mid in indexed:
                try:
                    mem_store.delete_by_memory_id(mid)
                    deleted_qdrant_memories += 1
                except Exception:
                    pass
    except Exception:
        pass
    return {
        "conversation_id": conversation_id,
        "deleted_records": deleted,
        "deleted_qdrant_documents": deleted_qdrant_docs,
        "deleted_qdrant_memories": deleted_qdrant_memories,
    }


def extract_user_visible_answer(value: Any) -> str:
    """Extract a user-readable string from any value.

    Handles nested dicts (final_payload), JSON strings, and raw text.
    Never returns a JSON blob that should stay internal.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        # Detect JSON strings that look like internal payloads
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return extract_user_visible_answer(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
        return value
    if isinstance(value, dict):
        # Extract the user-facing field from internal payload dicts
        for key in ("answer", "final_answer", "final_output", "content", "message", "text"):
            val = value.get(key)
            if isinstance(val, str) and val.strip():
                return val
        # If the dict looks like a final_payload, return empty rather than JSON
        if any(k in value for k in ("status", "route", "artifacts", "intent")):
            return ""
        # For other dicts without extractable text, return empty
        return ""
    if isinstance(value, (list, tuple)):
        # Don't stringify lists — they're not user-readable
        return ""
    return str(value)


def build_user_facing_answer(state: dict[str, Any]) -> str:
    final_payload = state.get("final_payload") or {}
    candidates = [
        state.get("answer"),
        final_payload.get("answer") if isinstance(final_payload, dict) else "",
        state.get("final_answer"),
        state.get("final_output"),
    ]
    for candidate in candidates:
        text = extract_user_visible_answer(candidate)
        if text and not _is_generic_completed_answer(text):
            return text

    status = state.get("status", "completed")
    errors = [str(item) for item in state.get("errors", []) if item] or ([str(state.get("error"))] if state.get("error") else [])
    if status == "failed" or errors:
        reason = errors[0] if errors else "runtime returned a failure state"
        return f"Run failed: {reason}. You can retry or ask me to inspect this Agent Run."

    route_plan = state.get("route_plan") or {}
    home_intent = state.get("home_intent") or {}
    intent = str(route_plan.get("intent") or home_intent.get("intent") or state.get("route") or "chat")
    risk_level = str(route_plan.get("risk_level") or home_intent.get("risk_level") or "L0")
    if status == "waiting_approval" or route_plan.get("needs_approval") or home_intent.get("needs_approval"):
        return f"Approval required: this is a {risk_level} risk action and must be approved before execution. I have not performed any external write or irreversible operation."

    # \u2500\u2500 Memory write: confirm the save \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    mem_result = state.get("memory_write_result") or {}
    if mem_result.get("success"):
        content = mem_result.get("content", "")
        return f"\u5df2\u8bb0\u4f4f\uff1a{content}"

    user_input = str(state.get("user_input") or "").strip()
    if _looks_like_greeting(user_input):
        return "\u4f60\u597d\uff0c\u6211\u662f\u4fe1\u606f\u5dee Agent OS \u52a9\u624b\u3002\u4f60\u53ef\u4ee5\u8ba9\u6211\u7814\u7a76\u4fe1\u606f\u3001\u751f\u6210\u6210\u679c\u6216\u6c89\u6dc0 Skill\u3002"
    if intent == "research":
        return "\u6211\u5df2\u8bc6\u522b\u8fd9\u662f\u7814\u7a76\u4efb\u52a1\uff0c\u5e76\u5b8c\u6210\u4e86\u521d\u6b65\u89c4\u5212\u3002\u5f53\u524d\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u5b8c\u6574\u7814\u7a76\u7ed3\u679c\u3002"
    if intent == "artifact":
        return "\u6211\u5df2\u8bc6\u522b\u8fd9\u662f\u6210\u679c\u751f\u6210\u4efb\u52a1\uff0c\u4f46\u5f53\u524d\u8fd8\u6ca1\u6709\u751f\u6210\u5b9e\u9645 Artifact\u3002"
    if intent == "tool":
        return "\u6211\u5df2\u8bc6\u522b\u8fd9\u662f\u5de5\u5177\u76f8\u5173\u4efb\u52a1\u3002\u5982\u679c\u6d89\u53ca\u5916\u90e8\u5199\u5165\u6216\u9ad8\u98ce\u9669\u52a8\u4f5c\uff0c\u4f1a\u5148\u8fdb\u5165\u5ba1\u6279\u72b6\u6001\u3002"

    return "\u6211\u5df2\u7ecf\u5b8c\u6210\u57fa\u7840\u5224\u65ad\u3002\u4f60\u53ef\u4ee5\u7ee7\u7eed\u8865\u5145\u76ee\u6807\uff0c\u6211\u4f1a\u6cbf\u7528\u5f53\u524d\u4f1a\u8bdd\u4e0a\u4e0b\u6587\u3002"

async def _stream_answer_deltas(db: Session, queue: asyncio.Queue | None, run_id: int, thread_id: str, user_id: int, answer: str, already_streamed: bool = False) -> None:
    if not queue:
        return
    # If the runtime already streamed answer_delta + answer_completed during
    # LLM generation, skip the fallback entirely — no duplicate events.
    if already_streamed:
        return
    # Fallback: runtime did not stream (e.g. final_response skipped, or LLM disabled).
    _queue_stream_event(queue, "answer_started", {}, run_id=run_id, thread_id=thread_id, node_name="final_response")
    await asyncio.sleep(0)  # yield so consumer sends answer_started before chunks arrive
    chunks = _chunk_answer_text(str(answer or ""))
    if not chunks:
        chunks = [answer]
    for index, chunk in enumerate(chunks, start=1):
        payload = {"text": chunk, "index": index}
        record_event(db, run_id, "answer_delta", payload, node_name="final_response", user_id=user_id, thread_id=thread_id)
        _queue_stream_event(queue, "answer_delta", payload, run_id=run_id, thread_id=thread_id, node_name="final_response")
        await asyncio.sleep(0)
    await asyncio.sleep(0)  # yield so consumer sends last answer_delta before completed
    completed_payload = {"answer": answer}
    record_event(db, run_id, "answer_completed", completed_payload, node_name="final_response", user_id=user_id, thread_id=thread_id)
    _queue_stream_event(queue, "answer_completed", completed_payload, run_id=run_id, thread_id=thread_id, node_name="final_response")
    await asyncio.sleep(0)  # yield so consumer sends answer_completed before caller queues more events


def _chunk_answer_text(text: str, max_chunk: int = 200) -> list[str]:
    """Split answer text into semantic chunks: paragraphs first, then sentences if needed."""
    # Split by double newlines (paragraphs) first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text]
    result: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chunk:
            result.append(para)
        else:
            # Split long paragraphs by sentence boundaries
            current = ""
            for char in para:
                current += char
                if char in "。！？!?；;" or len(current) >= max_chunk:
                    stripped = current.strip()
                    if stripped:
                        result.append(stripped)
                    current = ""
            remaining = current.strip()
            if remaining:
                result.append(remaining)
    return result


def _run_response(
    run_id: int,
    state: dict[str, Any],
    *,
    conversation=None,
    user_message=None,
    assistant_message=None,
    elapsed_ms: int = 0,
) -> dict[str, Any]:
    route_plan = state.get("route_plan") or {}
    answer = build_user_facing_answer(state)
    final_response = dict(state.get("final_payload") or {})
    final_response["answer"] = answer
    final_response.setdefault("thinking_summary", visible_thought_texts(state))
    final_response.setdefault("visible_thoughts", state.get("visible_thoughts", []))
    return {
        "run_id": run_id,
        "conversation_id": state.get("conversation_id", ""),
        "thread_id": state.get("thread_id", ""),
        "status": state.get("status", "completed"),
        "elapsed_ms": elapsed_ms,
        "answer": answer,
        "route": state.get("route"),
        "intent": route_plan.get("intent", state.get("route")),
        "route_plan": route_plan.get("route", []),
        "risk_level": route_plan.get("risk_level", "L0"),
        "final_output": answer,
        "final_answer": answer,
        "final_response": final_response,
        "final_payload": final_response,
        "visible_thoughts": state.get("visible_thoughts", []),
        "thinking_summary": visible_thought_texts(state),
        "langgraphstatus": state.get("langgraphstatus", {}),
        "user_message": _message_response(user_message) if user_message else None,
        "assistant_message": _message_response(assistant_message) if assistant_message else None,
        "conversation": _conversation_response(conversation) if conversation else None,
        "research": state.get("research", {}) or state.get("research_result", {}),
        "rag": state.get("rag", {}) or state.get("rag_result", {}),
        "artifacts": state.get("artifacts", []),
        "memory_updates": state.get("memory_updates", []),
        "skill_drafts": state.get("skill_drafts", []),
        "matched_skill": state.get("matched_skill"),
        "candidate_skills": state.get("candidate_skills", []),
        "created_skill_draft": state.get("created_skill_draft"),
        "reusable_score": (state.get("skill_reuse") or {}).get("reusable_score", 0),
        "tool_call": state.get("tool_call", {}),
        "tool_result": state.get("tool_result"),
        "evaluation": state.get("evaluation", {}),
        "errors": state.get("errors", []),
        "error": state.get("error", ""),
        "approval_required": state.get("approval_required", False),
        "approval_payload": state.get("approval_payload"),
        "agent_outputs": state.get("agent_outputs", []),
    }


def _get_or_create_conversation(db: Session, user_id: int, payload: dict[str, Any], user_input: str):
    conversation_id = str(payload.get("conversation_id") or "").strip()
    if conversation_id:
        return _require_conversation(db, user_id, conversation_id)
    return _create_conversation(db, user_id, payload, user_input)


def _create_conversation(db: Session, user_id: int, payload: dict[str, Any], title_seed: str):
    conversation_id = str(uuid4())
    thread_id = f"user:{user_id}:conversation:{conversation_id}"
    page_context = payload.get("page_context") or {}
    selected_feed_card_id = page_context.get("selected_feed_card_id") or page_context.get("feed_card_id")
    try:
        selected_feed_card_id = int(selected_feed_card_id) if selected_feed_card_id else None
    except (TypeError, ValueError):
        selected_feed_card_id = None
    selected_feed_card_title = str(page_context.get("selected_feed_card_title") or page_context.get("feed_card_title") or "")
    title = str(payload.get("title") or title_seed or "New Agent Run").strip()
    return AgentConversationRepository(db).create(
        conversation_id=conversation_id,
        user_id=user_id,
        title=_short_title(title),
        source=payload.get("source", "agent_page"),
        status="active",
        thread_id=thread_id,
        selected_feed_card_id=selected_feed_card_id,
        selected_feed_card_title=selected_feed_card_title[:512],
        metadata_json={"page_context": page_context},
        last_message_preview="",
        message_count=0,
    )


def _require_conversation(db: Session, user_id: int, conversation_id: str):
    conversation = AgentConversationRepository(db).get_by_conversation_id(user_id, conversation_id)
    if not conversation:
        raise ValueError("Agent conversation not found")
    return conversation


def _short_title(value: str) -> str:
    title = " ".join(value.split())
    if not title:
        return "New Agent Run"
    return title[:42]


def _looks_like_greeting(value: str) -> bool:
    lowered = value.lower()
    return lowered in {"hi", "hello", "hey"} or any(token in value for token in ("\u4f60\u597d", "\u60a8\u597d", "\u4f60\u662f\u8c01", "\u4f60\u662f\u8ab0"))


def _is_generic_completed_answer(value: str) -> bool:
    normalized = value.strip().rstrip(".\u3002").lower()
    if normalized in GENERIC_COMPLETED_ANSWERS:
        return True
    return normalized.startswith("\u5df2\u5b8c\u6210\u672c\u6b21 agent run") or normalized.startswith("\u5df2\u5b8c\u6210\u672c\u6b21 agent")

def _conversation_response(item, messages: list[Any] | None = None) -> dict[str, Any]:
    if not item:
        return {}
    data = {
        "id": item.id,
        "conversation_id": item.conversation_id,
        "thread_id": item.thread_id,
        "title": item.title,
        "source": item.source,
        "status": item.status,
        "selected_feed_card_id": item.selected_feed_card_id,
        "selected_feed_card_title": item.selected_feed_card_title,
        "metadata": item.metadata_json or {},
        "last_message_preview": item.last_message_preview,
        "last_run_id": item.last_run_id,
        "message_count": item.message_count,
        "created_at": item.created_at.isoformat() if item.created_at else "",
        "updated_at": item.updated_at.isoformat() if item.updated_at else "",
        "last_active_at": item.last_active_at.isoformat() if item.last_active_at else "",
    }
    if messages is not None:
        data["messages"] = [_message_response(message) for message in messages]
    return data


def _message_response(item) -> dict[str, Any]:
    if not item:
        return {}
    meta = item.metadata_json or {}
    return {
        "id": item.id,
        "message_id": item.message_id,
        "conversation_id": item.conversation_id,
        "run_id": item.run_id,
        "thread_id": item.thread_id,
        "role": item.role,
        "content": item.content,
        "status": item.status,
        "elapsed_ms": item.elapsed_ms,
        "langgraphstatus": item.langgraphstatus_json or {},
        "steps": item.steps_json or [],
        "error_message": item.error_message,
        "metadata": meta,
        "attachments": meta.get("attachments", []),
        "created_at": item.created_at.isoformat() if item.created_at else "",
        "updated_at": item.updated_at.isoformat() if item.updated_at else "",
    }


def _event_to_sse(item) -> dict[str, Any]:
    return {
        "event": item.event_type,
        "data": {
            "id": item.id,
            "run_id": item.run_id,
            "thread_id": item.thread_id,
            "user_id": item.user_id,
            "event_type": item.event_type,
            "node_name": item.node_name,
            "payload": item.payload_json or {},
            "created_at": item.created_at.isoformat() if item.created_at else "",
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _state_for_storage(state: dict[str, Any]) -> dict[str, Any]:
    data = dict(state)
    data.pop("_stream_queue", None)
    return data


def _build_approval_sse_payload(state: dict[str, Any], run_id: int) -> dict[str, Any]:
    """Build a rich SSE payload for the approval_required event."""
    approval_payload = state.get("approval_payload") or {}
    tool_call = state.get("tool_call") or {}
    route_plan = state.get("route_plan") or {}

    return {
        "run_id": run_id,
        "approval_id": approval_payload.get("approval_id"),
        "risk_level": approval_payload.get("risk_level") or route_plan.get("risk_level", "L3"),
        "tool_name": approval_payload.get("tool_name") or tool_call.get("tool_name", ""),
        "title": approval_payload.get("title") or f"需要你确认操作",
        "preview": approval_payload.get("preview") or tool_call.get("output", {}),
        "tool_args": approval_payload.get("tool_args", {}),
        "actions": ["approve", "reject"],
        "safety_notes": approval_payload.get("safety_notes", []),
        "status": "waiting_approval",
        "user_id": state.get("user_id"),
    }
