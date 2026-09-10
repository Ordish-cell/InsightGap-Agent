"""Conversation-scoped attachment inventory; snapshots establish membership only."""
from sqlalchemy import select
from src.web_app.models.orm import AgentChatMessage, Document


def load_file_context(bind, user_id, conversation_id):
    from sqlalchemy.orm import Session
    with Session(bind=bind) as db:
        return conversation_files(db, user_id, conversation_id), file_discussion(db, user_id, conversation_id)


def conversation_files(db, user_id, conversation_id):
    files, cursor = {}, 0
    while True:
        rows = db.execute(select(AgentChatMessage.id, AgentChatMessage.message_id, AgentChatMessage.metadata_json).where(
            AgentChatMessage.user_id == user_id, AgentChatMessage.conversation_id == conversation_id,
            AgentChatMessage.role == "user", AgentChatMessage.id > cursor).order_by(AgentChatMessage.id).limit(100)).all()
        if not rows:
            break
        for row in rows:
            cursor = row.id
            for attachment in (row.metadata_json or {}).get("attachments", []):
                if not isinstance(attachment, dict) or attachment.get("kind", "document") != "document":
                    continue
                document_id = attachment.get("document_id")
                if not isinstance(document_id, int):
                    continue
                files[document_id] = {"document_id": document_id, "filename": attachment.get("filename", ""),
                    "source_message_id": row.message_id, "source_message_order": row.id, "status": "unavailable"}
    ids = list(files)
    for offset in range(0, len(ids), 100):
        for doc in db.scalars(select(Document).where(Document.user_id == user_id, Document.id.in_(ids[offset:offset + 100]))):
            status = (doc.metadata_json or {}).get("ingest_status") or doc.status
            files[doc.id].update(filename=doc.filename, status=doc.status if doc.status in {"failed", "deleting"} else status)
    return sorted(files.values(), key=lambda item: (item["source_message_order"], item["document_id"]))


def file_discussion(db, user_id, conversation_id):
    rows = db.execute(select(AgentChatMessage.metadata_json).where(AgentChatMessage.user_id == user_id,
        AgentChatMessage.conversation_id == conversation_id, AgentChatMessage.role == "assistant",
        AgentChatMessage.status == "completed").order_by(AgentChatMessage.id.desc()).limit(20)).scalars()
    for metadata in rows:
        if (metadata or {}).get("file_context"):
            return metadata["file_context"]
    return {}
