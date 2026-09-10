"""Single-process serialization for deletion and read/modify/write boundaries."""
from threading import RLock
from contextlib import contextmanager
from sqlalchemy import select, inspect
from src.web_app.models.orm import AgentConversation, ConversationDeletionTask

asset_lock = RLock()


def guarded_transition(fn):
    from functools import wraps
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with asset_lock:
            return fn(*args, **kwargs)
    return wrapped


class ConversationDeletingError(RuntimeError):
    pass


def check_conversation(db, user_id, conversation_id, *, require_exists=False):
    if not conversation_id:
        return
    row = db.scalar(select(AgentConversation).where(AgentConversation.user_id == user_id,
        AgentConversation.conversation_id == conversation_id).execution_options(populate_existing=True))
    if row is None and not require_exists and inspect(db.get_bind()).has_table(ConversationDeletionTask.__tablename__):
        require_exists = db.scalar(select(ConversationDeletionTask.id).where(
            ConversationDeletionTask.user_id == user_id,
            ConversationDeletionTask.conversation_id == conversation_id).limit(1)) is not None
    if (row is None and require_exists) or (row is not None and row.status in {"deleting", "deleted"}):
        raise ConversationDeletingError("CONVERSATION_DELETING: 会话正在删除或已删除，不能继续写入。")


@contextmanager
def conversation_write(db, user_id, conversation_id, *, require_exists=False):
    with asset_lock:
        check_conversation(db, user_id, conversation_id, require_exists=require_exists)
        yield
