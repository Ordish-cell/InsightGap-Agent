from src.web_app.memory.base import BaseMemoryStore


class WorkingMemoryStore(BaseMemoryStore):
    ttl_seconds = 3600
