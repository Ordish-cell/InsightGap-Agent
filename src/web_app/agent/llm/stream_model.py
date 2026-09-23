"""Do not silently discard gateway errors embedded in a successful SSE response."""
from langchain_openai import ChatOpenAI

from .errors import ProviderStreamError


class NativeStreamChatOpenAI(ChatOpenAI):
    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        # Some compatible gateways emit Responses-style failures even on the
        # Chat Completions endpoint. LangChain otherwise ignores these chunks.
        if chunk.get("type") == "response.failed":
            error = (chunk.get("response") or {}).get("error") or {}
            message = str(error.get("message", "")) if isinstance(error, dict) else ""
            code = "model_input_rejected" if "DataInspectionFailed" in message else "model_provider_stream_failed"
            raise ProviderStreamError(code)
        return super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
