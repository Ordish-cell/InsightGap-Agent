import base64
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.web_app.agent.llm.config import get_llm_settings
from src.web_app.agent.llm.factory import _cached_chat_model
from src.web_app.core.config import settings

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB per image for base64 encoding
MAX_VISION_IMAGES_PER_MESSAGE = 5
MAX_TOTAL_VISION_BYTES = 40 * 1024 * 1024  # 40MB total base64 payload

logger = logging.getLogger(__name__)


def _get_vision_model() -> str:
    return getattr(settings, "qwen_vision_model", None) or "qwen3.6-plus"


def image_to_data_url(file_path: str, mime_type: str) -> str:
    data = Path(file_path).read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large for multimodal processing: {len(data)} bytes (max {MAX_IMAGE_BYTES})")
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


class QwenMultimodalService:

    async def analyze_images(
        self,
        prompt: str,
        images: list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        """Generate structured internal context for RAG / attachment_context injection."""
        if not images:
            return ""
        llm_settings = get_llm_settings()
        vision_model = model or _get_vision_model()
        truncated = self._truncate_images(images)
        message_content = self._build_image_message_content(prompt, truncated)

        try:
            chat_model = _cached_chat_model(
                provider=llm_settings.provider, model=vision_model,
                base_url=llm_settings.effective_base_url, api_key=llm_settings.effective_api_key,
                timeout=llm_settings.timeout_seconds, max_retries=llm_settings.max_retries,
                temperature=0.3, enabled=llm_settings.enabled, streaming=False,
            )
        except Exception as exc:
            logger.exception("Failed to create vision chat model")
            return f"[Image understanding unavailable: {exc}]"

        system_text = (
            "你是一个多模态分析助手。请仔细分析用户上传的图片，提取所有视觉信息。\n"
            "按以下结构输出：\n"
            "1. 图片整体描述\n"
            "2. 可见的文字/OCR内容（如果有）\n"
            "3. 与用户问题相关的细节\n"
            "请使用中文回答，简洁清晰。"
        )
        messages = [SystemMessage(content=system_text), HumanMessage(content=message_content)]

        try:
            response = await chat_model.ainvoke(messages)
            content = self._extract_response_text(response)
            parts = [f"Image: {img.get('filename', 'image')}" for img in images]
            header = "[Image Understanding]\n" + "\n".join(parts) + "\n\nDescription:\n"
            return header + content
        except Exception as exc:
            logger.exception("Image analysis failed")
            return f"[Image understanding failed: {exc}]"

    async def answer_image_question(
        self,
        prompt: str,
        images: list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        """User-facing natural language answer — no internal context markers."""
        if not images:
            return "没有可分析的图片。"
        llm_settings = get_llm_settings()
        vision_model = model or _get_vision_model()
        truncated = self._truncate_images(images)

        final_prompt = (
            "你是一个专业、准确的图片理解助手。\n\n"
            "请直接回答用户关于图片的问题，不要输出内部标签。\n"
            "不要输出如下内容：\n"
            "- [Image Understanding]\n"
            "- Image:\n"
            "- Description:\n"
            "- OCR:\n"
            "- Visible text:\n"
            "- Relevant details:\n\n"
            "如果图片里有文字，请自然地说明识别到的文字。\n"
            "如果用户只是要求分析图片，请从以下角度回答：\n"
            "1. 图片中主要是什么\n"
            "2. 关键细节\n"
            "3. 如果有符号/警告/界面元素，解释它的含义\n"
            "4. 给出必要建议\n\n"
            "用户问题：\n"
            f"{prompt}"
        )
        message_content = self._build_image_message_content(final_prompt, truncated)

        try:
            chat_model = _cached_chat_model(
                provider=llm_settings.provider, model=vision_model,
                base_url=llm_settings.effective_base_url, api_key=llm_settings.effective_api_key,
                timeout=llm_settings.timeout_seconds, max_retries=llm_settings.max_retries,
                temperature=0.3, enabled=llm_settings.enabled, streaming=False,
            )
        except Exception as exc:
            logger.exception("Failed to create vision chat model")
            return f"视觉模型不可用：{exc}"

        messages = [HumanMessage(content=message_content)]

        try:
            response = await chat_model.ainvoke(messages)
            return self._extract_response_text(response).strip()
        except Exception as exc:
            logger.exception("Image question answering failed")
            raise

    # ── helpers ──

    def _truncate_images(self, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        truncated = list(images[:MAX_VISION_IMAGES_PER_MESSAGE])
        if len(images) > MAX_VISION_IMAGES_PER_MESSAGE:
            logger.warning("Truncated %d images to limit of %d", len(images), MAX_VISION_IMAGES_PER_MESSAGE)
        total_bytes = 0
        for img in truncated:
            try:
                total_bytes += Path(img.get("file_path", "")).stat().st_size
            except Exception:
                pass
        if total_bytes > MAX_TOTAL_VISION_BYTES:
            logger.warning("Total image size %d exceeds limit %d, truncating", total_bytes, MAX_TOTAL_VISION_BYTES)
            limited: list[dict[str, Any]] = []
            running = 0
            for img in truncated:
                try:
                    s = Path(img.get("file_path", "")).stat().st_size
                except Exception:
                    s = 0
                if running + s > MAX_TOTAL_VISION_BYTES:
                    break
                limited.append(img)
                running += s
            truncated = limited
        return truncated

    def _build_image_message_content(self, prompt: str, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt or "请分析用户上传的图片。"}]
        for img in images:
            file_path = img.get("file_path", "")
            mime_type = img.get("mime_type", "image/png")
            filename = img.get("filename", "image")
            try:
                data_url = image_to_data_url(file_path, mime_type)
                message_content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}})
            except Exception as exc:
                logger.exception("Failed to encode image %s", filename)
                message_content.append({"type": "text", "text": f"\n[Image: {filename} — encoding failed: {exc}]"})
        return message_content

    def _extract_response_text(self, response: Any) -> str:
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content).strip()


qwen_multimodal_service = QwenMultimodalService()
