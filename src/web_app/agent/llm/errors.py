class LLMRouterError(RuntimeError):
    """Raised when model routing cannot resolve a usable model."""


class LLMUnavailableError(RuntimeError):
    """Raised when LLM config is disabled or missing credentials."""


class LLMInvocationError(RuntimeError):
    """Raised when a chat model call fails."""


class LLMParseError(RuntimeError):
    """Raised when a model response cannot be parsed or validated."""
