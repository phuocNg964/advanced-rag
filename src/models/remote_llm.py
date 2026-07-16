from typing import Any
import logging

from src.core.config import get_settings
from src.core.model_config import LLMProfile

logger = logging.getLogger(__name__)

_REMOTE_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "groq": ("langchain_groq", "ChatGroq", "groq_api_key"),
    "openai": ("langchain_openai", "ChatOpenAI", "openai_api_key"),
    "gemini": (
        "langchain_google_genai",
        "ChatGoogleGenerativeAI",
        "gemini_api_key",
    ),
}


def supports_remote_provider(provider: str) -> bool:
    return provider.lower() in _REMOTE_PROVIDERS


def instantiate_remote_llm(profile: LLMProfile) -> Any:
    settings = get_settings()
    provider = profile.provider.lower()

    module_name, class_name, key_field = _REMOTE_PROVIDERS[provider]
    api_key = getattr(settings, key_field, None)

    if not api_key:
        raise ValueError(f"{key_field.upper()} must be set when provider='{provider}'")

    ChatClass = _load_chat_class(module_name, class_name, provider)
    kwargs = {
        "model": profile.model,
        "temperature": profile.temperature,
        "api_key": api_key,
    }
    if provider == "gemini":
        kwargs["retries"] = profile.max_retries
    else:
        kwargs["max_retries"] = profile.max_retries

    if profile.model_kwargs:
        if provider == "gemini":
            kwargs.update(profile.model_kwargs)
        else:
            kwargs["model_kwargs"] = profile.model_kwargs

    logger.info("Instantiating %s Remote LLM: %s", provider, profile.model)
    return ChatClass(**kwargs)


def _load_chat_class(module_name: str, class_name: str, provider: str) -> Any:
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as exc:
        raise RuntimeError(
            f"provider='{provider}' requires {module_name}. "
            "Install runtime dependencies with `uv sync --frozen --no-dev`."
        ) from exc
    return getattr(module, class_name)
