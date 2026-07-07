from typing import Any
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.core.config import get_settings
from src.core.model_config import LLMProfile

logger = logging.getLogger(__name__)

_REMOTE_PROVIDERS = {
    "groq": (ChatGroq, "groq_api_key", "api_key"),
    "openai": (ChatOpenAI, "openai_api_key", "api_key"),
    "gemini": (ChatGoogleGenerativeAI, "gemini_api_key", "google_api_key"),
}


def supports_remote_provider(provider: str) -> bool:
    return provider.lower() in _REMOTE_PROVIDERS


def instantiate_remote_llm(profile: LLMProfile) -> Any:
    settings = get_settings()
    provider = profile.provider.lower()

    ChatClass, key_field, key_kwarg = _REMOTE_PROVIDERS[provider]
    api_key = getattr(settings, key_field, None)

    if not api_key:
        raise ValueError(f"{key_field.upper()} must be set when provider='{provider}'")

    kwargs = {
        "model": profile.model,
        "temperature": profile.temperature,
        "max_retries": profile.max_retries,
        key_kwarg: api_key,
    }
    if profile.model_kwargs:
        kwargs["model_kwargs"] = profile.model_kwargs

    logger.info("Instantiating %s Remote LLM: %s", provider, profile.model)
    return ChatClass(**kwargs)
