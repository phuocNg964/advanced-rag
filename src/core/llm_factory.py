import logging
from typing import Any

from src.core.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Provider → (ChatClass, api_key_field_name, api_key_kwarg_name)
_PROVIDERS = {
    "groq":   (ChatGroq,                  "groq_api_key",   "api_key"),
    "openai": (ChatOpenAI,                "openai_api_key", "api_key"),
    "gemini": (ChatGoogleGenerativeAI,    "gemini_api_key", "google_api_key"),
}

def get_llm(**kwargs: Any):
    """
    Pure dispatcher — instantiates the LLM class for the given provider.
    
    Every argument (provider, model, temperature, …) must be supplied
    explicitly by the caller. The only thing injected automatically is the
    API key from settings (if not already provided).
    """
    settings = get_settings()
    provider = kwargs.pop("provider", None)

    if not provider:
        raise ValueError("'provider' is required (e.g. 'groq', 'openai', 'gemini')")

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {', '.join(_PROVIDERS)}"
        )

    ChatClass, key_field, key_kwarg = _PROVIDERS[provider]
    api_key = getattr(settings, key_field, None)

    if not api_key:
        raise ValueError(f"{key_field.upper()} must be set when provider='{provider}'")

    # Inject API key (caller can still override)
    kwargs.setdefault(key_kwarg, api_key)
    logger.info(f"Instantiating {provider} LLM: {kwargs.get('model', '(default)')}") 
    return ChatClass(**kwargs)