import logging
from typing import Any

from src.core.config import get_settings
from src.core.model_config import load_models_config, LLMProfile
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

def _instantiate_llm(profile: LLMProfile) -> Any:
    settings = get_settings()
    provider = profile.provider.lower()

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {', '.join(_PROVIDERS)}"
        )

    ChatClass, key_field, key_kwarg = _PROVIDERS[provider]
    api_key = getattr(settings, key_field, None)

    if not api_key:
        raise ValueError(f"{key_field.upper()} must be set when provider='{provider}'")

    kwargs = {
        "model": profile.model,
        "temperature": profile.temperature,
        "max_retries": profile.max_retries,
        **profile.model_kwargs,
    }
    kwargs[key_kwarg] = api_key
    
    logger.info(f"Instantiating {provider} LLM: {profile.model}")
    return ChatClass(**kwargs)

def get_llm(profile_name: str) -> Any:
    """
    Instantiates an LLM using the external YAML configuration by profile name.
    Automatically chains fallback models if specified.
    """
    models_config = load_models_config()
    
    if profile_name not in models_config.llms:
        raise ValueError(f"Profile '{profile_name}' not found in models configuration.")
        
    profile = models_config.llms[profile_name]
    primary_llm = _instantiate_llm(profile)
    
    if profile.fallbacks:
        fallback_llms = []
        for fallback_name in profile.fallbacks:
            if fallback_name not in models_config.llms:
                logger.warning(f"Fallback profile '{fallback_name}' not found. Skipping.")
                continue
            fallback_profile = models_config.llms[fallback_name]
            fallback_llms.append(_instantiate_llm(fallback_profile))
            
        if fallback_llms:
            logger.info(f"Binding {len(fallback_llms)} fallback(s) to profile '{profile_name}'")
            return primary_llm.with_fallbacks(fallback_llms)
            
    return primary_llm
