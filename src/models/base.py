from typing import Any
import logging

from src.core.model_config import LLMProfile, load_models_config
from src.models.local_slm import instantiate_local_llm, supports_local_provider
from src.models.remote_llm import instantiate_remote_llm, supports_remote_provider

logger = logging.getLogger(__name__)


def _instantiate_llm(profile: LLMProfile) -> Any:
    provider = profile.provider.lower()
    if supports_remote_provider(provider):
        return instantiate_remote_llm(profile)
    if supports_local_provider(provider):
        return instantiate_local_llm(profile)
    raise ValueError(
        "Unknown provider '%s'. Supported providers: groq, openai, gemini, "
        "ollama, huggingface." % provider
    )


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

    fallback_llms = []
    for fallback_name in profile.fallbacks:
        fallback_profile = models_config.llms.get(fallback_name)
        if fallback_profile is None:
            logger.warning("Fallback profile '%s' not found. Skipping.", fallback_name)
            continue
        fallback_llms.append(_instantiate_llm(fallback_profile))

    if fallback_llms:
        logger.info(
            "Binding %s fallback(s) to profile '%s'",
            len(fallback_llms),
            profile_name,
        )
        return primary_llm.with_fallbacks(fallback_llms)

    return primary_llm
