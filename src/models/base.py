from abc import ABC, abstractmethod
from typing import Any
import logging
from src.core.model_config import load_models_config, LLMProfile

logger = logging.getLogger(__name__)

class BaseModelFactory(ABC):
    @classmethod
    @abstractmethod
    def supports(cls, provider: str) -> bool:
        pass

    @classmethod
    @abstractmethod
    def instantiate(cls, profile: LLMProfile) -> Any:
        pass

def _instantiate_llm(profile: LLMProfile) -> Any:
    from src.models.remote_llm import RemoteLLMFactory
    from src.models.local_slm import LocalSLMFactory

    provider = profile.provider.lower()
    
    if RemoteLLMFactory.supports(provider):
        return RemoteLLMFactory.instantiate(profile)
    elif LocalSLMFactory.supports(provider):
        return LocalSLMFactory.instantiate(profile)
    else:
        raise ValueError(f"Unknown provider '{provider}'.")

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
