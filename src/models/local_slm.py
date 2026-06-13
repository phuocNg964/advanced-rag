from typing import Any
import logging
from src.core.model_config import LLMProfile
from src.models.base import BaseModelFactory

# Import local models (e.g. HuggingFacePipeline, ChatOllama) if available
# from langchain_community.llms import HuggingFacePipeline
# from langchain_community.chat_models import ChatOllama

logger = logging.getLogger(__name__)

_LOCAL_PROVIDERS = ["huggingface", "ollama"]

class LocalSLMFactory(BaseModelFactory):
    @classmethod
    def supports(cls, provider: str) -> bool:
        return provider in _LOCAL_PROVIDERS

    @classmethod
    def instantiate(cls, profile: LLMProfile) -> Any:
        provider = profile.provider.lower()
        
        kwargs = {
            "model": profile.model,
            "temperature": profile.temperature,
            **profile.model_kwargs,
        }
        
        logger.info(f"Instantiating {provider} Local SLM: {profile.model}")
        
        if provider == "huggingface":
            # return HuggingFacePipeline.from_model_id(model_id=profile.model, task="text-generation", model_kwargs=kwargs)
            raise NotImplementedError("HuggingFace local SLM is stubbed and not fully implemented.")
        elif provider == "ollama":
            # return ChatOllama(model=profile.model, temperature=profile.temperature, **kwargs)
            raise NotImplementedError("Ollama local SLM is stubbed and not fully implemented.")
        else:
            raise ValueError(f"Unsupported local provider: {provider}")
