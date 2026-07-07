from typing import Any
import logging

from src.core.model_config import LLMProfile

logger = logging.getLogger(__name__)

_LOCAL_PROVIDERS = {"huggingface", "ollama"}


def supports_local_provider(provider: str) -> bool:
    return provider.lower() in _LOCAL_PROVIDERS


def instantiate_local_llm(profile: LLMProfile) -> Any:
    provider = profile.provider.lower()
    if provider == "ollama":
        return _instantiate_ollama(profile)
    if provider == "huggingface":
        return _instantiate_huggingface(profile)
    raise ValueError(f"Unsupported local provider: {provider}")


def _instantiate_ollama(profile: LLMProfile) -> Any:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError(
            "provider='ollama' requires langchain-ollama. "
            "Install runtime dependencies with `uv sync --frozen --no-dev`."
        ) from exc

    logger.info("Instantiating ollama Local LLM: %s", profile.model)
    return ChatOllama(
        model=profile.model,
        temperature=profile.temperature,
        **profile.model_kwargs,
    )


def _instantiate_huggingface(profile: LLMProfile) -> Any:
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    except ImportError as exc:
        raise RuntimeError(
            "provider='huggingface' requires langchain-huggingface. "
            "Install runtime dependencies with `uv sync --frozen --no-dev`."
        ) from exc

    options = dict(profile.model_kwargs)
    task = options.pop("task", "text-generation")
    model_kwargs = options.pop("model_kwargs", None)
    pipeline_kwargs = options.pop("pipeline_kwargs", None)

    pipeline_options = dict(pipeline_kwargs or {})
    if profile.temperature > 0:
        pipeline_options.setdefault("temperature", profile.temperature)

    logger.info("Instantiating huggingface Local LLM: %s", profile.model)
    llm = HuggingFacePipeline.from_model_id(
        model_id=profile.model,
        task=task,
        model_kwargs=model_kwargs,
        pipeline_kwargs=pipeline_options,
        **options,
    )
    return ChatHuggingFace(llm=llm, model_id=profile.model)
