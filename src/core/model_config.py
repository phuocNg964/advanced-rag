import yaml
from functools import lru_cache
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pathlib import Path
from src.core.config import get_settings

class LLMProfile(BaseModel):
    provider: str
    model: str
    temperature: float = 0.0
    max_retries: int = 2
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    fallbacks: List[str] = Field(default_factory=list)

class ModelsConfig(BaseModel):
    llms: Dict[str, LLMProfile]

@lru_cache
def load_models_config() -> ModelsConfig:
    config_path = get_settings().base_dir / "configs" / "model" / "models.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    return ModelsConfig.model_validate(data)
