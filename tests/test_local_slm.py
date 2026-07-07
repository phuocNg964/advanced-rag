import sys
from types import ModuleType

from src.core.model_config import LLMProfile
from src.models.local_slm import instantiate_local_llm, supports_local_provider


def test_supports_local_providers():
    assert supports_local_provider("ollama")
    assert supports_local_provider("huggingface")
    assert not supports_local_provider("groq")


def test_instantiate_ollama_uses_model_kwargs(monkeypatch):
    module = ModuleType("langchain_ollama")

    class FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.ChatOllama = FakeChatOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", module)

    llm = instantiate_local_llm(
        LLMProfile(
            provider="ollama",
            model="llama3.1:8b",
            temperature=0.2,
            model_kwargs={"base_url": "http://localhost:11434", "num_ctx": 4096},
        )
    )

    assert llm.kwargs == {
        "model": "llama3.1:8b",
        "temperature": 0.2,
        "base_url": "http://localhost:11434",
        "num_ctx": 4096,
    }


def test_instantiate_huggingface_maps_nested_kwargs(monkeypatch):
    module = ModuleType("langchain_huggingface")

    class FakeChatHuggingFace:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeHuggingFacePipeline:
        received = None

        @classmethod
        def from_model_id(cls, **kwargs):
            cls.received = kwargs
            return cls()

    module.ChatHuggingFace = FakeChatHuggingFace
    module.HuggingFacePipeline = FakeHuggingFacePipeline
    monkeypatch.setitem(sys.modules, "langchain_huggingface", module)

    llm = instantiate_local_llm(
        LLMProfile(
            provider="huggingface",
            model="microsoft/Phi-3-mini-4k-instruct",
            temperature=0.1,
            model_kwargs={
                "task": "text-generation",
                "device": -1,
                "pipeline_kwargs": {"max_new_tokens": 512},
            },
        )
    )

    assert FakeHuggingFacePipeline.received == {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "task": "text-generation",
        "model_kwargs": None,
        "pipeline_kwargs": {"temperature": 0.1, "max_new_tokens": 512},
        "device": -1,
    }
    assert isinstance(llm, FakeChatHuggingFace)
    assert llm.kwargs["model_id"] == "microsoft/Phi-3-mini-4k-instruct"
