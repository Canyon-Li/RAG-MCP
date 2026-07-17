"""
LLM Module.

This package contains LLM client abstractions and implementations:
- Base LLM class (text-only)
- Base Vision LLM class (multimodal: text + image)
- LLM factory
- Provider implementations (OpenAI, Azure, Ollama, DeepSeek, Qwen, Zhipu)
"""

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM, OpenAIVisionLLMError
from src.libs.llm.azure_llm import AzureLLM, AzureLLMError
from src.libs.llm.azure_vision_llm import AzureVisionLLM, AzureVisionLLMError
from src.libs.llm.ollama_vision_llm import OllamaVisionLLM, OllamaVisionLLMError
from src.libs.llm.deepseek_llm import DeepSeekLLM, DeepSeekLLMError
from src.libs.llm.ollama_llm import OllamaLLM, OllamaLLMError
from src.libs.llm.qwen_llm import QwenLLM, QwenLLMError
from src.libs.llm.zhipu_llm import ZhipuLLM, ZhipuLLMError

# Register text-only LLM providers with factory
LLMFactory.register_provider("openai", OpenAILLM)
LLMFactory.register_provider("azure", AzureLLM)
LLMFactory.register_provider("deepseek", DeepSeekLLM)
LLMFactory.register_provider("ollama", OllamaLLM)
LLMFactory.register_provider("qwen", QwenLLM)
LLMFactory.register_provider("zhipu", ZhipuLLM)

# Note: Vision LLM providers will be registered in task B9+

__all__ = [
    # Base classes
    "BaseLLM",
    "BaseVisionLLM",
    # Data types
    "ChatResponse",
    "Message",
    "ImageInput",
    # Factory
    "LLMFactory",
    # Text-only LLM implementations
    "OpenAILLM",
    "OpenAILLMError",
    "AzureLLM",
    "AzureLLMError",
    "DeepSeekLLM",
    "DeepSeekLLMError",
    "OllamaLLM",
    "OllamaLLMError",
    "QwenLLM",
    "QwenLLMError",
    "ZhipuLLM",
    "ZhipuLLMError",
    # Vision LLM implementations
    "OpenAIVisionLLM",
    "OpenAIVisionLLMError",
    "AzureVisionLLM",
    "AzureVisionLLMError",
    "OllamaVisionLLM",
    "OllamaVisionLLMError",
]
