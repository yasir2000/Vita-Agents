"""
LLM package initialization for Vita Agents.

This package provides comprehensive LLM integration with support for multiple providers
including Ollama, Hugging Face, Azure OpenAI, and Google AI.
"""

from .base import (
    BaseLLMProvider,
    LLMProviderType,
    LLMModelType,
    LLMCapability,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMModelInfo,
    LLMProviderConfig,
    LLMProviderStatus,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMModelNotFoundError,
    LLMQuotaExceededError,
    LLMProviderUnavailableError,
)

from .router import (
    LLMRouter,
    LLMFactory,
    RoutingStrategy,
    RoutingCriteria,
    ProviderMetrics,
    create_routing_criteria,
    create_healthcare_criteria,
)

# Provider imports
from .providers.ollama import OllamaProvider
from .providers.huggingface import HuggingFaceProvider
from .providers.azure_openai import AzureOpenAIProvider
from .providers.google import GoogleAIProvider

__all__ = [
    # Base classes and types
    "BaseLLMProvider",
    "LLMProviderType",
    "LLMModelType",
    "LLMCapability",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMModelInfo",
    "LLMProviderConfig",
    "LLMProviderStatus",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "LLMModelNotFoundError",
    "LLMQuotaExceededError",
    "LLMProviderUnavailableError",
    
    # Router and factory
    "LLMRouter",
    "LLMFactory",
    "RoutingStrategy",
    "RoutingCriteria",
    "ProviderMetrics",
    "create_routing_criteria",
    "create_healthcare_criteria",
    
    # Providers
    "OllamaProvider",
    "HuggingFaceProvider",
    "AzureOpenAIProvider",
    "GoogleAIProvider",
]

# Version information
__version__ = "0.1.0"