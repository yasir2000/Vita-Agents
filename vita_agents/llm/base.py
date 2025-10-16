"""
Abstract base classes and interfaces for LLM providers in Vita Agents.

This module defines the common interface that all LLM providers must implement,
ensuring consistent behavior across different language model backends.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"


class LLMModelType(str, Enum):
    """LLM model categories for healthcare use cases."""
    GENERAL = "general"  # General purpose models
    MEDICAL = "medical"  # Medical/healthcare specialized
    CODE = "code"  # Code generation models
    CLINICAL = "clinical"  # Clinical documentation
    RESEARCH = "research"  # Research and analysis
    COMPLIANCE = "compliance"  # Regulatory and compliance


class LLMCapability(str, Enum):
    """LLM capabilities for healthcare agents."""
    TEXT_GENERATION = "text_generation"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    QUESTION_ANSWERING = "question_answering"
    REASONING = "reasoning"
    FUNCTION_CALLING = "function_calling"
    STRUCTURED_OUTPUT = "structured_output"


class LLMMessage(BaseModel):
    """Standard message format for LLM interactions."""
    role: str = Field(..., description="Message role: system, user, assistant, function")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Name for function calls")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Function call data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LLMRequest(BaseModel):
    """Standard request format for LLM providers."""
    messages: List[LLMMessage] = Field(..., description="Conversation messages")
    model: Optional[str] = Field(None, description="Specific model to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    top_p: Optional[float] = Field(None, description="Top-p sampling")
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, description="Presence penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(default=False, description="Enable streaming")
    functions: Optional[List[Dict[str, Any]]] = Field(None, description="Available functions")
    function_call: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Function call mode")
    response_format: Optional[Dict[str, Any]] = Field(None, description="Response format specification")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    user: Optional[str] = Field(None, description="User identifier for tracking")


class LLMResponse(BaseModel):
    """Standard response format from LLM providers."""
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    finish_reason: str = Field(..., description="Reason for completion")
    response_time: float = Field(..., description="Response time in seconds")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Function call result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LLMModelInfo(BaseModel):
    """Information about an available LLM model."""
    name: str = Field(..., description="Model name/identifier")
    display_name: str = Field(..., description="Human-readable name")
    provider: LLMProviderType = Field(..., description="Provider type")
    model_type: LLMModelType = Field(..., description="Model category")
    capabilities: List[LLMCapability] = Field(..., description="Model capabilities")
    max_tokens: int = Field(..., description="Maximum context length")
    cost_per_1k_tokens: Optional[float] = Field(None, description="Cost per 1K tokens")
    supports_streaming: bool = Field(default=False, description="Supports streaming responses")
    supports_functions: bool = Field(default=False, description="Supports function calling")
    supports_structured_output: bool = Field(default=False, description="Supports structured JSON output")
    healthcare_optimized: bool = Field(default=False, description="Optimized for healthcare use cases")
    hipaa_compliant: bool = Field(default=False, description="HIPAA compliant deployment")
    local_deployment: bool = Field(default=False, description="Can be deployed locally")
    description: Optional[str] = Field(None, description="Model description")
    version: Optional[str] = Field(None, description="Model version")


class LLMProviderConfig(BaseModel):
    """Configuration for LLM providers."""
    provider_type: LLMProviderType = Field(..., description="Provider type")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    api_base: Optional[str] = Field(None, description="Custom API base URL")
    organization: Optional[str] = Field(None, description="Organization ID")
    project: Optional[str] = Field(None, description="Project ID")
    default_model: Optional[str] = Field(None, description="Default model to use")
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    rate_limit: Optional[int] = Field(None, description="Rate limit (requests per minute)")
    enabled: bool = Field(default=True, description="Whether provider is enabled")
    priority: int = Field(default=1, description="Provider priority (lower = higher priority)")
    custom_headers: Dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers")
    model_config: Dict[str, Any] = Field(default_factory=dict, description="Model-specific configuration")


class LLMProviderStatus(BaseModel):
    """Status information for an LLM provider."""
    provider_type: LLMProviderType = Field(..., description="Provider type")
    status: str = Field(..., description="Current status: healthy, degraded, unavailable")
    available_models: List[str] = Field(..., description="Available model names")
    last_check: datetime = Field(..., description="Last health check timestamp")
    response_time: Optional[float] = Field(None, description="Average response time")
    error_rate: float = Field(default=0.0, description="Error rate percentage")
    requests_per_minute: int = Field(default=0, description="Current request rate")
    quota_remaining: Optional[int] = Field(None, description="Remaining quota")
    quota_reset: Optional[datetime] = Field(None, description="Quota reset time")


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the interface that all LLM providers must implement
    to ensure consistent behavior across different backends.
    """
    
    def __init__(self, config: LLMProviderConfig):
        """Initialize the LLM provider with configuration."""
        self.config = config
        self.provider_type = config.provider_type
        self._initialized = False
        self._models_cache: Optional[List[LLMModelInfo]] = None
        self._status_cache: Optional[LLMProviderStatus] = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider and establish connections."""
        pass
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            request: The LLM request containing messages and parameters
            
        Returns:
            LLMResponse containing the generated content and metadata
        """
        pass
    
    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            request: The LLM request containing messages and parameters
            
        Yields:
            Partial response strings as they become available
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[LLMModelInfo]:
        """
        Get list of available models from this provider.
        
        Returns:
            List of available models with their capabilities
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> LLMProviderStatus:
        """
        Check the health status of the provider.
        
        Returns:
            Current status of the provider
        """
        pass
    
    async def validate_request(self, request: LLMRequest) -> None:
        """
        Validate the request before sending to the LLM.
        
        Args:
            request: The request to validate
            
        Raises:
            ValueError: If the request is invalid
        """
        if not request.messages:
            raise ValueError("Request must contain at least one message")
        
        if request.max_tokens and request.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if request.temperature is not None and not (0.0 <= request.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
    
    async def get_model_info(self, model_name: str) -> Optional[LLMModelInfo]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information if available
        """
        models = await self.get_available_models()
        for model in models:
            if model.name == model_name or model.display_name == model_name:
                return model
        return None
    
    def supports_capability(self, capability: LLMCapability) -> bool:
        """
        Check if this provider supports a specific capability.
        
        Args:
            capability: The capability to check
            
        Returns:
            True if the capability is supported
        """
        # Default implementation - subclasses should override
        return capability in [
            LLMCapability.TEXT_GENERATION,
            LLMCapability.QUESTION_ANSWERING
        ]
    
    async def estimate_cost(self, request: LLMRequest) -> Optional[float]:
        """
        Estimate the cost for a request.
        
        Args:
            request: The request to estimate cost for
            
        Returns:
            Estimated cost in USD, or None if unavailable
        """
        # Default implementation - subclasses can override
        return None
    
    async def shutdown(self) -> None:
        """Clean up resources and shutdown the provider."""
        self._initialized = False
        self._models_cache = None
        self._status_cache = None


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    def __init__(self, message: str, provider: str, error_code: Optional[str] = None):
        self.message = message
        self.provider = provider
        self.error_code = error_code
        super().__init__(f"[{provider}] {message}")


class LLMRateLimitError(LLMProviderError):
    """Raised when rate limits are exceeded."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Raised when authentication fails."""
    pass


class LLMModelNotFoundError(LLMProviderError):
    """Raised when a requested model is not available."""
    pass


class LLMQuotaExceededError(LLMProviderError):
    """Raised when quota limits are exceeded."""
    pass


class LLMProviderUnavailableError(LLMProviderError):
    """Raised when the provider is temporarily unavailable."""
    pass