"""
Azure OpenAI provider for enterprise healthcare deployments.

This module provides integration with Azure OpenAI Service,
offering enterprise-grade security and compliance features.
"""

import asyncio
import json
import time
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
from datetime import datetime

try:
    from openai import AsyncAzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

from ..base import (
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
    LLMAuthenticationError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMQuotaExceededError,
)

logger = logging.getLogger(__name__)


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI provider for enterprise healthcare deployments.
    
    Provides access to OpenAI models through Azure's enterprise-grade
    infrastructure with enhanced security and compliance features.
    """
    
    def __init__(self, config: LLMProviderConfig):
        """Initialize Azure OpenAI provider."""
        super().__init__(config)
        
        if not AZURE_OPENAI_AVAILABLE:
            raise LLMProviderError(
                "openai library not available. Install with: pip install openai",
                "azure_openai"
            )
        
        # Azure-specific configuration
        self.azure_endpoint = config.api_base or config.custom_headers.get("azure_endpoint")
        self.api_version = config.custom_headers.get("api_version", "2024-02-15-preview")
        self.deployment_name = config.custom_headers.get("deployment_name")
        
        if not self.azure_endpoint:
            raise LLMProviderError(
                "Azure endpoint is required for Azure OpenAI",
                "azure_openai"
            )
        
        if not config.api_key and not config.custom_headers.get("azure_ad_token"):
            raise LLMProviderError(
                "API key or Azure AD token is required",
                "azure_openai"
            )
        
        self.client: Optional[AsyncAzureOpenAI] = None
        
        # Azure OpenAI model configurations
        self.azure_models = {
            "gpt-4": {
                "display_name": "GPT-4",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.REASONING,
                    LLMCapability.FUNCTION_CALLING,
                    LLMCapability.STRUCTURED_OUTPUT,
                ],
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.03,
                "supports_functions": True,
            },
            "gpt-4-32k": {
                "display_name": "GPT-4 32K",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.REASONING,
                    LLMCapability.FUNCTION_CALLING,
                    LLMCapability.STRUCTURED_OUTPUT,
                ],
                "max_tokens": 32768,
                "cost_per_1k_tokens": 0.06,
                "supports_functions": True,
            },
            "gpt-4-turbo": {
                "display_name": "GPT-4 Turbo",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.REASONING,
                    LLMCapability.FUNCTION_CALLING,
                    LLMCapability.STRUCTURED_OUTPUT,
                ],
                "max_tokens": 128000,
                "cost_per_1k_tokens": 0.01,
                "supports_functions": True,
            },
            "gpt-35-turbo": {
                "display_name": "GPT-3.5 Turbo",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.FUNCTION_CALLING,
                ],
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.002,
                "supports_functions": True,
            },
            "gpt-35-turbo-16k": {
                "display_name": "GPT-3.5 Turbo 16K",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.FUNCTION_CALLING,
                ],
                "max_tokens": 16384,
                "cost_per_1k_tokens": 0.004,
                "supports_functions": True,
            },
            "text-embedding-ada-002": {
                "display_name": "Text Embedding Ada 002",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.EXTRACTION,
                    LLMCapability.CLASSIFICATION,
                ],
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.0001,
                "supports_functions": False,
            },
        }
    
    async def initialize(self) -> None:
        """Initialize the Azure OpenAI provider."""
        if self._initialized:
            return
        
        try:
            # Initialize Azure OpenAI client
            self.client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            
            # Test connection with a simple request
            await self._test_connection()
            
            self._initialized = True
            logger.info("Azure OpenAI provider initialized successfully")
            
        except Exception as e:
            await self.shutdown()
            raise LLMProviderError(f"Failed to initialize Azure OpenAI provider: {e}", "azure_openai")
    
    async def _test_connection(self) -> None:
        """Test connection to Azure OpenAI."""
        try:
            # Try to list models as a connectivity test
            response = await self.client.models.list()
            if not response.data:
                logger.warning("No models available in Azure OpenAI deployment")
        except Exception as e:
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                raise LLMAuthenticationError(f"Authentication failed: {e}", "azure_openai")
            else:
                raise LLMProviderError(f"Connection test failed: {e}", "azure_openai")
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Azure OpenAI."""
        if not self._initialized:
            await self.initialize()
        
        await self.validate_request(request)
        
        # Use deployment name if provided, otherwise use model name
        model = self.deployment_name or request.model or self.config.default_model or "gpt-35-turbo"
        
        # Convert our messages to OpenAI format
        messages = self._convert_messages(request.messages)
        
        start_time = time.time()
        
        try:
            # Build request parameters
            params = {
                "model": model,
                "messages": messages,
                "stream": request.stream,
            }
            
            # Add optional parameters
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.top_p is not None:
                params["top_p"] = request.top_p
            if request.frequency_penalty is not None:
                params["frequency_penalty"] = request.frequency_penalty
            if request.presence_penalty is not None:
                params["presence_penalty"] = request.presence_penalty
            if request.stop:
                params["stop"] = request.stop
            if request.functions:
                params["functions"] = request.functions
            if request.function_call:
                params["function_call"] = request.function_call
            if request.response_format:
                params["response_format"] = request.response_format
            if request.seed is not None:
                params["seed"] = request.seed
            if request.user:
                params["user"] = request.user
            
            if request.stream:
                # Streaming not supported in this sync method
                raise LLMProviderError("Use stream() method for streaming responses", "azure_openai")
            
            response = await self.client.chat.completions.create(**params)
            response_time = time.time() - start_time
            
            choice = response.choices[0]
            message = choice.message
            
            return LLMResponse(
                content=message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                finish_reason=choice.finish_reason,
                response_time=response_time,
                function_call=message.function_call.dict() if message.function_call else None,
                metadata={
                    "azure_endpoint": self.azure_endpoint,
                    "api_version": self.api_version,
                    "deployment": model,
                }
            )
            
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {e}", "azure_openai")
            elif "quota" in error_msg.lower():
                raise LLMQuotaExceededError(f"Quota exceeded: {e}", "azure_openai")
            elif "not found" in error_msg.lower() or "404" in error_msg:
                raise LLMModelNotFoundError(f"Model/deployment not found: {e}", "azure_openai")
            elif "authentication" in error_msg.lower() or "401" in error_msg:
                raise LLMAuthenticationError(f"Authentication error: {e}", "azure_openai")
            else:
                raise LLMProviderError(f"Azure OpenAI API error: {e}", "azure_openai")
    
    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response using Azure OpenAI."""
        if not self._initialized:
            await self.initialize()
        
        await self.validate_request(request)
        
        model = self.deployment_name or request.model or self.config.default_model or "gpt-35-turbo"
        messages = self._convert_messages(request.messages)
        
        try:
            params = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            
            # Add optional parameters
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.top_p is not None:
                params["top_p"] = request.top_p
            if request.frequency_penalty is not None:
                params["frequency_penalty"] = request.frequency_penalty
            if request.presence_penalty is not None:
                params["presence_penalty"] = request.presence_penalty
            if request.stop:
                params["stop"] = request.stop
            if request.seed is not None:
                params["seed"] = request.seed
            if request.user:
                params["user"] = request.user
            
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {e}", "azure_openai")
            elif "quota" in error_msg.lower():
                raise LLMQuotaExceededError(f"Quota exceeded: {e}", "azure_openai")
            else:
                raise LLMProviderError(f"Azure OpenAI streaming error: {e}", "azure_openai")
    
    async def get_available_models(self) -> List[LLMModelInfo]:
        """Get list of available models from Azure OpenAI."""
        if self._models_cache:
            return self._models_cache
        
        models = []
        
        # Get available models from Azure OpenAI
        try:
            if self._initialized and self.client:
                response = await self.client.models.list()
                available_model_names = {model.id for model in response.data}
            else:
                available_model_names = set()
        except Exception:
            # Fall back to our predefined list
            available_model_names = set()
        
        for model_name, model_info in self.azure_models.items():
            # Only include models that are available in the deployment
            if available_model_names and model_name not in available_model_names:
                continue
            
            models.append(LLMModelInfo(
                name=model_name,
                display_name=model_info["display_name"],
                provider=LLMProviderType.AZURE_OPENAI,
                model_type=model_info["model_type"],
                capabilities=model_info["capabilities"],
                max_tokens=model_info["max_tokens"],
                cost_per_1k_tokens=model_info["cost_per_1k_tokens"],
                supports_streaming=True,
                supports_functions=model_info["supports_functions"],
                supports_structured_output=model_info.get("supports_functions", False),
                healthcare_optimized=False,  # General purpose models
                hipaa_compliant=True,  # Azure provides HIPAA compliance
                local_deployment=False,
                description=f"Azure OpenAI {model_info['display_name']}",
            ))
        
        self._models_cache = models
        return models
    
    async def health_check(self) -> LLMProviderStatus:
        """Check health status of Azure OpenAI provider."""
        try:
            start_time = time.time()
            
            # Simple health check by listing models
            if self.client:
                response = await self.client.models.list()
                response_time = time.time() - start_time
                
                return LLMProviderStatus(
                    provider_type=LLMProviderType.AZURE_OPENAI,
                    status="healthy",
                    available_models=[model.id for model in response.data],
                    last_check=datetime.utcnow(),
                    response_time=response_time,
                    error_rate=0.0,
                    requests_per_minute=0,  # Azure manages this
                    quota_remaining=None,  # Not exposed by API
                    quota_reset=None,
                )
            else:
                return LLMProviderStatus(
                    provider_type=LLMProviderType.AZURE_OPENAI,
                    status="unavailable",
                    available_models=[],
                    last_check=datetime.utcnow(),
                    response_time=None,
                    error_rate=1.0,
                    requests_per_minute=0,
                )
                
        except Exception as e:
            logger.error(f"Azure OpenAI health check failed: {e}")
            return LLMProviderStatus(
                provider_type=LLMProviderType.AZURE_OPENAI,
                status="unavailable",
                available_models=[],
                last_check=datetime.utcnow(),
                response_time=None,
                error_rate=1.0,
                requests_per_minute=0,
            )
    
    def supports_capability(self, capability: LLMCapability) -> bool:
        """Check if Azure OpenAI supports a specific capability."""
        supported_capabilities = {
            LLMCapability.TEXT_GENERATION,
            LLMCapability.QUESTION_ANSWERING,
            LLMCapability.SUMMARIZATION,
            LLMCapability.CLASSIFICATION,
            LLMCapability.EXTRACTION,
            LLMCapability.REASONING,
            LLMCapability.FUNCTION_CALLING,
            LLMCapability.STRUCTURED_OUTPUT,
            LLMCapability.CODE_GENERATION,
        }
        return capability in supported_capabilities
    
    async def estimate_cost(self, request: LLMRequest) -> Optional[float]:
        """Estimate cost for Azure OpenAI request."""
        model_name = request.model or self.config.default_model or "gpt-35-turbo"
        model_info = self.azure_models.get(model_name)
        
        if not model_info:
            return None
        
        # Estimate token usage
        prompt_text = " ".join(msg.content for msg in request.messages)
        prompt_tokens = len(prompt_text.split()) * 1.3  # Rough estimation
        completion_tokens = request.max_tokens or 512
        
        total_tokens = prompt_tokens + completion_tokens
        cost_per_1k = model_info["cost_per_1k_tokens"]
        
        return (total_tokens / 1000) * cost_per_1k
    
    async def shutdown(self) -> None:
        """Shutdown the Azure OpenAI provider."""
        if self.client:
            await self.client.close()
            self.client = None
        await super().shutdown()
    
    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert our message format to OpenAI format."""
        openai_messages = []
        
        for message in messages:
            openai_msg = {
                "role": message.role,
                "content": message.content,
            }
            
            if message.name:
                openai_msg["name"] = message.name
            
            if message.function_call:
                openai_msg["function_call"] = message.function_call
            
            openai_messages.append(openai_msg)
        
        return openai_messages


# Azure-specific utilities

def create_azure_config(
    azure_endpoint: str,
    api_key: str,
    deployment_name: Optional[str] = None,
    api_version: str = "2024-02-15-preview"
) -> LLMProviderConfig:
    """
    Create Azure OpenAI configuration.
    
    Args:
        azure_endpoint: Azure OpenAI endpoint URL
        api_key: API key for authentication
        deployment_name: Optional deployment name
        api_version: API version to use
        
    Returns:
        Configured LLMProviderConfig for Azure OpenAI
    """
    custom_headers = {
        "api_version": api_version,
    }
    
    if deployment_name:
        custom_headers["deployment_name"] = deployment_name
    
    return LLMProviderConfig(
        provider_type=LLMProviderType.AZURE_OPENAI,
        api_key=api_key,
        api_base=azure_endpoint,
        custom_headers=custom_headers,
        default_model="gpt-35-turbo",
        timeout=60.0,
        max_retries=3,
        enabled=True,
        priority=1,
    )


def get_recommended_azure_models() -> List[str]:
    """Get list of recommended Azure OpenAI models for healthcare."""
    return [
        "gpt-4-turbo",           # Best reasoning and long context
        "gpt-4",                 # High quality responses
        "gpt-35-turbo-16k",      # Good balance of cost and capability
        "gpt-35-turbo",          # Most cost-effective for simple tasks
    ]