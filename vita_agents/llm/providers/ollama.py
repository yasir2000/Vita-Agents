"""
Ollama provider for local LLM deployment in Vita Agents.

This module provides integration with Ollama, enabling local deployment
of open-source language models for healthcare applications.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, AsyncGenerator
import aiohttp
import logging
from datetime import datetime

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
    LLMProviderUnavailableError,
    LLMModelNotFoundError,
)

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider for local LLM deployment.
    
    Supports running open-source models locally with Ollama,
    including healthcare-optimized models.
    """
    
    def __init__(self, config: LLMProviderConfig):
        """Initialize Ollama provider."""
        super().__init__(config)
        self.base_url = config.api_base or "http://localhost:11434"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Healthcare-specific model mappings
        self.healthcare_models = {
            "llama2:7b-chat": {
                "display_name": "Llama 2 7B Chat",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.REASONING,
                ],
                "max_tokens": 4096,
                "healthcare_optimized": False,
            },
            "llama2:13b-chat": {
                "display_name": "Llama 2 13B Chat",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.REASONING,
                    LLMCapability.CLASSIFICATION,
                ],
                "max_tokens": 4096,
                "healthcare_optimized": False,
            },
            "codellama:7b-instruct": {
                "display_name": "Code Llama 7B Instruct",
                "model_type": LLMModelType.CODE,
                "capabilities": [
                    LLMCapability.CODE_GENERATION,
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                ],
                "max_tokens": 4096,
                "healthcare_optimized": False,
            },
            "mistral:7b-instruct": {
                "display_name": "Mistral 7B Instruct",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.CLASSIFICATION,
                    LLMCapability.REASONING,
                ],
                "max_tokens": 8192,
                "healthcare_optimized": False,
            },
            "medllama2:7b": {
                "display_name": "MedLlama2 7B (Healthcare)",
                "model_type": LLMModelType.MEDICAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.CLASSIFICATION,
                    LLMCapability.EXTRACTION,
                    LLMCapability.SUMMARIZATION,
                ],
                "max_tokens": 4096,
                "healthcare_optimized": True,
            },
            "clinicallama:7b": {
                "display_name": "Clinical Llama 7B",
                "model_type": LLMModelType.CLINICAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.CLASSIFICATION,
                    LLMCapability.EXTRACTION,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.QUESTION_ANSWERING,
                ],
                "max_tokens": 4096,
                "healthcare_optimized": True,
            },
            "biogpt": {
                "display_name": "BioGPT (Medical)",
                "model_type": LLMModelType.MEDICAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.REASONING,
                ],
                "max_tokens": 1024,
                "healthcare_optimized": True,
            },
        }
    
    async def initialize(self) -> None:
        """Initialize the Ollama provider."""
        if self._initialized:
            return
        
        # Create aiohttp session
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=self.config.custom_headers
        )
        
        try:
            # Test connection to Ollama
            await self._test_connection()
            self._initialized = True
            logger.info("Ollama provider initialized successfully")
            
        except Exception as e:
            await self.shutdown()
            raise LLMProviderError(f"Failed to initialize Ollama provider: {e}", "ollama")
    
    async def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        if not self.session:
            raise LLMProviderError("Session not initialized", "ollama")
        
        try:
            async with self.session.get(f"{self.base_url}/api/version") as response:
                if response.status != 200:
                    raise LLMProviderUnavailableError(
                        f"Ollama server returned status {response.status}",
                        "ollama"
                    )
                
        except aiohttp.ClientError as e:
            raise LLMProviderUnavailableError(
                f"Cannot connect to Ollama server at {self.base_url}: {e}",
                "ollama"
            )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from Ollama model."""
        if not self._initialized:
            await self.initialize()
        
        await self.validate_request(request)
        
        model = request.model or self.config.default_model or "llama2:7b-chat"
        
        # Convert messages to Ollama format
        prompt = self._convert_messages_to_prompt(request.messages)
        
        start_time = time.time()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options(request),
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    if response.status == 404:
                        raise LLMModelNotFoundError(
                            f"Model {model} not found. Available models: {await self._get_model_names()}",
                            "ollama"
                        )
                    raise LLMProviderError(f"Ollama API error: {error_text}", "ollama")
                
                result = await response.json()
                response_time = time.time() - start_time
                
                return LLMResponse(
                    content=result.get("response", ""),
                    model=model,
                    usage={
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
                    },
                    finish_reason="stop" if result.get("done") else "length",
                    response_time=response_time,
                    metadata={
                        "eval_duration": result.get("eval_duration"),
                        "load_duration": result.get("load_duration"),
                        "prompt_eval_duration": result.get("prompt_eval_duration"),
                        "total_duration": result.get("total_duration"),
                        "context": result.get("context", []),
                    }
                )
                
        except aiohttp.ClientError as e:
            raise LLMProviderError(f"Network error: {e}", "ollama")
        except json.JSONDecodeError as e:
            raise LLMProviderError(f"Invalid JSON response: {e}", "ollama")
    
    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response from Ollama model."""
        if not self._initialized:
            await self.initialize()
        
        await self.validate_request(request)
        
        model = request.model or self.config.default_model or "llama2:7b-chat"
        prompt = self._convert_messages_to_prompt(request.messages)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": self._build_options(request),
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    if response.status == 404:
                        raise LLMModelNotFoundError(f"Model {model} not found", "ollama")
                    raise LLMProviderError(f"Ollama API error: {error_text}", "ollama")
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue
                            
        except aiohttp.ClientError as e:
            raise LLMProviderError(f"Network error: {e}", "ollama")
    
    async def get_available_models(self) -> List[LLMModelInfo]:
        """Get list of available models from Ollama."""
        if not self._initialized:
            await self.initialize()
        
        if self._models_cache:
            return self._models_cache
        
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    raise LLMProviderError("Failed to fetch available models", "ollama")
                
                data = await response.json()
                models = []
                
                for model_data in data.get("models", []):
                    model_name = model_data.get("name", "")
                    
                    # Get model info from our healthcare models mapping or use defaults
                    model_info = self.healthcare_models.get(model_name, {
                        "display_name": model_name.replace(":", " ").title(),
                        "model_type": LLMModelType.GENERAL,
                        "capabilities": [
                            LLMCapability.TEXT_GENERATION,
                            LLMCapability.QUESTION_ANSWERING,
                        ],
                        "max_tokens": 4096,
                        "healthcare_optimized": False,
                    })
                    
                    models.append(LLMModelInfo(
                        name=model_name,
                        display_name=model_info["display_name"],
                        provider=LLMProviderType.OLLAMA,
                        model_type=model_info["model_type"],
                        capabilities=model_info["capabilities"],
                        max_tokens=model_info["max_tokens"],
                        cost_per_1k_tokens=0.0,  # Local models are free
                        supports_streaming=True,
                        supports_functions=False,  # Ollama doesn't support function calling yet
                        supports_structured_output=False,
                        healthcare_optimized=model_info["healthcare_optimized"],
                        hipaa_compliant=True,  # Local deployment is HIPAA compliant
                        local_deployment=True,
                        description=f"Local {model_info['display_name']} deployed via Ollama",
                        version=model_data.get("digest", "")[:12],  # Short digest as version
                    ))
                
                self._models_cache = models
                return models
                
        except aiohttp.ClientError as e:
            raise LLMProviderError(f"Network error: {e}", "ollama")
        except json.JSONDecodeError as e:
            raise LLMProviderError(f"Invalid JSON response: {e}", "ollama")
    
    async def health_check(self) -> LLMProviderStatus:
        """Check health status of Ollama provider."""
        try:
            start_time = time.time()
            
            # Test basic connectivity
            async with self.session.get(f"{self.base_url}/api/version") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    models = await self._get_model_names()
                    return LLMProviderStatus(
                        provider_type=LLMProviderType.OLLAMA,
                        status="healthy",
                        available_models=models,
                        last_check=datetime.utcnow(),
                        response_time=response_time,
                        error_rate=0.0,
                        requests_per_minute=0,  # Local deployment
                        quota_remaining=None,  # No quota limits
                        quota_reset=None,
                    )
                else:
                    return LLMProviderStatus(
                        provider_type=LLMProviderType.OLLAMA,
                        status="degraded",
                        available_models=[],
                        last_check=datetime.utcnow(),
                        response_time=response_time,
                        error_rate=1.0,
                        requests_per_minute=0,
                    )
                    
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return LLMProviderStatus(
                provider_type=LLMProviderType.OLLAMA,
                status="unavailable",
                available_models=[],
                last_check=datetime.utcnow(),
                response_time=None,
                error_rate=1.0,
                requests_per_minute=0,
            )
    
    def supports_capability(self, capability: LLMCapability) -> bool:
        """Check if Ollama supports a specific capability."""
        supported_capabilities = {
            LLMCapability.TEXT_GENERATION,
            LLMCapability.QUESTION_ANSWERING,
            LLMCapability.SUMMARIZATION,
            LLMCapability.CLASSIFICATION,
            LLMCapability.EXTRACTION,
            LLMCapability.REASONING,
            LLMCapability.CODE_GENERATION,  # For code models
        }
        return capability in supported_capabilities
    
    async def estimate_cost(self, request: LLMRequest) -> Optional[float]:
        """Estimate cost for Ollama request."""
        # Local deployment is free (only infrastructure costs)
        return 0.0
    
    async def shutdown(self) -> None:
        """Shutdown the Ollama provider."""
        if self.session:
            await self.session.close()
            self.session = None
        await super().shutdown()
    
    def _convert_messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        """Convert messages to a single prompt for Ollama."""
        prompt_parts = []
        
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _build_options(self, request: LLMRequest) -> Dict[str, Any]:
        """Build options dictionary for Ollama API."""
        options = {}
        
        if request.temperature is not None:
            options["temperature"] = request.temperature
        
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        
        if request.top_p is not None:
            options["top_p"] = request.top_p
        
        if request.frequency_penalty is not None:
            options["frequency_penalty"] = request.frequency_penalty
        
        if request.presence_penalty is not None:
            options["presence_penalty"] = request.presence_penalty
        
        if request.stop:
            options["stop"] = request.stop
        
        if request.seed is not None:
            options["seed"] = request.seed
        
        return options
    
    async def _get_model_names(self) -> List[str]:
        """Get list of available model names."""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [model.get("name", "") for model in data.get("models", [])]
                return []
        except Exception:
            return []


# Healthcare-specific Ollama utilities

async def pull_healthcare_models(provider: OllamaProvider, models: List[str]) -> Dict[str, bool]:
    """
    Pull specific healthcare models to Ollama.
    
    Args:
        provider: Initialized Ollama provider
        models: List of model names to pull
        
    Returns:
        Dictionary mapping model names to success status
    """
    results = {}
    
    for model in models:
        try:
            logger.info(f"Pulling model {model}...")
            
            async with provider.session.post(
                f"{provider.base_url}/api/pull",
                json={"name": model}
            ) as response:
                
                if response.status == 200:
                    # Stream the pull progress
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                status = data.get("status", "")
                                if "pull complete" in status.lower():
                                    results[model] = True
                                    logger.info(f"Successfully pulled {model}")
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    results[model] = False
                    logger.error(f"Failed to pull {model}: HTTP {response.status}")
                    
        except Exception as e:
            results[model] = False
            logger.error(f"Error pulling {model}: {e}")
    
    return results


def get_recommended_healthcare_models() -> List[str]:
    """Get list of recommended healthcare models for Ollama."""
    return [
        "llama2:7b-chat",        # General purpose
        "mistral:7b-instruct",   # Good reasoning
        "codellama:7b-instruct", # For code generation
        # Add healthcare-specific models as they become available
        # "medllama2:7b",        # Uncomment when available
        # "clinicallama:7b",     # Uncomment when available
    ]