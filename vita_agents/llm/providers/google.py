"""
Google AI provider for Vita Agents.

This module provides integration with Google's AI models including
PaLM and Gemini for healthcare applications.
"""

import asyncio
import json
import time
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
from datetime import datetime

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

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


class GoogleAIProvider(BaseLLMProvider):
    """
    Google AI provider for PaLM and Gemini models.
    
    Provides access to Google's generative AI models including
    Gemini Pro and PaLM for healthcare applications.
    """
    
    def __init__(self, config: LLMProviderConfig):
        """Initialize Google AI provider."""
        super().__init__(config)
        
        if not GOOGLE_AI_AVAILABLE:
            raise LLMProviderError(
                "google-generativeai library not available. Install with: pip install google-generativeai",
                "google"
            )
        
        if not config.api_key:
            raise LLMProviderError(
                "API key is required for Google AI",
                "google"
            )
        
        # Configure Google AI
        genai.configure(api_key=config.api_key)
        
        # Google AI model configurations
        self.google_models = {
            "gemini-pro": {
                "display_name": "Gemini Pro",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.REASONING,
                    LLMCapability.CLASSIFICATION,
                    LLMCapability.EXTRACTION,
                ],
                "max_tokens": 32768,
                "cost_per_1k_tokens": 0.00025,  # Input: $0.00025/1K, Output: $0.0005/1K
                "supports_streaming": True,
            },
            "gemini-pro-vision": {
                "display_name": "Gemini Pro Vision",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.CLASSIFICATION,
                    LLMCapability.EXTRACTION,
                ],
                "max_tokens": 16384,
                "cost_per_1k_tokens": 0.00025,
                "supports_streaming": True,
            },
            "text-bison": {
                "display_name": "PaLM Text Bison",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.CLASSIFICATION,
                ],
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.0005,
                "supports_streaming": False,
            },
            "chat-bison": {
                "display_name": "PaLM Chat Bison",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.REASONING,
                ],
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.0005,
                "supports_streaming": False,
            },
            "embedding-gecko": {
                "display_name": "PaLM Embedding Gecko",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.EXTRACTION,
                    LLMCapability.CLASSIFICATION,
                ],
                "max_tokens": 3072,
                "cost_per_1k_tokens": 0.0001,
                "supports_streaming": False,
            },
        }
        
        self._models: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize the Google AI provider."""
        if self._initialized:
            return
        
        try:
            # Test connection by listing available models
            await self._test_connection()
            
            self._initialized = True
            logger.info("Google AI provider initialized successfully")
            
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Google AI provider: {e}", "google")
    
    async def _test_connection(self) -> None:
        """Test connection to Google AI."""
        try:
            # Try to list models as a connectivity test
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, lambda: list(genai.list_models()))
            
            if not models:
                logger.warning("No models available from Google AI")
                
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise LLMAuthenticationError(f"Authentication failed: {e}", "google")
            else:
                raise LLMProviderError(f"Connection test failed: {e}", "google")
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Google AI model."""
        if not self._initialized:
            await self.initialize()
        
        await self.validate_request(request)
        
        model_name = request.model or self.config.default_model or "gemini-pro"
        
        start_time = time.time()
        
        try:
            # Get or create model instance
            model = await self._get_model(model_name)
            
            # Convert messages to Google AI format
            if "gemini" in model_name.lower():
                content = await self._generate_gemini(model, request)
            else:
                content = await self._generate_palm(model, request)
            
            response_time = time.time() - start_time
            
            # Estimate token usage (Google AI doesn't always provide this)
            prompt_text = " ".join(msg.content for msg in request.messages)
            prompt_tokens = len(prompt_text.split()) * 1.3
            completion_tokens = len(content.split()) * 1.3
            
            return LLMResponse(
                content=content,
                model=model_name,
                usage={
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens),
                },
                finish_reason="stop",
                response_time=response_time,
                metadata={
                    "provider": "google",
                    "model_family": "gemini" if "gemini" in model_name else "palm",
                }
            )
            
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                raise LLMRateLimitError(f"Rate limit or quota exceeded: {e}", "google")
            elif "not found" in error_msg.lower() or "invalid model" in error_msg.lower():
                raise LLMModelNotFoundError(f"Model not found: {e}", "google")
            elif "authentication" in error_msg.lower():
                raise LLMAuthenticationError(f"Authentication error: {e}", "google")
            else:
                raise LLMProviderError(f"Google AI error: {e}", "google")
    
    async def _generate_gemini(self, model: Any, request: LLMRequest) -> str:
        """Generate using Gemini model."""
        # Convert messages to Gemini format
        contents = self._convert_messages_to_gemini(request.messages)
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9,
            max_output_tokens=request.max_tokens or 2048,
        )
        
        if request.stop:
            generation_config.stop_sequences = request.stop[:5]  # Gemini supports max 5
        
        # Configure safety settings for healthcare content
        safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_MEDICAL,
                "threshold": HarmBlockThreshold.BLOCK_NONE,  # Allow medical content
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        ]
        
        loop = asyncio.get_event_loop()
        
        def _generate():
            try:
                response = model.generate_content(
                    contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
                return response.text if response.text else ""
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
                raise
        
        return await loop.run_in_executor(None, _generate)
    
    async def _generate_palm(self, model: Any, request: LLMRequest) -> str:
        """Generate using PaLM model."""
        prompt = self._convert_messages_to_prompt(request.messages)
        
        loop = asyncio.get_event_loop()
        
        def _generate():
            try:
                if "chat" in model.name.lower():
                    # Use chat completion for chat models
                    response = model.generate_text(
                        prompt=prompt,
                        temperature=request.temperature or 0.7,
                        top_p=request.top_p or 0.9,
                        max_output_tokens=request.max_tokens or 1024,
                    )
                else:
                    # Use text completion for text models
                    response = model.generate_text(
                        prompt=prompt,
                        temperature=request.temperature or 0.7,
                        top_p=request.top_p or 0.9,
                        max_output_tokens=request.max_tokens or 1024,
                    )
                
                return response.result if response.result else ""
                
            except Exception as e:
                logger.error(f"PaLM generation failed: {e}")
                raise
        
        return await loop.run_in_executor(None, _generate)
    
    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response using Google AI."""
        if not self._initialized:
            await self.initialize()
        
        await self.validate_request(request)
        
        model_name = request.model or self.config.default_model or "gemini-pro"
        
        # Only Gemini models support streaming
        if "gemini" not in model_name.lower():
            # Fall back to regular generation for non-streaming models
            response = await self.generate(request)
            yield response.content
            return
        
        try:
            model = await self._get_model(model_name)
            contents = self._convert_messages_to_gemini(request.messages)
            
            generation_config = genai.types.GenerationConfig(
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.9,
                max_output_tokens=request.max_tokens or 2048,
            )
            
            safety_settings = [
                {
                    "category": HarmCategory.HARM_CATEGORY_MEDICAL,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
            ]
            
            loop = asyncio.get_event_loop()
            
            def _stream_generate():
                try:
                    response = model.generate_content(
                        contents,
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                        stream=True,
                    )
                    return response
                except Exception as e:
                    logger.error(f"Google AI streaming failed: {e}")
                    raise
            
            stream_response = await loop.run_in_executor(None, _stream_generate)
            
            for chunk in stream_response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            raise LLMProviderError(f"Google AI streaming error: {e}", "google")
    
    async def get_available_models(self) -> List[LLMModelInfo]:
        """Get list of available models from Google AI."""
        if self._models_cache:
            return self._models_cache
        
        models = []
        
        for model_name, model_info in self.google_models.items():
            models.append(LLMModelInfo(
                name=model_name,
                display_name=model_info["display_name"],
                provider=LLMProviderType.GOOGLE,
                model_type=model_info["model_type"],
                capabilities=model_info["capabilities"],
                max_tokens=model_info["max_tokens"],
                cost_per_1k_tokens=model_info["cost_per_1k_tokens"],
                supports_streaming=model_info["supports_streaming"],
                supports_functions=False,  # Google AI doesn't support function calling yet
                supports_structured_output=False,
                healthcare_optimized=False,  # General purpose models
                hipaa_compliant=False,  # Google AI is not HIPAA compliant by default
                local_deployment=False,
                description=f"Google {model_info['display_name']}",
            ))
        
        self._models_cache = models
        return models
    
    async def health_check(self) -> LLMProviderStatus:
        """Check health status of Google AI provider."""
        try:
            start_time = time.time()
            
            # Test by listing models
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, lambda: list(genai.list_models()))
            
            response_time = time.time() - start_time
            
            return LLMProviderStatus(
                provider_type=LLMProviderType.GOOGLE,
                status="healthy",
                available_models=[model.name for model in models],
                last_check=datetime.utcnow(),
                response_time=response_time,
                error_rate=0.0,
                requests_per_minute=0,  # Google manages this
                quota_remaining=None,  # Not exposed by API
                quota_reset=None,
            )
            
        except Exception as e:
            logger.error(f"Google AI health check failed: {e}")
            return LLMProviderStatus(
                provider_type=LLMProviderType.GOOGLE,
                status="unavailable",
                available_models=[],
                last_check=datetime.utcnow(),
                response_time=None,
                error_rate=1.0,
                requests_per_minute=0,
            )
    
    def supports_capability(self, capability: LLMCapability) -> bool:
        """Check if Google AI supports a specific capability."""
        supported_capabilities = {
            LLMCapability.TEXT_GENERATION,
            LLMCapability.QUESTION_ANSWERING,
            LLMCapability.SUMMARIZATION,
            LLMCapability.CLASSIFICATION,
            LLMCapability.EXTRACTION,
            LLMCapability.REASONING,
            LLMCapability.TRANSLATION,
        }
        return capability in supported_capabilities
    
    async def estimate_cost(self, request: LLMRequest) -> Optional[float]:
        """Estimate cost for Google AI request."""
        model_name = request.model or self.config.default_model or "gemini-pro"
        model_info = self.google_models.get(model_name)
        
        if not model_info:
            return None
        
        # Estimate token usage
        prompt_text = " ".join(msg.content for msg in request.messages)
        prompt_tokens = len(prompt_text.split()) * 1.3
        completion_tokens = request.max_tokens or 512
        
        total_tokens = prompt_tokens + completion_tokens
        cost_per_1k = model_info["cost_per_1k_tokens"]
        
        return (total_tokens / 1000) * cost_per_1k
    
    async def shutdown(self) -> None:
        """Shutdown the Google AI provider."""
        self._models.clear()
        await super().shutdown()
    
    async def _get_model(self, model_name: str) -> Any:
        """Get or create a model instance."""
        if model_name not in self._models:
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(None, lambda: genai.GenerativeModel(model_name))
            self._models[model_name] = model
        
        return self._models[model_name]
    
    def _convert_messages_to_gemini(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert messages to Gemini format."""
        contents = []
        
        for message in messages:
            if message.role == "system":
                # Gemini doesn't have a system role, prepend to first user message
                if contents and contents[-1]["role"] == "user":
                    contents[-1]["parts"][0]["text"] = f"System: {message.content}\n\n{contents[-1]['parts'][0]['text']}"
                else:
                    contents.append({
                        "role": "user",
                        "parts": [{"text": f"System: {message.content}"}]
                    })
            elif message.role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": message.content}]
                })
            elif message.role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": message.content}]
                })
        
        return contents
    
    def _convert_messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        """Convert messages to a single prompt for PaLM models."""
        prompt_parts = []
        
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"Human: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        return "\n\n".join(prompt_parts)


def get_recommended_google_models() -> List[str]:
    """Get list of recommended Google AI models for healthcare."""
    return [
        "gemini-pro",            # Best overall model
        "gemini-pro-vision",     # For multimodal tasks
        "text-bison",           # Good for text generation
        "chat-bison",           # Good for conversational AI
    ]