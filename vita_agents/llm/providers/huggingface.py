"""
Hugging Face Transformers provider for Vita Agents.

This module provides integration with Hugging Face Transformers library,
enabling both local and API-based access to open-source models.
"""

import asyncio
import logging
import time
import torch
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
from datetime import datetime

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        TextGenerationPipeline,
        pipeline,
        BitsAndBytesConfig,
    )
    from transformers.generation.utils import GenerationConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

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
    LLMModelNotFoundError,
    LLMProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """
    Hugging Face Transformers provider for local and API-based models.
    
    Supports both local model loading and Hugging Face Inference API.
    """
    
    def __init__(self, config: LLMProviderConfig):
        """Initialize Hugging Face provider."""
        super().__init__(config)
        
        if not TRANSFORMERS_AVAILABLE:
            raise LLMProviderError(
                "transformers library not available. Install with: pip install transformers torch",
                "huggingface"
            )
        
        self.use_api = config.api_key is not None
        self.api_base = config.api_base or "https://api-inference.huggingface.co"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Local model cache
        self._local_models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._pipelines: Dict[str, Any] = {}
        
        # Healthcare-specific model configurations
        self.healthcare_models = {
            # General medical models
            "microsoft/BioGPT": {
                "display_name": "BioGPT",
                "model_type": LLMModelType.MEDICAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                ],
                "max_tokens": 1024,
                "healthcare_optimized": True,
                "local_capable": True,
            },
            "microsoft/BioGPT-Large": {
                "display_name": "BioGPT Large",
                "model_type": LLMModelType.MEDICAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.REASONING,
                ],
                "max_tokens": 1024,
                "healthcare_optimized": True,
                "local_capable": True,
            },
            "microsoft/DialoGPT-medium": {
                "display_name": "DialoGPT Medium",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                ],
                "max_tokens": 1024,
                "healthcare_optimized": False,
                "local_capable": True,
            },
            "google/flan-t5-large": {
                "display_name": "FLAN-T5 Large",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.SUMMARIZATION,
                    LLMCapability.TRANSLATION,
                    LLMCapability.CLASSIFICATION,
                ],
                "max_tokens": 512,
                "healthcare_optimized": False,
                "local_capable": True,
            },
            "microsoft/DialoGPT-large": {
                "display_name": "DialoGPT Large",
                "model_type": LLMModelType.GENERAL,
                "capabilities": [
                    LLMCapability.TEXT_GENERATION,
                    LLMCapability.QUESTION_ANSWERING,
                    LLMCapability.REASONING,
                ],
                "max_tokens": 1024,
                "healthcare_optimized": False,
                "local_capable": True,
            },
            # Clinical NER models
            "d4data/biomedical-ner-all": {
                "display_name": "Biomedical NER",
                "model_type": LLMModelType.MEDICAL,
                "capabilities": [
                    LLMCapability.EXTRACTION,
                    LLMCapability.CLASSIFICATION,
                ],
                "max_tokens": 512,
                "healthcare_optimized": True,
                "local_capable": True,
            },
            # Code generation for healthcare
            "microsoft/CodeBERT-base": {
                "display_name": "CodeBERT Base",
                "model_type": LLMModelType.CODE,
                "capabilities": [
                    LLMCapability.CODE_GENERATION,
                    LLMCapability.CLASSIFICATION,
                ],
                "max_tokens": 512,
                "healthcare_optimized": False,
                "local_capable": True,
            },
        }
    
    async def initialize(self) -> None:
        """Initialize the Hugging Face provider."""
        if self._initialized:
            return
        
        try:
            if self.use_api:
                # Test API connectivity
                if not REQUESTS_AVAILABLE:
                    raise LLMProviderError(
                        "requests library not available for API mode",
                        "huggingface"
                    )
                await self._test_api_connection()
            else:
                # For local mode, just verify transformers is available
                logger.info(f"HuggingFace provider initialized for local inference on {self.device}")
            
            self._initialized = True
            logger.info("HuggingFace provider initialized successfully")
            
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize HuggingFace provider: {e}", "huggingface")
    
    async def _test_api_connection(self) -> None:
        """Test API connection."""
        try:
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            response = requests.get(
                f"{self.api_base}/models",
                headers=headers,
                timeout=self.config.timeout
            )
            if response.status_code != 200:
                raise LLMProviderUnavailableError(
                    f"HuggingFace API returned status {response.status_code}",
                    "huggingface"
                )
        except requests.RequestException as e:
            raise LLMProviderUnavailableError(
                f"Cannot connect to HuggingFace API: {e}",
                "huggingface"
            )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using HuggingFace model."""
        if not self._initialized:
            await self.initialize()
        
        await self.validate_request(request)
        
        model_name = request.model or self.config.default_model or "microsoft/BioGPT"
        
        start_time = time.time()
        
        if self.use_api:
            content = await self._generate_api(request, model_name)
        else:
            content = await self._generate_local(request, model_name)
        
        response_time = time.time() - start_time
        
        # Estimate token usage (rough approximation)
        prompt_text = self._messages_to_text(request.messages)
        prompt_tokens = len(prompt_text.split()) * 1.3  # Rough token estimation
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
                "device": self.device if not self.use_api else "api",
                "mode": "api" if self.use_api else "local",
            }
        )
    
    async def _generate_api(self, request: LLMRequest, model_name: str) -> str:
        """Generate using HuggingFace Inference API."""
        prompt = self._messages_to_text(request.messages)
        
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": request.max_tokens or 512,
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 0.9,
                "do_sample": True,
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/models/{model_name}",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code == 404:
                raise LLMModelNotFoundError(f"Model {model_name} not found", "huggingface")
            elif response.status_code != 200:
                raise LLMProviderError(
                    f"HuggingFace API error: {response.text}",
                    "huggingface"
                )
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").replace(prompt, "").strip()
            else:
                return result.get("generated_text", "").replace(prompt, "").strip()
                
        except requests.RequestException as e:
            raise LLMProviderError(f"API request failed: {e}", "huggingface")
    
    async def _generate_local(self, request: LLMRequest, model_name: str) -> str:
        """Generate using local HuggingFace model."""
        # Load model and tokenizer if not cached
        if model_name not in self._pipelines:
            await self._load_model(model_name)
        
        pipeline_obj = self._pipelines[model_name]
        prompt = self._messages_to_text(request.messages)
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _generate():
            try:
                generation_config = {
                    "max_new_tokens": request.max_tokens or 512,
                    "temperature": request.temperature or 0.7,
                    "top_p": request.top_p or 0.9,
                    "do_sample": True,
                    "pad_token_id": pipeline_obj.tokenizer.eos_token_id,
                }
                
                if request.stop:
                    generation_config["eos_token_id"] = pipeline_obj.tokenizer.eos_token_id
                
                result = pipeline_obj(prompt, **generation_config)
                
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get("generated_text", "")
                else:
                    generated = result.get("generated_text", "")
                
                # Remove the original prompt from the response
                return generated.replace(prompt, "").strip()
                
            except Exception as e:
                logger.error(f"Local generation failed: {e}")
                return ""
        
        return await loop.run_in_executor(None, _generate)
    
    async def _load_model(self, model_name: str) -> None:
        """Load model and tokenizer for local inference."""
        try:
            logger.info(f"Loading model {model_name}...")
            
            # Configure for efficient loading
            model_config = {}
            if self.device == "cuda" and torch.cuda.is_available():
                # Use quantization for large models to save GPU memory
                model_config["torch_dtype"] = torch.float16
                if "large" in model_name.lower():
                    model_config["load_in_8bit"] = True
            
            # Create text generation pipeline
            pipeline_obj = pipeline(
                "text-generation",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                **model_config
            )
            
            self._pipelines[model_name] = pipeline_obj
            logger.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            raise LLMProviderError(f"Failed to load model {model_name}: {e}", "huggingface")
    
    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response (not supported by most HF models)."""
        # Most HuggingFace models don't support streaming
        # Fall back to regular generation and yield the complete response
        response = await self.generate(request)
        yield response.content
    
    async def get_available_models(self) -> List[LLMModelInfo]:
        """Get list of available models."""
        if self._models_cache:
            return self._models_cache
        
        models = []
        
        for model_name, model_info in self.healthcare_models.items():
            models.append(LLMModelInfo(
                name=model_name,
                display_name=model_info["display_name"],
                provider=LLMProviderType.HUGGINGFACE,
                model_type=model_info["model_type"],
                capabilities=model_info["capabilities"],
                max_tokens=model_info["max_tokens"],
                cost_per_1k_tokens=0.0 if not self.use_api else None,
                supports_streaming=False,  # Most models don't support streaming
                supports_functions=False,
                supports_structured_output=False,
                healthcare_optimized=model_info["healthcare_optimized"],
                hipaa_compliant=not self.use_api,  # Local deployment is HIPAA compliant
                local_deployment=model_info["local_capable"] and not self.use_api,
                description=f"HuggingFace {model_info['display_name']} model",
            ))
        
        self._models_cache = models
        return models
    
    async def health_check(self) -> LLMProviderStatus:
        """Check health status of HuggingFace provider."""
        try:
            start_time = time.time()
            
            if self.use_api:
                # Test API connectivity
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                response = requests.get(
                    f"{self.api_base}/models",
                    headers=headers,
                    timeout=10
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    status = "healthy"
                    error_rate = 0.0
                else:
                    status = "degraded"
                    error_rate = 1.0
                    
            else:
                # For local mode, check if we can import transformers
                response_time = time.time() - start_time
                status = "healthy"
                error_rate = 0.0
            
            model_names = list(self.healthcare_models.keys())
            
            return LLMProviderStatus(
                provider_type=LLMProviderType.HUGGINGFACE,
                status=status,
                available_models=model_names,
                last_check=datetime.utcnow(),
                response_time=response_time,
                error_rate=error_rate,
                requests_per_minute=0,
                quota_remaining=None,
                quota_reset=None,
            )
            
        except Exception as e:
            logger.error(f"HuggingFace health check failed: {e}")
            return LLMProviderStatus(
                provider_type=LLMProviderType.HUGGINGFACE,
                status="unavailable",
                available_models=[],
                last_check=datetime.utcnow(),
                response_time=None,
                error_rate=1.0,
                requests_per_minute=0,
            )
    
    def supports_capability(self, capability: LLMCapability) -> bool:
        """Check if HuggingFace supports a specific capability."""
        supported_capabilities = {
            LLMCapability.TEXT_GENERATION,
            LLMCapability.QUESTION_ANSWERING,
            LLMCapability.SUMMARIZATION,
            LLMCapability.CLASSIFICATION,
            LLMCapability.EXTRACTION,
            LLMCapability.TRANSLATION,
            LLMCapability.CODE_GENERATION,
            LLMCapability.REASONING,
        }
        return capability in supported_capabilities
    
    async def estimate_cost(self, request: LLMRequest) -> Optional[float]:
        """Estimate cost for HuggingFace request."""
        if not self.use_api:
            return 0.0  # Local models are free
        
        # HuggingFace Inference API pricing varies by model
        # This is a rough estimation
        prompt_text = self._messages_to_text(request.messages)
        estimated_tokens = len(prompt_text.split()) * 1.3 + (request.max_tokens or 512)
        
        # Rough estimate: $0.002 per 1K tokens for most models
        return (estimated_tokens / 1000) * 0.002
    
    async def shutdown(self) -> None:
        """Shutdown the HuggingFace provider."""
        # Clear model cache to free memory
        for model_name in list(self._pipelines.keys()):
            del self._pipelines[model_name]
        
        self._pipelines.clear()
        self._local_models.clear()
        self._tokenizers.clear()
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        await super().shutdown()
    
    def _messages_to_text(self, messages: List[LLMMessage]) -> str:
        """Convert messages to text format."""
        text_parts = []
        
        for message in messages:
            if message.role == "system":
                text_parts.append(f"System: {message.content}")
            elif message.role == "user":
                text_parts.append(f"Human: {message.content}")
            elif message.role == "assistant":
                text_parts.append(f"Assistant: {message.content}")
        
        # Add assistant prompt for generation
        text_parts.append("Assistant:")
        
        return "\n\n".join(text_parts)


def get_recommended_healthcare_models() -> List[str]:
    """Get list of recommended healthcare models for HuggingFace."""
    return [
        "microsoft/BioGPT",                    # Medical text generation
        "microsoft/BioGPT-Large",              # Larger medical model
        "google/flan-t5-large",                # General purpose, good for QA
        "d4data/biomedical-ner-all",           # Medical NER
        "microsoft/CodeBERT-base",             # Code generation
    ]


def get_lightweight_models() -> List[str]:
    """Get list of lightweight models suitable for local deployment."""
    return [
        "microsoft/DialoGPT-medium",           # Conversational AI
        "google/flan-t5-base",                 # Instruction following
        "microsoft/CodeBERT-base",             # Code understanding
        "d4data/biomedical-ner-all",           # Medical NER
    ]