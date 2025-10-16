"""
LLM Router and Factory for intelligent model selection in Vita Agents.

This module provides intelligent routing and selection of LLM providers
based on task requirements, cost constraints, and availability.
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import random

from ..base import (
    BaseLLMProvider,
    LLMProviderType,
    LLMModelType,
    LLMCapability,
    LLMRequest,
    LLMResponse,
    LLMModelInfo,
    LLMProviderConfig,
    LLMProviderStatus,
    LLMProviderError,
    LLMProviderUnavailableError,
)

from ..providers.ollama import OllamaProvider
from ..providers.huggingface import HuggingFaceProvider
from ..providers.azure_openai import AzureOpenAIProvider
from ..providers.google import GoogleAIProvider

# Import existing providers if available
try:
    from openai import AsyncOpenAI
    from ...agents.base_agent import BaseAgent  # For existing OpenAI provider
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Strategies for routing LLM requests."""
    ROUND_ROBIN = "round_robin"
    LEAST_COST = "least_cost"
    BEST_PERFORMANCE = "best_performance"
    CAPABILITY_BASED = "capability_based"
    HEALTH_BASED = "health_based"
    PREFERENCE_BASED = "preference_based"
    LOAD_BALANCED = "load_balanced"


@dataclass
class RoutingCriteria:
    """Criteria for routing decisions."""
    required_capabilities: List[LLMCapability]
    max_cost_per_request: Optional[float] = None
    prefer_local: bool = False
    require_hipaa_compliance: bool = True
    require_healthcare_optimized: bool = False
    max_response_time: Optional[float] = None
    model_type_preference: Optional[LLMModelType] = None
    exclude_providers: List[LLMProviderType] = None


@dataclass
class ProviderMetrics:
    """Metrics for tracking provider performance."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    total_cost: float = 0.0
    last_request_time: Optional[datetime] = None
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    cost_per_request: float = 0.0


class LLMFactory:
    """
    Factory for creating and managing LLM provider instances.
    """
    
    def __init__(self):
        """Initialize the LLM factory."""
        self._provider_classes: Dict[LLMProviderType, Type[BaseLLMProvider]] = {
            LLMProviderType.OLLAMA: OllamaProvider,
            LLMProviderType.HUGGINGFACE: HuggingFaceProvider,
            LLMProviderType.AZURE_OPENAI: AzureOpenAIProvider,
            LLMProviderType.GOOGLE: GoogleAIProvider,
        }
        
        self._providers: Dict[LLMProviderType, BaseLLMProvider] = {}
        self._initialized = False
    
    def register_provider_class(
        self,
        provider_type: LLMProviderType,
        provider_class: Type[BaseLLMProvider]
    ) -> None:
        """Register a custom provider class."""
        self._provider_classes[provider_type] = provider_class
    
    async def create_provider(
        self,
        provider_type: LLMProviderType,
        config: LLMProviderConfig
    ) -> BaseLLMProvider:
        """Create a provider instance."""
        if provider_type not in self._provider_classes:
            raise LLMProviderError(
                f"Unknown provider type: {provider_type}",
                str(provider_type)
            )
        
        provider_class = self._provider_classes[provider_type]
        provider = provider_class(config)
        
        try:
            await provider.initialize()
            return provider
        except Exception as e:
            raise LLMProviderError(
                f"Failed to initialize {provider_type} provider: {e}",
                str(provider_type)
            )
    
    async def get_provider(self, provider_type: LLMProviderType) -> Optional[BaseLLMProvider]:
        """Get an existing provider instance."""
        return self._providers.get(provider_type)
    
    def add_provider(self, provider_type: LLMProviderType, provider: BaseLLMProvider) -> None:
        """Add a provider instance to the factory."""
        self._providers[provider_type] = provider
    
    def remove_provider(self, provider_type: LLMProviderType) -> None:
        """Remove a provider instance."""
        if provider_type in self._providers:
            del self._providers[provider_type]
    
    async def shutdown_all(self) -> None:
        """Shutdown all provider instances."""
        for provider in self._providers.values():
            try:
                await provider.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down provider: {e}")
        self._providers.clear()


class LLMRouter:
    """
    Intelligent router for selecting optimal LLM providers.
    """
    
    def __init__(
        self,
        factory: LLMFactory,
        routing_strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_BASED,
        health_check_interval: int = 300  # 5 minutes
    ):
        """Initialize the LLM router."""
        self.factory = factory
        self.routing_strategy = routing_strategy
        self.health_check_interval = health_check_interval
        
        # Provider metrics and health tracking
        self._provider_metrics: Dict[LLMProviderType, ProviderMetrics] = {}
        self._provider_health: Dict[LLMProviderType, LLMProviderStatus] = {}
        self._last_health_check: Optional[datetime] = None
        
        # Round robin state
        self._round_robin_index = 0
        
        # Circuit breaker state
        self._circuit_breaker: Dict[LLMProviderType, Dict[str, Any]] = {}
    
    async def route_request(
        self,
        request: LLMRequest,
        criteria: Optional[RoutingCriteria] = None
    ) -> BaseLLMProvider:
        """
        Route a request to the optimal provider.
        
        Args:
            request: The LLM request to route
            criteria: Optional routing criteria
            
        Returns:
            Selected provider instance
            
        Raises:
            LLMProviderUnavailableError: If no suitable provider is available
        """
        # Update health status if needed
        await self._update_health_if_needed()
        
        # Get available providers
        available_providers = await self._get_available_providers()
        
        if not available_providers:
            raise LLMProviderUnavailableError(
                "No LLM providers available",
                "router"
            )
        
        # Filter providers based on criteria
        if criteria:
            available_providers = self._filter_providers(available_providers, criteria)
        
        if not available_providers:
            raise LLMProviderUnavailableError(
                "No providers match the specified criteria",
                "router"
            )
        
        # Select provider based on routing strategy
        selected_provider_type = await self._select_provider(
            available_providers,
            request,
            criteria
        )
        
        provider = self.factory._providers[selected_provider_type]
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(selected_provider_type):
            # Try to find alternative
            alternative_providers = [p for p in available_providers if p != selected_provider_type]
            if alternative_providers:
                selected_provider_type = alternative_providers[0]
                provider = self.factory._providers[selected_provider_type]
            else:
                raise LLMProviderUnavailableError(
                    f"Provider {selected_provider_type} circuit breaker is open",
                    str(selected_provider_type)
                )
        
        return provider
    
    async def execute_request(
        self,
        request: LLMRequest,
        criteria: Optional[RoutingCriteria] = None,
        retry_on_failure: bool = True
    ) -> LLMResponse:
        """
        Execute a request with intelligent routing and error handling.
        
        Args:
            request: The LLM request to execute
            criteria: Optional routing criteria
            retry_on_failure: Whether to retry with different providers on failure
            
        Returns:
            LLM response
        """
        start_time = datetime.utcnow()
        attempted_providers = []
        
        while True:
            try:
                # Route request to provider
                provider = await self.route_request(request, criteria)
                provider_type = provider.provider_type
                
                if provider_type in attempted_providers:
                    # Avoid infinite loops
                    break
                
                attempted_providers.append(provider_type)
                
                # Execute request
                response = await provider.generate(request)
                
                # Update metrics on success
                await self._update_metrics_success(
                    provider_type,
                    start_time,
                    response.response_time,
                    await provider.estimate_cost(request) or 0.0
                )
                
                return response
                
            except Exception as e:
                logger.warning(f"Request failed on {provider_type}: {e}")
                
                # Update metrics on failure
                await self._update_metrics_failure(provider_type, start_time)
                
                # Update circuit breaker
                self._update_circuit_breaker(provider_type, False)
                
                if not retry_on_failure or len(attempted_providers) >= 3:
                    # Re-raise the last exception
                    raise
                
                # Try with next provider
                continue
        
        # If we get here, all providers failed
        raise LLMProviderUnavailableError(
            f"All providers failed. Attempted: {attempted_providers}",
            "router"
        )
    
    async def get_provider_recommendations(
        self,
        criteria: RoutingCriteria
    ) -> List[Dict[str, Any]]:
        """
        Get provider recommendations based on criteria.
        
        Args:
            criteria: Routing criteria
            
        Returns:
            List of provider recommendations with scores
        """
        recommendations = []
        
        for provider_type, provider in self.factory._providers.items():
            try:
                models = await provider.get_available_models()
                health = await provider.health_check()
                metrics = self._provider_metrics.get(provider_type, ProviderMetrics())
                
                score = await self._calculate_provider_score(
                    provider_type,
                    models,
                    health,
                    metrics,
                    criteria
                )
                
                recommendations.append({
                    "provider_type": provider_type,
                    "score": score,
                    "health_status": health.status,
                    "available_models": len(models),
                    "avg_response_time": metrics.avg_response_time,
                    "success_rate": metrics.success_rate,
                    "cost_per_request": metrics.cost_per_request,
                })
                
            except Exception as e:
                logger.error(f"Error evaluating provider {provider_type}: {e}")
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations
    
    async def _get_available_providers(self) -> List[LLMProviderType]:
        """Get list of available provider types."""
        available = []
        
        for provider_type, provider in self.factory._providers.items():
            try:
                health = self._provider_health.get(provider_type)
                if health and health.status in ["healthy", "degraded"]:
                    available.append(provider_type)
            except Exception:
                continue
        
        return available
    
    def _filter_providers(
        self,
        providers: List[LLMProviderType],
        criteria: RoutingCriteria
    ) -> List[LLMProviderType]:
        """Filter providers based on criteria."""
        filtered = []
        
        for provider_type in providers:
            provider = self.factory._providers[provider_type]
            
            # Check excluded providers
            if criteria.exclude_providers and provider_type in criteria.exclude_providers:
                continue
            
            # Check required capabilities
            if criteria.required_capabilities:
                supports_all = all(
                    provider.supports_capability(cap)
                    for cap in criteria.required_capabilities
                )
                if not supports_all:
                    continue
            
            # Check HIPAA compliance
            if criteria.require_hipaa_compliance:
                # Local providers and Azure are HIPAA compliant
                if provider_type not in [
                    LLMProviderType.OLLAMA,
                    LLMProviderType.AZURE_OPENAI,
                ] and not criteria.prefer_local:
                    continue
            
            # Check local preference
            if criteria.prefer_local and provider_type not in [
                LLMProviderType.OLLAMA,
                LLMProviderType.HUGGINGFACE,  # If configured for local use
            ]:
                continue
            
            filtered.append(provider_type)
        
        return filtered
    
    async def _select_provider(
        self,
        providers: List[LLMProviderType],
        request: LLMRequest,
        criteria: Optional[RoutingCriteria]
    ) -> LLMProviderType:
        """Select a provider based on routing strategy."""
        if len(providers) == 1:
            return providers[0]
        
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(providers)
        
        elif self.routing_strategy == RoutingStrategy.LEAST_COST:
            return await self._select_least_cost(providers, request)
        
        elif self.routing_strategy == RoutingStrategy.BEST_PERFORMANCE:
            return self._select_best_performance(providers)
        
        elif self.routing_strategy == RoutingStrategy.HEALTH_BASED:
            return self._select_health_based(providers)
        
        elif self.routing_strategy == RoutingStrategy.LOAD_BALANCED:
            return self._select_load_balanced(providers)
        
        else:  # CAPABILITY_BASED or default
            return await self._select_capability_based(providers, request, criteria)
    
    def _select_round_robin(self, providers: List[LLMProviderType]) -> LLMProviderType:
        """Select provider using round robin."""
        provider = providers[self._round_robin_index % len(providers)]
        self._round_robin_index += 1
        return provider
    
    async def _select_least_cost(
        self,
        providers: List[LLMProviderType],
        request: LLMRequest
    ) -> LLMProviderType:
        """Select provider with lowest estimated cost."""
        best_provider = providers[0]
        best_cost = float('inf')
        
        for provider_type in providers:
            provider = self.factory._providers[provider_type]
            try:
                cost = await provider.estimate_cost(request)
                if cost is not None and cost < best_cost:
                    best_cost = cost
                    best_provider = provider_type
            except Exception:
                continue
        
        return best_provider
    
    def _select_best_performance(self, providers: List[LLMProviderType]) -> LLMProviderType:
        """Select provider with best performance metrics."""
        best_provider = providers[0]
        best_score = -1.0
        
        for provider_type in providers:
            metrics = self._provider_metrics.get(provider_type, ProviderMetrics())
            
            # Calculate performance score (higher is better)
            score = metrics.success_rate * 0.7 + (1.0 / (metrics.avg_response_time + 1)) * 0.3
            
            if score > best_score:
                best_score = score
                best_provider = provider_type
        
        return best_provider
    
    def _select_health_based(self, providers: List[LLMProviderType]) -> LLMProviderType:
        """Select provider based on health status."""
        # Prefer healthy providers over degraded ones
        healthy_providers = []
        degraded_providers = []
        
        for provider_type in providers:
            health = self._provider_health.get(provider_type)
            if health:
                if health.status == "healthy":
                    healthy_providers.append(provider_type)
                elif health.status == "degraded":
                    degraded_providers.append(provider_type)
        
        if healthy_providers:
            return random.choice(healthy_providers)
        elif degraded_providers:
            return random.choice(degraded_providers)
        else:
            return random.choice(providers)
    
    def _select_load_balanced(self, providers: List[LLMProviderType]) -> LLMProviderType:
        """Select provider with lowest current load."""
        # For now, use success rate as a proxy for load
        # In production, this could be based on active request counts
        return self._select_best_performance(providers)
    
    async def _select_capability_based(
        self,
        providers: List[LLMProviderType],
        request: LLMRequest,
        criteria: Optional[RoutingCriteria]
    ) -> LLMProviderType:
        """Select provider based on capabilities and requirements."""
        scores = {}
        
        for provider_type in providers:
            provider = self.factory._providers[provider_type]
            models = await provider.get_available_models()
            
            score = 0.0
            
            # Score based on model availability
            score += len(models) * 0.1
            
            # Score based on healthcare optimization
            healthcare_models = [m for m in models if m.healthcare_optimized]
            if healthcare_models:
                score += 2.0
            
            # Score based on local deployment (if preferred)
            if criteria and criteria.prefer_local:
                local_models = [m for m in models if m.local_deployment]
                if local_models:
                    score += 3.0
            
            # Score based on HIPAA compliance
            if criteria and criteria.require_hipaa_compliance:
                hipaa_models = [m for m in models if m.hipaa_compliant]
                if hipaa_models:
                    score += 2.0
            
            # Score based on performance metrics
            metrics = self._provider_metrics.get(provider_type, ProviderMetrics())
            score += metrics.success_rate * 1.0
            score += (1.0 / (metrics.avg_response_time + 1)) * 0.5
            
            scores[provider_type] = score
        
        # Return provider with highest score
        return max(scores.keys(), key=lambda k: scores[k])
    
    async def _update_health_if_needed(self) -> None:
        """Update provider health status if needed."""
        now = datetime.utcnow()
        
        if (self._last_health_check is None or 
            (now - self._last_health_check).seconds >= self.health_check_interval):
            
            await self._update_all_health()
            self._last_health_check = now
    
    async def _update_all_health(self) -> None:
        """Update health status for all providers."""
        tasks = []
        
        for provider_type, provider in self.factory._providers.items():
            task = asyncio.create_task(self._update_provider_health(provider_type, provider))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _update_provider_health(
        self,
        provider_type: LLMProviderType,
        provider: BaseLLMProvider
    ) -> None:
        """Update health status for a single provider."""
        try:
            health = await provider.health_check()
            self._provider_health[provider_type] = health
        except Exception as e:
            logger.error(f"Health check failed for {provider_type}: {e}")
            # Create a failed health status
            self._provider_health[provider_type] = LLMProviderStatus(
                provider_type=provider_type,
                status="unavailable",
                available_models=[],
                last_check=datetime.utcnow(),
                response_time=None,
                error_rate=1.0,
                requests_per_minute=0,
            )
    
    async def _update_metrics_success(
        self,
        provider_type: LLMProviderType,
        start_time: datetime,
        response_time: float,
        cost: float
    ) -> None:
        """Update metrics on successful request."""
        if provider_type not in self._provider_metrics:
            self._provider_metrics[provider_type] = ProviderMetrics()
        
        metrics = self._provider_metrics[provider_type]
        metrics.total_requests += 1
        metrics.successful_requests += 1
        metrics.total_response_time += response_time
        metrics.total_cost += cost
        metrics.last_request_time = datetime.utcnow()
        
        # Update averages
        metrics.avg_response_time = metrics.total_response_time / metrics.total_requests
        metrics.success_rate = metrics.successful_requests / metrics.total_requests
        metrics.cost_per_request = metrics.total_cost / metrics.total_requests
        
        # Update circuit breaker
        self._update_circuit_breaker(provider_type, True)
    
    async def _update_metrics_failure(
        self,
        provider_type: LLMProviderType,
        start_time: datetime
    ) -> None:
        """Update metrics on failed request."""
        if provider_type not in self._provider_metrics:
            self._provider_metrics[provider_type] = ProviderMetrics()
        
        metrics = self._provider_metrics[provider_type]
        metrics.total_requests += 1
        metrics.failed_requests += 1
        metrics.last_request_time = datetime.utcnow()
        
        # Update averages
        if metrics.total_requests > 0:
            metrics.success_rate = metrics.successful_requests / metrics.total_requests
    
    def _update_circuit_breaker(self, provider_type: LLMProviderType, success: bool) -> None:
        """Update circuit breaker state."""
        if provider_type not in self._circuit_breaker:
            self._circuit_breaker[provider_type] = {
                "state": "closed",  # closed, open, half_open
                "failure_count": 0,
                "last_failure_time": None,
                "failure_threshold": 5,
                "timeout": 60,  # seconds
            }
        
        cb = self._circuit_breaker[provider_type]
        
        if success:
            cb["failure_count"] = 0
            if cb["state"] == "half_open":
                cb["state"] = "closed"
        else:
            cb["failure_count"] += 1
            cb["last_failure_time"] = datetime.utcnow()
            
            if cb["failure_count"] >= cb["failure_threshold"]:
                cb["state"] = "open"
    
    def _is_circuit_breaker_open(self, provider_type: LLMProviderType) -> bool:
        """Check if circuit breaker is open for a provider."""
        if provider_type not in self._circuit_breaker:
            return False
        
        cb = self._circuit_breaker[provider_type]
        
        if cb["state"] == "closed":
            return False
        elif cb["state"] == "open":
            # Check if timeout has passed
            if cb["last_failure_time"]:
                time_since_failure = (datetime.utcnow() - cb["last_failure_time"]).seconds
                if time_since_failure >= cb["timeout"]:
                    cb["state"] = "half_open"
                    return False
            return True
        else:  # half_open
            return False
    
    async def _calculate_provider_score(
        self,
        provider_type: LLMProviderType,
        models: List[LLMModelInfo],
        health: LLMProviderStatus,
        metrics: ProviderMetrics,
        criteria: RoutingCriteria
    ) -> float:
        """Calculate a score for a provider based on various factors."""
        score = 0.0
        
        # Health score (0-30 points)
        if health.status == "healthy":
            score += 30
        elif health.status == "degraded":
            score += 15
        else:
            score += 0
        
        # Performance score (0-25 points)
        score += metrics.success_rate * 20
        if metrics.avg_response_time > 0:
            score += min(5, 5.0 / metrics.avg_response_time)
        
        # Capability score (0-20 points)
        capability_matches = 0
        for capability in criteria.required_capabilities:
            for model in models:
                if capability in model.capabilities:
                    capability_matches += 1
                    break
        
        if criteria.required_capabilities:
            capability_score = (capability_matches / len(criteria.required_capabilities)) * 20
            score += capability_score
        
        # Healthcare optimization score (0-15 points)
        healthcare_models = [m for m in models if m.healthcare_optimized]
        if healthcare_models:
            score += min(15, len(healthcare_models) * 3)
        
        # HIPAA compliance score (0-10 points)
        if criteria.require_hipaa_compliance:
            hipaa_models = [m for m in models if m.hipaa_compliant]
            if hipaa_models:
                score += 10
        
        return score


# Utility functions

def create_routing_criteria(
    capabilities: List[LLMCapability],
    **kwargs
) -> RoutingCriteria:
    """Convenience function to create routing criteria."""
    return RoutingCriteria(
        required_capabilities=capabilities,
        **kwargs
    )


def create_healthcare_criteria(
    prefer_local: bool = False,
    require_hipaa: bool = True
) -> RoutingCriteria:
    """Create criteria optimized for healthcare use cases."""
    return RoutingCriteria(
        required_capabilities=[
            LLMCapability.TEXT_GENERATION,
            LLMCapability.QUESTION_ANSWERING,
        ],
        prefer_local=prefer_local,
        require_hipaa_compliance=require_hipaa,
        require_healthcare_optimized=True,
        model_type_preference=LLMModelType.MEDICAL,
    )