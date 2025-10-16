"""
LLM Integration Manager for Vita Agents
Supports multiple LLM providers including local Ollama models
"""

import json
import os
import requests
import asyncio
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    LOCAL_MODEL = "local_model"

@dataclass
class LLMModel:
    name: str
    provider: LLMProvider
    endpoint: str
    context_length: int
    capabilities: List[str]
    healthcare_optimized: bool = False
    cost_per_token: float = 0.0
    
class LLMManager:
    def __init__(self):
        self.models = {}
        self.active_model = None
        self.config = self._load_config()
        self._initialize_models()
    
    def _load_config(self) -> Dict:
        """Load LLM configuration from config file or environment"""
        config_path = "config/llm_config.json"
        default_config = {
            "ollama": {
                "base_url": "http://localhost:11434",
                "enabled": True
            },
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "enabled": bool(os.getenv("OPENAI_API_KEY"))
            },
            "anthropic": {
                "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                "enabled": bool(os.getenv("ANTHROPIC_API_KEY"))
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default config
            os.makedirs("config", exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _initialize_models(self):
        """Initialize available LLM models"""
        # Ollama Models (Local)
        if self.config.get("ollama", {}).get("enabled"):
            self._discover_ollama_models()
        
        # OpenAI Models
        if self.config.get("openai", {}).get("enabled"):
            self._add_openai_models()
        
        # Anthropic Models
        if self.config.get("anthropic", {}).get("enabled"):
            self._add_anthropic_models()
        
        # Healthcare-specific models
        self._add_healthcare_models()
    
    def _discover_ollama_models(self):
        """Discover available Ollama models"""
        try:
            base_url = self.config["ollama"]["base_url"]
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                for model_info in data.get("models", []):
                    model_name = model_info["name"]
                    
                    # Determine healthcare optimization based on model name
                    healthcare_optimized = any(keyword in model_name.lower() for keyword in 
                                             ["med", "bio", "clinical", "health", "llama2-med", "meditron"])
                    
                    self.models[f"ollama:{model_name}"] = LLMModel(
                        name=model_name,
                        provider=LLMProvider.OLLAMA,
                        endpoint=f"{base_url}/api/generate",
                        context_length=self._get_context_length(model_name),
                        capabilities=["text_generation", "medical_analysis", "clinical_reasoning"],
                        healthcare_optimized=healthcare_optimized,
                        cost_per_token=0.0  # Local models are free
                    )
                    
                logger.info(f"Discovered {len(data.get('models', []))} Ollama models")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")
    
    def _add_openai_models(self):
        """Add OpenAI models"""
        openai_models = [
            ("gpt-4-turbo", 128000, ["text_generation", "medical_analysis", "reasoning"], 0.01),
            ("gpt-4", 8192, ["text_generation", "medical_analysis", "reasoning"], 0.03),
            ("gpt-3.5-turbo", 16384, ["text_generation", "medical_analysis"], 0.001),
        ]
        
        for name, context, capabilities, cost in openai_models:
            self.models[f"openai:{name}"] = LLMModel(
                name=name,
                provider=LLMProvider.OPENAI,
                endpoint="https://api.openai.com/v1/chat/completions",
                context_length=context,
                capabilities=capabilities,
                healthcare_optimized=False,
                cost_per_token=cost
            )
    
    def _add_anthropic_models(self):
        """Add Anthropic models"""
        anthropic_models = [
            ("claude-3-opus", 200000, ["text_generation", "medical_analysis", "reasoning"], 0.015),
            ("claude-3-sonnet", 200000, ["text_generation", "medical_analysis"], 0.003),
            ("claude-3-haiku", 200000, ["text_generation"], 0.00025),
        ]
        
        for name, context, capabilities, cost in anthropic_models:
            self.models[f"anthropic:{name}"] = LLMModel(
                name=name,
                provider=LLMProvider.ANTHROPIC,
                endpoint="https://api.anthropic.com/v1/messages",
                context_length=context,
                capabilities=capabilities,
                healthcare_optimized=False,
                cost_per_token=cost
            )
    
    def _add_healthcare_models(self):
        """Add specialized healthcare models"""
        healthcare_models = [
            {
                "name": "med-llama2-7b",
                "provider": LLMProvider.HUGGINGFACE,
                "endpoint": "https://api-inference.huggingface.co/models/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                "context_length": 4096,
                "capabilities": ["medical_ner", "clinical_reasoning", "drug_interactions"],
                "healthcare_optimized": True
            },
            {
                "name": "clinical-bert",
                "provider": LLMProvider.HUGGINGFACE,
                "endpoint": "https://api-inference.huggingface.co/models/emilyalsentzer/Bio_ClinicalBERT",
                "context_length": 512,
                "capabilities": ["medical_ner", "clinical_classification"],
                "healthcare_optimized": True
            }
        ]
        
        for model_info in healthcare_models:
            model_key = f"{model_info['provider'].value}:{model_info['name']}"
            self.models[model_key] = LLMModel(**model_info)
    
    def _get_context_length(self, model_name: str) -> int:
        """Estimate context length based on model name"""
        model_name_lower = model_name.lower()
        if "llama2" in model_name_lower:
            return 4096
        elif "mistral" in model_name_lower:
            return 8192
        elif "codellama" in model_name_lower:
            return 16384
        elif "llama3" in model_name_lower:
            return 8192
        else:
            return 2048  # Default
    
    def get_available_models(self) -> Dict[str, LLMModel]:
        """Get all available models"""
        return self.models
    
    def get_healthcare_models(self) -> Dict[str, LLMModel]:
        """Get models optimized for healthcare"""
        return {k: v for k, v in self.models.items() if v.healthcare_optimized}
    
    def set_active_model(self, model_key: str) -> bool:
        """Set the active model"""
        if model_key in self.models:
            self.active_model = model_key
            logger.info(f"Set active model: {model_key}")
            return True
        return False
    
    def get_active_model(self) -> Optional[LLMModel]:
        """Get the currently active model"""
        if self.active_model and self.active_model in self.models:
            return self.models[self.active_model]
        return None
    
    async def generate_response(self, prompt: str, context: str = "", temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response using the active model"""
        if not self.active_model:
            return {"error": "No active model selected"}
        
        model = self.models[self.active_model]
        
        try:
            if model.provider == LLMProvider.OLLAMA:
                return await self._generate_ollama(prompt, context, temperature, max_tokens, model)
            elif model.provider == LLMProvider.OPENAI:
                return await self._generate_openai(prompt, context, temperature, max_tokens, model)
            elif model.provider == LLMProvider.ANTHROPIC:
                return await self._generate_anthropic(prompt, context, temperature, max_tokens, model)
            else:
                return {"error": f"Provider {model.provider} not implemented"}
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": str(e)}
    
    async def _generate_ollama(self, prompt: str, context: str, temperature: float, max_tokens: int, model: LLMModel) -> Dict[str, Any]:
        """Generate response using Ollama"""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        payload = {
            "model": model.name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        async with asyncio.timeout(30):  # 30 second timeout
            response = requests.post(model.endpoint, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "response": data.get("response", ""),
                    "model": model.name,
                    "provider": model.provider.value,
                    "tokens_used": len(data.get("response", "").split()),
                    "cost": 0.0
                }
            else:
                return {"error": f"Ollama API error: {response.status_code}"}
    
    async def _generate_openai(self, prompt: str, context: str, temperature: float, max_tokens: int, model: LLMModel) -> Dict[str, Any]:
        """Generate response using OpenAI"""
        # This would require the OpenAI API key and proper implementation
        return {"error": "OpenAI integration requires API key configuration"}
    
    async def _generate_anthropic(self, prompt: str, context: str, temperature: float, max_tokens: int, model: LLMModel) -> Dict[str, Any]:
        """Generate response using Anthropic"""
        # This would require the Anthropic API key and proper implementation
        return {"error": "Anthropic integration requires API key configuration"}
    
    def get_model_recommendations(self, use_case: str) -> List[str]:
        """Get model recommendations for specific use cases"""
        recommendations = []
        
        use_case_lower = use_case.lower()
        
        if "diagnosis" in use_case_lower or "clinical" in use_case_lower:
            # Prefer healthcare-optimized models for clinical tasks
            healthcare_models = [k for k, v in self.models.items() if v.healthcare_optimized]
            recommendations.extend(healthcare_models[:3])
            
            # Add general purpose models with good reasoning
            general_models = [k for k, v in self.models.items() 
                            if "reasoning" in v.capabilities and not v.healthcare_optimized]
            recommendations.extend(general_models[:2])
        
        elif "drug" in use_case_lower or "interaction" in use_case_lower:
            # Models good at drug interactions
            drug_models = [k for k, v in self.models.items() 
                          if "drug_interactions" in v.capabilities]
            recommendations.extend(drug_models)
        
        else:
            # General purpose recommendations
            all_models = list(self.models.keys())
            recommendations.extend(all_models[:5])
        
        return recommendations[:5]  # Return top 5 recommendations

# Global LLM manager instance
llm_manager = LLMManager()

# Clinical prompt templates
CLINICAL_PROMPTS = {
    "diagnosis": """
You are an expert clinical decision support system. Analyze the following patient presentation and provide a differential diagnosis.

Patient Information:
- Age: {age}
- Gender: {gender}
- Chief Complaint: {chief_complaint}
- History of Present Illness: {hpi}
- Vital Signs: {vitals}
- Physical Exam: {physical_exam}

Please provide:
1. Top 3 differential diagnoses with probability estimates
2. Recommended diagnostic tests
3. Immediate treatment considerations
4. Red flags or concerning features

Respond in a structured, clinical format suitable for healthcare professionals.
""",
    
    "drug_interaction": """
You are a clinical pharmacist reviewing potential drug interactions.

Current Medications:
{current_medications}

New Medication to Add:
{new_medication}

Please analyze:
1. Potential drug-drug interactions
2. Severity level of interactions
3. Clinical significance
4. Monitoring recommendations
5. Alternative medications if contraindicated

Provide evidence-based recommendations with clinical rationale.
""",
    
    "treatment_plan": """
You are creating a comprehensive treatment plan for the following patient:

Diagnosis: {diagnosis}
Patient: {age}-year-old {gender}
Comorbidities: {comorbidities}
Current Medications: {medications}

Create a detailed treatment plan including:
1. Pharmacological interventions
2. Non-pharmacological approaches
3. Monitoring parameters
4. Follow-up schedule
5. Patient education points

Ensure all recommendations follow current clinical guidelines.
"""
}