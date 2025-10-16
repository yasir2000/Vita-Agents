"""
Medical Foundation Models Provider for Advanced Healthcare AI.

This module provides integration with state-of-the-art medical foundation models
including Med-PaLM 2, ChatGPT-4 Medical, BioGPT, ClinicalBERT, and multimodal
medical AI systems.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import structlog
from pydantic import BaseModel, Field

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        pipeline, BertTokenizer, BertModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import google.generativeai as genai
    from google.ai.generativelanguage_v1beta import GenerativeServiceClient
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = structlog.get_logger(__name__)


class ModelType(Enum):
    """Types of medical foundation models."""
    MED_PALM_2 = "med_palm_2"
    CHATGPT_4_MEDICAL = "chatgpt_4_medical"
    BIO_GPT = "bio_gpt"
    CLINICAL_BERT = "clinical_bert"
    MULTIMODAL_MEDICAL = "multimodal_medical"
    FLAMINGO_MEDICAL = "flamingo_medical"


class MedicalTask(Enum):
    """Types of medical AI tasks."""
    CLINICAL_REASONING = "clinical_reasoning"
    DIAGNOSIS_SUPPORT = "diagnosis_support"
    DRUG_DISCOVERY = "drug_discovery"
    MEDICAL_QA = "medical_qa"
    CLINICAL_NOTE_ANALYSIS = "clinical_note_analysis"
    MEDICAL_IMAGE_ANALYSIS = "medical_image_analysis"
    BIOMARKER_PREDICTION = "biomarker_prediction"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"


class MedicalFoundationModelRequest(BaseModel):
    """Request for medical foundation model inference."""
    
    model_type: ModelType
    task: MedicalTask
    input_text: str
    patient_context: Optional[Dict[str, Any]] = None
    clinical_data: Optional[Dict[str, Any]] = None
    image_data: Optional[List[str]] = None  # Base64 encoded images
    lab_results: Optional[List[Dict[str, Any]]] = None
    medication_history: Optional[List[str]] = None
    temperature: float = 0.3
    max_tokens: int = 1024
    top_p: float = 0.9


class MedicalFoundationModelResponse(BaseModel):
    """Response from medical foundation model."""
    
    model_type: ModelType
    task: MedicalTask
    response_text: str
    confidence_score: float
    clinical_reasoning: Optional[str] = None
    evidence_citations: List[str] = Field(default_factory=list)
    safety_warnings: List[str] = Field(default_factory=list)
    differential_diagnoses: Optional[List[Dict[str, Any]]] = None
    treatment_options: Optional[List[Dict[str, Any]]] = None
    follow_up_recommendations: Optional[List[str]] = None
    processing_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseMedicalFoundationModel(ABC):
    """Base class for medical foundation models."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    @abstractmethod
    async def generate_response(
        self, 
        request: MedicalFoundationModelRequest
    ) -> MedicalFoundationModelResponse:
        """Generate response using the medical foundation model."""
        pass
    
    @abstractmethod
    async def validate_clinical_context(self, context: Dict[str, Any]) -> bool:
        """Validate clinical context for safety."""
        pass
    
    def _format_clinical_prompt(
        self, 
        task: MedicalTask, 
        input_text: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format prompt for clinical tasks."""
        
        base_prompts = {
            MedicalTask.CLINICAL_REASONING: """
You are an expert clinical AI assistant. Analyze the following clinical scenario and provide detailed clinical reasoning.

Clinical Scenario: {input_text}

{patient_context}

Please provide:
1. Clinical assessment and differential diagnoses
2. Relevant clinical reasoning and evidence
3. Recommended diagnostic workup
4. Treatment considerations
5. Safety considerations and contraindications

Response should be evidence-based and include clinical reasoning.
            """,
            
            MedicalTask.DIAGNOSIS_SUPPORT: """
You are a medical diagnostic AI assistant. Analyze the presented case and provide diagnostic support.

Patient Presentation: {input_text}

{patient_context}

Please provide:
1. Primary differential diagnoses with likelihood scores
2. Supporting and contradicting evidence for each diagnosis
3. Additional diagnostic tests recommended
4. Red flags or urgent considerations
5. Confidence assessment

Ensure all recommendations are evidence-based and appropriate.
            """,
            
            MedicalTask.MEDICAL_QA: """
You are a medical knowledge AI. Answer the following medical question with accurate, evidence-based information.

Question: {input_text}

{patient_context}

Please provide:
1. Comprehensive answer based on current medical evidence
2. Relevant clinical guidelines or recommendations
3. Any important caveats or contraindications
4. References to supporting literature when applicable

Ensure accuracy and clinical relevance.
            """,
            
            MedicalTask.TREATMENT_RECOMMENDATION: """
You are a clinical treatment AI assistant. Provide evidence-based treatment recommendations.

Clinical Situation: {input_text}

{patient_context}

Please provide:
1. First-line treatment recommendations
2. Alternative treatment options
3. Contraindications and safety considerations
4. Monitoring parameters
5. Expected outcomes and follow-up

All recommendations should follow current clinical guidelines.
            """
        }
        
        prompt_template = base_prompts.get(task, "Analyze the following medical scenario: {input_text}")
        
        # Format patient context
        context_str = ""
        if patient_context:
            context_str = f"Patient Context:\n"
            for key, value in patient_context.items():
                context_str += f"- {key}: {value}\n"
        
        return prompt_template.format(
            input_text=input_text,
            patient_context=context_str
        )


class MedPaLM2Provider(BaseMedicalFoundationModel):
    """Google's Med-PaLM 2 medical foundation model provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("med_palm_2", config)
        
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("Google AI library not available. Install with: pip install google-generativeai")
        
        self.api_key = config.get("google_api_key")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel('gemini-pro')  # Using Gemini as Med-PaLM proxy
        
    async def generate_response(
        self, 
        request: MedicalFoundationModelRequest
    ) -> MedicalFoundationModelResponse:
        """Generate response using Med-PaLM 2."""
        
        start_time = datetime.utcnow()
        
        try:
            # Format clinical prompt
            prompt = self._format_clinical_prompt(
                request.task, 
                request.input_text,
                request.patient_context
            )
            
            # Add medical context enhancement
            enhanced_prompt = f"""
You are Med-PaLM 2, a state-of-the-art medical AI with expert-level performance.
Your responses should be:
- Clinically accurate and evidence-based
- Appropriate for healthcare professionals
- Include confidence assessments
- Highlight safety considerations

{prompt}

Important: This is for clinical decision support. Always emphasize the need for professional medical judgment.
            """
            
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                enhanced_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=request.temperature,
                    max_output_tokens=request.max_tokens,
                    top_p=request.top_p
                )
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Parse response for clinical components
            response_text = response.text
            clinical_reasoning = self._extract_clinical_reasoning(response_text)
            safety_warnings = self._extract_safety_warnings(response_text)
            
            return MedicalFoundationModelResponse(
                model_type=ModelType.MED_PALM_2,
                task=request.task,
                response_text=response_text,
                confidence_score=0.85,  # High confidence for Med-PaLM 2
                clinical_reasoning=clinical_reasoning,
                safety_warnings=safety_warnings,
                processing_time_ms=processing_time,
                metadata={
                    "model_version": "med_palm_2_proxy",
                    "prompt_tokens": len(enhanced_prompt.split()),
                    "completion_tokens": len(response_text.split())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Med-PaLM 2 generation failed: {e}")
            raise
    
    async def validate_clinical_context(self, context: Dict[str, Any]) -> bool:
        """Validate clinical context for Med-PaLM 2."""
        required_fields = ["patient_age", "gender"]
        return all(field in context for field in required_fields)
    
    def _extract_clinical_reasoning(self, text: str) -> str:
        """Extract clinical reasoning from response."""
        # Simple extraction - in production would use more sophisticated NLP
        lines = text.split('\n')
        reasoning_lines = [line for line in lines if 'reasoning' in line.lower() or 'rationale' in line.lower()]
        return '\n'.join(reasoning_lines) if reasoning_lines else ""
    
    def _extract_safety_warnings(self, text: str) -> List[str]:
        """Extract safety warnings from response."""
        warnings = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['contraindication', 'warning', 'caution', 'avoid']):
                warnings.append(line.strip())
        return warnings


class ChatGPT4MedicalProvider(BaseMedicalFoundationModel):
    """OpenAI's ChatGPT-4 adapted for medical applications."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("chatgpt_4_medical", config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.client = openai.AsyncOpenAI(
            api_key=config.get("openai_api_key"),
            organization=config.get("openai_organization")
        )
        
        self.model_name = config.get("model_name", "gpt-4-turbo-preview")
    
    async def generate_response(
        self, 
        request: MedicalFoundationModelRequest
    ) -> MedicalFoundationModelResponse:
        """Generate response using ChatGPT-4 Medical."""
        
        start_time = datetime.utcnow()
        
        try:
            # Format medical prompt
            prompt = self._format_clinical_prompt(
                request.task, 
                request.input_text,
                request.patient_context
            )
            
            # Add medical specialization system message
            system_message = """
You are a highly specialized medical AI assistant based on ChatGPT-4, specifically adapted for healthcare applications. You have extensive knowledge of:

- Clinical medicine and pathophysiology
- Diagnostic reasoning and differential diagnosis
- Evidence-based treatment guidelines
- Drug interactions and pharmacology
- Medical imaging interpretation
- Laboratory result analysis

Your responses should:
- Be clinically accurate and evidence-based
- Include confidence levels for recommendations
- Highlight safety considerations and contraindications
- Reference relevant medical literature when appropriate
- Emphasize the importance of clinical judgment

Always remind users that AI assistance does not replace professional medical judgment.
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Add clinical context to messages if available
            if request.clinical_data:
                clinical_context = f"Additional Clinical Data: {json.dumps(request.clinical_data, indent=2)}"
                messages.append({"role": "user", "content": clinical_context})
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            response_text = response.choices[0].message.content
            
            # Extract clinical components
            differential_diagnoses = self._extract_differential_diagnoses(response_text)
            treatment_options = self._extract_treatment_options(response_text)
            follow_up_recommendations = self._extract_follow_up_recommendations(response_text)
            
            return MedicalFoundationModelResponse(
                model_type=ModelType.CHATGPT_4_MEDICAL,
                task=request.task,
                response_text=response_text,
                confidence_score=0.88,  # High confidence for GPT-4 medical
                differential_diagnoses=differential_diagnoses,
                treatment_options=treatment_options,
                follow_up_recommendations=follow_up_recommendations,
                processing_time_ms=processing_time,
                metadata={
                    "model": self.model_name,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
            
        except Exception as e:
            self.logger.error(f"ChatGPT-4 Medical generation failed: {e}")
            raise
    
    async def validate_clinical_context(self, context: Dict[str, Any]) -> bool:
        """Validate clinical context for ChatGPT-4 Medical."""
        # More flexible validation for GPT-4
        return isinstance(context, dict) and len(context) > 0
    
    def _extract_differential_diagnoses(self, text: str) -> List[Dict[str, Any]]:
        """Extract differential diagnoses from response."""
        # Simplified extraction - would use more sophisticated NLP in production
        diagnoses = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if 'differential' in line.lower() and 'diagnos' in line.lower():
                # Look for numbered or bulleted list following
                for j in range(i+1, min(i+6, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and (next_line.startswith(('1.', '2.', '3.', '-', '•'))):
                        # Extract diagnosis name and add to list
                        diagnosis_text = next_line.split('.', 1)[-1].strip()
                        if diagnosis_text:
                            diagnoses.append({
                                "diagnosis": diagnosis_text,
                                "likelihood": "moderate",  # Default
                                "supporting_evidence": []
                            })
        
        return diagnoses[:5]  # Limit to top 5
    
    def _extract_treatment_options(self, text: str) -> List[Dict[str, Any]]:
        """Extract treatment options from response."""
        treatments = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if 'treatment' in line.lower() and any(keyword in line.lower() for keyword in ['option', 'recommend', 'therapy']):
                # Look for following treatments
                for j in range(i+1, min(i+6, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and (next_line.startswith(('1.', '2.', '3.', '-', '•'))):
                        treatment_text = next_line.split('.', 1)[-1].strip()
                        if treatment_text:
                            treatments.append({
                                "treatment": treatment_text,
                                "category": "first_line",  # Default
                                "evidence_level": "moderate"
                            })
        
        return treatments[:5]  # Limit to top 5
    
    def _extract_follow_up_recommendations(self, text: str) -> List[str]:
        """Extract follow-up recommendations from response."""
        recommendations = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if 'follow' in line.lower() and 'up' in line.lower():
                # Look for following recommendations
                for j in range(i+1, min(i+4, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and (next_line.startswith(('1.', '2.', '3.', '-', '•'))):
                        rec_text = next_line.split('.', 1)[-1].strip()
                        if rec_text:
                            recommendations.append(rec_text)
        
        return recommendations


class BioGPTProvider(BaseMedicalFoundationModel):
    """Microsoft's BioGPT for biomedical text generation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("bio_gpt", config)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers torch")
        
        self.model_name = config.get("model_name", "microsoft/biogpt")
        self.device = config.get("device", "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def generate_response(
        self, 
        request: MedicalFoundationModelRequest
    ) -> MedicalFoundationModelResponse:
        """Generate response using BioGPT."""
        
        start_time = datetime.utcnow()
        
        try:
            # Format biomedical prompt
            prompt = self._format_biomedical_prompt(request.task, request.input_text)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    inputs,
                    max_length=inputs.shape[1] + request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            response_text = response_text[len(prompt):].strip()
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return MedicalFoundationModelResponse(
                model_type=ModelType.BIO_GPT,
                task=request.task,
                response_text=response_text,
                confidence_score=0.75,  # Moderate confidence for BioGPT
                processing_time_ms=processing_time,
                metadata={
                    "model_name": self.model_name,
                    "device": self.device,
                    "input_tokens": inputs.shape[1],
                    "output_tokens": outputs.shape[1] - inputs.shape[1]
                }
            )
            
        except Exception as e:
            self.logger.error(f"BioGPT generation failed: {e}")
            raise
    
    async def validate_clinical_context(self, context: Dict[str, Any]) -> bool:
        """Validate clinical context for BioGPT."""
        # BioGPT is more focused on biomedical text, less strict validation
        return True
    
    def _format_biomedical_prompt(self, task: MedicalTask, input_text: str) -> str:
        """Format prompt specifically for biomedical tasks."""
        
        if task == MedicalTask.DRUG_DISCOVERY:
            return f"Drug discovery analysis: {input_text}. Potential therapeutic targets and mechanisms:"
        elif task == MedicalTask.BIOMARKER_PREDICTION:
            return f"Biomarker analysis: {input_text}. Relevant biomarkers and clinical significance:"
        elif task == MedicalTask.CLINICAL_NOTE_ANALYSIS:
            return f"Clinical note analysis: {input_text}. Key medical findings and insights:"
        else:
            return f"Biomedical analysis: {input_text}. Clinical interpretation:"


class ClinicalBERTProvider(BaseMedicalFoundationModel):
    """ClinicalBERT for medical text understanding and analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("clinical_bert", config)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers torch")
        
        self.model_name = config.get("model_name", "emilyalsentzer/Bio_ClinicalBERT")
        self.device = config.get("device", "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Initialize text classification pipeline
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
    
    async def generate_response(
        self, 
        request: MedicalFoundationModelRequest
    ) -> MedicalFoundationModelResponse:
        """Generate response using ClinicalBERT."""
        
        start_time = datetime.utcnow()
        
        try:
            # ClinicalBERT is primarily for understanding/classification
            # We'll provide analysis and embeddings
            
            # Get text embeddings
            inputs = self.tokenizer(
                request.input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = await asyncio.to_thread(self.model, **inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Perform clinical text analysis
            analysis_results = await self._analyze_clinical_text(request.input_text)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            response_text = f"""
Clinical Text Analysis Results:

Medical Entities Detected: {len(analysis_results.get('entities', []))}
Clinical Concepts: {', '.join(analysis_results.get('concepts', []))}
Sentiment: {analysis_results.get('sentiment', 'neutral')}
Confidence: {analysis_results.get('confidence', 0.0):.2f}

Key Findings:
{analysis_results.get('summary', 'No specific findings identified.')}
            """.strip()
            
            return MedicalFoundationModelResponse(
                model_type=ModelType.CLINICAL_BERT,
                task=request.task,
                response_text=response_text,
                confidence_score=analysis_results.get('confidence', 0.7),
                processing_time_ms=processing_time,
                metadata={
                    "model_name": self.model_name,
                    "embedding_dimensions": embeddings.shape[1],
                    "entities_detected": len(analysis_results.get('entities', [])),
                    "clinical_concepts": analysis_results.get('concepts', [])
                }
            )
            
        except Exception as e:
            self.logger.error(f"ClinicalBERT analysis failed: {e}")
            raise
    
    async def validate_clinical_context(self, context: Dict[str, Any]) -> bool:
        """Validate clinical context for ClinicalBERT."""
        return True  # ClinicalBERT can analyze any clinical text
    
    async def _analyze_clinical_text(self, text: str) -> Dict[str, Any]:
        """Analyze clinical text for entities and concepts."""
        
        # Simple medical entity detection (in production would use more sophisticated NER)
        medical_terms = [
            'diagnosis', 'symptom', 'medication', 'treatment', 'patient', 'condition',
            'disease', 'therapy', 'clinical', 'medical', 'health', 'doctor', 'physician',
            'hospital', 'pain', 'blood', 'pressure', 'heart', 'lung', 'brain'
        ]
        
        entities = []
        concepts = []
        
        text_lower = text.lower()
        for term in medical_terms:
            if term in text_lower:
                entities.append(term)
                concepts.append(term.capitalize())
        
        # Simple sentiment analysis
        positive_indicators = ['improved', 'better', 'normal', 'stable', 'good']
        negative_indicators = ['worse', 'deteriorated', 'abnormal', 'elevated', 'critical']
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = 0.8
        elif negative_count > positive_count:
            sentiment = "negative" 
            confidence = 0.8
        else:
            sentiment = "neutral"
            confidence = 0.6
        
        return {
            'entities': entities,
            'concepts': list(set(concepts)),
            'sentiment': sentiment,
            'confidence': confidence,
            'summary': f"Clinical text contains {len(entities)} medical entities with {sentiment} sentiment."
        }


class MultimodalMedicalProvider(BaseMedicalFoundationModel):
    """Multimodal medical AI combining text, imaging, and lab data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("multimodal_medical", config)
        
        # Initialize multiple model providers for different modalities
        self.text_provider = None
        self.image_provider = None
        
        # Set up text model (using ChatGPT-4 as backbone)
        if config.get("openai_api_key"):
            self.text_provider = ChatGPT4MedicalProvider(config)
        
        # Set up image analysis (placeholder for medical imaging AI)
        self.image_analysis_enabled = config.get("enable_image_analysis", False)
    
    async def generate_response(
        self, 
        request: MedicalFoundationModelRequest
    ) -> MedicalFoundationModelResponse:
        """Generate multimodal medical response."""
        
        start_time = datetime.utcnow()
        
        try:
            # Combine multiple modalities
            modality_results = {}
            
            # Process text component
            if self.text_provider:
                text_response = await self.text_provider.generate_response(request)
                modality_results['text'] = {
                    'response': text_response.response_text,
                    'confidence': text_response.confidence_score
                }
            
            # Process image component if available
            if request.image_data and self.image_analysis_enabled:
                image_analysis = await self._analyze_medical_images(request.image_data)
                modality_results['imaging'] = image_analysis
            
            # Process lab data if available
            if request.lab_results:
                lab_analysis = await self._analyze_lab_data(request.lab_results)
                modality_results['laboratory'] = lab_analysis
            
            # Integrate findings across modalities
            integrated_response = await self._integrate_multimodal_findings(
                modality_results, request.task
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return MedicalFoundationModelResponse(
                model_type=ModelType.MULTIMODAL_MEDICAL,
                task=request.task,
                response_text=integrated_response['summary'],
                confidence_score=integrated_response['confidence'],
                clinical_reasoning=integrated_response.get('reasoning'),
                differential_diagnoses=integrated_response.get('diagnoses'),
                treatment_options=integrated_response.get('treatments'),
                processing_time_ms=processing_time,
                metadata={
                    "modalities_processed": list(modality_results.keys()),
                    "integration_method": "consensus_fusion",
                    "individual_results": modality_results
                }
            )
            
        except Exception as e:
            self.logger.error(f"Multimodal medical analysis failed: {e}")
            raise
    
    async def validate_clinical_context(self, context: Dict[str, Any]) -> bool:
        """Validate clinical context for multimodal analysis."""
        # Multimodal analysis benefits from rich context
        required_fields = ["patient_id"]
        return all(field in context for field in required_fields)
    
    async def _analyze_medical_images(self, image_data: List[str]) -> Dict[str, Any]:
        """Analyze medical images (placeholder implementation)."""
        
        # Placeholder for medical image analysis
        # In production, would integrate with specialized medical imaging AI
        
        return {
            "images_analyzed": len(image_data),
            "findings": ["Image analysis placeholder - would integrate with medical imaging AI"],
            "confidence": 0.7,
            "modality": "imaging"
        }
    
    async def _analyze_lab_data(self, lab_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze laboratory data."""
        
        abnormal_results = []
        critical_values = []
        
        for lab in lab_results:
            test_name = lab.get('test_name', 'Unknown')
            value = lab.get('value', 0)
            reference_range = lab.get('reference_range', '')
            
            # Simple abnormality detection
            if 'high' in reference_range.lower() or value > 100:  # Simplified logic
                abnormal_results.append(f"{test_name}: {value}")
                
                if value > 200:  # Simplified critical threshold
                    critical_values.append(f"{test_name}: {value} (CRITICAL)")
        
        return {
            "total_tests": len(lab_results),
            "abnormal_results": abnormal_results,
            "critical_values": critical_values,
            "confidence": 0.85,
            "modality": "laboratory"
        }
    
    async def _integrate_multimodal_findings(
        self, 
        modality_results: Dict[str, Any],
        task: MedicalTask
    ) -> Dict[str, Any]:
        """Integrate findings from multiple modalities."""
        
        summary_parts = []
        confidence_scores = []
        
        # Integrate text findings
        if 'text' in modality_results:
            text_result = modality_results['text']
            summary_parts.append(f"Clinical Analysis: {text_result['response']}")
            confidence_scores.append(text_result['confidence'])
        
        # Integrate imaging findings
        if 'imaging' in modality_results:
            imaging_result = modality_results['imaging']
            summary_parts.append(f"Imaging Findings: {', '.join(imaging_result['findings'])}")
            confidence_scores.append(imaging_result['confidence'])
        
        # Integrate lab findings
        if 'laboratory' in modality_results:
            lab_result = modality_results['laboratory']
            if lab_result['abnormal_results']:
                summary_parts.append(f"Laboratory Abnormalities: {', '.join(lab_result['abnormal_results'])}")
            if lab_result['critical_values']:
                summary_parts.append(f"CRITICAL VALUES: {', '.join(lab_result['critical_values'])}")
            confidence_scores.append(lab_result['confidence'])
        
        # Calculate integrated confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Create integrated summary
        integrated_summary = f"""
MULTIMODAL MEDICAL ANALYSIS

{chr(10).join(summary_parts)}

INTEGRATION SUMMARY:
- {len(modality_results)} modalities analyzed
- Overall confidence: {avg_confidence:.2f}
- Recommendation: Correlate all findings with clinical presentation

Note: This multimodal analysis integrates findings from multiple data sources. 
Clinical correlation and professional judgment are essential for interpretation.
        """.strip()
        
        return {
            'summary': integrated_summary,
            'confidence': avg_confidence,
            'reasoning': 'Integrated analysis across multiple medical data modalities',
            'diagnoses': [],  # Would be populated by more sophisticated integration
            'treatments': []   # Would be populated by more sophisticated integration
        }


class MedicalFoundationModelManager:
    """Manager for medical foundation models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[ModelType, BaseMedicalFoundationModel] = {}
        self.logger = structlog.get_logger(__name__)
        
        # Initialize available providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available medical foundation model providers."""
        
        # Initialize Med-PaLM 2 (via Google AI)
        if self.config.get("google_api_key"):
            try:
                self.providers[ModelType.MED_PALM_2] = MedPaLM2Provider(self.config)
                self.logger.info("Med-PaLM 2 provider initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Med-PaLM 2: {e}")
        
        # Initialize ChatGPT-4 Medical
        if self.config.get("openai_api_key"):
            try:
                self.providers[ModelType.CHATGPT_4_MEDICAL] = ChatGPT4MedicalProvider(self.config)
                self.logger.info("ChatGPT-4 Medical provider initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ChatGPT-4 Medical: {e}")
        
        # Initialize BioGPT
        if self.config.get("enable_biogpt", False):
            try:
                self.providers[ModelType.BIO_GPT] = BioGPTProvider(self.config)
                self.logger.info("BioGPT provider initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize BioGPT: {e}")
        
        # Initialize ClinicalBERT
        if self.config.get("enable_clinical_bert", False):
            try:
                self.providers[ModelType.CLINICAL_BERT] = ClinicalBERTProvider(self.config)
                self.logger.info("ClinicalBERT provider initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ClinicalBERT: {e}")
        
        # Initialize Multimodal Medical
        if self.config.get("enable_multimodal", False):
            try:
                self.providers[ModelType.MULTIMODAL_MEDICAL] = MultimodalMedicalProvider(self.config)
                self.logger.info("Multimodal Medical provider initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Multimodal Medical: {e}")
    
    async def generate_response(
        self, 
        request: MedicalFoundationModelRequest
    ) -> MedicalFoundationModelResponse:
        """Generate response using specified medical foundation model."""
        
        provider = self.providers.get(request.model_type)
        if not provider:
            raise ValueError(f"Provider for {request.model_type.value} not available")
        
        # Validate clinical context
        if request.patient_context:
            is_valid = await provider.validate_clinical_context(request.patient_context)
            if not is_valid:
                raise ValueError("Invalid clinical context for the specified model")
        
        return await provider.generate_response(request)
    
    def get_available_models(self) -> List[ModelType]:
        """Get list of available medical foundation models."""
        return list(self.providers.keys())
    
    async def get_model_capabilities(self, model_type: ModelType) -> Dict[str, Any]:
        """Get capabilities of a specific model."""
        
        provider = self.providers.get(model_type)
        if not provider:
            return {}
        
        # Define model capabilities
        capabilities = {
            ModelType.MED_PALM_2: {
                "tasks": [MedicalTask.CLINICAL_REASONING, MedicalTask.MEDICAL_QA, MedicalTask.DIAGNOSIS_SUPPORT],
                "strengths": ["Clinical accuracy", "Evidence-based responses", "Safety considerations"],
                "limitations": ["Text-only", "Requires clinical context"],
                "confidence_level": "high"
            },
            ModelType.CHATGPT_4_MEDICAL: {
                "tasks": [MedicalTask.CLINICAL_REASONING, MedicalTask.TREATMENT_RECOMMENDATION, MedicalTask.MEDICAL_QA],
                "strengths": ["Versatile clinical reasoning", "Comprehensive responses", "Good clinical judgment"],
                "limitations": ["May require fact-checking", "Context length limits"],
                "confidence_level": "high"
            },
            ModelType.BIO_GPT: {
                "tasks": [MedicalTask.DRUG_DISCOVERY, MedicalTask.BIOMARKER_PREDICTION, MedicalTask.CLINICAL_NOTE_ANALYSIS],
                "strengths": ["Biomedical knowledge", "Research-oriented", "Scientific accuracy"],
                "limitations": ["Limited clinical application", "Specialized vocabulary"],
                "confidence_level": "moderate"
            },
            ModelType.CLINICAL_BERT: {
                "tasks": [MedicalTask.CLINICAL_NOTE_ANALYSIS, MedicalTask.MEDICAL_QA],
                "strengths": ["Text understanding", "Entity extraction", "Clinical concept recognition"],
                "limitations": ["Analysis-focused", "Limited generation capability"],
                "confidence_level": "moderate"
            },
            ModelType.MULTIMODAL_MEDICAL: {
                "tasks": [MedicalTask.DIAGNOSIS_SUPPORT, MedicalTask.MEDICAL_IMAGE_ANALYSIS, MedicalTask.CLINICAL_REASONING],
                "strengths": ["Multimodal integration", "Comprehensive analysis", "Rich clinical insights"],
                "limitations": ["Complexity", "Resource intensive"],
                "confidence_level": "high"
            }
        }
        
        return capabilities.get(model_type, {})


# Factory function for easy instantiation
def create_medical_foundation_model_manager(config: Dict[str, Any]) -> MedicalFoundationModelManager:
    """Create a medical foundation model manager with given configuration."""
    return MedicalFoundationModelManager(config)