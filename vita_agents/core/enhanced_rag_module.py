"""
🧠 Enhanced RAG Module - Retrieval-Augmented Generation for Healthcare
=======================================================================

This module implements advanced RAG capabilities for clinical decision support,
following the healthcare AI agent framework requirements.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import structlog
from pydantic import BaseModel, Field
import json
import hashlib

from ..core.agent import BaseAgent, AgentMessage, MessageType, Priority

logger = structlog.get_logger(__name__)


class KnowledgeSource(str, Enum):
    """Types of knowledge sources for RAG"""
    CLINICAL_GUIDELINES = "clinical_guidelines"
    RESEARCH_PAPERS = "research_papers"
    DRUG_DATABASE = "drug_database"
    DIAGNOSTIC_CRITERIA = "diagnostic_criteria"
    TREATMENT_PROTOCOLS = "treatment_protocols"
    PATIENT_HISTORY = "patient_history"
    MEDICAL_LITERATURE = "medical_literature"


class EvidenceQuality(str, Enum):
    """Quality levels for retrieved evidence"""
    HIGH_QUALITY = "high_quality"      # Systematic reviews, RCTs
    MODERATE_QUALITY = "moderate"      # Cohort studies, guidelines
    LOW_QUALITY = "low"               # Case studies, expert opinion
    UNCERTAIN = "uncertain"           # Conflicting evidence


class RetrievedEvidence(BaseModel):
    """Retrieved evidence item from knowledge base"""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: KnowledgeSource
    content: str
    relevance_score: float = Field(description="Relevance to query (0-1)")
    quality_level: EvidenceQuality
    confidence: float = Field(description="Confidence in evidence (0-1)")
    
    # Metadata
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    study_type: Optional[str] = None
    
    # Clinical context
    patient_population: Optional[str] = None
    clinical_context: List[str] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)
    
    # Supporting data
    statistical_measures: Dict[str, Any] = Field(default_factory=dict)
    clinical_outcomes: List[str] = Field(default_factory=list)


class RAGQuery(BaseModel):
    """RAG query with clinical context"""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str
    clinical_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Patient context
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    
    # Query constraints
    max_results: int = 10
    min_relevance_score: float = 0.6
    preferred_sources: List[KnowledgeSource] = Field(default_factory=list)
    exclude_sources: List[KnowledgeSource] = Field(default_factory=list)
    
    # Temporal constraints
    max_age_years: Optional[int] = None  # Maximum age of evidence
    require_recent: bool = False


class RAGResponse(BaseModel):
    """RAG response with retrieved evidence and reasoning"""
    query_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Retrieved evidence
    evidence_items: List[RetrievedEvidence] = Field(default_factory=list)
    total_evidence_found: int = 0
    
    # Synthesis
    synthesized_answer: str
    confidence_score: float = Field(description="Overall confidence in answer (0-1)")
    evidence_quality_score: float = Field(description="Quality of supporting evidence (0-1)")
    
    # Reasoning path
    reasoning_steps: List[str] = Field(default_factory=list)
    key_considerations: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    
    # Clinical recommendations
    recommendations: List[str] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)
    monitoring_requirements: List[str] = Field(default_factory=list)
    
    # Quality indicators
    evidence_consistency: float = Field(description="Consistency across evidence (0-1)")
    clinical_applicability: float = Field(description="Applicability to patient (0-1)")


class EnhancedRAGModule:
    """
    Enhanced RAG module for healthcare AI agents.
    
    Implements retrieval-augmented generation with clinical reasoning,
    evidence synthesis, and contextual adaptation.
    """
    
    def __init__(self):
        self.knowledge_bases = {
            KnowledgeSource.CLINICAL_GUIDELINES: self._load_clinical_guidelines(),
            KnowledgeSource.RESEARCH_PAPERS: self._load_research_papers(),
            KnowledgeSource.DRUG_DATABASE: self._load_drug_database(),
            KnowledgeSource.DIAGNOSTIC_CRITERIA: self._load_diagnostic_criteria(),
            KnowledgeSource.TREATMENT_PROTOCOLS: self._load_treatment_protocols()
        }
        
        # Vector embeddings for semantic search (simplified)
        self.embeddings_cache = {}
        self.similarity_threshold = 0.7
        
        # Evidence quality weights
        self.quality_weights = {
            EvidenceQuality.HIGH_QUALITY: 1.0,
            EvidenceQuality.MODERATE_QUALITY: 0.8,
            EvidenceQuality.LOW_QUALITY: 0.6,
            EvidenceQuality.UNCERTAIN: 0.4
        }

    def _load_clinical_guidelines(self) -> Dict[str, Any]:
        """Load clinical guidelines knowledge base"""
        return {
            "chest_pain_guidelines": {
                "content": "For acute chest pain, evaluate using HEART score. Consider immediate ECG, troponin levels.",
                "quality": EvidenceQuality.HIGH_QUALITY,
                "source": "ACC/AHA Guidelines",
                "clinical_context": ["emergency_medicine", "cardiology"],
                "recommendations": ["ECG within 10 minutes", "Serial troponins", "Risk stratification"]
            },
            "stroke_guidelines": {
                "content": "For acute stroke, use NIHSS scoring. tPA within 4.5 hours if eligible.",
                "quality": EvidenceQuality.HIGH_QUALITY,
                "source": "AHA/ASA Stroke Guidelines",
                "clinical_context": ["emergency_medicine", "neurology"],
                "recommendations": ["CT head immediately", "tPA if <4.5h", "Neurology consultation"]
            },
            "diabetes_management": {
                "content": "Type 2 diabetes management: HbA1c goal <7% for most adults. Metformin first-line.",
                "quality": EvidenceQuality.HIGH_QUALITY,
                "source": "ADA Standards of Care",
                "clinical_context": ["endocrinology", "primary_care"],
                "recommendations": ["HbA1c every 3 months", "Annual eye exam", "ACE inhibitor if hypertensive"]
            }
        }

    def _load_research_papers(self) -> Dict[str, Any]:
        """Load research papers knowledge base"""
        return {
            "covid_treatment_2023": {
                "content": "Paxlovid reduces hospitalization by 30% in high-risk COVID-19 patients when started within 5 days.",
                "quality": EvidenceQuality.HIGH_QUALITY,
                "study_type": "randomized_controlled_trial",
                "patient_population": "high_risk_adults",
                "statistical_measures": {"relative_risk_reduction": 0.3, "nnt": 15}
            },
            "anticoagulation_afib": {
                "content": "DOACs preferred over warfarin for stroke prevention in atrial fibrillation (CHA2DS2-VASc ≥2).",
                "quality": EvidenceQuality.HIGH_QUALITY,
                "study_type": "meta_analysis",
                "clinical_outcomes": ["reduced_stroke", "reduced_bleeding", "improved_mortality"]
            }
        }

    def _load_drug_database(self) -> Dict[str, Any]:
        """Load drug information database"""
        return {
            "metformin": {
                "content": "Metformin 500-1000mg BID. First-line for T2DM. Monitor renal function.",
                "contraindications": ["eGFR <30", "severe_heart_failure", "metabolic_acidosis"],
                "interactions": ["contrast_agents", "alcohol"],
                "monitoring": ["renal_function", "b12_levels"]
            },
            "lisinopril": {
                "content": "ACE inhibitor. Start 5-10mg daily. Monitor potassium and creatinine.",
                "contraindications": ["pregnancy", "angioedema_history", "bilateral_renal_stenosis"],
                "monitoring": ["renal_function", "potassium", "blood_pressure"]
            }
        }

    def _load_diagnostic_criteria(self) -> Dict[str, Any]:
        """Load diagnostic criteria database"""
        return {
            "heart_failure": {
                "content": "Heart failure diagnosis requires symptoms + signs + objective evidence of cardiac dysfunction.",
                "criteria": ["dyspnea", "fatigue", "ankle_swelling", "elevated_bnp", "echo_abnormalities"],
                "classification": "framingham_criteria"
            },
            "depression": {
                "content": "Major depression: ≥5 symptoms for ≥2 weeks including depressed mood or anhedonia.",
                "criteria": ["depressed_mood", "anhedonia", "weight_change", "sleep_disturbance", "fatigue"],
                "classification": "dsm5"
            }
        }

    def _load_treatment_protocols(self) -> Dict[str, Any]:
        """Load treatment protocol database"""
        return {
            "sepsis_protocol": {
                "content": "Sepsis-3: qSOFA ≥2 + infection. Bundle: cultures, antibiotics <1h, fluids, vasopressors.",
                "steps": ["blood_cultures", "broad_spectrum_antibiotics", "iv_fluids", "vasopressors_if_needed"],
                "timeframes": {"antibiotics": "1_hour", "cultures": "before_antibiotics"}
            }
        }

    async def retrieve_and_generate(self, query: RAGQuery) -> RAGResponse:
        """Main RAG pipeline: retrieve evidence and generate response"""
        
        # Step 1: Retrieve relevant evidence
        evidence_items = await self._retrieve_evidence(query)
        
        # Step 2: Rank and filter evidence
        ranked_evidence = await self._rank_evidence(evidence_items, query)
        
        # Step 3: Synthesize evidence into coherent response
        synthesized_answer = await self._synthesize_evidence(ranked_evidence, query)
        
        # Step 4: Generate clinical reasoning
        reasoning_steps = await self._generate_reasoning(ranked_evidence, query)
        
        # Step 5: Extract recommendations and considerations
        recommendations, contraindications = await self._extract_recommendations(ranked_evidence, query)
        
        # Step 6: Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(ranked_evidence)
        
        return RAGResponse(
            query_id=query.query_id,
            evidence_items=ranked_evidence,
            total_evidence_found=len(evidence_items),
            synthesized_answer=synthesized_answer,
            confidence_score=quality_metrics["confidence"],
            evidence_quality_score=quality_metrics["quality"],
            reasoning_steps=reasoning_steps,
            recommendations=recommendations,
            contraindications=contraindications,
            evidence_consistency=quality_metrics["consistency"],
            clinical_applicability=quality_metrics["applicability"]
        )

    async def _retrieve_evidence(self, query: RAGQuery) -> List[RetrievedEvidence]:
        """Retrieve relevant evidence from knowledge bases"""
        evidence_items = []
        query_lower = query.query_text.lower()
        
        # Search across all knowledge bases
        for source, kb in self.knowledge_bases.items():
            if query.exclude_sources and source in query.exclude_sources:
                continue
            
            for item_id, item_data in kb.items():
                content = item_data.get("content", "")
                
                # Simple keyword-based relevance (in production, use semantic embeddings)
                relevance_score = self._calculate_relevance(query_lower, content.lower())
                
                if relevance_score >= query.min_relevance_score:
                    evidence = RetrievedEvidence(
                        evidence_id=item_id,
                        source=source,
                        content=content,
                        relevance_score=relevance_score,
                        quality_level=item_data.get("quality", EvidenceQuality.MODERATE_QUALITY),
                        confidence=relevance_score * self.quality_weights.get(
                            item_data.get("quality", EvidenceQuality.MODERATE_QUALITY), 0.7
                        ),
                        clinical_context=item_data.get("clinical_context", []),
                        contraindications=item_data.get("contraindications", []),
                        clinical_outcomes=item_data.get("clinical_outcomes", []),
                        statistical_measures=item_data.get("statistical_measures", {})
                    )
                    evidence_items.append(evidence)
        
        return evidence_items

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        query_words = set(query.split())
        content_words = set(content.split())
        
        if not query_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))
        
        return intersection / union if union > 0 else 0.0

    async def _rank_evidence(self, evidence_items: List[RetrievedEvidence], query: RAGQuery) -> List[RetrievedEvidence]:
        """Rank evidence by relevance, quality, and clinical applicability"""
        
        # Apply patient-specific filtering
        filtered_evidence = []
        for evidence in evidence_items:
            if self._is_clinically_applicable(evidence, query):
                filtered_evidence.append(evidence)
        
        # Sort by composite score (relevance * quality * applicability)
        def composite_score(evidence: RetrievedEvidence) -> float:
            quality_weight = self.quality_weights.get(evidence.quality_level, 0.7)
            applicability = self._calculate_applicability(evidence, query)
            return evidence.relevance_score * quality_weight * applicability
        
        ranked_evidence = sorted(filtered_evidence, key=composite_score, reverse=True)
        
        # Return top results
        return ranked_evidence[:query.max_results]

    def _is_clinically_applicable(self, evidence: RetrievedEvidence, query: RAGQuery) -> bool:
        """Check if evidence is applicable to patient context"""
        
        # Check contraindications
        if evidence.contraindications:
            patient_conditions = query.medical_history + query.current_medications + query.allergies
            for contraindication in evidence.contraindications:
                if any(contraindication.lower() in condition.lower() for condition in patient_conditions):
                    return False
        
        # Check age appropriateness (simplified)
        if evidence.patient_population and query.patient_age:
            if "pediatric" in evidence.patient_population.lower() and query.patient_age >= 18:
                return False
            if "geriatric" in evidence.patient_population.lower() and query.patient_age < 65:
                return False
        
        return True

    def _calculate_applicability(self, evidence: RetrievedEvidence, query: RAGQuery) -> float:
        """Calculate clinical applicability score"""
        score = 1.0
        
        # Adjust for clinical context match
        if evidence.clinical_context and query.clinical_context:
            query_contexts = query.clinical_context.get("specialties", [])
            if query_contexts:
                if any(ctx in evidence.clinical_context for ctx in query_contexts):
                    score *= 1.2
                else:
                    score *= 0.8
        
        # Adjust for patient population match
        if evidence.patient_population and query.patient_age:
            # Age-based adjustments (simplified)
            if "adult" in evidence.patient_population.lower() and 18 <= query.patient_age <= 65:
                score *= 1.1
        
        return min(score, 1.0)

    async def _synthesize_evidence(self, evidence_items: List[RetrievedEvidence], query: RAGQuery) -> str:
        """Synthesize evidence into coherent clinical answer"""
        if not evidence_items:
            return "No relevant evidence found for the query."
        
        # Group evidence by type
        guidelines = [e for e in evidence_items if e.source == KnowledgeSource.CLINICAL_GUIDELINES]
        research = [e for e in evidence_items if e.source == KnowledgeSource.RESEARCH_PAPERS]
        drugs = [e for e in evidence_items if e.source == KnowledgeSource.DRUG_DATABASE]
        
        synthesis_parts = []
        
        if guidelines:
            synthesis_parts.append(f"Clinical guidelines recommend: {guidelines[0].content}")
        
        if research:
            synthesis_parts.append(f"Research evidence shows: {research[0].content}")
        
        if drugs:
            synthesis_parts.append(f"Medication considerations: {drugs[0].content}")
        
        # Add evidence strength
        high_quality_count = len([e for e in evidence_items if e.quality_level == EvidenceQuality.HIGH_QUALITY])
        if high_quality_count > 0:
            synthesis_parts.append(f"This recommendation is supported by {high_quality_count} high-quality evidence source(s).")
        
        return " ".join(synthesis_parts)

    async def _generate_reasoning(self, evidence_items: List[RetrievedEvidence], query: RAGQuery) -> List[str]:
        """Generate step-by-step clinical reasoning"""
        reasoning_steps = []
        
        if evidence_items:
            reasoning_steps.append(f"1. Identified {len(evidence_items)} relevant evidence sources")
            
            # Quality assessment
            high_quality = len([e for e in evidence_items if e.quality_level == EvidenceQuality.HIGH_QUALITY])
            if high_quality > 0:
                reasoning_steps.append(f"2. {high_quality} sources are high-quality evidence")
            
            # Clinical context
            if query.clinical_context:
                reasoning_steps.append("3. Considered patient-specific clinical context")
            
            # Safety considerations
            contraindications = []
            for evidence in evidence_items:
                contraindications.extend(evidence.contraindications)
            
            if contraindications:
                reasoning_steps.append("4. Evaluated contraindications and safety considerations")
            
            reasoning_steps.append("5. Synthesized evidence into clinical recommendation")
        
        return reasoning_steps

    async def _extract_recommendations(self, evidence_items: List[RetrievedEvidence], query: RAGQuery) -> Tuple[List[str], List[str]]:
        """Extract clinical recommendations and contraindications"""
        recommendations = []
        contraindications = []
        
        for evidence in evidence_items:
            # Extract recommendations from drug database
            if evidence.source == KnowledgeSource.DRUG_DATABASE:
                if "monitoring" in evidence.content.lower():
                    recommendations.append(f"Monitor: {evidence.content}")
            
            # Extract contraindications
            contraindications.extend(evidence.contraindications)
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        contraindications = list(set(contraindications))
        
        return recommendations, contraindications

    async def _calculate_quality_metrics(self, evidence_items: List[RetrievedEvidence]) -> Dict[str, float]:
        """Calculate quality metrics for the evidence set"""
        if not evidence_items:
            return {"confidence": 0.0, "quality": 0.0, "consistency": 0.0, "applicability": 0.0}
        
        # Average confidence
        avg_confidence = sum(e.confidence for e in evidence_items) / len(evidence_items)
        
        # Quality score
        quality_scores = [self.quality_weights.get(e.quality_level, 0.7) for e in evidence_items]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # Consistency (simplified - in practice, would compare content semantically)
        consistency = 0.8  # Placeholder
        
        # Applicability
        applicability = 0.9  # Placeholder
        
        return {
            "confidence": avg_confidence,
            "quality": avg_quality,
            "consistency": consistency,
            "applicability": applicability
        }


# Example integration with existing agents
class RAGEnhancedAgent(BaseAgent):
    """Base class for agents with RAG capabilities"""
    
    def __init__(self, agent_id: str, name: str, description: str):
        super().__init__(agent_id, name, description)
        self.rag_module = EnhancedRAGModule()
    
    async def query_with_rag(self, query_text: str, clinical_context: Dict[str, Any] = None) -> RAGResponse:
        """Query the knowledge base with RAG"""
        rag_query = RAGQuery(
            query_text=query_text,
            clinical_context=clinical_context or {}
        )
        
        return await self.rag_module.retrieve_and_generate(rag_query)


# Example usage
async def test_rag_module():
    """Test the enhanced RAG module"""
    rag_module = EnhancedRAGModule()
    
    query = RAGQuery(
        query_text="chest pain management emergency department",
        clinical_context={"specialties": ["emergency_medicine"]},
        patient_age=58,
        medical_history=["hypertension", "diabetes"]
    )
    
    response = await rag_module.retrieve_and_generate(query)
    
    print("RAG Response:")
    print(f"Answer: {response.synthesized_answer}")
    print(f"Confidence: {response.confidence_score:.2f}")
    print(f"Evidence Quality: {response.evidence_quality_score:.2f}")
    print(f"Evidence Found: {len(response.evidence_items)}")
    
    if response.recommendations:
        print("Recommendations:")
        for rec in response.recommendations:
            print(f"  - {rec}")


if __name__ == "__main__":
    asyncio.run(test_rag_module())