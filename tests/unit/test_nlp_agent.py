"""
Unit tests for Natural Language Processing Agent.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from vita_agents.agents.nlp_agent import (
    NaturalLanguageProcessingAgent,
    EntityType,
    SentimentType,
    MedicalEntity,
    ClinicalConcept
)
from vita_agents.core.agent import TaskRequest
from vita_agents.core.config import Settings


class TestNaturalLanguageProcessingAgent:
    """Test NLP Agent functionality."""
    
    @pytest.fixture
    def settings(self):
        """Mock settings fixture."""
        return Settings()
    
    @pytest.fixture
    def mock_database(self):
        """Mock database fixture."""
        return AsyncMock()
    
    @pytest.fixture
    def agent(self, settings, mock_database):
        """Create agent fixture."""
        return NaturalLanguageProcessingAgent("test-nlp", settings, mock_database)
    
    @pytest.fixture
    def sample_clinical_note(self):
        """Sample clinical note for testing."""
        return """
        Chief Complaint: Patient presents with chest pain and shortness of breath.
        
        History: 45-year-old male with history of diabetes and hypertension.
        Current medications include metformin 500mg twice daily and lisinopril 10mg daily.
        
        Vital Signs: BP 140/90 mmHg, HR 85 bpm, Temp 98.6°F, RR 18, O2 Sat 98%
        
        Assessment: Possible cardiac workup needed. Rule out myocardial infarction.
        
        Plan: Order EKG, cardiac enzymes, and chest X-ray. Continue current medications.
        """
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.agent_id == "test-nlp"
        assert agent.agent_type == "nlp"
        assert len(agent.task_handlers) > 0
        assert hasattr(agent, 'medication_patterns')
        assert hasattr(agent, 'condition_patterns')
    
    @pytest.mark.asyncio
    async def test_extract_entities(self, agent, sample_clinical_note):
        """Test entity extraction from clinical text."""
        task = TaskRequest(
            task_id="test-task",
            task_type="extract_entities",
            parameters={
                "text": sample_clinical_note,
                "entity_types": [EntityType.MEDICATION, EntityType.MEASUREMENT]
            }
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert "entities" in response.result
        assert "entity_count" in response.result
        assert "entity_types_found" in response.result
        
        # Should find medications like metformin and lisinopril
        entities = response.result["entities"]
        medication_entities = [e for e in entities if e["entity_type"] == EntityType.MEDICATION.value]
        assert len(medication_entities) >= 1
    
    @pytest.mark.asyncio
    async def test_analyze_clinical_note(self, agent, sample_clinical_note):
        """Test comprehensive clinical note analysis."""
        task = TaskRequest(
            task_id="test-task",
            task_type="analyze_clinical_note",
            parameters={
                "text": sample_clinical_note,
                "note_id": "note123"
            }
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        result = response.result
        
        # Check all required fields are present
        assert "note_id" in result
        assert "entities" in result
        assert "concepts" in result
        assert "sentiment" in result
        assert "summary" in result
        assert "key_findings" in result
        assert "recommendations" in result
        assert "quality_score" in result
        assert "confidence_score" in result
        
        assert result["note_id"] == "note123"
        assert isinstance(result["quality_score"], float)
        assert 0.0 <= result["quality_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_extract_medications(self, agent):
        """Test medication extraction."""
        text = "Patient is taking metformin 500mg twice daily and aspirin 81mg once daily."
        
        task = TaskRequest(
            task_id="test-task",
            task_type="extract_medications",
            parameters={"text": text}
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert "medications" in response.result
        assert "medication_count" in response.result
        
        medications = response.result["medications"]
        # Should find metformin and aspirin
        assert len(medications) >= 1
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, agent):
        """Test sentiment analysis."""
        # Test concerning text
        concerning_text = "Patient in severe distress with critical condition"
        
        task = TaskRequest(
            task_id="test-task",
            task_type="analyze_sentiment",
            parameters={"text": concerning_text}
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        sentiment_result = response.result
        
        assert "sentiment" in sentiment_result
        assert "confidence" in sentiment_result
        assert sentiment_result["sentiment"] in [s.value for s in SentimentType]
    
    @pytest.mark.asyncio
    async def test_generate_summary(self, agent, sample_clinical_note):
        """Test clinical text summarization."""
        task = TaskRequest(
            task_id="test-task",
            task_type="generate_summary",
            parameters={
                "text": sample_clinical_note,
                "max_length": 100
            }
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        result = response.result
        
        assert "summary" in result
        assert "original_length" in result
        assert "summary_length" in result
        assert "compression_ratio" in result
        
        assert len(result["summary"]) <= 100  # Respects max_length
        assert result["compression_ratio"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_validate_note_quality(self, agent, sample_clinical_note):
        """Test note quality validation."""
        task = TaskRequest(
            task_id="test-task",
            task_type="validate_note_quality",
            parameters={"text": sample_clinical_note}
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        result = response.result
        
        assert "quality_score" in result
        assert "quality_grade" in result
        assert "quality_issues" in result
        assert "recommendations" in result
        
        assert 0.0 <= result["quality_score"] <= 1.0
        assert result["quality_grade"] in ["Excellent", "Good", "Fair", "Poor"]
    
    @pytest.mark.asyncio
    async def test_anonymize_text(self, agent):
        """Test text anonymization."""
        text_with_phi = "Patient John Doe (555-123-4567) with SSN 123-45-6789 was seen today."
        
        task = TaskRequest(
            task_id="test-task",
            task_type="anonymize_text",
            parameters={"text": text_with_phi}
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        result = response.result
        
        assert "anonymized_text" in result
        assert "phi_entities_removed" in result
        assert "phi_types" in result
        
        # Anonymized text should not contain original PHI
        anonymized = result["anonymized_text"]
        assert "555-123-4567" not in anonymized
        assert "123-45-6789" not in anonymized
        assert "[PHONE]" in anonymized or "[SSN]" in anonymized
    
    @pytest.mark.asyncio
    async def test_standardize_terminology(self, agent):
        """Test medical terminology standardization."""
        text = "Patient has diabetes and takes aspirin daily."
        
        task = TaskRequest(
            task_id="test-task",
            task_type="standardize_terminology",
            parameters={
                "text": text,
                "target_standard": "SNOMED"
            }
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        result = response.result
        
        assert "standardized_terms" in result
        assert "target_standard" in result
        assert "mapping_count" in result
        assert result["target_standard"] == "SNOMED"
    
    @pytest.mark.asyncio
    async def test_detect_clinical_concepts(self, agent, sample_clinical_note):
        """Test clinical concept detection."""
        task = TaskRequest(
            task_id="test-task",
            task_type="detect_clinical_concepts",
            parameters={"text": sample_clinical_note}
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        result = response.result
        
        assert "concepts" in result
        assert "concept_count" in result
        assert "concept_types" in result
    
    @pytest.mark.asyncio
    async def test_extract_medication_entities(self, agent):
        """Test medication entity extraction method."""
        text = "Patient takes metformin and aspirin daily."
        entities = agent._extract_medication_entities(text)
        
        # Should find at least metformin and aspirin
        medication_names = [entity.text.lower() for entity in entities]
        assert any("metformin" in name for name in medication_names)
        assert any("aspirin" in name for name in medication_names)
    
    @pytest.mark.asyncio
    async def test_extract_measurement_entities(self, agent):
        """Test measurement entity extraction method."""
        text = "BP 120/80 mmHg, HR 70 bpm, Temp 98.6°F"
        entities = agent._extract_measurement_entities(text)
        
        # Should find blood pressure, heart rate, temperature
        assert len(entities) >= 1
        measurement_texts = [entity.text for entity in entities]
        assert any("120/80" in text for text in measurement_texts)
    
    @pytest.mark.asyncio
    async def test_analyze_clinical_sentiment_concerning(self, agent):
        """Test sentiment analysis for concerning content."""
        concerning_text = "Patient in severe pain with critical condition"
        sentiment = agent._analyze_clinical_sentiment(concerning_text)
        
        assert sentiment["sentiment"] == SentimentType.CONCERNING.value
        assert sentiment["concerning_indicators"] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_clinical_sentiment_positive(self, agent):
        """Test sentiment analysis for positive content."""
        positive_text = "Patient is improving well and stable condition"
        sentiment = agent._analyze_clinical_sentiment(positive_text)
        
        assert sentiment["sentiment"] == SentimentType.POSITIVE.value
        assert sentiment["positive_indicators"] > 0
    
    @pytest.mark.asyncio
    async def test_assess_note_quality_good(self, agent):
        """Test note quality assessment for good quality note."""
        good_note = """
        Chief complaint: Patient presents with chest pain.
        History: Detailed patient history here.
        Assessment: Clinical assessment provided.
        Plan: Treatment plan outlined.
        Vital signs: BP 120/80, HR 70 bpm.
        """
        
        quality_score = agent._assess_note_quality(good_note)
        assert quality_score > 0.7  # Should be high quality
    
    @pytest.mark.asyncio
    async def test_assess_note_quality_poor(self, agent):
        """Test note quality assessment for poor quality note."""
        poor_note = "Brief note."
        
        quality_score = agent._assess_note_quality(poor_note)
        assert quality_score < 0.5  # Should be low quality
    
    @pytest.mark.asyncio
    async def test_identify_phi_entities(self, agent):
        """Test PHI entity identification."""
        text_with_phi = "Call patient at 555-123-4567 or SSN 123-45-6789"
        phi_entities = agent._identify_phi_entities(text_with_phi)
        
        phi_types = [entity["type"] for entity in phi_entities]
        assert "phone" in phi_types
        assert "ssn" in phi_types
    
    @pytest.mark.asyncio
    async def test_replace_phi_entities(self, agent):
        """Test PHI entity replacement."""
        text = "Patient phone: 555-123-4567"
        phi_entities = [
            {
                "text": "555-123-4567",
                "type": "phone",
                "start": 15,
                "end": 27
            }
        ]
        
        anonymized = agent._replace_phi_entities(text, phi_entities)
        assert "555-123-4567" not in anonymized
        assert "[PHONE]" in anonymized
    
    @pytest.mark.asyncio
    async def test_unknown_task_type(self, agent):
        """Test handling of unknown task type."""
        task = TaskRequest(
            task_id="test-task",
            task_type="unknown_nlp_task",
            parameters={}
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "failed"
        assert "Unknown task type" in response.error


class TestMedicalEntity:
    """Test MedicalEntity data model."""
    
    def test_entity_creation(self):
        """Test creating a medical entity."""
        entity = MedicalEntity(
            text="metformin",
            entity_type=EntityType.MEDICATION,
            start_pos=10,
            end_pos=19,
            confidence=0.95,
            normalized_form="metformin",
            codes={"RxNorm": "6809"}
        )
        
        assert entity.text == "metformin"
        assert entity.entity_type == EntityType.MEDICATION
        assert entity.start_pos == 10
        assert entity.end_pos == 19
        assert entity.confidence == 0.95
        assert entity.codes["RxNorm"] == "6809"
    
    def test_entity_default_codes(self):
        """Test entity with default codes."""
        entity = MedicalEntity(
            text="diabetes",
            entity_type=EntityType.CONDITION,
            start_pos=0,
            end_pos=8,
            confidence=0.8
        )
        
        assert entity.codes == {}  # Default empty dict


class TestClinicalConcept:
    """Test ClinicalConcept data model."""
    
    def test_concept_creation(self):
        """Test creating a clinical concept."""
        concept = ClinicalConcept(
            concept="chest pain",
            concept_type=EntityType.SYMPTOM,
            modifiers=["acute", "severe"],
            negated=False,
            uncertain=True,
            severity="moderate"
        )
        
        assert concept.concept == "chest pain"
        assert concept.concept_type == EntityType.SYMPTOM
        assert "acute" in concept.modifiers
        assert concept.uncertain == True
        assert concept.severity == "moderate"
    
    def test_concept_defaults(self):
        """Test concept with default values."""
        concept = ClinicalConcept(
            concept="hypertension",
            concept_type=EntityType.CONDITION
        )
        
        assert concept.modifiers == []  # Default empty list
        assert concept.negated == False  # Default False
        assert concept.uncertain == False  # Default False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])