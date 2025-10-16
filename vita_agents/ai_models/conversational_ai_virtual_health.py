"""
Conversational AI & Virtual Health Assistant for Healthcare.

This module provides comprehensive conversational AI capabilities including medical
chatbots, virtual health assistants, patient education AI, symptom checking,
appointment scheduling automation, and multilingual healthcare communication.
"""

import asyncio
import json
import re
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
import structlog
from pydantic import BaseModel, Field
from collections import defaultdict, deque
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import difflib

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import googletrans
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False

try:
    import speech_recognition as sr
    import pyttsx3
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

logger = structlog.get_logger(__name__)


class ConversationType(Enum):
    """Types of conversations."""
    SYMPTOM_CHECKER = "symptom_checker"
    APPOINTMENT_SCHEDULING = "appointment_scheduling"
    MEDICATION_REMINDER = "medication_reminder"
    HEALTH_EDUCATION = "health_education"
    GENERAL_INQUIRY = "general_inquiry"
    EMERGENCY_TRIAGE = "emergency_triage"
    MENTAL_HEALTH_SUPPORT = "mental_health_support"
    CHRONIC_CARE_MANAGEMENT = "chronic_care_management"
    PREVENTIVE_CARE = "preventive_care"
    LAB_RESULTS_EXPLANATION = "lab_results_explanation"


class ConversationMode(Enum):
    """Conversation interaction modes."""
    TEXT = "text"
    VOICE = "voice"
    MULTIMODAL = "multimodal"
    VIDEO = "video"


class UrgencyLevel(Enum):
    """Urgency levels for medical conversations."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"


class ConversationIntent(Enum):
    """Conversation intents."""
    GREETING = "greeting"
    SYMPTOM_REPORTING = "symptom_reporting"
    APPOINTMENT_REQUEST = "appointment_request"
    MEDICATION_QUESTION = "medication_question"
    HEALTH_INFORMATION = "health_information"
    EMERGENCY_HELP = "emergency_help"
    FOLLOW_UP = "follow_up"
    COMPLAINT = "complaint"
    PRAISE = "praise"
    GOODBYE = "goodbye"


@dataclass
class ConversationContext:
    """Context for ongoing conversation."""
    patient_id: str
    conversation_id: str
    conversation_type: ConversationType
    conversation_mode: ConversationMode
    language: Language
    urgency_level: UrgencyLevel = UrgencyLevel.LOW
    patient_demographics: Dict[str, Any] = field(default_factory=dict)
    medical_history: Dict[str, Any] = field(default_factory=dict)
    current_medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    previous_conversations: List[str] = field(default_factory=list)
    session_data: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConversationMessage:
    """Individual message in conversation."""
    message_id: str
    sender: str  # user, assistant, system
    content: str
    intent: Optional[ConversationIntent] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    language: Language = Language.ENGLISH
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_notes: List[str] = field(default_factory=list)


@dataclass
class SymptomReport:
    """Structured symptom report."""
    reported_symptoms: List[str]
    symptom_details: Dict[str, Any]
    severity_scores: Dict[str, int]  # 1-10 scale
    duration: Dict[str, str]
    associated_factors: List[str]
    previous_episodes: bool
    current_medications_affecting: List[str]
    red_flag_symptoms: List[str]
    differential_diagnosis: List[str]
    recommended_actions: List[str]
    urgency_assessment: UrgencyLevel


@dataclass
class AppointmentRequest:
    """Appointment scheduling request."""
    appointment_type: str
    preferred_provider: Optional[str] = None
    preferred_date_range: Tuple[datetime, datetime] = None
    preferred_times: List[time] = field(default_factory=list)
    urgency: UrgencyLevel = UrgencyLevel.LOW
    reason: str = ""
    special_requirements: List[str] = field(default_factory=list)
    insurance_information: Dict[str, Any] = field(default_factory=dict)
    contact_preferences: Dict[str, Any] = field(default_factory=dict)


class ConversationRequest(BaseModel):
    """Request for conversational AI interaction."""
    
    patient_id: str
    message: str
    conversation_context: ConversationContext
    interaction_mode: ConversationMode = ConversationMode.TEXT
    include_audio: bool = False
    audio_data: Optional[bytes] = None
    request_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationResponse(BaseModel):
    """Response from conversational AI."""
    
    conversation_id: str
    response_message: str
    intent_detected: Optional[ConversationIntent] = None
    confidence: float = 0.0
    urgency_level: UrgencyLevel = UrgencyLevel.LOW
    recommendations: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    extracted_information: Dict[str, Any] = Field(default_factory=dict)
    appointment_suggestions: Optional[AppointmentRequest] = None
    symptom_analysis: Optional[SymptomReport] = None
    educational_content: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    escalation_required: bool = False
    audio_response: Optional[bytes] = None
    response_timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseConversationalAgent(ABC):
    """Base class for conversational AI agents."""
    
    def __init__(self, agent_name: str, version: str = "1.0.0"):
        self.agent_name = agent_name
        self.version = version
        self.logger = structlog.get_logger(__name__)
        
        # Initialize NLP components
        self.sentiment_analyzer = None
        self.nlp_model = None
        self.translator = None
        
        # Conversation history
        self.conversation_history = defaultdict(list)
        
        # Performance metrics
        self.metrics = {
            'total_conversations': 0,
            'successful_interactions': 0,
            'escalations': 0,
            'average_response_time': 0.0
        }
    
    async def initialize(self):
        """Initialize the conversational agent."""
        
        try:
            # Initialize sentiment analyzer
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize NLP model if available
            if SPACY_AVAILABLE:
                try:
                    import spacy
                    self.nlp_model = spacy.load("en_core_web_sm")
                except OSError:
                    self.logger.warning("SpaCy English model not found")
            
            # Initialize translator if available
            if GOOGLETRANS_AVAILABLE:
                self.translator = Translator()
            
            self.logger.info(f"Conversational agent {self.agent_name} initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize conversational agent: {e}")
    
    @abstractmethod
    async def process_conversation(
        self, 
        request: ConversationRequest
    ) -> ConversationResponse:
        """Process conversation request."""
        pass
    
    def detect_intent(self, message: str, context: ConversationContext) -> Tuple[ConversationIntent, float]:
        """Detect intent from user message."""
        
        message_lower = message.lower()
        
        # Intent patterns
        intent_patterns = {
            ConversationIntent.GREETING: [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                r'\b(how are you|greetings)\b'
            ],
            ConversationIntent.SYMPTOM_REPORTING: [
                r'\b(i feel|i have|experiencing|symptoms|pain|ache|hurt)\b',
                r'\b(headache|fever|cough|nausea|dizzy|tired)\b',
                r'\b(my (head|stomach|chest|back) (hurts|aches))\b'
            ],
            ConversationIntent.APPOINTMENT_REQUEST: [
                r'\b(schedule|book|make|need) (an )?appointment\b',
                r'\b(see (a )?doctor|visit|consultation)\b',
                r'\b(available (times|slots)|when can i)\b'
            ],
            ConversationIntent.MEDICATION_QUESTION: [
                r'\b(medication|medicine|prescription|pills|drug)\b',
                r'\b(side effects|dosage|when to take)\b',
                r'\b(forgot to take|missed dose)\b'
            ],
            ConversationIntent.EMERGENCY_HELP: [
                r'\b(emergency|urgent|help|911|chest pain|can\'t breathe)\b',
                r'\b(severe pain|bleeding|unconscious|heart attack)\b'
            ],
            ConversationIntent.HEALTH_INFORMATION: [
                r'\b(what is|tell me about|information about|learn about)\b',
                r'\b(condition|disease|treatment|therapy)\b'
            ],
            ConversationIntent.GOODBYE: [
                r'\b(goodbye|bye|thank you|thanks|have a good day)\b',
                r'\b(that\'s all|no more questions|end conversation)\b'
            ]
        }
        
        # Calculate confidence scores for each intent
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, message_lower))
                score += matches
            
            # Normalize score
            if score > 0:
                intent_scores[intent] = min(1.0, score / len(patterns))
        
        if not intent_scores:
            return ConversationIntent.GENERAL_INQUIRY, 0.5
        
        # Return intent with highest score
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], best_intent[1]
    
    def extract_symptoms(self, message: str) -> Dict[str, Any]:
        """Extract symptoms from user message."""
        
        # Common symptoms dictionary
        symptoms_dict = {
            'headache': ['headache', 'head pain', 'migraine'],
            'fever': ['fever', 'temperature', 'hot', 'feverish'],
            'cough': ['cough', 'coughing', 'hack'],
            'nausea': ['nausea', 'sick to stomach', 'queasy'],
            'fatigue': ['tired', 'exhausted', 'fatigue', 'weak'],
            'pain': ['pain', 'ache', 'hurt', 'sore'],
            'shortness_of_breath': ['short of breath', 'difficulty breathing', 'can\'t breathe'],
            'dizziness': ['dizzy', 'lightheaded', 'spinning'],
            'chest_pain': ['chest pain', 'chest hurt', 'heart pain'],
            'abdominal_pain': ['stomach pain', 'belly hurt', 'abdominal pain']
        }
        
        detected_symptoms = []
        message_lower = message.lower()
        
        for symptom, keywords in symptoms_dict.items():
            for keyword in keywords:
                if keyword in message_lower:
                    detected_symptoms.append(symptom)
                    break
        
        # Extract severity indicators
        severity_indicators = {
            'mild': ['little', 'slight', 'mild', 'minor'],
            'moderate': ['moderate', 'medium', 'some'],
            'severe': ['severe', 'intense', 'terrible', 'awful', 'excruciating']
        }
        
        detected_severity = 'unknown'
        for severity, keywords in severity_indicators.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_severity = severity
                break
        
        # Extract duration indicators
        duration_patterns = [
            r'for (\d+) (days?|hours?|weeks?|months?)',
            r'since (yesterday|today|last week|last month)',
            r'(\d+) (days?|hours?) ago'
        ]
        
        detected_duration = 'unknown'
        for pattern in duration_patterns:
            match = re.search(pattern, message_lower)
            if match:
                detected_duration = match.group(0)
                break
        
        return {
            'symptoms': list(set(detected_symptoms)),
            'severity': detected_severity,
            'duration': detected_duration,
            'raw_text': message
        }
    
    def assess_urgency(self, symptoms: List[str], message: str) -> UrgencyLevel:
        """Assess urgency level based on symptoms and message content."""
        
        # Emergency keywords
        emergency_keywords = [
            'chest pain', 'heart attack', 'stroke', 'can\'t breathe', 'bleeding heavily',
            'unconscious', 'severe pain', 'suicide', 'overdose', '911'
        ]
        
        # High urgency symptoms
        high_urgency_symptoms = [
            'chest_pain', 'shortness_of_breath', 'severe_pain'
        ]
        
        # Moderate urgency symptoms
        moderate_urgency_symptoms = [
            'fever', 'persistent_cough', 'severe_headache'
        ]
        
        message_lower = message.lower()
        
        # Check for emergency keywords
        if any(keyword in message_lower for keyword in emergency_keywords):
            return UrgencyLevel.EMERGENCY
        
        # Check for high urgency symptoms
        if any(symptom in symptoms for symptom in high_urgency_symptoms):
            return UrgencyLevel.HIGH
        
        # Check for moderate urgency symptoms
        if any(symptom in symptoms for symptom in moderate_urgency_symptoms):
            return UrgencyLevel.MODERATE
        
        # Check for severity indicators
        if 'severe' in message_lower or 'intense' in message_lower:
            return UrgencyLevel.HIGH
        elif 'moderate' in message_lower:
            return UrgencyLevel.MODERATE
        
        return UrgencyLevel.LOW
    
    def generate_empathetic_response(
        self, 
        intent: ConversationIntent, 
        context: ConversationContext,
        extracted_info: Dict[str, Any]
    ) -> str:
        """Generate empathetic response based on intent and context."""
        
        patient_name = context.patient_demographics.get('first_name', 'there')
        
        if intent == ConversationIntent.GREETING:
            return f"Hello {patient_name}! I'm here to help you with your health questions today. How are you feeling?"
        
        elif intent == ConversationIntent.SYMPTOM_REPORTING:
            symptoms = extracted_info.get('symptoms', [])
            if symptoms:
                symptom_text = ', '.join(symptoms).replace('_', ' ')
                return f"I understand you're experiencing {symptom_text}. That must be concerning for you. Let me help you understand what might be going on and what steps you should take."
            else:
                return "I hear that you're not feeling well. Can you tell me more about what symptoms you're experiencing?"
        
        elif intent == ConversationIntent.EMERGENCY_HELP:
            return "I understand this is urgent. If this is a life-threatening emergency, please call 911 immediately or go to the nearest emergency room. If you're experiencing chest pain, difficulty breathing, severe bleeding, or signs of stroke, do not wait."
        
        elif intent == ConversationIntent.APPOINTMENT_REQUEST:
            return f"I'd be happy to help you schedule an appointment, {patient_name}. What type of appointment do you need, and do you have any preferred dates or times?"
        
        elif intent == ConversationIntent.MEDICATION_QUESTION:
            return "I understand you have questions about your medication. Medication management is important for your health. What specific questions do you have?"
        
        elif intent == ConversationIntent.HEALTH_INFORMATION:
            return "I'm here to provide you with reliable health information. What would you like to learn about today?"
        
        elif intent == ConversationIntent.GOODBYE:
            return f"Thank you for using our health assistant, {patient_name}. Take care of yourself, and don't hesitate to reach out if you have any more questions!"
        
        else:
            return f"I'm here to help you with your health needs, {patient_name}. Could you please tell me more about what you're looking for today?"
    
    async def translate_message(self, message: str, target_language: Language) -> str:
        """Translate message to target language."""
        
        if not self.translator or target_language == Language.ENGLISH:
            return message
        
        try:
            translated = self.translator.translate(message, dest=target_language.value)
            return translated.text
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return message
    
    def analyze_sentiment(self, message: str) -> Dict[str, float]:
        """Analyze sentiment of message."""
        
        if not self.sentiment_analyzer:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(message)
            return scores
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}


class SymptomCheckerAgent(BaseConversationalAgent):
    """Specialized agent for symptom checking and triage."""
    
    def __init__(self):
        super().__init__("symptom_checker", "v2.1.0")
        
        # Medical knowledge base (simplified)
        self.symptom_conditions = {
            'headache': {
                'common_causes': ['tension', 'migraine', 'dehydration', 'stress'],
                'red_flags': ['sudden severe headache', 'headache with fever', 'worst headache ever'],
                'questions': ['How severe is the pain?', 'Where is the pain located?', 'Any visual changes?']
            },
            'chest_pain': {
                'common_causes': ['muscle strain', 'acid reflux', 'anxiety'],
                'red_flags': ['crushing chest pain', 'pain radiating to arm', 'shortness of breath'],
                'questions': ['Is the pain crushing or stabbing?', 'Does it radiate anywhere?', 'Any shortness of breath?']
            },
            'fever': {
                'common_causes': ['viral infection', 'bacterial infection', 'inflammation'],
                'red_flags': ['temperature > 103°F', 'fever with stiff neck', 'severe dehydration'],
                'questions': ['What is your temperature?', 'How long have you had the fever?', 'Any other symptoms?']
            }
        }
    
    async def process_conversation(
        self, 
        request: ConversationRequest
    ) -> ConversationResponse:
        """Process symptom checking conversation."""
        
        try:
            # Detect intent
            intent, confidence = self.detect_intent(request.message, request.conversation_context)
            
            # Extract symptoms
            extracted_symptoms = self.extract_symptoms(request.message)
            
            # Assess urgency
            urgency = self.assess_urgency(extracted_symptoms['symptoms'], request.message)
            
            # Generate symptom analysis
            symptom_analysis = await self._analyze_symptoms(extracted_symptoms, request.conversation_context)
            
            # Generate response
            response_message = await self._generate_symptom_response(
                intent, extracted_symptoms, symptom_analysis, request.conversation_context
            )
            
            # Generate follow-up questions
            follow_up_questions = self._generate_symptom_follow_up_questions(extracted_symptoms['symptoms'])
            
            # Generate recommendations
            recommendations = self._generate_symptom_recommendations(symptom_analysis, urgency)
            
            # Determine if escalation is needed
            escalation_required = urgency in [UrgencyLevel.URGENT, UrgencyLevel.EMERGENCY]
            
            return ConversationResponse(
                conversation_id=request.conversation_context.conversation_id,
                response_message=response_message,
                intent_detected=intent,
                confidence=confidence,
                urgency_level=urgency,
                recommendations=recommendations,
                follow_up_questions=follow_up_questions,
                extracted_information=extracted_symptoms,
                symptom_analysis=symptom_analysis,
                escalation_required=escalation_required,
                response_timestamp=datetime.utcnow(),
                metadata={
                    'agent_type': 'symptom_checker',
                    'analysis_version': self.version
                }
            )
            
        except Exception as e:
            self.logger.error(f"Symptom checker processing failed: {e}")
            raise
    
    async def _analyze_symptoms(
        self, 
        extracted_symptoms: Dict[str, Any], 
        context: ConversationContext
    ) -> SymptomReport:
        """Analyze reported symptoms."""
        
        symptoms = extracted_symptoms['symptoms']
        severity = extracted_symptoms['severity']
        duration = extracted_symptoms['duration']
        
        # Assess each symptom
        severity_scores = {}
        differential_diagnosis = []
        red_flag_symptoms = []
        recommended_actions = []
        
        for symptom in symptoms:
            # Get symptom information
            symptom_info = self.symptom_conditions.get(symptom, {})
            
            # Assign severity score (simplified)
            if severity == 'severe':
                severity_scores[symptom] = 8
            elif severity == 'moderate':
                severity_scores[symptom] = 5
            elif severity == 'mild':
                severity_scores[symptom] = 3
            else:
                severity_scores[symptom] = 5  # Default
            
            # Check for red flags
            red_flags = symptom_info.get('red_flags', [])
            message_lower = extracted_symptoms['raw_text'].lower()
            for red_flag in red_flags:
                if any(word in message_lower for word in red_flag.split()):
                    red_flag_symptoms.append(red_flag)
            
            # Add potential conditions
            common_causes = symptom_info.get('common_causes', [])
            differential_diagnosis.extend(common_causes)
        
        # Remove duplicates
        differential_diagnosis = list(set(differential_diagnosis))
        red_flag_symptoms = list(set(red_flag_symptoms))
        
        # Generate recommendations based on findings
        if red_flag_symptoms:
            recommended_actions.append("Seek immediate medical attention")
            urgency_assessment = UrgencyLevel.URGENT
        elif any(score >= 7 for score in severity_scores.values()):
            recommended_actions.append("Consider seeing a healthcare provider today")
            urgency_assessment = UrgencyLevel.HIGH
        elif any(score >= 5 for score in severity_scores.values()):
            recommended_actions.append("Monitor symptoms and consider medical consultation")
            urgency_assessment = UrgencyLevel.MODERATE
        else:
            recommended_actions.append("Continue monitoring symptoms")
            urgency_assessment = UrgencyLevel.LOW
        
        # Add general care recommendations
        recommended_actions.extend([
            "Stay hydrated",
            "Get adequate rest",
            "Track symptoms over time"
        ])
        
        return SymptomReport(
            reported_symptoms=symptoms,
            symptom_details=extracted_symptoms,
            severity_scores=severity_scores,
            duration={s: duration for s in symptoms},
            associated_factors=[],
            previous_episodes=False,  # Would need more information
            current_medications_affecting=[],
            red_flag_symptoms=red_flag_symptoms,
            differential_diagnosis=differential_diagnosis,
            recommended_actions=recommended_actions,
            urgency_assessment=urgency_assessment
        )
    
    async def _generate_symptom_response(
        self, 
        intent: ConversationIntent, 
        extracted_symptoms: Dict[str, Any],
        symptom_analysis: SymptomReport,
        context: ConversationContext
    ) -> str:
        """Generate response for symptom reporting."""
        
        if intent == ConversationIntent.EMERGENCY_HELP or symptom_analysis.urgency_assessment == UrgencyLevel.EMERGENCY:
            return "⚠️ Based on your symptoms, this may require immediate medical attention. If this is a life-threatening emergency, please call 911 now or go to the nearest emergency room immediately."
        
        symptoms = extracted_symptoms['symptoms']
        if not symptoms:
            return "I understand you're not feeling well. To better help you, could you describe your specific symptoms? For example, are you experiencing pain, fever, nausea, or other discomfort?"
        
        symptom_text = ', '.join(s.replace('_', ' ') for s in symptoms)
        
        response_parts = [
            f"I understand you're experiencing {symptom_text}. Let me help you understand what this might mean."
        ]
        
        # Add urgency-appropriate guidance
        if symptom_analysis.urgency_assessment == UrgencyLevel.HIGH:
            response_parts.append("Based on your symptoms, I recommend seeking medical attention soon.")
        elif symptom_analysis.urgency_assessment == UrgencyLevel.MODERATE:
            response_parts.append("Your symptoms warrant monitoring and possible medical consultation.")
        
        # Add red flag warning if applicable
        if symptom_analysis.red_flag_symptoms:
            response_parts.append("⚠️ I've identified some concerning features that should be evaluated by a healthcare provider promptly.")
        
        # Add potential causes (educational)
        if symptom_analysis.differential_diagnosis:
            common_causes = symptom_analysis.differential_diagnosis[:3]  # Top 3
            causes_text = ', '.join(common_causes)
            response_parts.append(f"Some common causes of these symptoms include: {causes_text}.")
        
        response_parts.append("Please remember that this is educational information and not a substitute for professional medical advice.")
        
        return " ".join(response_parts)
    
    def _generate_symptom_follow_up_questions(self, symptoms: List[str]) -> List[str]:
        """Generate follow-up questions for better symptom assessment."""
        
        questions = []
        
        for symptom in symptoms:
            symptom_info = self.symptom_conditions.get(symptom, {})
            symptom_questions = symptom_info.get('questions', [])
            questions.extend(symptom_questions)
        
        # Add general follow-up questions
        if symptoms:
            questions.extend([
                "How long have you been experiencing these symptoms?",
                "Have you taken any medications for this?",
                "Have you experienced these symptoms before?",
                "Are there any activities that make the symptoms better or worse?"
            ])
        
        # Remove duplicates and limit to 5 questions
        unique_questions = list(dict.fromkeys(questions))
        return unique_questions[:5]
    
    def _generate_symptom_recommendations(
        self, 
        symptom_analysis: SymptomReport, 
        urgency: UrgencyLevel
    ) -> List[str]:
        """Generate recommendations based on symptom analysis."""
        
        recommendations = []
        
        # Add urgency-based recommendations
        if urgency == UrgencyLevel.EMERGENCY:
            recommendations.extend([
                "Call 911 or go to emergency room immediately",
                "Do not drive yourself - call ambulance or have someone drive you",
                "If available, chew aspirin (unless allergic) for chest pain"
            ])
        elif urgency == UrgencyLevel.HIGH:
            recommendations.extend([
                "Contact your healthcare provider today",
                "Consider visiting urgent care if primary care unavailable",
                "Monitor symptoms closely for any worsening"
            ])
        elif urgency == UrgencyLevel.MODERATE:
            recommendations.extend([
                "Schedule appointment with healthcare provider within 1-2 days",
                "Keep a symptom diary to track changes",
                "Contact provider if symptoms worsen"
            ])
        else:
            recommendations.extend([
                "Monitor symptoms for 24-48 hours",
                "Consider self-care measures",
                "Contact healthcare provider if symptoms persist or worsen"
            ])
        
        # Add symptom-specific recommendations
        recommendations.extend(symptom_analysis.recommended_actions)
        
        return recommendations


class AppointmentSchedulingAgent(BaseConversationalAgent):
    """Specialized agent for appointment scheduling."""
    
    def __init__(self):
        super().__init__("appointment_scheduler", "v1.8.0")
        
        # Available appointment types
        self.appointment_types = {
            'routine_checkup': {'duration': 30, 'urgency': UrgencyLevel.LOW},
            'follow_up': {'duration': 20, 'urgency': UrgencyLevel.LOW},
            'urgent_care': {'duration': 45, 'urgency': UrgencyLevel.HIGH},
            'specialist_consultation': {'duration': 60, 'urgency': UrgencyLevel.MODERATE},
            'diagnostic_test': {'duration': 90, 'urgency': UrgencyLevel.MODERATE},
            'vaccination': {'duration': 15, 'urgency': UrgencyLevel.LOW},
            'mental_health': {'duration': 50, 'urgency': UrgencyLevel.MODERATE}
        }
        
        # Mock provider availability (in production, would integrate with scheduling system)
        self.provider_availability = {
            'Dr. Smith': ['2024-10-18', '2024-10-19', '2024-10-21'],
            'Dr. Johnson': ['2024-10-17', '2024-10-20', '2024-10-22'],
            'Dr. Williams': ['2024-10-18', '2024-10-21', '2024-10-23']
        }
    
    async def process_conversation(
        self, 
        request: ConversationRequest
    ) -> ConversationResponse:
        """Process appointment scheduling conversation."""
        
        try:
            # Detect intent
            intent, confidence = self.detect_intent(request.message, request.conversation_context)
            
            # Extract appointment information
            appointment_info = self._extract_appointment_details(request.message)
            
            # Generate appointment suggestions
            appointment_suggestions = await self._generate_appointment_suggestions(
                appointment_info, request.conversation_context
            )
            
            # Generate response
            response_message = await self._generate_scheduling_response(
                intent, appointment_info, appointment_suggestions, request.conversation_context
            )
            
            # Generate follow-up questions
            follow_up_questions = self._generate_scheduling_follow_up_questions(appointment_info)
            
            # Generate recommendations
            recommendations = self._generate_scheduling_recommendations(appointment_info)
            
            return ConversationResponse(
                conversation_id=request.conversation_context.conversation_id,
                response_message=response_message,
                intent_detected=intent,
                confidence=confidence,
                urgency_level=appointment_info.urgency,
                recommendations=recommendations,
                follow_up_questions=follow_up_questions,
                extracted_information=appointment_info.__dict__,
                appointment_suggestions=appointment_suggestions,
                escalation_required=appointment_info.urgency == UrgencyLevel.EMERGENCY,
                response_timestamp=datetime.utcnow(),
                metadata={
                    'agent_type': 'appointment_scheduler',
                    'available_slots_found': len(appointment_suggestions.__dict__ if appointment_suggestions else {})
                }
            )
            
        except Exception as e:
            self.logger.error(f"Appointment scheduling processing failed: {e}")
            raise
    
    def _extract_appointment_details(self, message: str) -> AppointmentRequest:
        """Extract appointment details from message."""
        
        message_lower = message.lower()
        
        # Detect appointment type
        appointment_type = 'routine_checkup'  # Default
        type_patterns = {
            'urgent_care': ['urgent', 'soon', 'asap', 'emergency'],
            'follow_up': ['follow up', 'follow-up', 'check back'],
            'specialist_consultation': ['specialist', 'cardiologist', 'dermatologist'],
            'routine_checkup': ['checkup', 'physical', 'routine', 'annual'],
            'vaccination': ['vaccine', 'vaccination', 'shot', 'flu shot'],
            'mental_health': ['therapy', 'counseling', 'mental health', 'psychiatrist']
        }
        
        for apt_type, keywords in type_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                appointment_type = apt_type
                break
        
        # Detect urgency
        urgency = UrgencyLevel.LOW
        if any(word in message_lower for word in ['urgent', 'emergency', 'asap', 'soon']):
            urgency = UrgencyLevel.HIGH
        elif any(word in message_lower for word in ['this week', 'soon', 'quickly']):
            urgency = UrgencyLevel.MODERATE
        
        # Extract preferred provider
        preferred_provider = None
        provider_patterns = [
            r'(dr\.?\s+\w+)',
            r'(doctor\s+\w+)',
            r'(with\s+dr\.?\s+\w+)'
        ]
        
        for pattern in provider_patterns:
            match = re.search(pattern, message_lower)
            if match:
                preferred_provider = match.group(1).title()
                break
        
        # Extract reason
        reason_keywords = ['for', 'because', 'regarding', 'about']
        reason = ""
        for keyword in reason_keywords:
            if keyword in message_lower:
                reason_start = message_lower.find(keyword)
                reason = message[reason_start:reason_start + 100]  # Extract up to 100 chars
                break
        
        return AppointmentRequest(
            appointment_type=appointment_type,
            preferred_provider=preferred_provider,
            urgency=urgency,
            reason=reason.strip(),
            special_requirements=[],
            insurance_information={},
            contact_preferences={}
        )
    
    async def _generate_appointment_suggestions(
        self, 
        appointment_info: AppointmentRequest, 
        context: ConversationContext
    ) -> AppointmentRequest:
        """Generate appointment suggestions based on request."""
        
        # Get appointment type details
        apt_details = self.appointment_types.get(appointment_info.appointment_type, {})
        
        # Generate preferred date range based on urgency
        now = datetime.utcnow()
        if appointment_info.urgency == UrgencyLevel.HIGH:
            start_date = now + timedelta(days=1)
            end_date = now + timedelta(days=3)
        elif appointment_info.urgency == UrgencyLevel.MODERATE:
            start_date = now + timedelta(days=3)
            end_date = now + timedelta(days=7)
        else:
            start_date = now + timedelta(days=7)
            end_date = now + timedelta(days=14)
        
        # Suggest available providers if none specified
        if not appointment_info.preferred_provider:
            available_providers = list(self.provider_availability.keys())
            appointment_info.preferred_provider = available_providers[0] if available_providers else None
        
        # Set preferred times based on appointment type
        if appointment_info.appointment_type == 'routine_checkup':
            preferred_times = [time(9, 0), time(11, 0), time(14, 0), time(16, 0)]
        elif appointment_info.appointment_type == 'urgent_care':
            preferred_times = [time(8, 0), time(10, 0), time(13, 0), time(15, 0)]
        else:
            preferred_times = [time(10, 0), time(14, 0), time(16, 0)]
        
        appointment_info.preferred_date_range = (start_date, end_date)
        appointment_info.preferred_times = preferred_times
        
        return appointment_info
    
    async def _generate_scheduling_response(
        self, 
        intent: ConversationIntent,
        appointment_info: AppointmentRequest,
        suggestions: AppointmentRequest,
        context: ConversationContext
    ) -> str:
        """Generate scheduling response."""
        
        patient_name = context.patient_demographics.get('first_name', 'there')
        
        if intent == ConversationIntent.APPOINTMENT_REQUEST:
            response_parts = [
                f"I'd be happy to help you schedule a {appointment_info.appointment_type.replace('_', ' ')}, {patient_name}."
            ]
            
            # Add urgency-appropriate scheduling
            if appointment_info.urgency == UrgencyLevel.HIGH:
                response_parts.append("Since this is urgent, I can offer you appointments within the next few days.")
            elif appointment_info.urgency == UrgencyLevel.MODERATE:
                response_parts.append("I can schedule this within the next week.")
            else:
                response_parts.append("I have several options available in the coming weeks.")
            
            # Add provider information
            if suggestions.preferred_provider:
                response_parts.append(f"You mentioned {suggestions.preferred_provider} - let me check their availability.")
            else:
                response_parts.append("I can suggest some excellent providers based on your needs.")
            
            # Add next steps
            response_parts.append("To complete your scheduling, I'll need to confirm a few details with you.")
            
            return " ".join(response_parts)
        
        else:
            return self.generate_empathetic_response(intent, context, {})
    
    def _generate_scheduling_follow_up_questions(self, appointment_info: AppointmentRequest) -> List[str]:
        """Generate follow-up questions for scheduling."""
        
        questions = []
        
        # Basic scheduling questions
        if not appointment_info.preferred_provider:
            questions.append("Do you have a preferred provider or would you like me to suggest one?")
        
        if not appointment_info.preferred_date_range:
            questions.append("What days work best for you?")
        
        if not appointment_info.preferred_times:
            questions.append("Do you prefer morning, afternoon, or evening appointments?")
        
        # Insurance and contact questions
        questions.extend([
            "What insurance will you be using for this appointment?",
            "What's the best phone number to confirm your appointment?",
            "Do you need any special accommodations for your visit?"
        ])
        
        return questions[:4]  # Limit to 4 questions
    
    def _generate_scheduling_recommendations(self, appointment_info: AppointmentRequest) -> List[str]:
        """Generate scheduling recommendations."""
        
        recommendations = []
        
        # Urgency-based recommendations
        if appointment_info.urgency == UrgencyLevel.HIGH:
            recommendations.extend([
                "Consider urgent care if no appointments available today",
                "Prepare list of current symptoms and medications",
                "Bring photo ID and insurance card"
            ])
        else:
            recommendations.extend([
                "Arrive 15 minutes early for your appointment",
                "Bring current medication list and insurance information",
                "Prepare any questions you want to ask during your visit"
            ])
        
        # Appointment type specific recommendations
        if appointment_info.appointment_type == 'routine_checkup':
            recommendations.extend([
                "Consider fasting if lab work might be needed",
                "Bring list of any health concerns to discuss"
            ])
        elif appointment_info.appointment_type == 'specialist_consultation':
            recommendations.extend([
                "Bring any relevant test results or imaging",
                "Prepare detailed symptom history"
            ])
        
        return recommendations


class VirtualHealthAssistantManager:
    """Manager for virtual health assistant capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Initialize specialized agents
        self.agents = {
            ConversationType.SYMPTOM_CHECKER: SymptomCheckerAgent(),
            ConversationType.APPOINTMENT_SCHEDULING: AppointmentSchedulingAgent(),
        }
        
        # Conversation sessions
        self.active_conversations = {}
        
        # Performance metrics
        self.performance_metrics = {
            'total_conversations': 0,
            'successful_completions': 0,
            'escalations_to_human': 0,
            'average_conversation_length': 0.0,
            'user_satisfaction_scores': []
        }
    
    async def initialize(self):
        """Initialize the virtual health assistant manager."""
        
        try:
            # Initialize all agents
            for agent in self.agents.values():
                await agent.initialize()
            
            self.logger.info("Virtual Health Assistant Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Virtual Health Assistant Manager: {e}")
            raise
    
    async def start_conversation(
        self, 
        patient_id: str, 
        conversation_type: ConversationType,
        language: Language = Language.ENGLISH,
        mode: ConversationMode = ConversationMode.TEXT
    ) -> ConversationContext:
        """Start a new conversation session."""
        
        conversation_id = f"conv_{patient_id}_{datetime.utcnow().timestamp()}"
        
        context = ConversationContext(
            patient_id=patient_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            conversation_mode=mode,
            language=language
        )
        
        self.active_conversations[conversation_id] = context
        self.performance_metrics['total_conversations'] += 1
        
        return context
    
    async def process_message(self, request: ConversationRequest) -> ConversationResponse:
        """Process a conversation message."""
        
        try:
            # Get appropriate agent for conversation type
            agent = self.agents.get(request.conversation_context.conversation_type)
            if not agent:
                # Use symptom checker as default
                agent = self.agents[ConversationType.SYMPTOM_CHECKER]
            
            # Process with agent
            response = await agent.process_conversation(request)
            
            # Update conversation context
            self._update_conversation_context(request, response)
            
            # Handle escalation if needed
            if response.escalation_required:
                await self._handle_escalation(request.conversation_context, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Message processing failed: {e}")
            raise
    
    def _update_conversation_context(
        self, 
        request: ConversationRequest, 
        response: ConversationResponse
    ):
        """Update conversation context with new information."""
        
        context = self.active_conversations.get(request.conversation_context.conversation_id)
        if context:
            # Update session data
            context.session_data.update(response.extracted_information)
            
            # Update urgency level
            if response.urgency_level.value > context.urgency_level.value:
                context.urgency_level = response.urgency_level
            
            # Store conversation history
            context.previous_conversations.append({
                'user_message': request.message,
                'assistant_response': response.response_message,
                'timestamp': response.response_timestamp.isoformat(),
                'intent': response.intent_detected.value if response.intent_detected else None
            })
    
    async def _handle_escalation(
        self, 
        context: ConversationContext, 
        response: ConversationResponse
    ):
        """Handle escalation to human agent."""
        
        try:
            # Log escalation
            self.logger.warning(f"Escalation required for conversation {context.conversation_id}")
            
            # Update metrics
            self.performance_metrics['escalations_to_human'] += 1
            
            # In production, this would:
            # 1. Alert human agents
            # 2. Transfer conversation context
            # 3. Notify patient of transfer
            
            # For now, just log the escalation details
            escalation_details = {
                'conversation_id': context.conversation_id,
                'patient_id': context.patient_id,
                'urgency_level': response.urgency_level.value,
                'reason': 'High urgency or emergency detection',
                'last_message': response.response_message,
                'escalation_time': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Escalation details: {escalation_details}")
            
        except Exception as e:
            self.logger.error(f"Escalation handling failed: {e}")
    
    def get_conversation_history(self, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history."""
        
        context = self.active_conversations.get(conversation_id)
        return context.previous_conversations if context else None
    
    def end_conversation(self, conversation_id: str) -> bool:
        """End a conversation session."""
        
        try:
            context = self.active_conversations.pop(conversation_id, None)
            if context:
                # Calculate conversation metrics
                conversation_length = len(context.previous_conversations)
                
                # Update performance metrics
                current_avg = self.performance_metrics['average_conversation_length']
                total_convs = self.performance_metrics['total_conversations']
                self.performance_metrics['average_conversation_length'] = (
                    (current_avg * (total_convs - 1)) + conversation_length
                ) / total_convs
                
                # Mark as successful completion if not escalated
                if context.urgency_level != UrgencyLevel.EMERGENCY:
                    self.performance_metrics['successful_completions'] += 1
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to end conversation {conversation_id}: {e}")
            return False
    
    def get_active_conversations(self) -> List[Dict[str, Any]]:
        """Get list of active conversations."""
        
        return [
            {
                'conversation_id': context.conversation_id,
                'patient_id': context.patient_id,
                'conversation_type': context.conversation_type.value,
                'language': context.language.value,
                'urgency_level': context.urgency_level.value,
                'started_at': context.started_at.isoformat(),
                'message_count': len(context.previous_conversations)
            }
            for context in self.active_conversations.values()
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        
        return {
            'total_conversations': self.performance_metrics['total_conversations'],
            'successful_completions': self.performance_metrics['successful_completions'],
            'escalations_to_human': self.performance_metrics['escalations_to_human'],
            'active_conversations': len(self.active_conversations),
            'average_conversation_length': self.performance_metrics['average_conversation_length'],
            'success_rate': (
                self.performance_metrics['successful_completions'] / 
                max(1, self.performance_metrics['total_conversations'])
            ) * 100,
            'escalation_rate': (
                self.performance_metrics['escalations_to_human'] / 
                max(1, self.performance_metrics['total_conversations'])
            ) * 100
        }
    
    async def generate_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Generate summary of completed conversation."""
        
        context = self.active_conversations.get(conversation_id)
        if not context:
            return {'error': 'Conversation not found'}
        
        conversation_history = context.previous_conversations
        
        if not conversation_history:
            return {'error': 'No conversation history available'}
        
        # Extract key information
        detected_intents = [msg.get('intent') for msg in conversation_history if msg.get('intent')]
        intent_counts = Counter(detected_intents)
        
        # Calculate conversation duration
        start_time = context.started_at
        end_time = datetime.utcnow()
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Identify main topics discussed
        all_messages = ' '.join([msg['user_message'] for msg in conversation_history])
        main_topics = self._extract_main_topics(all_messages)
        
        return {
            'conversation_id': conversation_id,
            'patient_id': context.patient_id,
            'conversation_type': context.conversation_type.value,
            'duration_minutes': duration_minutes,
            'message_count': len(conversation_history),
            'final_urgency_level': context.urgency_level.value,
            'detected_intents': dict(intent_counts),
            'main_topics': main_topics,
            'session_data_collected': list(context.session_data.keys()),
            'escalation_occurred': context.urgency_level == UrgencyLevel.EMERGENCY,
            'completion_status': 'escalated' if context.urgency_level == UrgencyLevel.EMERGENCY else 'completed'
        }
    
    def _extract_main_topics(self, text: str) -> List[str]:
        """Extract main topics from conversation text."""
        
        # Simple keyword-based topic extraction
        medical_topics = {
            'symptoms': ['pain', 'fever', 'headache', 'nausea', 'cough', 'tired'],
            'medications': ['medication', 'pills', 'prescription', 'dosage'],
            'appointments': ['appointment', 'schedule', 'doctor', 'visit'],
            'emergency': ['emergency', 'urgent', 'severe', 'can\'t breathe'],
            'mental_health': ['anxious', 'depressed', 'stress', 'worry', 'mental']
        }
        
        detected_topics = []
        text_lower = text.lower()
        
        for topic, keywords in medical_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics


# Factory function
def create_virtual_health_assistant_manager(config: Dict[str, Any]) -> VirtualHealthAssistantManager:
    """Create virtual health assistant manager with configuration."""
    return VirtualHealthAssistantManager(config)