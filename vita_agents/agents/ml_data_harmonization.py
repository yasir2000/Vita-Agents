"""
Machine Learning-Based Data Harmonization for Vita Agents
Advanced AI-driven data integration, conflict resolution, and quality improvement
"""

import asyncio
import json
import hashlib
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import structlog

# ML and Data Science Libraries
try:
    import sklearn
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from ..core.agent import HealthcareAgent
from ..core.security import HIPAACompliantAgent, AuditAction, ComplianceLevel


class MLHarmonizationMethod(Enum):
    """Machine learning methods for data harmonization"""
    CLUSTERING_BASED = "clustering_based"
    SIMILARITY_LEARNING = "similarity_learning"
    ENSEMBLE_MATCHING = "ensemble_matching"
    DEEP_LEARNING = "deep_learning"
    HYBRID_APPROACH = "hybrid_approach"


class ConflictResolutionML(Enum):
    """ML-based conflict resolution strategies"""
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ENSEMBLE_VOTING = "ensemble_voting"
    NEURAL_ARBITRATION = "neural_arbitration"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_INFERENCE = "bayesian_inference"


class DataQualityMetric(Enum):
    """Data quality metrics for ML assessment"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


@dataclass
class MLModelMetadata:
    """Metadata for ML models used in harmonization"""
    model_id: str
    model_type: str
    version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    model_parameters: Dict[str, Any]
    validation_score: float
    last_updated: datetime


@dataclass
class SimilarityScore:
    """Similarity score between data records"""
    record_1_id: str
    record_2_id: str
    overall_similarity: float
    field_similarities: Dict[str, float]
    confidence: float
    method_used: str
    computed_at: datetime


@dataclass
class DataQualityAssessment:
    """Comprehensive data quality assessment"""
    record_id: str
    source_id: str
    quality_scores: Dict[DataQualityMetric, float]
    overall_quality: float
    issues_detected: List[str]
    recommendations: List[str]
    confidence: float
    assessed_at: datetime


@dataclass
class RecordLinkage:
    """Record linkage result"""
    linkage_id: str
    records: List[str]  # Record IDs
    linkage_confidence: float
    linkage_type: str  # exact, probable, possible
    linking_features: List[str]
    method_used: str
    created_at: datetime


@dataclass
class SemanticMapping:
    """Semantic mapping between different terminologies"""
    source_concept: str
    target_concept: str
    mapping_score: float
    mapping_type: str  # exact, broad, narrow, related
    source_system: str
    target_system: str
    validation_status: str


class AdvancedRecordLinkage:
    """Advanced ML-based record linkage system"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.models = {}
        self.feature_extractors = {}
        self.similarity_thresholds = {
            'exact': 0.95,
            'probable': 0.85,
            'possible': 0.70
        }
    
    async def initialize_models(self):
        """Initialize ML models for record linkage"""
        
        if SKLEARN_AVAILABLE:
            # Initialize clustering models for blocking
            self.models['blocking_kmeans'] = KMeans(n_clusters=100, random_state=42)
            self.models['blocking_dbscan'] = DBSCAN(eps=0.3, min_samples=2)
            
            # Initialize classification models for linkage prediction
            self.models['linkage_classifier'] = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            
            # Initialize anomaly detection for quality assessment
            self.models['quality_detector'] = IsolationForest(
                contamination=0.1, random_state=42
            )
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.matcher = Matcher(self.nlp.vocab)
                
                # Add patterns for medical entities
                self._add_medical_patterns()
                
            except OSError:
                self.logger.warning("spaCy model not found, using basic text processing")
                self.nlp = None
        
        self.logger.info("Advanced record linkage models initialized")
    
    async def link_records(
        self, 
        records: List[Dict[str, Any]], 
        linkage_method: MLHarmonizationMethod = MLHarmonizationMethod.ENSEMBLE_MATCHING
    ) -> List[RecordLinkage]:
        """Perform advanced record linkage using ML"""
        
        try:
            if linkage_method == MLHarmonizationMethod.CLUSTERING_BASED:
                return await self._clustering_based_linkage(records)
            elif linkage_method == MLHarmonizationMethod.SIMILARITY_LEARNING:
                return await self._similarity_learning_linkage(records)
            elif linkage_method == MLHarmonizationMethod.ENSEMBLE_MATCHING:
                return await self._ensemble_matching_linkage(records)
            elif linkage_method == MLHarmonizationMethod.DEEP_LEARNING:
                return await self._deep_learning_linkage(records)
            else:
                return await self._hybrid_linkage(records)
                
        except Exception as e:
            self.logger.error(f"Record linkage failed: {e}")
            raise
    
    async def _clustering_based_linkage(self, records: List[Dict[str, Any]]) -> List[RecordLinkage]:
        """Clustering-based record linkage"""
        
        if not SKLEARN_AVAILABLE:
            raise ValueError("scikit-learn not available for clustering")
        
        # Extract features for clustering
        features = await self._extract_linkage_features(records)
        
        # Perform clustering to identify potential matches
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Use DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.3, min_samples=2)
        clusters = dbscan.fit_predict(scaled_features)
        
        # Create linkages from clusters
        linkages = []
        cluster_groups = {}
        
        for i, cluster_id in enumerate(clusters):
            if cluster_id != -1:  # Not noise
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(i)
        
        for cluster_id, record_indices in cluster_groups.items():
            if len(record_indices) > 1:
                record_ids = [records[i]['id'] for i in record_indices]
                
                # Calculate linkage confidence
                cluster_features = scaled_features[record_indices]
                confidence = self._calculate_cluster_confidence(cluster_features)
                
                linkage = RecordLinkage(
                    linkage_id=f"cluster_{cluster_id}_{datetime.utcnow().timestamp()}",
                    records=record_ids,
                    linkage_confidence=confidence,
                    linkage_type=self._determine_linkage_type(confidence),
                    linking_features=['demographic', 'clinical'],
                    method_used='clustering_dbscan',
                    created_at=datetime.utcnow()
                )
                linkages.append(linkage)
        
        return linkages
    
    async def _similarity_learning_linkage(self, records: List[Dict[str, Any]]) -> List[RecordLinkage]:
        """Similarity learning-based record linkage"""
        
        linkages = []
        
        # Calculate pairwise similarities
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                similarity = await self._calculate_record_similarity(records[i], records[j])
                
                if similarity.overall_similarity >= self.similarity_thresholds['possible']:
                    linkage = RecordLinkage(
                        linkage_id=f"sim_{i}_{j}_{datetime.utcnow().timestamp()}",
                        records=[records[i]['id'], records[j]['id']],
                        linkage_confidence=similarity.overall_similarity,
                        linkage_type=self._determine_linkage_type(similarity.overall_similarity),
                        linking_features=list(similarity.field_similarities.keys()),
                        method_used='similarity_learning',
                        created_at=datetime.utcnow()
                    )
                    linkages.append(linkage)
        
        return linkages
    
    async def _ensemble_matching_linkage(self, records: List[Dict[str, Any]]) -> List[RecordLinkage]:
        """Ensemble-based record linkage combining multiple methods"""
        
        # Get results from multiple methods
        clustering_results = await self._clustering_based_linkage(records)
        similarity_results = await self._similarity_learning_linkage(records)
        
        # Combine results using ensemble voting
        combined_linkages = {}
        
        # Process clustering results
        for linkage in clustering_results:
            key = frozenset(linkage.records)
            if key not in combined_linkages:
                combined_linkages[key] = {
                    'records': linkage.records,
                    'confidences': [],
                    'methods': [],
                    'features': set()
                }
            combined_linkages[key]['confidences'].append(linkage.linkage_confidence * 0.6)  # Weight
            combined_linkages[key]['methods'].append('clustering')
            combined_linkages[key]['features'].update(linkage.linking_features)
        
        # Process similarity results
        for linkage in similarity_results:
            key = frozenset(linkage.records)
            if key not in combined_linkages:
                combined_linkages[key] = {
                    'records': linkage.records,
                    'confidences': [],
                    'methods': [],
                    'features': set()
                }
            combined_linkages[key]['confidences'].append(linkage.linkage_confidence * 0.4)  # Weight
            combined_linkages[key]['methods'].append('similarity')
            combined_linkages[key]['features'].update(linkage.linking_features)
        
        # Create final ensemble linkages
        final_linkages = []
        for key, data in combined_linkages.items():
            if len(data['confidences']) > 0:
                ensemble_confidence = np.mean(data['confidences'])
                
                if ensemble_confidence >= self.similarity_thresholds['possible']:
                    linkage = RecordLinkage(
                        linkage_id=f"ensemble_{hash(key)}_{datetime.utcnow().timestamp()}",
                        records=list(data['records']),
                        linkage_confidence=ensemble_confidence,
                        linkage_type=self._determine_linkage_type(ensemble_confidence),
                        linking_features=list(data['features']),
                        method_used=f"ensemble_{'+'.join(data['methods'])}",
                        created_at=datetime.utcnow()
                    )
                    final_linkages.append(linkage)
        
        return final_linkages
    
    async def _deep_learning_linkage(self, records: List[Dict[str, Any]]) -> List[RecordLinkage]:
        """Deep learning-based record linkage"""
        
        if not PYTORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, falling back to ensemble matching")
            return await self._ensemble_matching_linkage(records)
        
        # This would implement a more sophisticated deep learning approach
        # For now, fall back to ensemble matching
        return await self._ensemble_matching_linkage(records)
    
    async def _extract_linkage_features(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for record linkage"""
        
        features = []
        
        for record in records:
            record_features = []
            
            # Demographic features
            patient_data = record.get('patient', {})
            
            # Age (normalized)
            age = patient_data.get('age', 0)
            record_features.append(age / 100.0 if age else 0)
            
            # Gender (encoded)
            gender = patient_data.get('gender', '').lower()
            gender_encoding = {'male': 1, 'female': 2, 'other': 3, '': 0}
            record_features.append(gender_encoding.get(gender, 0))
            
            # Name similarity features (using soundex or similar)
            name = patient_data.get('name', {})
            first_name = name.get('given', [''])[0] if name.get('given') else ''
            last_name = name.get('family', '')
            
            # Simple hash-based encoding for names
            record_features.append(hash(first_name.lower()) % 1000 / 1000.0)
            record_features.append(hash(last_name.lower()) % 1000 / 1000.0)
            
            # Clinical features
            conditions = record.get('conditions', [])
            medications = record.get('medications', [])
            
            record_features.append(len(conditions) / 10.0)  # Normalized condition count
            record_features.append(len(medications) / 10.0)  # Normalized medication count
            
            # Temporal features
            last_visit = record.get('last_visit_date')
            if last_visit:
                try:
                    visit_date = datetime.fromisoformat(last_visit.replace('Z', '+00:00'))
                    days_ago = (datetime.utcnow() - visit_date).days
                    record_features.append(min(days_ago / 365.0, 5.0))  # Years ago, capped at 5
                except:
                    record_features.append(0)
            else:
                record_features.append(0)
            
            features.append(record_features)
        
        return np.array(features)
    
    async def _calculate_record_similarity(
        self, 
        record1: Dict[str, Any], 
        record2: Dict[str, Any]
    ) -> SimilarityScore:
        """Calculate similarity between two records"""
        
        field_similarities = {}
        
        # Patient demographic similarity
        patient1 = record1.get('patient', {})
        patient2 = record2.get('patient', {})
        
        # Name similarity
        name1 = self._get_full_name(patient1)
        name2 = self._get_full_name(patient2)
        
        if FUZZYWUZZY_AVAILABLE and name1 and name2:
            field_similarities['name'] = fuzz.ratio(name1, name2) / 100.0
        else:
            field_similarities['name'] = 1.0 if name1 == name2 else 0.0
        
        # Date of birth similarity
        dob1 = patient1.get('birthDate', '')
        dob2 = patient2.get('birthDate', '')
        field_similarities['birthDate'] = 1.0 if dob1 == dob2 and dob1 else 0.0
        
        # Gender similarity
        gender1 = patient1.get('gender', '').lower()
        gender2 = patient2.get('gender', '').lower()
        field_similarities['gender'] = 1.0 if gender1 == gender2 and gender1 else 0.0
        
        # Clinical data similarity
        conditions1 = set(record1.get('conditions', []))
        conditions2 = set(record2.get('conditions', []))
        
        if conditions1 or conditions2:
            field_similarities['conditions'] = len(conditions1 & conditions2) / len(conditions1 | conditions2)
        else:
            field_similarities['conditions'] = 0.0
        
        # Overall similarity (weighted average)
        weights = {
            'name': 0.3,
            'birthDate': 0.3,
            'gender': 0.2,
            'conditions': 0.2
        }
        
        overall_similarity = sum(
            field_similarities.get(field, 0) * weight 
            for field, weight in weights.items()
        )
        
        # Calculate confidence based on data completeness
        completeness1 = sum(1 for v in [name1, dob1, gender1] if v)
        completeness2 = sum(1 for v in [name2, dob2, gender2] if v)
        confidence = min(completeness1, completeness2) / 3.0
        
        return SimilarityScore(
            record_1_id=record1.get('id', ''),
            record_2_id=record2.get('id', ''),
            overall_similarity=overall_similarity,
            field_similarities=field_similarities,
            confidence=confidence,
            method_used='fuzzy_matching',
            computed_at=datetime.utcnow()
        )
    
    def _get_full_name(self, patient: Dict[str, Any]) -> str:
        """Extract full name from patient data"""
        name = patient.get('name', {})
        if isinstance(name, list) and name:
            name = name[0]
        
        given = name.get('given', [])
        family = name.get('family', '')
        
        if isinstance(given, list):
            given = ' '.join(given)
        
        return f"{given} {family}".strip()
    
    def _calculate_cluster_confidence(self, cluster_features: np.ndarray) -> float:
        """Calculate confidence for a cluster of records"""
        if len(cluster_features) < 2:
            return 0.0
        
        # Calculate average pairwise distance within cluster
        distances = []
        for i in range(len(cluster_features)):
            for j in range(i + 1, len(cluster_features)):
                distance = np.linalg.norm(cluster_features[i] - cluster_features[j])
                distances.append(distance)
        
        avg_distance = np.mean(distances)
        
        # Convert distance to confidence (lower distance = higher confidence)
        confidence = max(0, 1 - avg_distance)
        return confidence
    
    def _determine_linkage_type(self, confidence: float) -> str:
        """Determine linkage type based on confidence"""
        if confidence >= self.similarity_thresholds['exact']:
            return 'exact'
        elif confidence >= self.similarity_thresholds['probable']:
            return 'probable'
        elif confidence >= self.similarity_thresholds['possible']:
            return 'possible'
        else:
            return 'unlikely'
    
    def _add_medical_patterns(self):
        """Add medical entity patterns to spaCy matcher"""
        if not self.matcher:
            return
        
        # Medical condition patterns
        condition_patterns = [
            [{"LOWER": {"IN": ["diabetes", "hypertension", "asthma", "copd"]}},
             {"LOWER": "mellitus", "OP": "?"}],
            [{"LOWER": "heart"}, {"LOWER": {"IN": ["failure", "disease", "attack"]}}],
            [{"LOWER": "chronic"}, {"LOWER": {"IN": ["pain", "fatigue", "kidney"]}}]
        ]
        
        self.matcher.add("MEDICAL_CONDITION", condition_patterns)


class SmartConflictResolver:
    """ML-based conflict resolution system"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.resolution_models = {}
        self.conflict_history = []
        
    async def initialize_models(self):
        """Initialize ML models for conflict resolution"""
        
        if SKLEARN_AVAILABLE:
            # Model for predicting best source based on conflict characteristics
            self.resolution_models['source_selector'] = RandomForestClassifier(
                n_estimators=50, random_state=42
            )
            
            # Model for confidence estimation
            self.resolution_models['confidence_estimator'] = LogisticRegression(
                random_state=42
            )
    
    async def resolve_conflicts(
        self, 
        conflicts: List[Dict[str, Any]], 
        method: ConflictResolutionML = ConflictResolutionML.ENSEMBLE_VOTING
    ) -> List[Dict[str, Any]]:
        """Resolve data conflicts using ML methods"""
        
        resolved_conflicts = []
        
        for conflict in conflicts:
            if method == ConflictResolutionML.CONFIDENCE_WEIGHTED:
                resolution = await self._confidence_weighted_resolution(conflict)
            elif method == ConflictResolutionML.ENSEMBLE_VOTING:
                resolution = await self._ensemble_voting_resolution(conflict)
            elif method == ConflictResolutionML.NEURAL_ARBITRATION:
                resolution = await self._neural_arbitration_resolution(conflict)
            else:
                resolution = await self._ensemble_voting_resolution(conflict)
            
            resolved_conflicts.append(resolution)
        
        return resolved_conflicts
    
    async def _confidence_weighted_resolution(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using confidence-weighted voting"""
        
        conflicting_values = conflict.get('conflicting_values', {})
        confidence_scores = conflict.get('confidence_scores', {})
        
        if not conflicting_values or not confidence_scores:
            return conflict
        
        # Calculate weighted scores
        weighted_scores = {}
        for source_id, value in conflicting_values.items():
            confidence = confidence_scores.get(source_id, 0.5)
            weighted_scores[source_id] = confidence
        
        # Select source with highest confidence
        best_source = max(weighted_scores.items(), key=lambda x: x[1])[0]
        resolution_value = conflicting_values[best_source]
        
        conflict.update({
            'resolved': True,
            'resolution': resolution_value,
            'resolution_rationale': f'Selected based on highest confidence ({weighted_scores[best_source]:.3f})',
            'resolution_method': 'confidence_weighted',
            'resolution_confidence': weighted_scores[best_source]
        })
        
        return conflict
    
    async def _ensemble_voting_resolution(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using ensemble voting"""
        
        conflicting_values = conflict.get('conflicting_values', {})
        confidence_scores = conflict.get('confidence_scores', {})
        
        # Count occurrences of each value
        value_counts = {}
        value_confidences = {}
        
        for source_id, value in conflicting_values.items():
            value_str = str(value)
            confidence = confidence_scores.get(source_id, 0.5)
            
            if value_str not in value_counts:
                value_counts[value_str] = 0
                value_confidences[value_str] = []
            
            value_counts[value_str] += 1
            value_confidences[value_str].append(confidence)
        
        # Select value with highest weighted vote
        best_value = None
        best_score = 0
        
        for value_str, count in value_counts.items():
            avg_confidence = np.mean(value_confidences[value_str])
            weighted_score = count * avg_confidence
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_value = value_str
        
        # Convert back to original type if needed
        resolution_value = best_value
        for source_id, value in conflicting_values.items():
            if str(value) == best_value:
                resolution_value = value
                break
        
        conflict.update({
            'resolved': True,
            'resolution': resolution_value,
            'resolution_rationale': f'Ensemble voting winner with score {best_score:.3f}',
            'resolution_method': 'ensemble_voting',
            'resolution_confidence': best_score / len(conflicting_values)
        })
        
        return conflict
    
    async def _neural_arbitration_resolution(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using neural arbitration (placeholder)"""
        
        # For now, fall back to ensemble voting
        # In a full implementation, this would use a trained neural network
        return await self._ensemble_voting_resolution(conflict)


class DataQualityAnalyzer:
    """ML-based data quality assessment system"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.quality_models = {}
        self.quality_thresholds = {
            DataQualityMetric.COMPLETENESS: 0.8,
            DataQualityMetric.ACCURACY: 0.9,
            DataQualityMetric.CONSISTENCY: 0.85,
            DataQualityMetric.TIMELINESS: 0.7,
            DataQualityMetric.VALIDITY: 0.95,
            DataQualityMetric.UNIQUENESS: 0.98
        }
    
    async def initialize_models(self):
        """Initialize ML models for quality assessment"""
        
        if SKLEARN_AVAILABLE:
            # Anomaly detection for data quality issues
            self.quality_models['anomaly_detector'] = IsolationForest(
                contamination=0.1, random_state=42
            )
            
            # Classification model for quality prediction
            self.quality_models['quality_classifier'] = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
    
    async def assess_data_quality(
        self, 
        records: List[Dict[str, Any]], 
        source_info: Dict[str, Any]
    ) -> List[DataQualityAssessment]:
        """Perform comprehensive data quality assessment"""
        
        quality_assessments = []
        
        for record in records:
            assessment = await self._assess_single_record_quality(record, source_info)
            quality_assessments.append(assessment)
        
        return quality_assessments
    
    async def _assess_single_record_quality(
        self, 
        record: Dict[str, Any], 
        source_info: Dict[str, Any]
    ) -> DataQualityAssessment:
        """Assess quality of a single record"""
        
        quality_scores = {}
        issues_detected = []
        recommendations = []
        
        # Completeness assessment
        completeness_score = await self._assess_completeness(record)
        quality_scores[DataQualityMetric.COMPLETENESS] = completeness_score
        
        if completeness_score < self.quality_thresholds[DataQualityMetric.COMPLETENESS]:
            issues_detected.append("Low data completeness")
            recommendations.append("Request additional data fields from source system")
        
        # Accuracy assessment
        accuracy_score = await self._assess_accuracy(record)
        quality_scores[DataQualityMetric.ACCURACY] = accuracy_score
        
        if accuracy_score < self.quality_thresholds[DataQualityMetric.ACCURACY]:
            issues_detected.append("Data accuracy concerns detected")
            recommendations.append("Validate data against external sources")
        
        # Consistency assessment
        consistency_score = await self._assess_consistency(record)
        quality_scores[DataQualityMetric.CONSISTENCY] = consistency_score
        
        if consistency_score < self.quality_thresholds[DataQualityMetric.CONSISTENCY]:
            issues_detected.append("Internal data inconsistencies found")
            recommendations.append("Review and standardize data entry procedures")
        
        # Timeliness assessment
        timeliness_score = await self._assess_timeliness(record)
        quality_scores[DataQualityMetric.TIMELINESS] = timeliness_score
        
        if timeliness_score < self.quality_thresholds[DataQualityMetric.TIMELINESS]:
            issues_detected.append("Data may be outdated")
            recommendations.append("Implement more frequent data updates")
        
        # Validity assessment
        validity_score = await self._assess_validity(record)
        quality_scores[DataQualityMetric.VALIDITY] = validity_score
        
        if validity_score < self.quality_thresholds[DataQualityMetric.VALIDITY]:
            issues_detected.append("Invalid data values detected")
            recommendations.append("Implement stronger data validation rules")
        
        # Overall quality score (weighted average)
        weights = {
            DataQualityMetric.COMPLETENESS: 0.25,
            DataQualityMetric.ACCURACY: 0.25,
            DataQualityMetric.CONSISTENCY: 0.2,
            DataQualityMetric.TIMELINESS: 0.1,
            DataQualityMetric.VALIDITY: 0.2
        }
        
        overall_quality = sum(
            quality_scores.get(metric, 0) * weight 
            for metric, weight in weights.items()
        )
        
        # Calculate confidence based on data availability
        confidence = min(1.0, len([s for s in quality_scores.values() if s > 0]) / len(weights))
        
        return DataQualityAssessment(
            record_id=record.get('id', ''),
            source_id=source_info.get('source_id', ''),
            quality_scores=quality_scores,
            overall_quality=overall_quality,
            issues_detected=issues_detected,
            recommendations=recommendations,
            confidence=confidence,
            assessed_at=datetime.utcnow()
        )
    
    async def _assess_completeness(self, record: Dict[str, Any]) -> float:
        """Assess data completeness"""
        
        required_fields = [
            'patient.name', 'patient.birthDate', 'patient.gender',
            'patient.identifier', 'encounter.date'
        ]
        
        present_fields = 0
        
        for field_path in required_fields:
            if self._get_nested_value(record, field_path) is not None:
                present_fields += 1
        
        return present_fields / len(required_fields) if required_fields else 0.0
    
    async def _assess_accuracy(self, record: Dict[str, Any]) -> float:
        """Assess data accuracy using various heuristics"""
        
        accuracy_checks = []
        
        # Check date validity
        birth_date = self._get_nested_value(record, 'patient.birthDate')
        if birth_date:
            try:
                birth_datetime = datetime.fromisoformat(birth_date.replace('Z', '+00:00'))
                if birth_datetime > datetime.utcnow():
                    accuracy_checks.append(0.0)  # Future birth date
                elif birth_datetime < datetime(1900, 1, 1):
                    accuracy_checks.append(0.5)  # Very old birth date
                else:
                    accuracy_checks.append(1.0)
            except:
                accuracy_checks.append(0.0)  # Invalid date format
        
        # Check age consistency
        age = self._get_nested_value(record, 'patient.age')
        if age and birth_date:
            try:
                birth_datetime = datetime.fromisoformat(birth_date.replace('Z', '+00:00'))
                calculated_age = (datetime.utcnow() - birth_datetime).days // 365
                if abs(calculated_age - age) <= 1:
                    accuracy_checks.append(1.0)
                else:
                    accuracy_checks.append(0.5)
            except:
                accuracy_checks.append(0.5)
        
        # Check name validity (basic heuristics)
        name = self._get_nested_value(record, 'patient.name.family')
        if name:
            if len(name) > 1 and name.isalpha():
                accuracy_checks.append(1.0)
            else:
                accuracy_checks.append(0.7)
        
        return np.mean(accuracy_checks) if accuracy_checks else 0.8  # Default
    
    async def _assess_consistency(self, record: Dict[str, Any]) -> float:
        """Assess internal data consistency"""
        
        consistency_checks = []
        
        # Check gender consistency across different fields
        gender1 = self._get_nested_value(record, 'patient.gender')
        gender2 = self._get_nested_value(record, 'demographics.gender')
        
        if gender1 and gender2:
            consistency_checks.append(1.0 if gender1.lower() == gender2.lower() else 0.0)
        
        # Check date consistency
        admission_date = self._get_nested_value(record, 'encounter.admission_date')
        discharge_date = self._get_nested_value(record, 'encounter.discharge_date')
        
        if admission_date and discharge_date:
            try:
                admission_dt = datetime.fromisoformat(admission_date.replace('Z', '+00:00'))
                discharge_dt = datetime.fromisoformat(discharge_date.replace('Z', '+00:00'))
                consistency_checks.append(1.0 if discharge_dt >= admission_dt else 0.0)
            except:
                consistency_checks.append(0.5)
        
        return np.mean(consistency_checks) if consistency_checks else 0.9  # Default
    
    async def _assess_timeliness(self, record: Dict[str, Any]) -> float:
        """Assess data timeliness"""
        
        last_updated = self._get_nested_value(record, 'meta.lastUpdated')
        if not last_updated:
            return 0.5  # Unknown update time
        
        try:
            updated_datetime = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            days_old = (datetime.utcnow() - updated_datetime).days
            
            # Score based on age (more recent = higher score)
            if days_old <= 1:
                return 1.0
            elif days_old <= 7:
                return 0.9
            elif days_old <= 30:
                return 0.7
            elif days_old <= 90:
                return 0.5
            else:
                return 0.3
        except:
            return 0.5
    
    async def _assess_validity(self, record: Dict[str, Any]) -> float:
        """Assess data validity using validation rules"""
        
        validity_checks = []
        
        # Validate gender values
        gender = self._get_nested_value(record, 'patient.gender')
        if gender:
            valid_genders = ['male', 'female', 'other', 'unknown']
            validity_checks.append(1.0 if gender.lower() in valid_genders else 0.0)
        
        # Validate phone number format (basic check)
        phone = self._get_nested_value(record, 'patient.telecom.value')
        if phone:
            # Simple phone validation
            if any(char.isdigit() for char in phone) and len(phone) >= 10:
                validity_checks.append(1.0)
            else:
                validity_checks.append(0.5)
        
        # Validate medical codes (basic format check)
        conditions = record.get('conditions', [])
        for condition in conditions:
            code = condition.get('code', {}).get('coding', [{}])[0].get('code', '')
            if code and (len(code) >= 3):  # Basic code format check
                validity_checks.append(1.0)
            else:
                validity_checks.append(0.7)
        
        return np.mean(validity_checks) if validity_checks else 0.9  # Default
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


class MLDataHarmonizationAgent(HIPAACompliantAgent, HealthcareAgent):
    """Advanced ML-based data harmonization agent"""
    
    def __init__(self, agent_id: str, settings, db_manager=None):
        super().__init__(agent_id, settings, db_manager)
        
        self.capabilities = [
            "ml_record_linkage",
            "smart_conflict_resolution", 
            "data_quality_assessment",
            "semantic_mapping",
            "anomaly_detection",
            "predictive_harmonization",
            "adaptive_learning"
        ]
        
        self.logger = structlog.get_logger(__name__)
        
        # Initialize ML components
        self.record_linkage = AdvancedRecordLinkage()
        self.conflict_resolver = SmartConflictResolver()
        self.quality_analyzer = DataQualityAnalyzer()
        
        # Model performance tracking
        self.performance_metrics = {
            'linkage_accuracy': 0.0,
            'resolution_accuracy': 0.0,
            'quality_prediction_accuracy': 0.0,
            'processing_time': 0.0,
            'last_updated': datetime.utcnow()
        }
    
    async def initialize(self):
        """Initialize the ML data harmonization agent"""
        
        try:
            await self.record_linkage.initialize_models()
            await self.conflict_resolver.initialize_models()
            await self.quality_analyzer.initialize_models()
            
            self.logger.info("ML Data Harmonization Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML agent: {e}")
            raise
    
    async def _process_healthcare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process healthcare data with ML-based harmonization"""
        
        start_time = datetime.utcnow()
        
        try:
            records = data.get("records", [])
            harmonization_method = MLHarmonizationMethod(
                data.get("method", MLHarmonizationMethod.HYBRID_APPROACH.value)
            )
            conflict_resolution_method = ConflictResolutionML(
                data.get("conflict_resolution", ConflictResolutionML.ENSEMBLE_VOTING.value)
            )
            
            # Step 1: Perform advanced record linkage
            linkages = await self.record_linkage.link_records(records, harmonization_method)
            
            # Step 2: Assess data quality
            quality_assessments = await self.quality_analyzer.assess_data_quality(
                records, data.get("source_info", {})
            )
            
            # Step 3: Detect conflicts
            conflicts = await self._detect_advanced_conflicts(records, linkages)
            
            # Step 4: Resolve conflicts using ML
            resolved_conflicts = await self.conflict_resolver.resolve_conflicts(
                conflicts, conflict_resolution_method
            )
            
            # Step 5: Generate harmonized dataset
            harmonized_data = await self._generate_harmonized_dataset(
                records, linkages, resolved_conflicts, quality_assessments
            )
            
            # Step 6: Calculate performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.performance_metrics['processing_time'] = processing_time
            self.performance_metrics['last_updated'] = datetime.utcnow()
            
            return {
                "harmonized_data": harmonized_data,
                "record_linkages": [linkage.__dict__ for linkage in linkages],
                "quality_assessments": [qa.__dict__ for qa in quality_assessments],
                "conflicts_resolved": resolved_conflicts,
                "performance_metrics": self.performance_metrics,
                "method_used": harmonization_method.value,
                "processing_time_seconds": processing_time,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"ML data harmonization failed: {e}")
            raise
    
    async def harmonize_patient_records(
        self,
        records: List[Dict[str, Any]],
        user_id: str,
        user_permissions: List[str],
        access_reason: str,
        harmonization_method: MLHarmonizationMethod = MLHarmonizationMethod.HYBRID_APPROACH,
        conflict_resolution: ConflictResolutionML = ConflictResolutionML.ENSEMBLE_VOTING
    ) -> Dict[str, Any]:
        """Harmonize patient records using ML methods"""
        
        harmonization_data = {
            "records": records,
            "method": harmonization_method.value,
            "conflict_resolution": conflict_resolution.value,
            "source_info": {
                "source_id": "multiple_sources",
                "harmonization_timestamp": datetime.utcnow().isoformat()
            }
        }
        
        return await self.secure_process_data(
            data=harmonization_data,
            user_id=user_id,
            user_permissions=user_permissions,
            access_reason=access_reason,
            action=AuditAction.TRANSFORM,
            resource_type="MLDataHarmonization",
            patient_id=self._extract_patient_id(records)
        )
    
    async def _detect_advanced_conflicts(
        self, 
        records: List[Dict[str, Any]], 
        linkages: List[RecordLinkage]
    ) -> List[Dict[str, Any]]:
        """Detect conflicts using advanced ML techniques"""
        
        conflicts = []
        
        # Group linked records
        linked_groups = {}
        for linkage in linkages:
            group_id = linkage.linkage_id
            linked_groups[group_id] = linkage.records
        
        # Check for conflicts within linked groups
        for group_id, record_ids in linked_groups.items():
            group_records = [r for r in records if r.get('id') in record_ids]
            
            if len(group_records) > 1:
                group_conflicts = await self._find_conflicts_in_group(group_records)
                conflicts.extend(group_conflicts)
        
        return conflicts
    
    async def _find_conflicts_in_group(self, group_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find conflicts within a group of linked records"""
        
        conflicts = []
        
        # Check common fields for conflicts
        conflict_fields = [
            'patient.name.family',
            'patient.name.given',
            'patient.birthDate',
            'patient.gender',
            'patient.address'
        ]
        
        for field_path in conflict_fields:
            values = {}
            confidences = {}
            
            for i, record in enumerate(group_records):
                value = self._get_nested_value(record, field_path)
                if value is not None:
                    source_id = record.get('source', {}).get('id', f'source_{i}')
                    values[source_id] = value
                    # Calculate confidence based on data completeness and source reliability
                    confidences[source_id] = self._calculate_source_confidence(record)
            
            # Check if there are conflicting values
            if len(set(str(v) for v in values.values())) > 1:
                conflict = {
                    'conflict_id': f"conflict_{hash(field_path)}_{datetime.utcnow().timestamp()}",
                    'field_path': field_path,
                    'patient_id': group_records[0].get('patient', {}).get('id', ''),
                    'conflicting_values': values,
                    'confidence_scores': confidences,
                    'conflict_type': 'value_mismatch',
                    'created_at': datetime.utcnow().isoformat(),
                    'resolved': False
                }
                conflicts.append(conflict)
        
        return conflicts
    
    async def _generate_harmonized_dataset(
        self,
        records: List[Dict[str, Any]],
        linkages: List[RecordLinkage],
        resolved_conflicts: List[Dict[str, Any]],
        quality_assessments: List[DataQualityAssessment]
    ) -> Dict[str, Any]:
        """Generate the final harmonized dataset"""
        
        harmonized_records = []
        processed_record_ids = set()
        
        # Create quality lookup
        quality_lookup = {qa.record_id: qa for qa in quality_assessments}
        
        # Process linked record groups
        for linkage in linkages:
            if linkage.linkage_type in ['exact', 'probable']:
                # Merge linked records
                linked_records = [r for r in records if r.get('id') in linkage.records]
                
                if linked_records:
                    merged_record = await self._merge_linked_records(
                        linked_records, resolved_conflicts, quality_lookup
                    )
                    harmonized_records.append(merged_record)
                    processed_record_ids.update(linkage.records)
        
        # Add unlinked records
        for record in records:
            if record.get('id') not in processed_record_ids:
                # Apply quality improvements
                improved_record = await self._improve_record_quality(
                    record, quality_lookup.get(record.get('id'))
                )
                harmonized_records.append(improved_record)
        
        # Calculate harmonization statistics
        total_original_records = len(records)
        total_harmonized_records = len(harmonized_records)
        deduplication_rate = (total_original_records - total_harmonized_records) / total_original_records
        
        # Calculate overall quality score
        quality_scores = [qa.overall_quality for qa in quality_assessments]
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0
        
        return {
            'records': harmonized_records,
            'statistics': {
                'original_record_count': total_original_records,
                'harmonized_record_count': total_harmonized_records,
                'deduplication_rate': deduplication_rate,
                'average_quality_score': avg_quality_score,
                'linkages_created': len(linkages),
                'conflicts_resolved': len([c for c in resolved_conflicts if c.get('resolved')])
            },
            'harmonization_metadata': {
                'method': 'ml_based_harmonization',
                'timestamp': datetime.utcnow().isoformat(),
                'agent_version': '2.0.0'
            }
        }
    
    async def _merge_linked_records(
        self,
        linked_records: List[Dict[str, Any]],
        resolved_conflicts: List[Dict[str, Any]],
        quality_lookup: Dict[str, DataQualityAssessment]
    ) -> Dict[str, Any]:
        """Merge linked records into a single harmonized record"""
        
        # Start with the highest quality record as base
        base_record = max(
            linked_records,
            key=lambda r: quality_lookup.get(r.get('id'), DataQualityAssessment(
                record_id='', source_id='', quality_scores={}, overall_quality=0.0,
                issues_detected=[], recommendations=[], confidence=0.0, assessed_at=datetime.utcnow()
            )).overall_quality
        )
        
        merged_record = base_record.copy()
        
        # Apply conflict resolutions
        conflict_lookup = {}
        for conflict in resolved_conflicts:
            if conflict.get('resolved') and 'resolution' in conflict:
                field_path = conflict.get('field_path', '')
                conflict_lookup[field_path] = conflict['resolution']
        
        # Merge data from other records
        for record in linked_records:
            if record.get('id') != base_record.get('id'):
                merged_record = await self._merge_record_data(
                    merged_record, record, conflict_lookup
                )
        
        # Add harmonization metadata
        merged_record['_harmonization'] = {
            'merged_from': [r.get('id') for r in linked_records],
            'merge_timestamp': datetime.utcnow().isoformat(),
            'conflicts_resolved': len([c for c in resolved_conflicts if c.get('resolved')]),
            'primary_source': base_record.get('source', {}).get('id', 'unknown')
        }
        
        return merged_record
    
    async def _merge_record_data(
        self,
        target_record: Dict[str, Any],
        source_record: Dict[str, Any],
        conflict_resolutions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge data from source record into target record"""
        
        # This is a simplified merge - in practice, this would be more sophisticated
        for key, value in source_record.items():
            if key not in target_record or target_record[key] is None:
                target_record[key] = value
            elif key in conflict_resolutions:
                target_record[key] = conflict_resolutions[key]
        
        return target_record
    
    async def _improve_record_quality(
        self,
        record: Dict[str, Any],
        quality_assessment: Optional[DataQualityAssessment]
    ) -> Dict[str, Any]:
        """Improve record quality based on assessment"""
        
        if not quality_assessment:
            return record
        
        improved_record = record.copy()
        
        # Apply quality improvements based on detected issues
        for issue in quality_assessment.issues_detected:
            if "completeness" in issue.lower():
                improved_record = await self._fill_missing_data(improved_record)
            elif "consistency" in issue.lower():
                improved_record = await self._fix_inconsistencies(improved_record)
            elif "validity" in issue.lower():
                improved_record = await self._validate_and_correct(improved_record)
        
        return improved_record
    
    async def _fill_missing_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing data using ML predictions or rules"""
        
        # Placeholder for missing data imputation
        # In practice, this would use trained ML models
        return record
    
    async def _fix_inconsistencies(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Fix data inconsistencies"""
        
        # Placeholder for consistency fixes
        return record
    
    async def _validate_and_correct(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and correct invalid data"""
        
        # Placeholder for data validation and correction
        return record
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _calculate_source_confidence(self, record: Dict[str, Any]) -> float:
        """Calculate confidence score for a data source"""
        
        source_info = record.get('source', {})
        reliability = source_info.get('reliability_score', 0.5)
        
        # Factor in data completeness
        patient_data = record.get('patient', {})
        required_fields = ['name', 'birthDate', 'gender']
        completeness = sum(1 for field in required_fields if patient_data.get(field)) / len(required_fields)
        
        return (reliability * 0.7) + (completeness * 0.3)
    
    def _extract_patient_id(self, records: List[Dict[str, Any]]) -> Optional[str]:
        """Extract patient ID for audit purposes"""
        
        for record in records:
            patient_id = record.get('patient', {}).get('id')
            if patient_id:
                return patient_id
        
        return None


# Factory function
def create_ml_harmonization_agent(agent_id: str, settings, db_manager=None) -> MLDataHarmonizationAgent:
    """Create ML-based data harmonization agent"""
    return MLDataHarmonizationAgent(agent_id, settings, db_manager)