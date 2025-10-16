"""
Data Harmonization Agent for Vita Agents
Normalizes and harmonizes healthcare data from multiple sources
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Set
import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import difflib

import structlog

from ..core.agent import HealthcareAgent
from ..core.security import HIPAACompliantAgent, AuditAction, ComplianceLevel
from ..core.config import Settings


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving data conflicts"""
    LATEST_WINS = "latest_wins"
    MOST_COMPLETE = "most_complete"
    HIGHEST_CONFIDENCE = "highest_confidence"
    MANUAL_REVIEW = "manual_review"
    SOURCE_PRIORITY = "source_priority"


class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class DataSource:
    """Data source information"""
    source_id: str
    name: str
    system_type: str  # EHR, lab, pharmacy, etc.
    reliability_score: float  # 0.0 to 1.0
    last_updated: datetime
    data_standards: List[str]  # FHIR, HL7, etc.
    priority: int  # 1=highest priority


@dataclass
class DataConflict:
    """Represents a conflict between data sources"""
    conflict_id: str
    field_path: str
    patient_id: str
    conflicting_values: Dict[str, Any]  # source_id -> value
    conflict_type: str
    resolution_strategy: ConflictResolutionStrategy
    confidence_scores: Dict[str, float]
    created_at: datetime
    resolved: bool = False
    resolution: Optional[Any] = None
    resolution_rationale: Optional[str] = None


@dataclass
class HarmonizationResult:
    """Result of data harmonization process"""
    harmonized_data: Dict[str, Any]
    conflicts_detected: List[DataConflict]
    conflicts_resolved: List[DataConflict]
    data_quality_score: float
    completeness_score: float
    source_contribution: Dict[str, float]  # percentage from each source
    harmonization_timestamp: datetime


class DataHarmonizationAgent(HIPAACompliantAgent, HealthcareAgent):
    """Agent for harmonizing healthcare data from multiple sources"""
    
    def __init__(self, agent_id: str, settings: Settings, db_manager=None):
        super().__init__(agent_id, settings, db_manager)
        
        self.capabilities = [
            "data_normalization",
            "conflict_resolution",
            "duplicate_detection", 
            "data_quality_assessment",
            "schema_mapping",
            "terminology_harmonization",
            "record_linkage",
            "data_completeness_analysis"
        ]
        
        self.logger = structlog.get_logger(__name__)
        
        # Initialize harmonization components
        self._load_terminology_mappings()
        self._load_schema_mappings()
        self._initialize_conflict_resolution_rules()
    
    async def _process_healthcare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process multiple data sources for harmonization"""
        try:
            data_sources = data.get("data_sources", [])
            patient_id = data.get("patient_id")
            harmonization_rules = data.get("harmonization_rules", {})
            
            # Step 1: Normalize data from each source
            normalized_sources = await self._normalize_data_sources(data_sources)
            
            # Step 2: Detect conflicts between sources
            conflicts = await self._detect_conflicts(normalized_sources, patient_id)
            
            # Step 3: Resolve conflicts using configured strategies
            resolved_conflicts = await self._resolve_conflicts(conflicts, harmonization_rules)
            
            # Step 4: Merge data sources
            harmonized_data = await self._merge_data_sources(
                normalized_sources, resolved_conflicts
            )
            
            # Step 5: Assess data quality
            quality_assessment = await self._assess_harmonized_quality(
                harmonized_data, normalized_sources
            )
            
            return {
                "harmonized_data": harmonized_data,
                "conflicts_detected": [c.__dict__ for c in conflicts],
                "conflicts_resolved": [c.__dict__ for c in resolved_conflicts],
                "quality_assessment": quality_assessment,
                "source_count": len(data_sources),
                "processed_at": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            self.logger.error("Data harmonization failed", error=str(e))
            raise
    
    async def harmonize_patient_data(
        self,
        data_sources: List[Dict[str, Any]],
        patient_id: str,
        user_id: str,
        user_permissions: List[str],
        access_reason: str
    ) -> Dict[str, Any]:
        """Harmonize patient data from multiple sources"""
        
        harmonization_data = {
            "data_sources": data_sources,
            "patient_id": patient_id,
            "harmonization_rules": self._get_default_harmonization_rules()
        }
        
        return await self.secure_process_data(
            data=harmonization_data,
            user_id=user_id,
            user_permissions=user_permissions,
            access_reason=access_reason,
            action=AuditAction.TRANSFORM,
            resource_type="DataHarmonization",
            patient_id=patient_id
        )
    
    async def _normalize_data_sources(self, data_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize data from different sources to common format"""
        normalized_sources = []
        
        for source_data in data_sources:
            source_info = DataSource(**source_data.get("source_info", {}))
            raw_data = source_data.get("data", {})
            
            # Normalize based on source type and format
            normalized_data = await self._normalize_single_source(raw_data, source_info)
            
            normalized_sources.append({
                "source_info": source_info,
                "normalized_data": normalized_data,
                "original_data": raw_data
            })
        
        return normalized_sources
    
    async def _normalize_single_source(self, raw_data: Dict[str, Any], source: DataSource) -> Dict[str, Any]:
        """Normalize data from a single source"""
        
        if "FHIR" in source.data_standards:
            return await self._normalize_fhir_data(raw_data)
        elif "HL7" in source.data_standards:
            return await self._normalize_hl7_data(raw_data)
        elif source.system_type.lower() in ["epic", "cerner", "allscripts"]:
            return await self._normalize_ehr_data(raw_data, source.system_type)
        else:
            return await self._normalize_generic_data(raw_data)
    
    async def _normalize_fhir_data(self, fhir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize FHIR data to common format"""
        normalized = {}
        
        if fhir_data.get("resourceType") == "Patient":
            normalized = {
                "patient_id": fhir_data.get("id"),
                "name": self._extract_patient_name(fhir_data),
                "birth_date": fhir_data.get("birthDate"),
                "gender": fhir_data.get("gender"),
                "identifiers": self._extract_identifiers(fhir_data),
                "addresses": self._extract_addresses(fhir_data),
                "telecoms": self._extract_telecoms(fhir_data),
                "active": fhir_data.get("active", True)
            }
        elif fhir_data.get("resourceType") == "Observation":
            normalized = {
                "observation_id": fhir_data.get("id"),
                "patient_reference": self._extract_patient_reference(fhir_data),
                "code": self._extract_coding(fhir_data.get("code", {})),
                "value": self._extract_observation_value(fhir_data),
                "effective_date": fhir_data.get("effectiveDateTime"),
                "status": fhir_data.get("status"),
                "category": self._extract_coding(fhir_data.get("category", [{}])[0] if fhir_data.get("category") else {})
            }
        elif fhir_data.get("resourceType") == "Medication":
            normalized = {
                "medication_id": fhir_data.get("id"),
                "code": self._extract_coding(fhir_data.get("code", {})),
                "form": self._extract_coding(fhir_data.get("form", {})),
                "ingredients": self._extract_medication_ingredients(fhir_data),
                "status": fhir_data.get("status")
            }
        
        # Add metadata
        normalized["_source_format"] = "FHIR"
        normalized["_last_updated"] = fhir_data.get("meta", {}).get("lastUpdated")
        
        return normalized
    
    async def _normalize_hl7_data(self, hl7_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize HL7 data to common format"""
        # Simplified HL7 normalization
        normalized = {}
        
        if hl7_data.get("message_type") == "ADT":
            # Patient demographics from HL7 ADT
            pid_segment = hl7_data.get("PID", {})
            normalized = {
                "patient_id": pid_segment.get("patient_id"),
                "name": self._parse_hl7_name(pid_segment.get("name")),
                "birth_date": self._parse_hl7_date(pid_segment.get("birth_date")),
                "gender": self._map_hl7_gender(pid_segment.get("gender")),
                "identifiers": self._parse_hl7_identifiers(pid_segment),
                "addresses": self._parse_hl7_addresses(pid_segment),
                "telecoms": self._parse_hl7_telecoms(pid_segment)
            }
        elif hl7_data.get("message_type") == "ORU":
            # Observation results from HL7 ORU
            obx_segments = hl7_data.get("OBX", [])
            normalized = {
                "observations": [self._parse_hl7_observation(obx) for obx in obx_segments]
            }
        
        normalized["_source_format"] = "HL7"
        normalized["_message_timestamp"] = hl7_data.get("timestamp")
        
        return normalized
    
    async def _normalize_ehr_data(self, ehr_data: Dict[str, Any], system_type: str) -> Dict[str, Any]:
        """Normalize EHR-specific data formats"""
        normalized = {}
        
        if system_type.lower() == "epic":
            normalized = await self._normalize_epic_data(ehr_data)
        elif system_type.lower() == "cerner":
            normalized = await self._normalize_cerner_data(ehr_data)
        elif system_type.lower() == "allscripts":
            normalized = await self._normalize_allscripts_data(ehr_data)
        
        normalized["_source_format"] = f"EHR_{system_type.upper()}"
        
        return normalized
    
    async def _normalize_generic_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize generic/unknown format data"""
        # Apply generic field mappings
        field_mappings = self.schema_mappings.get("generic", {})
        
        normalized = {}
        for raw_field, value in raw_data.items():
            mapped_field = field_mappings.get(raw_field, raw_field)
            normalized[mapped_field] = value
        
        normalized["_source_format"] = "GENERIC"
        
        return normalized
    
    async def _detect_conflicts(self, normalized_sources: List[Dict[str, Any]], patient_id: str) -> List[DataConflict]:
        """Detect conflicts between normalized data sources"""
        conflicts = []
        
        if len(normalized_sources) < 2:
            return conflicts
        
        # Compare each pair of sources
        for i, source1 in enumerate(normalized_sources):
            for source2 in normalized_sources[i+1:]:
                source_conflicts = await self._compare_sources(
                    source1, source2, patient_id
                )
                conflicts.extend(source_conflicts)
        
        return conflicts
    
    async def _compare_sources(
        self, source1: Dict[str, Any], source2: Dict[str, Any], patient_id: str
    ) -> List[DataConflict]:
        """Compare two data sources and identify conflicts"""
        conflicts = []
        
        data1 = source1["normalized_data"]
        data2 = source2["normalized_data"]
        source1_id = source1["source_info"].source_id
        source2_id = source2["source_info"].source_id
        
        # Compare common fields
        common_fields = set(data1.keys()) & set(data2.keys())
        
        for field in common_fields:
            if field.startswith("_"):  # Skip metadata fields
                continue
                
            value1 = data1[field]
            value2 = data2[field]
            
            if not self._values_equivalent(value1, value2):
                conflict = DataConflict(
                    conflict_id=f"{patient_id}_{field}_{source1_id}_{source2_id}",
                    field_path=field,
                    patient_id=patient_id,
                    conflicting_values={source1_id: value1, source2_id: value2},
                    conflict_type=self._classify_conflict(value1, value2),
                    resolution_strategy=self._get_resolution_strategy(field),
                    confidence_scores={
                        source1_id: source1["source_info"].reliability_score,
                        source2_id: source2["source_info"].reliability_score
                    },
                    created_at=datetime.utcnow()
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _resolve_conflicts(
        self, conflicts: List[DataConflict], harmonization_rules: Dict[str, Any]
    ) -> List[DataConflict]:
        """Resolve detected conflicts using specified strategies"""
        resolved_conflicts = []
        
        for conflict in conflicts:
            try:
                resolved_conflict = await self._resolve_single_conflict(conflict, harmonization_rules)
                resolved_conflicts.append(resolved_conflict)
            except Exception as e:
                self.logger.error(
                    "Failed to resolve conflict",
                    conflict_id=conflict.conflict_id,
                    error=str(e)
                )
                # Mark as requiring manual review
                conflict.resolution_strategy = ConflictResolutionStrategy.MANUAL_REVIEW
                resolved_conflicts.append(conflict)
        
        return resolved_conflicts
    
    async def _resolve_single_conflict(
        self, conflict: DataConflict, rules: Dict[str, Any]
    ) -> DataConflict:
        """Resolve a single conflict using appropriate strategy"""
        
        strategy = conflict.resolution_strategy
        
        if strategy == ConflictResolutionStrategy.LATEST_WINS:
            conflict.resolution = await self._resolve_by_latest(conflict)
            conflict.resolution_rationale = "Selected most recent value"
            
        elif strategy == ConflictResolutionStrategy.MOST_COMPLETE:
            conflict.resolution = await self._resolve_by_completeness(conflict)
            conflict.resolution_rationale = "Selected most complete value"
            
        elif strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            conflict.resolution = await self._resolve_by_confidence(conflict)
            conflict.resolution_rationale = "Selected value from most reliable source"
            
        elif strategy == ConflictResolutionStrategy.SOURCE_PRIORITY:
            conflict.resolution = await self._resolve_by_priority(conflict)
            conflict.resolution_rationale = "Selected value from highest priority source"
            
        else:
            # Manual review required
            conflict.resolution = None
            conflict.resolution_rationale = "Requires manual review"
        
        conflict.resolved = conflict.resolution is not None
        
        return conflict
    
    async def _merge_data_sources(
        self, normalized_sources: List[Dict[str, Any]], resolved_conflicts: List[DataConflict]
    ) -> Dict[str, Any]:
        """Merge normalized sources using conflict resolutions"""
        
        merged_data = {}
        conflict_resolutions = {c.field_path: c.resolution for c in resolved_conflicts if c.resolved}
        
        # Start with the highest priority source as base
        base_source = max(
            normalized_sources,
            key=lambda s: s["source_info"].priority
        )
        merged_data = base_source["normalized_data"].copy()
        
        # Apply data from other sources
        for source in normalized_sources:
            if source == base_source:
                continue
                
            source_data = source["normalized_data"]
            
            for field, value in source_data.items():
                if field.startswith("_"):
                    continue
                    
                if field in conflict_resolutions:
                    # Use conflict resolution
                    merged_data[field] = conflict_resolutions[field]
                elif field not in merged_data or merged_data[field] is None:
                    # Fill missing data
                    merged_data[field] = value
        
        # Add harmonization metadata
        merged_data["_harmonization"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "source_count": len(normalized_sources),
            "conflicts_resolved": len([c for c in resolved_conflicts if c.resolved]),
            "source_contributions": self._calculate_source_contributions(
                normalized_sources, merged_data
            )
        }
        
        return merged_data
    
    async def _assess_harmonized_quality(
        self, harmonized_data: Dict[str, Any], sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess quality of harmonized data"""
        
        # Calculate completeness
        completeness = self._calculate_completeness(harmonized_data)
        
        # Calculate consistency score
        consistency = self._calculate_consistency(harmonized_data, sources)
        
        # Calculate overall quality score
        quality_score = (completeness + consistency) / 2
        
        # Determine quality level
        if quality_score >= 0.9:
            quality_level = DataQualityLevel.EXCELLENT
        elif quality_score >= 0.8:
            quality_level = DataQualityLevel.GOOD
        elif quality_score >= 0.6:
            quality_level = DataQualityLevel.FAIR
        elif quality_score >= 0.4:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.UNUSABLE
        
        return {
            "overall_score": quality_score,
            "quality_level": quality_level.value,
            "completeness_score": completeness,
            "consistency_score": consistency,
            "assessment_timestamp": datetime.utcnow().isoformat()
        }
    
    def _load_terminology_mappings(self):
        """Load terminology mapping tables"""
        self.terminology_mappings = {
            "gender": {
                # HL7 to FHIR mappings
                "M": "male",
                "F": "female",
                "U": "unknown",
                "O": "other"
            },
            "units": {
                # Common unit mappings
                "mg/dl": "mg/dL",
                "mmol/l": "mmol/L",
                "bpm": "/min"
            }
        }
    
    def _load_schema_mappings(self):
        """Load schema mapping configurations"""
        self.schema_mappings = {
            "epic": {
                "patient_id": "PatientID",
                "medical_record_number": "MRN",
                "first_name": "FirstName",
                "last_name": "LastName"
            },
            "cerner": {
                "patient_id": "PersonID",
                "medical_record_number": "MRN",
                "given_name": "GivenName",
                "family_name": "FamilyName"
            },
            "generic": {
                "pt_id": "patient_id",
                "dob": "birth_date",
                "sex": "gender"
            }
        }
    
    def _initialize_conflict_resolution_rules(self):
        """Initialize default conflict resolution rules"""
        self.conflict_resolution_rules = {
            "patient_id": ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
            "name": ConflictResolutionStrategy.MOST_COMPLETE,
            "birth_date": ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
            "gender": ConflictResolutionStrategy.SOURCE_PRIORITY,
            "addresses": ConflictResolutionStrategy.LATEST_WINS,
            "telecoms": ConflictResolutionStrategy.MOST_COMPLETE,
            "medications": ConflictResolutionStrategy.LATEST_WINS,
            "observations": ConflictResolutionStrategy.LATEST_WINS
        }
    
    def _get_default_harmonization_rules(self) -> Dict[str, Any]:
        """Get default harmonization rules"""
        return {
            "conflict_resolution_strategies": self.conflict_resolution_rules,
            "require_manual_review_threshold": 0.5,
            "quality_score_threshold": 0.7
        }
    
    # Helper methods for data extraction and parsing
    def _extract_patient_name(self, fhir_patient: Dict[str, Any]) -> Dict[str, str]:
        """Extract patient name from FHIR Patient resource"""
        names = fhir_patient.get("name", [])
        if not names:
            return {}
        
        # Use first official name
        name = next((n for n in names if n.get("use") == "official"), names[0])
        
        return {
            "family": " ".join(name.get("family", [])) if isinstance(name.get("family"), list) else name.get("family", ""),
            "given": " ".join(name.get("given", [])),
            "prefix": " ".join(name.get("prefix", [])),
            "suffix": " ".join(name.get("suffix", []))
        }
    
    def _extract_identifiers(self, fhir_resource: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract identifiers from FHIR resource"""
        identifiers = fhir_resource.get("identifier", [])
        
        extracted = []
        for identifier in identifiers:
            extracted.append({
                "system": identifier.get("system", ""),
                "value": identifier.get("value", ""),
                "type": identifier.get("type", {}).get("coding", [{}])[0].get("code", ""),
                "use": identifier.get("use", "")
            })
        
        return extracted
    
    def _extract_addresses(self, fhir_resource: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract addresses from FHIR resource"""
        addresses = fhir_resource.get("address", [])
        
        extracted = []
        for address in addresses:
            extracted.append({
                "line": address.get("line", []),
                "city": address.get("city", ""),
                "state": address.get("state", ""),
                "postal_code": address.get("postalCode", ""),
                "country": address.get("country", ""),
                "use": address.get("use", "")
            })
        
        return extracted
    
    def _extract_telecoms(self, fhir_resource: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract telecoms from FHIR resource"""
        telecoms = fhir_resource.get("telecom", [])
        
        extracted = []
        for telecom in telecoms:
            extracted.append({
                "system": telecom.get("system", ""),
                "value": telecom.get("value", ""),
                "use": telecom.get("use", "")
            })
        
        return extracted
    
    def _extract_patient_reference(self, fhir_resource: Dict[str, Any]) -> str:
        """Extract patient reference from FHIR resource"""
        subject = fhir_resource.get("subject", {})
        return subject.get("reference", "")
    
    def _extract_coding(self, coding_element: Dict[str, Any]) -> Dict[str, str]:
        """Extract coding information"""
        if not coding_element:
            return {}
        
        coding = coding_element.get("coding", [])
        if not coding:
            return {}
        
        first_coding = coding[0]
        return {
            "system": first_coding.get("system", ""),
            "code": first_coding.get("code", ""),
            "display": first_coding.get("display", "")
        }
    
    def _extract_observation_value(self, fhir_observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract observation value"""
        value_keys = [k for k in fhir_observation.keys() if k.startswith("value")]
        
        if not value_keys:
            return {}
        
        value_key = value_keys[0]
        value = fhir_observation[value_key]
        
        if isinstance(value, dict) and "value" in value:
            return {
                "value": value["value"],
                "unit": value.get("unit", ""),
                "system": value.get("system", ""),
                "type": value_key
            }
        
        return {"value": value, "type": value_key}
    
    def _extract_medication_ingredients(self, fhir_medication: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract medication ingredients"""
        ingredients = fhir_medication.get("ingredient", [])
        
        extracted = []
        for ingredient in ingredients:
            item = ingredient.get("itemCodeableConcept", {})
            strength = ingredient.get("strength", {})
            
            extracted.append({
                "item": self._extract_coding(item),
                "strength": strength.get("numerator", {}).get("value", 0),
                "strength_unit": strength.get("numerator", {}).get("unit", "")
            })
        
        return extracted
    
    # HL7 parsing helpers
    def _parse_hl7_name(self, hl7_name: str) -> Dict[str, str]:
        """Parse HL7 name format"""
        if not hl7_name:
            return {}
        
        # HL7 name format: Family^Given^Middle^Suffix^Prefix
        parts = hl7_name.split("^")
        
        return {
            "family": parts[0] if len(parts) > 0 else "",
            "given": parts[1] if len(parts) > 1 else "",
            "middle": parts[2] if len(parts) > 2 else "",
            "suffix": parts[3] if len(parts) > 3 else "",
            "prefix": parts[4] if len(parts) > 4 else ""
        }
    
    def _parse_hl7_date(self, hl7_date: str) -> str:
        """Parse HL7 date format to ISO format"""
        if not hl7_date:
            return ""
        
        # HL7 date format: YYYYMMDD or YYYYMMDDHHMMSS
        if len(hl7_date) >= 8:
            year = hl7_date[0:4]
            month = hl7_date[4:6]
            day = hl7_date[6:8]
            return f"{year}-{month}-{day}"
        
        return hl7_date
    
    def _map_hl7_gender(self, hl7_gender: str) -> str:
        """Map HL7 gender to FHIR format"""
        return self.terminology_mappings["gender"].get(hl7_gender, "unknown")
    
    def _parse_hl7_identifiers(self, pid_segment: Dict[str, Any]) -> List[Dict[str, str]]:
        """Parse HL7 identifiers"""
        # Simplified - would need full HL7 parsing in real implementation
        identifiers = []
        
        if pid_segment.get("patient_id"):
            identifiers.append({
                "system": "hospital_id",
                "value": pid_segment["patient_id"],
                "type": "MR"
            })
        
        return identifiers
    
    def _parse_hl7_addresses(self, pid_segment: Dict[str, Any]) -> List[Dict[str, str]]:
        """Parse HL7 addresses"""
        # Simplified implementation
        return []
    
    def _parse_hl7_telecoms(self, pid_segment: Dict[str, Any]) -> List[Dict[str, str]]:
        """Parse HL7 telecoms"""
        # Simplified implementation
        return []
    
    def _parse_hl7_observation(self, obx_segment: Dict[str, Any]) -> Dict[str, Any]:
        """Parse HL7 OBX segment to observation"""
        return {
            "code": obx_segment.get("observation_id", ""),
            "value": obx_segment.get("observation_value", ""),
            "units": obx_segment.get("units", ""),
            "reference_range": obx_segment.get("reference_range", ""),
            "status": obx_segment.get("observation_result_status", "")
        }
    
    # EHR-specific normalization
    async def _normalize_epic_data(self, epic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Epic-specific data format"""
        mappings = self.schema_mappings["epic"]
        normalized = {}
        
        for epic_field, value in epic_data.items():
            normalized_field = mappings.get(epic_field, epic_field)
            normalized[normalized_field] = value
        
        return normalized
    
    async def _normalize_cerner_data(self, cerner_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Cerner-specific data format"""
        mappings = self.schema_mappings["cerner"]
        normalized = {}
        
        for cerner_field, value in cerner_data.items():
            normalized_field = mappings.get(cerner_field, cerner_field)
            normalized[normalized_field] = value
        
        return normalized
    
    async def _normalize_allscripts_data(self, allscripts_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Allscripts-specific data format"""
        # Simplified implementation
        return allscripts_data
    
    # Conflict detection and resolution helpers
    def _values_equivalent(self, value1: Any, value2: Any) -> bool:
        """Check if two values are equivalent (handling different formats)"""
        if value1 == value2:
            return True
        
        # Handle None/empty equivalence
        if not value1 and not value2:
            return True
        
        # Handle string case differences
        if isinstance(value1, str) and isinstance(value2, str):
            return value1.lower().strip() == value2.lower().strip()
        
        # Handle date format differences
        if self._is_date_string(value1) and self._is_date_string(value2):
            return self._normalize_date(value1) == self._normalize_date(value2)
        
        # Handle numeric equivalence
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return abs(value1 - value2) < 0.001
        
        return False
    
    def _classify_conflict(self, value1: Any, value2: Any) -> str:
        """Classify the type of conflict"""
        if type(value1) != type(value2):
            return "type_mismatch"
        
        if isinstance(value1, str) and isinstance(value2, str):
            similarity = difflib.SequenceMatcher(None, value1, value2).ratio()
            if similarity > 0.8:
                return "minor_difference"
            else:
                return "major_difference"
        
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            relative_diff = abs(value1 - value2) / max(abs(value1), abs(value2), 1)
            if relative_diff < 0.1:
                return "minor_numeric_difference"
            else:
                return "major_numeric_difference"
        
        return "value_conflict"
    
    def _get_resolution_strategy(self, field_path: str) -> ConflictResolutionStrategy:
        """Get resolution strategy for field"""
        return self.conflict_resolution_rules.get(
            field_path, 
            ConflictResolutionStrategy.MANUAL_REVIEW
        )
    
    async def _resolve_by_latest(self, conflict: DataConflict) -> Any:
        """Resolve conflict by selecting latest value"""
        # Would need timestamp information to implement properly
        # For now, return first value
        return list(conflict.conflicting_values.values())[0]
    
    async def _resolve_by_completeness(self, conflict: DataConflict) -> Any:
        """Resolve conflict by selecting most complete value"""
        values = conflict.conflicting_values.values()
        
        # Select value with most non-null/non-empty fields
        if all(isinstance(v, dict) for v in values):
            return max(values, key=lambda v: len([k for k, val in v.items() if val]))
        
        # For simple values, select non-empty one
        non_empty = [v for v in values if v]
        return non_empty[0] if non_empty else list(values)[0]
    
    async def _resolve_by_confidence(self, conflict: DataConflict) -> Any:
        """Resolve conflict by selecting value from most confident source"""
        max_confidence = max(conflict.confidence_scores.values())
        
        for source_id, confidence in conflict.confidence_scores.items():
            if confidence == max_confidence:
                return conflict.conflicting_values[source_id]
        
        return list(conflict.conflicting_values.values())[0]
    
    async def _resolve_by_priority(self, conflict: DataConflict) -> Any:
        """Resolve conflict by source priority"""
        # Would need source priority information
        # For now, return first value
        return list(conflict.conflicting_values.values())[0]
    
    def _calculate_source_contributions(
        self, sources: List[Dict[str, Any]], merged_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate percentage contribution from each source"""
        contributions = {}
        total_fields = len([k for k in merged_data.keys() if not k.startswith("_")])
        
        for source in sources:
            source_id = source["source_info"].source_id
            source_data = source["normalized_data"]
            
            contributed_fields = 0
            for field, value in source_data.items():
                if field.startswith("_"):
                    continue
                if field in merged_data and merged_data[field] == value:
                    contributed_fields += 1
            
            contributions[source_id] = contributed_fields / total_fields if total_fields > 0 else 0
        
        return contributions
    
    def _calculate_completeness(self, harmonized_data: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        total_fields = len([k for k in harmonized_data.keys() if not k.startswith("_")])
        filled_fields = len([
            k for k, v in harmonized_data.items() 
            if not k.startswith("_") and v is not None and v != ""
        ])
        
        return filled_fields / total_fields if total_fields > 0 else 0
    
    def _calculate_consistency(
        self, harmonized_data: Dict[str, Any], sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate data consistency score"""
        # Simplified consistency calculation
        # In real implementation, would check for logical consistency
        return 0.85  # Placeholder
    
    def _is_date_string(self, value: Any) -> bool:
        """Check if value is a date string"""
        if not isinstance(value, str):
            return False
        
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True
        except (ValueError, TypeError):
            return False
    
    def _normalize_date(self, date_string: str) -> str:
        """Normalize date string to ISO format"""
        try:
            dt = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
            return dt.date().isoformat()
        except (ValueError, TypeError):
            return date_string


# Export the data harmonization agent
__all__ = ["DataHarmonizationAgent", "DataConflict", "HarmonizationResult", "ConflictResolutionStrategy"]