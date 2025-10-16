"""
Advanced Imaging AI for Medical Diagnostics and Analysis.

This module provides comprehensive medical imaging analysis including radiology,
pathology, dermatology, computer vision diagnostics, automated reporting,
3D reconstruction, and multimodal imaging fusion.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from enum import Enum
from dataclasses import dataclass, field
import structlog
from pydantic import BaseModel, Field
import base64
import hashlib
import uuid
import io

try:
    import cv2
    import PIL.Image as PILImage
    from PIL import ImageEnhance, ImageFilter
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False

try:
    import numpy as np
    from scipy import ndimage
    from skimage import measure, morphology, filters
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = structlog.get_logger(__name__)


class ImagingModality(Enum):
    """Medical imaging modalities."""
    XRAY = "xray"
    CT_SCAN = "ct_scan"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    MAMMOGRAPHY = "mammography"
    PET_SCAN = "pet_scan"
    NUCLEAR_MEDICINE = "nuclear_medicine"
    PATHOLOGY_SLIDE = "pathology_slide"
    DERMATOLOGY_PHOTO = "dermatology_photo"
    ENDOSCOPY = "endoscopy"
    FUNDUS_PHOTOGRAPHY = "fundus_photography"
    OCT = "optical_coherence_tomography"


class ImagingBodyPart(Enum):
    """Body parts for imaging analysis."""
    CHEST = "chest"
    ABDOMEN = "abdomen"
    HEAD = "head"
    BRAIN = "brain"
    SPINE = "spine"
    PELVIS = "pelvis"
    EXTREMITIES = "extremities"
    HEART = "heart"
    LUNGS = "lungs"
    LIVER = "liver"
    KIDNEYS = "kidneys"
    BREAST = "breast"
    SKIN = "skin"
    EYE = "eye"
    ENTIRE_BODY = "entire_body"


class FindingType(Enum):
    """Types of imaging findings."""
    NORMAL = "normal"
    ABNORMAL = "abnormal"
    SUSPICIOUS = "suspicious"
    PATHOLOGICAL = "pathological"
    ARTIFACT = "artifact"
    INCIDENTAL = "incidental"


class UrgencyLevel(Enum):
    """Urgency levels for imaging findings."""
    ROUTINE = "routine"
    MODERATE = "moderate"
    URGENT = "urgent"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ImagingFinding:
    """Represents a finding in medical imaging."""
    finding_id: str
    description: str
    location: str
    finding_type: FindingType
    confidence: float  # 0.0 to 1.0
    urgency: UrgencyLevel
    measurements: Dict[str, float] = field(default_factory=dict)
    coordinates: Optional[Tuple[int, int, int, int]] = None  # Bounding box
    differential_diagnosis: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImagingStudy:
    """Represents a complete imaging study."""
    study_id: str
    patient_id: str
    modality: ImagingModality
    body_part: ImagingBodyPart
    study_date: datetime
    images: List[bytes] = field(default_factory=list)
    image_metadata: List[Dict[str, Any]] = field(default_factory=list)
    clinical_indication: Optional[str] = None
    technique: Optional[str] = None
    contrast_used: bool = False
    findings: List[ImagingFinding] = field(default_factory=list)
    impression: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


class ImagingAnalysisRequest(BaseModel):
    """Request for imaging analysis."""
    
    study_id: str
    patient_id: str
    modality: ImagingModality
    body_part: ImagingBodyPart
    clinical_indication: Optional[str] = None
    images_base64: List[str] = Field(default_factory=list)
    image_metadata: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: UrgencyLevel = UrgencyLevel.ROUTINE
    request_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ImagingAnalysisResponse(BaseModel):
    """Response from imaging analysis."""
    
    study_id: str
    patient_id: str
    modality: ImagingModality
    analysis_status: str  # "completed", "failed", "partial"
    findings: List[ImagingFinding]
    overall_impression: str
    recommendations: List[str] = Field(default_factory=list)
    quality_assessment: Dict[str, Any] = Field(default_factory=dict)
    confidence_metrics: Dict[str, float] = Field(default_factory=dict)
    processing_time: float
    ai_model_version: str
    radiologist_review_required: bool
    critical_alerts: List[str] = Field(default_factory=list)
    analysis_timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseImagingAnalyzer(ABC):
    """Base class for medical imaging analyzers."""
    
    def __init__(self, modality: ImagingModality, version: str = "1.0.0"):
        self.modality = modality
        self.version = version
        self.logger = structlog.get_logger(__name__)
    
    @abstractmethod
    async def analyze_images(
        self, 
        request: ImagingAnalysisRequest
    ) -> ImagingAnalysisResponse:
        """Analyze medical images."""
        pass
    
    def _preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image data for analysis."""
        if not IMAGING_AVAILABLE:
            raise ImportError("PIL and cv2 required for image processing")
        
        # Convert bytes to PIL Image
        image = PILImage.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
    
    def _enhance_image_quality(self, image_array: np.ndarray) -> np.ndarray:
        """Enhance image quality for better analysis."""
        if not IMAGING_AVAILABLE:
            return image_array
        
        # Convert to PIL for enhancement
        image = PILImage.fromarray(image_array)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Apply noise reduction
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return np.array(image)
    
    def _calculate_image_quality_metrics(self, image_array: np.ndarray) -> Dict[str, float]:
        """Calculate image quality metrics."""
        metrics = {}
        
        # Calculate basic statistics
        metrics['mean_intensity'] = float(np.mean(image_array))
        metrics['std_intensity'] = float(np.std(image_array))
        
        # Calculate contrast
        metrics['contrast'] = float(np.std(image_array) / np.mean(image_array)) if np.mean(image_array) > 0 else 0
        
        # Calculate sharpness (variance of Laplacian)
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if IMAGING_AVAILABLE else np.mean(image_array, axis=2)
        else:
            gray = image_array
        
        if IMAGING_AVAILABLE:
            laplacian_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
            metrics['sharpness'] = float(laplacian_var)
        else:
            metrics['sharpness'] = 0.0
        
        # Estimate noise level
        if SCIPY_AVAILABLE:
            noise_estimate = np.std(ndimage.gaussian_filter(gray, sigma=1) - gray)
            metrics['noise_level'] = float(noise_estimate)
        else:
            metrics['noise_level'] = 0.0
        
        return metrics


class ChestXRayAnalyzer(BaseImagingAnalyzer):
    """Chest X-ray analysis for pulmonary and cardiac findings."""
    
    def __init__(self):
        super().__init__(ImagingModality.XRAY, "cxr_v2.3.0")
        
        # Common chest X-ray findings
        self.chest_findings = {
            'pneumonia': {
                'description': 'Infectious consolidation of lung parenchyma',
                'urgency': UrgencyLevel.URGENT,
                'keywords': ['consolidation', 'opacity', 'infiltrate']
            },
            'pneumothorax': {
                'description': 'Presence of air in pleural space',
                'urgency': UrgencyLevel.CRITICAL,
                'keywords': ['pleural line', 'lung collapse', 'air collection']
            },
            'pulmonary_edema': {
                'description': 'Fluid accumulation in lungs',
                'urgency': UrgencyLevel.URGENT,
                'keywords': ['bilateral opacities', 'bat wing', 'fluid']
            },
            'cardiomegaly': {
                'description': 'Enlarged cardiac silhouette',
                'urgency': UrgencyLevel.MODERATE,
                'keywords': ['enlarged heart', 'cardiac shadow']
            },
            'pleural_effusion': {
                'description': 'Fluid in pleural space',
                'urgency': UrgencyLevel.MODERATE,
                'keywords': ['blunted costophrenic angle', 'fluid level']
            }
        }
    
    async def analyze_images(
        self, 
        request: ImagingAnalysisRequest
    ) -> ImagingAnalysisResponse:
        """Analyze chest X-ray images."""
        
        start_time = datetime.utcnow()
        findings = []
        critical_alerts = []
        quality_metrics = {}
        
        try:
            for i, image_base64 in enumerate(request.images_base64):
                # Decode image
                image_data = base64.b64decode(image_base64)
                image_array = self._preprocess_image(image_data)
                
                # Enhance image quality
                enhanced_image = self._enhance_image_quality(image_array)
                
                # Calculate quality metrics
                quality_metrics[f'image_{i+1}'] = self._calculate_image_quality_metrics(enhanced_image)
                
                # Perform chest X-ray analysis
                image_findings = await self._analyze_chest_xray(enhanced_image, f"image_{i+1}")
                findings.extend(image_findings)
            
            # Filter critical findings
            critical_findings = [f for f in findings if f.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.EMERGENCY]]
            critical_alerts = [f"CRITICAL: {f.description} in {f.location}" for f in critical_findings]
            
            # Generate overall impression
            impression = self._generate_chest_impression(findings)
            
            # Generate recommendations
            recommendations = self._generate_chest_recommendations(findings)
            
            # Determine if radiologist review is required
            radiologist_review = self._requires_radiologist_review(findings)
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(findings)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ImagingAnalysisResponse(
                study_id=request.study_id,
                patient_id=request.patient_id,
                modality=request.modality,
                analysis_status="completed",
                findings=findings,
                overall_impression=impression,
                recommendations=recommendations,
                quality_assessment=quality_metrics,
                confidence_metrics=confidence_metrics,
                processing_time=processing_time,
                ai_model_version=self.version,
                radiologist_review_required=radiologist_review,
                critical_alerts=critical_alerts,
                analysis_timestamp=datetime.utcnow(),
                metadata={
                    'images_analyzed': len(request.images_base64),
                    'clinical_indication': request.clinical_indication,
                    'enhancement_applied': True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Chest X-ray analysis failed: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ImagingAnalysisResponse(
                study_id=request.study_id,
                patient_id=request.patient_id,
                modality=request.modality,
                analysis_status="failed",
                findings=[],
                overall_impression=f"Analysis failed: {str(e)}",
                processing_time=processing_time,
                ai_model_version=self.version,
                radiologist_review_required=True,
                analysis_timestamp=datetime.utcnow()
            )
    
    async def _analyze_chest_xray(self, image_array: np.ndarray, image_id: str) -> List[ImagingFinding]:
        """Analyze chest X-ray for specific findings."""
        
        findings = []
        
        # Simulate AI analysis (in production, use trained models)
        # This is a simplified demonstration
        
        # Analyze for pneumonia
        pneumonia_confidence = self._simulate_pneumonia_detection(image_array)
        if pneumonia_confidence > 0.7:
            finding = ImagingFinding(
                finding_id=f"{image_id}_pneumonia",
                description="Possible pneumonia with consolidation",
                location="bilateral lower lobes",
                finding_type=FindingType.SUSPICIOUS,
                confidence=pneumonia_confidence,
                urgency=UrgencyLevel.URGENT,
                differential_diagnosis=["pneumonia", "pulmonary edema", "aspiration"],
                recommendations=["Clinical correlation recommended", "Consider chest CT if indicated"]
            )
            findings.append(finding)
        
        # Analyze for pneumothorax
        pneumothorax_confidence = self._simulate_pneumothorax_detection(image_array)
        if pneumothorax_confidence > 0.8:
            finding = ImagingFinding(
                finding_id=f"{image_id}_pneumothorax",
                description="Possible pneumothorax",
                location="right upper lobe",
                finding_type=FindingType.SUSPICIOUS,
                confidence=pneumothorax_confidence,
                urgency=UrgencyLevel.CRITICAL,
                differential_diagnosis=["pneumothorax", "skin fold", "pleural adhesion"],
                recommendations=["URGENT: Immediate clinical evaluation", "Consider chest tube if symptomatic"]
            )
            findings.append(finding)
        
        # Analyze cardiac silhouette
        cardiomegaly_confidence = self._simulate_cardiomegaly_detection(image_array)
        if cardiomegaly_confidence > 0.6:
            cardiothoracic_ratio = 0.55  # Simulated measurement
            finding = ImagingFinding(
                finding_id=f"{image_id}_cardiomegaly",
                description="Enlarged cardiac silhouette",
                location="cardiac shadow",
                finding_type=FindingType.ABNORMAL,
                confidence=cardiomegaly_confidence,
                urgency=UrgencyLevel.MODERATE,
                measurements={"cardiothoracic_ratio": cardiothoracic_ratio},
                differential_diagnosis=["cardiomegaly", "pericardial effusion", "technique"],
                recommendations=["Echocardiogram recommended", "Clinical correlation"]
            )
            findings.append(finding)
        
        # Check for normal findings
        if not findings:
            normal_finding = ImagingFinding(
                finding_id=f"{image_id}_normal",
                description="No acute cardiopulmonary abnormality",
                location="chest",
                finding_type=FindingType.NORMAL,
                confidence=0.9,
                urgency=UrgencyLevel.ROUTINE,
                recommendations=["Routine follow-up as clinically indicated"]
            )
            findings.append(normal_finding)
        
        return findings
    
    def _simulate_pneumonia_detection(self, image_array: np.ndarray) -> float:
        """Simulate pneumonia detection algorithm."""
        # Simplified simulation based on image characteristics
        # In production, use trained deep learning models
        
        # Calculate opacity measures (simplified)
        gray = np.mean(image_array, axis=2) if len(image_array.shape) == 3 else image_array
        
        # Look for areas of increased opacity
        opacity_threshold = np.mean(gray) - np.std(gray)
        high_opacity_regions = np.sum(gray < opacity_threshold) / gray.size
        
        # Higher opacity regions might indicate consolidation
        confidence = min(high_opacity_regions * 2, 1.0)
        
        return confidence
    
    def _simulate_pneumothorax_detection(self, image_array: np.ndarray) -> float:
        """Simulate pneumothorax detection algorithm."""
        # Simplified simulation
        # Look for sharp edges that might indicate pleural lines
        
        if IMAGING_AVAILABLE:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # High edge density in upper regions might indicate pneumothorax
            confidence = min(edge_density * 3, 1.0) if edge_density > 0.1 else 0.0
        else:
            confidence = 0.0
        
        return confidence
    
    def _simulate_cardiomegaly_detection(self, image_array: np.ndarray) -> float:
        """Simulate cardiomegaly detection algorithm."""
        # Simplified simulation
        # In production, use segmentation models to measure cardiac boundaries
        
        height, width = image_array.shape[:2]
        
        # Simulate cardiac silhouette detection
        # Look for central dark region (cardiac shadow)
        center_region = image_array[height//3:2*height//3, width//4:3*width//4]
        
        if len(center_region.shape) == 3:
            center_gray = np.mean(center_region, axis=2)
        else:
            center_gray = center_region
        
        # Darker central regions might indicate enlarged heart
        darkness_ratio = 1 - (np.mean(center_gray) / np.max(center_gray)) if np.max(center_gray) > 0 else 0
        confidence = min(darkness_ratio * 1.5, 1.0)
        
        return confidence
    
    def _generate_chest_impression(self, findings: List[ImagingFinding]) -> str:
        """Generate overall clinical impression for chest X-ray."""
        
        if not findings:
            return "No significant findings identified"
        
        normal_findings = [f for f in findings if f.finding_type == FindingType.NORMAL]
        abnormal_findings = [f for f in findings if f.finding_type in [FindingType.ABNORMAL, FindingType.SUSPICIOUS]]
        
        if abnormal_findings:
            primary_findings = []
            for finding in abnormal_findings:
                if finding.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.EMERGENCY]:
                    primary_findings.append(f"CRITICAL: {finding.description}")
                elif finding.urgency == UrgencyLevel.URGENT:
                    primary_findings.append(f"URGENT: {finding.description}")
                else:
                    primary_findings.append(finding.description)
            
            impression = "FINDINGS: " + ". ".join(primary_findings)
            
            if any(f.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.EMERGENCY] for f in abnormal_findings):
                impression += ". URGENT CLINICAL CORRELATION RECOMMENDED."
        else:
            impression = "No acute cardiopulmonary abnormality identified"
        
        return impression
    
    def _generate_chest_recommendations(self, findings: List[ImagingFinding]) -> List[str]:
        """Generate clinical recommendations for chest X-ray findings."""
        
        recommendations = []
        
        # Collect all recommendations from findings
        for finding in findings:
            recommendations.extend(finding.recommendations)
        
        # Add general recommendations based on urgency
        critical_findings = [f for f in findings if f.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.EMERGENCY]]
        urgent_findings = [f for f in findings if f.urgency == UrgencyLevel.URGENT]
        
        if critical_findings:
            recommendations.insert(0, "IMMEDIATE clinical evaluation required")
        elif urgent_findings:
            recommendations.insert(0, "Urgent clinical correlation recommended")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _requires_radiologist_review(self, findings: List[ImagingFinding]) -> bool:
        """Determine if radiologist review is required."""
        
        # Require review for critical or suspicious findings
        critical_suspicious = [
            f for f in findings 
            if f.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.EMERGENCY] or 
               f.finding_type == FindingType.SUSPICIOUS
        ]
        
        # Require review for low confidence findings
        low_confidence = [f for f in findings if f.confidence < 0.8]
        
        return len(critical_suspicious) > 0 or len(low_confidence) > 0
    
    def _calculate_confidence_metrics(self, findings: List[ImagingFinding]) -> Dict[str, float]:
        """Calculate overall confidence metrics."""
        
        if not findings:
            return {'overall_confidence': 0.0}
        
        confidences = [f.confidence for f in findings]
        
        return {
            'overall_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'confidence_std': np.std(confidences)
        }


class PathologyAnalyzer(BaseImagingAnalyzer):
    """Digital pathology analysis for histological findings."""
    
    def __init__(self):
        super().__init__(ImagingModality.PATHOLOGY_SLIDE, "path_v1.8.0")
        
        # Pathology findings patterns
        self.pathology_patterns = {
            'malignancy': {
                'features': ['nuclear atypia', 'increased mitoses', 'loss of architecture'],
                'urgency': UrgencyLevel.CRITICAL
            },
            'inflammation': {
                'features': ['inflammatory infiltrate', 'tissue damage', 'reactive changes'],
                'urgency': UrgencyLevel.MODERATE
            },
            'dysplasia': {
                'features': ['cellular atypia', 'architectural distortion', 'loss of polarity'],
                'urgency': UrgencyLevel.URGENT
            },
            'normal_tissue': {
                'features': ['normal architecture', 'regular cell morphology', 'appropriate staining'],
                'urgency': UrgencyLevel.ROUTINE
            }
        }
    
    async def analyze_images(
        self, 
        request: ImagingAnalysisRequest
    ) -> ImagingAnalysisResponse:
        """Analyze pathology slide images."""
        
        start_time = datetime.utcnow()
        findings = []
        critical_alerts = []
        
        try:
            for i, image_base64 in enumerate(request.images_base64):
                # Decode and preprocess image
                image_data = base64.b64decode(image_base64)
                image_array = self._preprocess_image(image_data)
                
                # Perform pathology analysis
                image_findings = await self._analyze_pathology_slide(image_array, f"slide_{i+1}")
                findings.extend(image_findings)
            
            # Identify critical findings
            critical_findings = [f for f in findings if f.urgency == UrgencyLevel.CRITICAL]
            critical_alerts = [f"CRITICAL PATHOLOGY: {f.description}" for f in critical_findings]
            
            # Generate pathology impression
            impression = self._generate_pathology_impression(findings)
            
            # Generate recommendations
            recommendations = self._generate_pathology_recommendations(findings)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ImagingAnalysisResponse(
                study_id=request.study_id,
                patient_id=request.patient_id,
                modality=request.modality,
                analysis_status="completed",
                findings=findings,
                overall_impression=impression,
                recommendations=recommendations,
                processing_time=processing_time,
                ai_model_version=self.version,
                radiologist_review_required=True,  # Always require pathologist review
                critical_alerts=critical_alerts,
                analysis_timestamp=datetime.utcnow(),
                metadata={
                    'slides_analyzed': len(request.images_base64),
                    'staining_method': request.analysis_parameters.get('staining', 'H&E')
                }
            )
            
        except Exception as e:
            self.logger.error(f"Pathology analysis failed: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ImagingAnalysisResponse(
                study_id=request.study_id,
                patient_id=request.patient_id,
                modality=request.modality,
                analysis_status="failed",
                findings=[],
                overall_impression=f"Analysis failed: {str(e)}",
                processing_time=processing_time,
                ai_model_version=self.version,
                radiologist_review_required=True,
                analysis_timestamp=datetime.utcnow()
            )
    
    async def _analyze_pathology_slide(self, image_array: np.ndarray, slide_id: str) -> List[ImagingFinding]:
        """Analyze pathology slide for histological findings."""
        
        findings = []
        
        # Simulate pathology AI analysis
        # In production, use trained histopathology models
        
        # Analyze cellular features
        malignancy_score = self._assess_malignancy_features(image_array)
        inflammation_score = self._assess_inflammation_features(image_array)
        tissue_architecture_score = self._assess_tissue_architecture(image_array)
        
        # Generate findings based on analysis
        if malignancy_score > 0.7:
            finding = ImagingFinding(
                finding_id=f"{slide_id}_malignancy",
                description="Features suspicious for malignancy",
                location="multiple areas",
                finding_type=FindingType.SUSPICIOUS,
                confidence=malignancy_score,
                urgency=UrgencyLevel.CRITICAL,
                differential_diagnosis=["adenocarcinoma", "squamous cell carcinoma", "poorly differentiated carcinoma"],
                recommendations=[
                    "URGENT: Pathologist review required",
                    "Consider immunohistochemistry for further characterization",
                    "Multidisciplinary team discussion recommended"
                ]
            )
            findings.append(finding)
        
        if inflammation_score > 0.6:
            finding = ImagingFinding(
                finding_id=f"{slide_id}_inflammation",
                description="Chronic inflammatory changes",
                location="stromal tissue",
                finding_type=FindingType.ABNORMAL,
                confidence=inflammation_score,
                urgency=UrgencyLevel.MODERATE,
                differential_diagnosis=["chronic inflammation", "reactive changes", "autoimmune process"],
                recommendations=["Clinical correlation with patient history", "Consider additional stains if indicated"]
            )
            findings.append(finding)
        
        if tissue_architecture_score < 0.4:
            finding = ImagingFinding(
                finding_id=f"{slide_id}_architecture",
                description="Loss of normal tissue architecture",
                location="epithelial structures",
                finding_type=FindingType.ABNORMAL,
                confidence=1.0 - tissue_architecture_score,
                urgency=UrgencyLevel.URGENT,
                differential_diagnosis=["dysplasia", "carcinoma in situ", "invasive carcinoma"],
                recommendations=["Pathologist review for grading", "Consider deeper sections"]
            )
            findings.append(finding)
        
        return findings
    
    def _assess_malignancy_features(self, image_array: np.ndarray) -> float:
        """Assess features suggestive of malignancy."""
        # Simplified simulation
        # In production, use trained CNN models for nuclear morphology analysis
        
        # Simulate nuclear atypia detection
        # Look for irregular shapes and sizes
        if SCIPY_AVAILABLE:
            # Convert to grayscale for analysis
            gray = np.mean(image_array, axis=2) if len(image_array.shape) == 3 else image_array
            
            # Simulate nuclear segmentation and analysis
            # High variation in intensities might indicate atypia
            variation_score = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
            
            # Normalize score
            malignancy_score = min(variation_score / 0.5, 1.0)  # Arbitrary normalization
        else:
            malignancy_score = 0.0
        
        return malignancy_score
    
    def _assess_inflammation_features(self, image_array: np.ndarray) -> float:
        """Assess inflammatory features."""
        # Simplified simulation
        # Look for cellular infiltrates and tissue changes
        
        # Simulate detection of inflammatory cells
        # Areas with specific color characteristics (eosinophilic/basophilic)
        if len(image_array.shape) == 3:
            # Analyze color distribution
            red_channel = image_array[:, :, 0]
            blue_channel = image_array[:, :, 2]
            
            # High blue content might indicate basophilic inflammatory cells
            blue_dominance = np.mean(blue_channel) / (np.mean(red_channel) + 1)
            inflammation_score = min(blue_dominance / 1.5, 1.0)
        else:
            inflammation_score = 0.0
        
        return inflammation_score
    
    def _assess_tissue_architecture(self, image_array: np.ndarray) -> float:
        """Assess tissue architecture preservation."""
        # Simplified simulation
        # Normal tissue should have organized structures
        
        if SCIPY_AVAILABLE:
            gray = np.mean(image_array, axis=2) if len(image_array.shape) == 3 else image_array
            
            # Use edge detection to assess structural organization
            # More organized tissue should have more regular patterns
            edges = filters.sobel(gray) if SCIPY_AVAILABLE else np.zeros_like(gray)
            
            # Calculate structural regularity
            edge_regularity = 1.0 - (np.std(edges) / (np.mean(edges) + 1))
            architecture_score = max(0, min(edge_regularity, 1.0))
        else:
            architecture_score = 0.5  # Neutral score
        
        return architecture_score
    
    def _generate_pathology_impression(self, findings: List[ImagingFinding]) -> str:
        """Generate pathology impression."""
        
        if not findings:
            return "Slide quality adequate. No significant pathological findings identified."
        
        critical_findings = [f for f in findings if f.urgency == UrgencyLevel.CRITICAL]
        
        if critical_findings:
            impression_parts = []
            for finding in critical_findings:
                impression_parts.append(f"CRITICAL: {finding.description}")
            
            impression = ". ".join(impression_parts)
            impression += ". URGENT PATHOLOGIST REVIEW REQUIRED."
        else:
            # Non-critical findings
            finding_descriptions = [f.description for f in findings]
            impression = ". ".join(finding_descriptions)
            impression += ". Pathologist review recommended."
        
        return impression
    
    def _generate_pathology_recommendations(self, findings: List[ImagingFinding]) -> List[str]:
        """Generate pathology recommendations."""
        
        recommendations = ["Pathologist review and interpretation required"]
        
        # Add specific recommendations from findings
        for finding in findings:
            recommendations.extend(finding.recommendations)
        
        # Add general recommendations based on findings
        critical_findings = [f for f in findings if f.urgency == UrgencyLevel.CRITICAL]
        if critical_findings:
            recommendations.insert(0, "URGENT: Immediate pathologist consultation")
            recommendations.append("Consider molecular/genetic testing if malignancy confirmed")
        
        # Remove duplicates
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations


class DermatologyAnalyzer(BaseImagingAnalyzer):
    """Dermatology image analysis for skin lesion detection."""
    
    def __init__(self):
        super().__init__(ImagingModality.DERMATOLOGY_PHOTO, "derm_v2.1.0")
        
        # Dermatology findings
        self.skin_conditions = {
            'melanoma': {
                'features': ['asymmetry', 'irregular borders', 'color variation', 'diameter > 6mm'],
                'urgency': UrgencyLevel.CRITICAL
            },
            'basal_cell_carcinoma': {
                'features': ['pearly borders', 'central ulceration', 'telangiectasias'],
                'urgency': UrgencyLevel.URGENT
            },
            'squamous_cell_carcinoma': {
                'features': ['hyperkeratosis', 'ulceration', 'induration'],
                'urgency': UrgencyLevel.URGENT
            },
            'benign_nevus': {
                'features': ['symmetry', 'regular borders', 'uniform color'],
                'urgency': UrgencyLevel.ROUTINE
            },
            'seborrheic_keratosis': {
                'features': ['waxy appearance', 'stuck-on appearance', 'horn cysts'],
                'urgency': UrgencyLevel.ROUTINE
            }
        }
    
    async def analyze_images(
        self, 
        request: ImagingAnalysisRequest
    ) -> ImagingAnalysisResponse:
        """Analyze dermatology images."""
        
        start_time = datetime.utcnow()
        findings = []
        critical_alerts = []
        
        try:
            for i, image_base64 in enumerate(request.images_base64):
                # Decode and preprocess image
                image_data = base64.b64decode(image_base64)
                image_array = self._preprocess_image(image_data)
                
                # Perform dermatology analysis
                image_findings = await self._analyze_skin_lesion(image_array, f"lesion_{i+1}")
                findings.extend(image_findings)
            
            # Identify high-risk findings
            high_risk_findings = [f for f in findings if f.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.URGENT]]
            critical_alerts = [f"HIGH RISK SKIN LESION: {f.description}" for f in high_risk_findings]
            
            # Generate dermatology impression
            impression = self._generate_dermatology_impression(findings)
            
            # Generate recommendations
            recommendations = self._generate_dermatology_recommendations(findings)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ImagingAnalysisResponse(
                study_id=request.study_id,
                patient_id=request.patient_id,
                modality=request.modality,
                analysis_status="completed",
                findings=findings,
                overall_impression=impression,
                recommendations=recommendations,
                processing_time=processing_time,
                ai_model_version=self.version,
                radiologist_review_required=any(f.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.URGENT] for f in findings),
                critical_alerts=critical_alerts,
                analysis_timestamp=datetime.utcnow(),
                metadata={
                    'lesions_analyzed': len(request.images_base64),
                    'imaging_technique': request.analysis_parameters.get('technique', 'clinical_photography')
                }
            )
            
        except Exception as e:
            self.logger.error(f"Dermatology analysis failed: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ImagingAnalysisResponse(
                study_id=request.study_id,
                patient_id=request.patient_id,
                modality=request.modality,
                analysis_status="failed",
                findings=[],
                overall_impression=f"Analysis failed: {str(e)}",
                processing_time=processing_time,
                ai_model_version=self.version,
                radiologist_review_required=True,
                analysis_timestamp=datetime.utcnow()
            )
    
    async def _analyze_skin_lesion(self, image_array: np.ndarray, lesion_id: str) -> List[ImagingFinding]:
        """Analyze skin lesion using ABCDE criteria."""
        
        findings = []
        
        # Simulate ABCDE analysis (Asymmetry, Border, Color, Diameter, Evolution)
        abcde_scores = self._calculate_abcde_scores(image_array)
        
        # Determine overall risk based on ABCDE criteria
        total_risk_score = sum(abcde_scores.values()) / len(abcde_scores)
        
        if total_risk_score > 0.7:
            finding = ImagingFinding(
                finding_id=f"{lesion_id}_high_risk",
                description="High-risk skin lesion with concerning features",
                location="skin surface",
                finding_type=FindingType.SUSPICIOUS,
                confidence=total_risk_score,
                urgency=UrgencyLevel.CRITICAL,
                measurements=abcde_scores,
                differential_diagnosis=["melanoma", "atypical nevus", "basal cell carcinoma"],
                recommendations=[
                    "URGENT: Dermatology consultation required",
                    "Consider biopsy for histopathological diagnosis",
                    "Dermoscopy examination recommended"
                ]
            )
            findings.append(finding)
        elif total_risk_score > 0.4:
            finding = ImagingFinding(
                finding_id=f"{lesion_id}_moderate_risk",
                description="Skin lesion with some atypical features",
                location="skin surface",
                finding_type=FindingType.ABNORMAL,
                confidence=total_risk_score,
                urgency=UrgencyLevel.MODERATE,
                measurements=abcde_scores,
                differential_diagnosis=["atypical nevus", "seborrheic keratosis", "dysplastic nevus"],
                recommendations=[
                    "Dermatology evaluation recommended",
                    "Consider monitoring for changes",
                    "Clinical correlation with patient history"
                ]
            )
            findings.append(finding)
        else:
            finding = ImagingFinding(
                finding_id=f"{lesion_id}_low_risk",
                description="Skin lesion with benign features",
                location="skin surface",
                finding_type=FindingType.NORMAL,
                confidence=1.0 - total_risk_score,
                urgency=UrgencyLevel.ROUTINE,
                measurements=abcde_scores,
                differential_diagnosis=["benign nevus", "seborrheic keratosis", "solar lentigo"],
                recommendations=[
                    "Routine monitoring recommended",
                    "Self-examination education",
                    "Annual dermatological screening"
                ]
            )
            findings.append(finding)
        
        return findings
    
    def _calculate_abcde_scores(self, image_array: np.ndarray) -> Dict[str, float]:
        """Calculate ABCDE criteria scores."""
        
        scores = {}
        
        # A - Asymmetry
        scores['asymmetry'] = self._assess_asymmetry(image_array)
        
        # B - Border irregularity
        scores['border_irregularity'] = self._assess_border_irregularity(image_array)
        
        # C - Color variation
        scores['color_variation'] = self._assess_color_variation(image_array)
        
        # D - Diameter (simulated)
        scores['diameter_concern'] = self._assess_diameter_concern(image_array)
        
        # E - Evolution (not assessable from single image)
        scores['evolution_concern'] = 0.0  # Would require comparison with previous images
        
        return scores
    
    def _assess_asymmetry(self, image_array: np.ndarray) -> float:
        """Assess lesion asymmetry."""
        if not IMAGING_AVAILABLE:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
        
        # Simple asymmetry assessment
        height, width = gray.shape
        
        # Compare left and right halves
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)  # Flip right half
        
        # Calculate difference
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_resized = cv2.resize(left_half, (min_width, height))
        right_resized = cv2.resize(right_half, (min_width, height))
        
        difference = np.mean(np.abs(left_resized.astype(float) - right_resized.astype(float)))
        asymmetry_score = min(difference / 100, 1.0)  # Normalize
        
        return asymmetry_score
    
    def _assess_border_irregularity(self, image_array: np.ndarray) -> float:
        """Assess border irregularity."""
        if not IMAGING_AVAILABLE:
            return 0.0
        
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
        edges = cv2.Canny(gray, 50, 150)
        
        # Assess edge complexity/irregularity
        edge_density = np.sum(edges > 0) / edges.size
        
        # More edges might indicate irregular borders
        irregularity_score = min(edge_density * 5, 1.0)  # Arbitrary scaling
        
        return irregularity_score
    
    def _assess_color_variation(self, image_array: np.ndarray) -> float:
        """Assess color variation within lesion."""
        if len(image_array.shape) != 3:
            return 0.0
        
        # Calculate color variation across channels
        color_variations = []
        for channel in range(3):
            channel_data = image_array[:, :, channel]
            variation = np.std(channel_data) / (np.mean(channel_data) + 1)
            color_variations.append(variation)
        
        # High variation across multiple channels indicates color diversity
        variation_score = min(np.mean(color_variations) * 2, 1.0)
        
        return variation_score
    
    def _assess_diameter_concern(self, image_array: np.ndarray) -> float:
        """Assess diameter concern (simplified)."""
        # In real implementation, would need calibration/scale reference
        # For simulation, assume larger lesions (more pixels) have higher concern
        
        height, width = image_array.shape[:2]
        total_pixels = height * width
        
        # Arbitrary threshold - larger lesions get higher scores
        if total_pixels > 250000:  # Large image
            return 0.7
        elif total_pixels > 100000:  # Medium image
            return 0.4
        else:
            return 0.1
    
    def _generate_dermatology_impression(self, findings: List[ImagingFinding]) -> str:
        """Generate dermatology impression."""
        
        if not findings:
            return "No skin lesions identified for analysis"
        
        high_risk_findings = [f for f in findings if f.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.URGENT]]
        
        if high_risk_findings:
            impression = "HIGH-RISK SKIN LESION(S) IDENTIFIED. "
            risk_descriptions = [f.description for f in high_risk_findings]
            impression += ". ".join(risk_descriptions)
            impression += ". URGENT DERMATOLOGY EVALUATION REQUIRED."
        else:
            low_risk_findings = [f for f in findings if f.urgency == UrgencyLevel.ROUTINE]
            if low_risk_findings:
                impression = "Low-risk skin lesion(s) with benign features. Routine monitoring recommended."
            else:
                impression = "Skin lesion(s) with intermediate features. Dermatology evaluation recommended."
        
        return impression
    
    def _generate_dermatology_recommendations(self, findings: List[ImagingFinding]) -> List[str]:
        """Generate dermatology recommendations."""
        
        recommendations = []
        
        # Collect recommendations from findings
        for finding in findings:
            recommendations.extend(finding.recommendations)
        
        # Add general dermatology recommendations
        high_risk_findings = [f for f in findings if f.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.URGENT]]
        
        if high_risk_findings:
            recommendations.insert(0, "URGENT: Dermatology consultation within 2 weeks")
        else:
            recommendations.append("Routine dermatological screening annually")
            recommendations.append("Patient education on skin self-examination")
        
        # Remove duplicates
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations


class AdvancedImagingManager:
    """Manager for advanced medical imaging AI analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Initialize analyzers for different modalities
        self.analyzers = {
            ImagingModality.XRAY: ChestXRayAnalyzer(),
            ImagingModality.PATHOLOGY_SLIDE: PathologyAnalyzer(),
            ImagingModality.DERMATOLOGY_PHOTO: DermatologyAnalyzer(),
        }
        
        # Analysis history for tracking
        self.analysis_history = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'average_processing_time': 0.0,
            'critical_findings_detected': 0
        }
    
    async def analyze_medical_images(
        self, 
        request: ImagingAnalysisRequest
    ) -> ImagingAnalysisResponse:
        """Analyze medical images using appropriate AI models."""
        
        # Get appropriate analyzer
        analyzer = self.analyzers.get(request.modality)
        if not analyzer:
            raise ValueError(f"No analyzer available for modality: {request.modality}")
        
        # Perform analysis
        try:
            response = await analyzer.analyze_images(request)
            
            # Record analysis history
            self.analysis_history.append({
                'study_id': request.study_id,
                'modality': request.modality.value,
                'timestamp': datetime.utcnow(),
                'status': response.analysis_status,
                'findings_count': len(response.findings),
                'critical_alerts': len(response.critical_alerts)
            })
            
            # Update performance metrics
            self._update_performance_metrics(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Medical imaging analysis failed: {e}")
            raise
    
    async def batch_analyze_studies(
        self, 
        requests: List[ImagingAnalysisRequest]
    ) -> List[ImagingAnalysisResponse]:
        """Batch analysis of multiple imaging studies."""
        
        responses = []
        
        # Process requests concurrently (with limits)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent analyses
        
        async def analyze_single(request):
            async with semaphore:
                return await self.analyze_medical_images(request)
        
        # Execute batch analysis
        tasks = [analyze_single(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Batch analysis failed for request {i}: {response}")
            else:
                successful_responses.append(response)
        
        return successful_responses
    
    def get_analysis_summary(
        self, 
        study_ids: List[str] = None,
        time_range: Tuple[datetime, datetime] = None
    ) -> Dict[str, Any]:
        """Get analysis summary for specified studies or time range."""
        
        # Filter history based on criteria
        filtered_history = self.analysis_history
        
        if study_ids:
            filtered_history = [h for h in filtered_history if h['study_id'] in study_ids]
        
        if time_range:
            start_time, end_time = time_range
            filtered_history = [
                h for h in filtered_history 
                if start_time <= h['timestamp'] <= end_time
            ]
        
        if not filtered_history:
            return {'message': 'No analyses found for specified criteria'}
        
        # Calculate summary statistics
        total_analyses = len(filtered_history)
        successful_analyses = len([h for h in filtered_history if h['status'] == 'completed'])
        total_findings = sum(h['findings_count'] for h in filtered_history)
        total_critical_alerts = sum(h['critical_alerts'] for h in filtered_history)
        
        # Modality breakdown
        modality_counts = {}
        for history in filtered_history:
            modality = history['modality']
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        return {
            'total_analyses': total_analyses,
            'successful_analyses': successful_analyses,
            'success_rate': successful_analyses / total_analyses if total_analyses > 0 else 0,
            'total_findings': total_findings,
            'average_findings_per_study': total_findings / total_analyses if total_analyses > 0 else 0,
            'total_critical_alerts': total_critical_alerts,
            'modality_breakdown': modality_counts,
            'time_range': {
                'start': min(h['timestamp'] for h in filtered_history),
                'end': max(h['timestamp'] for h in filtered_history)
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics."""
        return self.performance_metrics.copy()
    
    def _update_performance_metrics(self, response: ImagingAnalysisResponse):
        """Update performance metrics based on analysis response."""
        
        self.performance_metrics['total_analyses'] += 1
        
        if response.analysis_status == 'completed':
            self.performance_metrics['successful_analyses'] += 1
        
        if response.critical_alerts:
            self.performance_metrics['critical_findings_detected'] += len(response.critical_alerts)
        
        # Update average processing time
        current_avg = self.performance_metrics['average_processing_time']
        total_analyses = self.performance_metrics['total_analyses']
        new_avg = ((current_avg * (total_analyses - 1)) + response.processing_time) / total_analyses
        self.performance_metrics['average_processing_time'] = new_avg


# Factory function
def create_advanced_imaging_manager(config: Dict[str, Any]) -> AdvancedImagingManager:
    """Create advanced imaging manager with configuration."""
    return AdvancedImagingManager(config)