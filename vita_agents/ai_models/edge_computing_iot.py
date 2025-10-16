"""
Edge Computing & IoT Integration for Real-time Healthcare Monitoring.

This module provides comprehensive edge computing capabilities for healthcare IoT devices
including real-time wearable data processing, edge AI for monitoring devices, IoT sensor
integration, offline-capable AI models, and edge-to-cloud synchronization.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
import structlog
from pydantic import BaseModel, Field
import threading
import queue
import time
from collections import deque, defaultdict
import hashlib
import gzip
import pickle

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    import bluetooth
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

try:
    import tensorflow_lite as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

logger = structlog.get_logger(__name__)


class DeviceType(Enum):
    """Types of IoT healthcare devices."""
    WEARABLE_FITNESS = "wearable_fitness"
    SMARTWATCH = "smartwatch"
    GLUCOSE_MONITOR = "glucose_monitor"
    BLOOD_PRESSURE_MONITOR = "blood_pressure_monitor"
    PULSE_OXIMETER = "pulse_oximeter"
    ECG_MONITOR = "ecg_monitor"
    TEMPERATURE_SENSOR = "temperature_sensor"
    WEIGHT_SCALE = "weight_scale"
    MEDICATION_DISPENSER = "medication_dispenser"
    SLEEP_TRACKER = "sleep_tracker"
    ACTIVITY_TRACKER = "activity_tracker"
    HEART_RATE_MONITOR = "heart_rate_monitor"
    FALL_DETECTOR = "fall_detector"
    ENVIRONMENTAL_SENSOR = "environmental_sensor"


class DataType(Enum):
    """Types of sensor data."""
    HEART_RATE = "heart_rate"
    BLOOD_PRESSURE = "blood_pressure"
    BLOOD_GLUCOSE = "blood_glucose"
    OXYGEN_SATURATION = "oxygen_saturation"
    BODY_TEMPERATURE = "body_temperature"
    WEIGHT = "weight"
    STEPS = "steps"
    CALORIES = "calories"
    SLEEP_STAGES = "sleep_stages"
    ECG_WAVEFORM = "ecg_waveform"
    ACCELERATION = "acceleration"
    GYROSCOPE = "gyroscope"
    GPS_LOCATION = "gps_location"
    AMBIENT_TEMPERATURE = "ambient_temperature"
    HUMIDITY = "humidity"
    AIR_QUALITY = "air_quality"


class ProcessingMode(Enum):
    """Edge processing modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    OFFLINE = "offline"
    HYBRID = "hybrid"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SensorReading:
    """Represents a sensor reading from IoT device."""
    device_id: str
    device_type: DeviceType
    data_type: DataType
    value: Union[float, int, str, List, Dict]
    unit: str
    timestamp: datetime
    quality_score: float = 1.0  # Data quality indicator (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    raw_data: Optional[bytes] = None


@dataclass
class EdgeDevice:
    """Represents an edge computing device."""
    device_id: str
    device_name: str
    device_type: DeviceType
    capabilities: List[str]
    battery_level: Optional[float] = None
    connection_status: str = "unknown"
    last_seen: Optional[datetime] = None
    firmware_version: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    processing_capacity: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeAlert:
    """Alert generated at edge device."""
    alert_id: str
    device_id: str
    patient_id: str
    alert_level: AlertLevel
    message: str
    data_value: Any
    threshold: Optional[float] = None
    timestamp: datetime
    requires_immediate_action: bool = False
    clinical_context: Dict[str, Any] = field(default_factory=dict)
    processed_at_edge: bool = True


class EdgeDataRequest(BaseModel):
    """Request for edge data processing."""
    
    device_id: str
    sensor_readings: List[SensorReading]
    processing_mode: ProcessingMode
    patient_id: str
    clinical_context: Dict[str, Any] = Field(default_factory=dict)
    processing_parameters: Dict[str, Any] = Field(default_factory=dict)
    request_timestamp: datetime = Field(default_factory=datetime.utcnow)


class EdgeProcessingResponse(BaseModel):
    """Response from edge processing."""
    
    device_id: str
    patient_id: str
    processed_readings: List[SensorReading]
    alerts: List[EdgeAlert] = Field(default_factory=list)
    insights: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    sync_required: bool = False
    processing_latency_ms: float = 0.0
    battery_impact: str = "low"
    next_sync_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseEdgeProcessor(ABC):
    """Base class for edge processors."""
    
    def __init__(self, processor_name: str, version: str = "1.0.0"):
        self.processor_name = processor_name
        self.version = version
        self.logger = structlog.get_logger(__name__)
        
        # Processing metrics
        self.processing_metrics = {
            'total_readings_processed': 0,
            'alerts_generated': 0,
            'average_processing_time_ms': 0.0,
            'error_rate': 0.0
        }
    
    @abstractmethod
    async def process_data(
        self, 
        request: EdgeDataRequest
    ) -> EdgeProcessingResponse:
        """Process sensor data at edge."""
        pass
    
    def _calculate_data_quality(self, reading: SensorReading) -> float:
        """Calculate data quality score for sensor reading."""
        
        quality_factors = []
        
        # Timestamp freshness (within last 5 minutes = high quality)
        time_diff = (datetime.utcnow() - reading.timestamp).total_seconds()
        freshness_score = max(0, 1 - (time_diff / 300))  # 5 minutes = 300 seconds
        quality_factors.append(freshness_score)
        
        # Value reasonableness based on data type
        reasonableness_score = self._assess_value_reasonableness(reading)
        quality_factors.append(reasonableness_score)
        
        # Metadata completeness
        metadata_score = len(reading.metadata) / 5  # Assume 5 metadata fields is ideal
        metadata_score = min(1.0, metadata_score)
        quality_factors.append(metadata_score)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _assess_value_reasonableness(self, reading: SensorReading) -> float:
        """Assess if sensor value is reasonable for the data type."""
        
        if not isinstance(reading.value, (int, float)):
            return 0.8  # Non-numeric data gets moderate score
        
        value = float(reading.value)
        
        # Define reasonable ranges for different data types
        reasonable_ranges = {
            DataType.HEART_RATE: (30, 200),
            DataType.BLOOD_PRESSURE: (60, 250),  # Systolic
            DataType.BLOOD_GLUCOSE: (50, 400),  # mg/dL
            DataType.OXYGEN_SATURATION: (70, 100),
            DataType.BODY_TEMPERATURE: (95, 110),  # Fahrenheit
            DataType.WEIGHT: (50, 500),  # pounds
            DataType.STEPS: (0, 50000),
        }
        
        range_limits = reasonable_ranges.get(reading.data_type, (0, float('inf')))
        min_val, max_val = range_limits
        
        if min_val <= value <= max_val:
            return 1.0
        elif value < min_val * 0.5 or value > max_val * 1.5:
            return 0.2  # Very unreasonable
        else:
            return 0.5  # Somewhat unreasonable
    
    def _detect_anomalies(
        self, 
        readings: List[SensorReading], 
        historical_data: Optional[List[SensorReading]] = None
    ) -> List[SensorReading]:
        """Detect anomalous readings using simple statistical methods."""
        
        anomalies = []
        
        if not historical_data or len(historical_data) < 10:
            # Simple threshold-based detection for new devices
            return self._threshold_based_anomaly_detection(readings)
        
        # Statistical anomaly detection using historical data
        for reading in readings:
            if self._is_statistical_anomaly(reading, historical_data):
                anomalies.append(reading)
        
        return anomalies
    
    def _threshold_based_anomaly_detection(self, readings: List[SensorReading]) -> List[SensorReading]:
        """Simple threshold-based anomaly detection."""
        
        anomalies = []
        
        # Define critical thresholds for different data types
        critical_thresholds = {
            DataType.HEART_RATE: {'low': 40, 'high': 150},
            DataType.BLOOD_PRESSURE: {'low': 80, 'high': 180},  # Systolic
            DataType.BLOOD_GLUCOSE: {'low': 70, 'high': 250},
            DataType.OXYGEN_SATURATION: {'low': 90, 'high': 100},
            DataType.BODY_TEMPERATURE: {'low': 97, 'high': 102},
        }
        
        for reading in readings:
            if not isinstance(reading.value, (int, float)):
                continue
                
            thresholds = critical_thresholds.get(reading.data_type)
            if thresholds:
                value = float(reading.value)
                if value < thresholds['low'] or value > thresholds['high']:
                    anomalies.append(reading)
        
        return anomalies
    
    def _is_statistical_anomaly(
        self, 
        reading: SensorReading, 
        historical_data: List[SensorReading]
    ) -> bool:
        """Determine if reading is statistical anomaly using historical data."""
        
        if not isinstance(reading.value, (int, float)):
            return False
        
        # Filter historical data for same data type
        same_type_data = [
            float(r.value) for r in historical_data 
            if r.data_type == reading.data_type and isinstance(r.value, (int, float))
        ]
        
        if len(same_type_data) < 5:
            return False
        
        # Calculate statistical metrics
        mean_val = np.mean(same_type_data)
        std_val = np.std(same_type_data)
        
        if std_val == 0:
            return reading.value != mean_val
        
        # Z-score based anomaly detection
        z_score = abs((float(reading.value) - mean_val) / std_val)
        
        # Threshold of 3 standard deviations
        return z_score > 3.0


class WearableDataProcessor(BaseEdgeProcessor):
    """Processor for wearable device data."""
    
    def __init__(self):
        super().__init__("wearable_processor", "v2.3.0")
        
        # Cache for recent readings (for trend analysis)
        self.reading_cache = defaultdict(lambda: deque(maxlen=100))
        
        # Activity patterns cache
        self.activity_patterns = defaultdict(dict)
    
    async def process_data(
        self, 
        request: EdgeDataRequest
    ) -> EdgeProcessingResponse:
        """Process wearable device data."""
        
        start_time = time.time()
        
        try:
            processed_readings = []
            alerts = []
            insights = {}
            recommendations = []
            
            # Process each reading
            for reading in request.sensor_readings:
                # Calculate data quality
                reading.quality_score = self._calculate_data_quality(reading)
                
                # Process based on data type
                processed_reading = await self._process_wearable_reading(reading, request)
                processed_readings.append(processed_reading)
                
                # Cache for trend analysis
                self.reading_cache[f"{request.device_id}_{reading.data_type.value}"].append(reading)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(request.sensor_readings)
            
            # Generate alerts for anomalies
            for anomaly in anomalies:
                alert = await self._generate_wearable_alert(anomaly, request)
                if alert:
                    alerts.append(alert)
            
            # Analyze activity patterns
            activity_insights = await self._analyze_activity_patterns(request)
            insights.update(activity_insights)
            
            # Generate health recommendations
            recommendations = self._generate_wearable_recommendations(processed_readings, insights)
            
            # Determine if cloud sync is needed
            sync_required = self._should_sync_to_cloud(alerts, processed_readings)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Update metrics
            self.processing_metrics['total_readings_processed'] += len(request.sensor_readings)
            self.processing_metrics['alerts_generated'] += len(alerts)
            
            return EdgeProcessingResponse(
                device_id=request.device_id,
                patient_id=request.patient_id,
                processed_readings=processed_readings,
                alerts=alerts,
                insights=insights,
                recommendations=recommendations,
                sync_required=sync_required,
                processing_latency_ms=processing_time,
                battery_impact="low",
                next_sync_time=self._calculate_next_sync_time(sync_required),
                metadata={
                    'anomalies_detected': len(anomalies),
                    'cache_size': sum(len(cache) for cache in self.reading_cache.values()),
                    'processing_version': self.version
                }
            )
            
        except Exception as e:
            self.logger.error(f"Wearable data processing failed: {e}")
            raise
    
    async def _process_wearable_reading(
        self, 
        reading: SensorReading, 
        request: EdgeDataRequest
    ) -> SensorReading:
        """Process individual wearable reading."""
        
        processed_reading = reading
        processed_reading.processed = True
        
        # Add processing metadata
        processed_reading.metadata.update({
            'processed_at': datetime.utcnow().isoformat(),
            'processor_version': self.version,
            'edge_device': request.device_id
        })
        
        # Data type specific processing
        if reading.data_type == DataType.HEART_RATE:
            processed_reading = await self._process_heart_rate(reading, request)
        elif reading.data_type == DataType.STEPS:
            processed_reading = await self._process_steps_data(reading, request)
        elif reading.data_type == DataType.SLEEP_STAGES:
            processed_reading = await self._process_sleep_data(reading, request)
        elif reading.data_type == DataType.ACCELERATION:
            processed_reading = await self._process_acceleration_data(reading, request)
        
        return processed_reading
    
    async def _process_heart_rate(
        self, 
        reading: SensorReading, 
        request: EdgeDataRequest
    ) -> SensorReading:
        """Process heart rate data with advanced analytics."""
        
        if not isinstance(reading.value, (int, float)):
            return reading
        
        hr_value = float(reading.value)
        
        # Get recent heart rate readings for trend analysis
        recent_readings = self.reading_cache[f"{request.device_id}_heart_rate"]
        
        if len(recent_readings) >= 5:
            recent_hr_values = [float(r.value) for r in list(recent_readings)[-5:] if isinstance(r.value, (int, float))]
            
            if recent_hr_values:
                # Calculate heart rate variability (simplified)
                hr_variability = np.std(recent_hr_values)
                
                # Detect trends
                hr_trend = self._calculate_heart_rate_trend(recent_hr_values)
                
                reading.metadata.update({
                    'hr_variability': hr_variability,
                    'hr_trend': hr_trend,
                    'resting_hr_estimate': min(recent_hr_values),
                    'max_hr_estimate': max(recent_hr_values)
                })
        
        # Classify heart rate zone
        hr_zone = self._classify_heart_rate_zone(hr_value, request.clinical_context)
        reading.metadata['hr_zone'] = hr_zone
        
        return reading
    
    def _calculate_heart_rate_trend(self, hr_values: List[float]) -> str:
        """Calculate heart rate trend from recent values."""
        
        if len(hr_values) < 3:
            return "insufficient_data"
        
        # Simple trend calculation
        recent_avg = np.mean(hr_values[-3:])
        earlier_avg = np.mean(hr_values[:-3])
        
        diff_percentage = ((recent_avg - earlier_avg) / earlier_avg) * 100
        
        if diff_percentage > 10:
            return "increasing"
        elif diff_percentage < -10:
            return "decreasing"
        else:
            return "stable"
    
    def _classify_heart_rate_zone(self, hr_value: float, clinical_context: Dict[str, Any]) -> str:
        """Classify heart rate into zones."""
        
        # Estimate max heart rate (simplified formula)
        age = clinical_context.get('age', 30)
        estimated_max_hr = 220 - age
        
        # Heart rate zones as percentage of max HR
        if hr_value < estimated_max_hr * 0.5:
            return "resting"
        elif hr_value < estimated_max_hr * 0.6:
            return "fat_burn"
        elif hr_value < estimated_max_hr * 0.7:
            return "aerobic"
        elif hr_value < estimated_max_hr * 0.85:
            return "anaerobic"
        else:
            return "maximum"
    
    async def _process_steps_data(
        self, 
        reading: SensorReading, 
        request: EdgeDataRequest
    ) -> SensorReading:
        """Process step count data."""
        
        if not isinstance(reading.value, (int, float)):
            return reading
        
        steps = int(reading.value)
        
        # Calculate daily step goal progress
        daily_goal = request.clinical_context.get('daily_step_goal', 10000)
        goal_progress = (steps / daily_goal) * 100
        
        # Estimate calories burned (simplified calculation)
        weight_lbs = request.clinical_context.get('weight_lbs', 150)
        estimated_calories = steps * 0.04 * (weight_lbs / 150)
        
        reading.metadata.update({
            'daily_goal': daily_goal,
            'goal_progress_percentage': goal_progress,
            'estimated_calories_burned': estimated_calories,
            'activity_level': self._classify_activity_level(steps)
        })
        
        return reading
    
    def _classify_activity_level(self, steps: int) -> str:
        """Classify activity level based on step count."""
        
        if steps < 5000:
            return "sedentary"
        elif steps < 7500:
            return "low_active"
        elif steps < 10000:
            return "somewhat_active"
        elif steps < 12500:
            return "active"
        else:
            return "highly_active"
    
    async def _process_sleep_data(
        self, 
        reading: SensorReading, 
        request: EdgeDataRequest
    ) -> SensorReading:
        """Process sleep stage data."""
        
        # Sleep data might be complex (stages, duration, quality)
        if isinstance(reading.value, dict):
            sleep_data = reading.value
            
            # Calculate sleep efficiency
            total_sleep_time = sleep_data.get('total_sleep_minutes', 0)
            time_in_bed = sleep_data.get('time_in_bed_minutes', total_sleep_time)
            
            sleep_efficiency = (total_sleep_time / time_in_bed) * 100 if time_in_bed > 0 else 0
            
            # Analyze sleep stages
            deep_sleep_percentage = sleep_data.get('deep_sleep_percentage', 0)
            rem_sleep_percentage = sleep_data.get('rem_sleep_percentage', 0)
            
            reading.metadata.update({
                'sleep_efficiency': sleep_efficiency,
                'sleep_quality_score': self._calculate_sleep_quality_score(sleep_data),
                'deep_sleep_adequate': deep_sleep_percentage >= 15,
                'rem_sleep_adequate': rem_sleep_percentage >= 20
            })
        
        return reading
    
    def _calculate_sleep_quality_score(self, sleep_data: Dict[str, Any]) -> float:
        """Calculate overall sleep quality score."""
        
        factors = []
        
        # Duration factor (7-9 hours optimal)
        duration_hours = sleep_data.get('total_sleep_minutes', 0) / 60
        if 7 <= duration_hours <= 9:
            duration_score = 1.0
        elif 6 <= duration_hours <= 10:
            duration_score = 0.8
        else:
            duration_score = 0.5
        factors.append(duration_score)
        
        # Efficiency factor
        efficiency = sleep_data.get('sleep_efficiency', 0)
        efficiency_score = min(1.0, efficiency / 85)  # 85% is good efficiency
        factors.append(efficiency_score)
        
        # Deep sleep factor
        deep_sleep_pct = sleep_data.get('deep_sleep_percentage', 0)
        deep_sleep_score = min(1.0, deep_sleep_pct / 20)  # 20% is good
        factors.append(deep_sleep_score)
        
        return sum(factors) / len(factors)
    
    async def _process_acceleration_data(
        self, 
        reading: SensorReading, 
        request: EdgeDataRequest
    ) -> SensorReading:
        """Process acceleration data for fall detection and activity recognition."""
        
        if not isinstance(reading.value, (list, dict)):
            return reading
        
        # Extract acceleration components
        if isinstance(reading.value, dict):
            x = reading.value.get('x', 0)
            y = reading.value.get('y', 0)
            z = reading.value.get('z', 0)
        else:
            x, y, z = reading.value[:3] if len(reading.value) >= 3 else (0, 0, 0)
        
        # Calculate magnitude of acceleration
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        # Fall detection (simplified)
        fall_threshold = 2.5  # g-force threshold
        potential_fall = magnitude > fall_threshold
        
        # Activity recognition (very simplified)
        activity = self._classify_activity_from_acceleration(magnitude, x, y, z)
        
        reading.metadata.update({
            'acceleration_magnitude': magnitude,
            'potential_fall_detected': potential_fall,
            'activity_classification': activity,
            'movement_intensity': self._classify_movement_intensity(magnitude)
        })
        
        return reading
    
    def _classify_activity_from_acceleration(self, magnitude: float, x: float, y: float, z: float) -> str:
        """Classify activity based on acceleration patterns."""
        
        # Very simplified activity classification
        if magnitude < 0.5:
            return "resting"
        elif magnitude < 1.2:
            return "walking"
        elif magnitude < 2.0:
            return "moderate_activity"
        else:
            return "vigorous_activity"
    
    def _classify_movement_intensity(self, magnitude: float) -> str:
        """Classify movement intensity."""
        
        if magnitude < 0.3:
            return "sedentary"
        elif magnitude < 1.0:
            return "light"
        elif magnitude < 1.8:
            return "moderate"
        else:
            return "vigorous"
    
    async def _generate_wearable_alert(
        self, 
        anomaly: SensorReading, 
        request: EdgeDataRequest
    ) -> Optional[EdgeAlert]:
        """Generate alert for wearable data anomaly."""
        
        if not isinstance(anomaly.value, (int, float)):
            return None
        
        value = float(anomaly.value)
        alert_level = AlertLevel.WARNING
        requires_immediate_action = False
        
        # Determine alert level based on data type and value
        if anomaly.data_type == DataType.HEART_RATE:
            if value < 40 or value > 150:
                alert_level = AlertLevel.CRITICAL
                requires_immediate_action = True
            elif value < 50 or value > 120:
                alert_level = AlertLevel.WARNING
        
        elif anomaly.data_type == DataType.OXYGEN_SATURATION:
            if value < 90:
                alert_level = AlertLevel.CRITICAL
                requires_immediate_action = True
            elif value < 95:
                alert_level = AlertLevel.WARNING
        
        elif anomaly.data_type == DataType.BODY_TEMPERATURE:
            if value > 102 or value < 96:
                alert_level = AlertLevel.CRITICAL
                requires_immediate_action = True
            elif value > 100.5 or value < 97:
                alert_level = AlertLevel.WARNING
        
        # Check for fall detection
        if anomaly.data_type == DataType.ACCELERATION and anomaly.metadata.get('potential_fall_detected'):
            alert_level = AlertLevel.EMERGENCY
            requires_immediate_action = True
        
        message = f"{anomaly.data_type.value.replace('_', ' ').title()} anomaly detected: {value} {anomaly.unit}"
        
        return EdgeAlert(
            alert_id=f"wearable_alert_{anomaly.device_id}_{datetime.utcnow().timestamp()}",
            device_id=request.device_id,
            patient_id=request.patient_id,
            alert_level=alert_level,
            message=message,
            data_value=value,
            timestamp=datetime.utcnow(),
            requires_immediate_action=requires_immediate_action,
            clinical_context=request.clinical_context,
            processed_at_edge=True
        )
    
    async def _analyze_activity_patterns(self, request: EdgeDataRequest) -> Dict[str, Any]:
        """Analyze activity patterns from wearable data."""
        
        insights = {}
        
        # Analyze step patterns
        step_readings = [r for r in request.sensor_readings if r.data_type == DataType.STEPS]
        if step_readings:
            total_steps = sum(int(r.value) for r in step_readings if isinstance(r.value, (int, float)))
            insights['daily_step_total'] = total_steps
            insights['step_goal_achievement'] = total_steps >= request.clinical_context.get('daily_step_goal', 10000)
        
        # Analyze heart rate patterns
        hr_readings = [r for r in request.sensor_readings if r.data_type == DataType.HEART_RATE]
        if hr_readings:
            hr_values = [float(r.value) for r in hr_readings if isinstance(r.value, (int, float))]
            if hr_values:
                insights['heart_rate_analysis'] = {
                    'average_hr': np.mean(hr_values),
                    'min_hr': min(hr_values),
                    'max_hr': max(hr_values),
                    'hr_variability': np.std(hr_values)
                }
        
        # Activity level assessment
        activity_level = self._assess_overall_activity_level(request.sensor_readings)
        insights['overall_activity_level'] = activity_level
        
        return insights
    
    def _assess_overall_activity_level(self, readings: List[SensorReading]) -> str:
        """Assess overall activity level from multiple sensor readings."""
        
        activity_scores = []
        
        # Score from steps
        step_readings = [r for r in readings if r.data_type == DataType.STEPS]
        if step_readings:
            total_steps = sum(int(r.value) for r in step_readings if isinstance(r.value, (int, float)))
            step_score = min(1.0, total_steps / 10000)  # Normalize to 10k steps
            activity_scores.append(step_score)
        
        # Score from heart rate zones
        hr_readings = [r for r in readings if r.data_type == DataType.HEART_RATE]
        active_hr_readings = [r for r in hr_readings if r.metadata.get('hr_zone') in ['aerobic', 'anaerobic', 'maximum']]
        if hr_readings:
            hr_activity_score = len(active_hr_readings) / len(hr_readings)
            activity_scores.append(hr_activity_score)
        
        # Score from acceleration data
        accel_readings = [r for r in readings if r.data_type == DataType.ACCELERATION]
        active_accel_readings = [r for r in accel_readings if r.metadata.get('movement_intensity') in ['moderate', 'vigorous']]
        if accel_readings:
            accel_activity_score = len(active_accel_readings) / len(accel_readings)
            activity_scores.append(accel_activity_score)
        
        if not activity_scores:
            return "unknown"
        
        overall_score = sum(activity_scores) / len(activity_scores)
        
        if overall_score < 0.3:
            return "sedentary"
        elif overall_score < 0.5:
            return "lightly_active"
        elif overall_score < 0.7:
            return "moderately_active"
        else:
            return "very_active"
    
    def _generate_wearable_recommendations(
        self, 
        readings: List[SensorReading], 
        insights: Dict[str, Any]
    ) -> List[str]:
        """Generate health recommendations based on wearable data."""
        
        recommendations = []
        
        # Step-based recommendations
        if 'daily_step_total' in insights:
            steps = insights['daily_step_total']
            goal_achieved = insights.get('step_goal_achievement', False)
            
            if not goal_achieved:
                remaining_steps = 10000 - steps
                recommendations.append(f"Try to take {remaining_steps} more steps to reach your daily goal")
            else:
                recommendations.append("Great job reaching your step goal! Consider increasing it next week")
        
        # Heart rate recommendations
        hr_analysis = insights.get('heart_rate_analysis', {})
        if hr_analysis:
            avg_hr = hr_analysis.get('average_hr', 0)
            if avg_hr > 100:
                recommendations.append("Your heart rate seems elevated. Consider rest and relaxation")
            elif avg_hr < 60:
                recommendations.append("Your heart rate is quite low. Monitor for any symptoms")
        
        # Activity level recommendations
        activity_level = insights.get('overall_activity_level', 'unknown')
        if activity_level == 'sedentary':
            recommendations.append("Consider adding light physical activity throughout the day")
        elif activity_level == 'very_active':
            recommendations.append("Excellent activity level! Don't forget to include rest and recovery")
        
        # Sleep recommendations (if sleep data available)
        sleep_readings = [r for r in readings if r.data_type == DataType.SLEEP_STAGES]
        if sleep_readings:
            for reading in sleep_readings:
                sleep_quality = reading.metadata.get('sleep_quality_score', 0)
                if sleep_quality < 0.7:
                    recommendations.append("Focus on improving sleep quality through consistent bedtime routine")
        
        return recommendations
    
    def _should_sync_to_cloud(self, alerts: List[EdgeAlert], readings: List[SensorReading]) -> bool:
        """Determine if data should be synced to cloud immediately."""
        
        # Sync if there are critical or emergency alerts
        critical_alerts = [a for a in alerts if a.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]
        if critical_alerts:
            return True
        
        # Sync if there are readings requiring immediate attention
        immediate_attention_readings = [r for r in readings if r.metadata.get('requires_immediate_attention', False)]
        if immediate_attention_readings:
            return True
        
        # Sync based on data volume (batch processing)
        if len(readings) > 100:
            return True
        
        return False
    
    def _calculate_next_sync_time(self, sync_required: bool) -> Optional[datetime]:
        """Calculate next sync time based on current conditions."""
        
        if sync_required:
            return datetime.utcnow()  # Immediate sync
        else:
            # Regular sync every 15 minutes
            return datetime.utcnow() + timedelta(minutes=15)


class IoTDeviceManager:
    """Manager for IoT devices and edge processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Device registry
        self.registered_devices = {}
        
        # Edge processors
        self.processors = {
            DeviceType.WEARABLE_FITNESS: WearableDataProcessor(),
            DeviceType.SMARTWATCH: WearableDataProcessor(),
            DeviceType.ACTIVITY_TRACKER: WearableDataProcessor(),
            DeviceType.HEART_RATE_MONITOR: WearableDataProcessor(),
        }
        
        # Data queues for different processing modes
        self.real_time_queue = queue.Queue()
        self.batch_queue = queue.Queue()
        
        # Connection managers
        self.mqtt_client = None
        self.bluetooth_connections = {}
        
        # Processing threads
        self.processing_threads = {}
        self.running = False
        
        # Performance metrics
        self.performance_metrics = {
            'devices_connected': 0,
            'total_data_processed': 0,
            'alerts_generated': 0,
            'sync_operations': 0,
            'average_latency_ms': 0.0
        }
    
    async def initialize(self):
        """Initialize the IoT device manager."""
        
        try:
            # Initialize MQTT client if available
            if MQTT_AVAILABLE:
                await self._initialize_mqtt()
            
            # Initialize Bluetooth if available
            if BLUETOOTH_AVAILABLE:
                await self._initialize_bluetooth()
            
            # Start processing threads
            await self._start_processing_threads()
            
            self.running = True
            self.logger.info("IoT Device Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize IoT Device Manager: {e}")
            raise
    
    async def register_device(self, device: EdgeDevice) -> bool:
        """Register a new IoT device."""
        
        try:
            # Validate device capabilities
            if not self._validate_device_capabilities(device):
                self.logger.warning(f"Device {device.device_id} has invalid capabilities")
                return False
            
            # Add to registry
            self.registered_devices[device.device_id] = device
            
            # Update connection status
            device.connection_status = "registered"
            device.last_seen = datetime.utcnow()
            
            # Initialize device-specific settings
            await self._initialize_device_settings(device)
            
            self.performance_metrics['devices_connected'] += 1
            
            self.logger.info(f"Device {device.device_id} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register device {device.device_id}: {e}")
            return False
    
    async def process_device_data(self, request: EdgeDataRequest) -> EdgeProcessingResponse:
        """Process data from IoT device."""
        
        start_time = time.time()
        
        try:
            # Validate device is registered
            if request.device_id not in self.registered_devices:
                raise ValueError(f"Device {request.device_id} not registered")
            
            device = self.registered_devices[request.device_id]
            
            # Update device last seen
            device.last_seen = datetime.utcnow()
            device.connection_status = "active"
            
            # Get appropriate processor
            processor = self.processors.get(device.device_type)
            if not processor:
                raise ValueError(f"No processor available for device type {device.device_type}")
            
            # Process based on mode
            if request.processing_mode == ProcessingMode.REAL_TIME:
                response = await processor.process_data(request)
            elif request.processing_mode == ProcessingMode.BATCH:
                # Add to batch queue
                self.batch_queue.put(request)
                response = self._create_batch_response(request)
            else:
                response = await processor.process_data(request)
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_metrics['total_data_processed'] += len(request.sensor_readings)
            self.performance_metrics['alerts_generated'] += len(response.alerts)
            
            # Update average latency
            current_avg = self.performance_metrics['average_latency_ms']
            total_processed = self.performance_metrics['total_data_processed']
            self.performance_metrics['average_latency_ms'] = ((current_avg * (total_processed - 1)) + processing_time) / total_processed
            
            # Handle cloud sync if required
            if response.sync_required:
                await self._sync_to_cloud(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to process device data: {e}")
            raise
    
    def _validate_device_capabilities(self, device: EdgeDevice) -> bool:
        """Validate device capabilities."""
        
        required_capabilities = {
            DeviceType.WEARABLE_FITNESS: ['heart_rate', 'steps', 'activity_tracking'],
            DeviceType.SMARTWATCH: ['heart_rate', 'steps', 'notifications'],
            DeviceType.GLUCOSE_MONITOR: ['glucose_measurement', 'data_storage'],
            DeviceType.BLOOD_PRESSURE_MONITOR: ['bp_measurement', 'data_storage'],
        }
        
        required = required_capabilities.get(device.device_type, [])
        return all(cap in device.capabilities for cap in required)
    
    async def _initialize_device_settings(self, device: EdgeDevice):
        """Initialize device-specific settings."""
        
        # Set default configuration based on device type
        default_configs = {
            DeviceType.WEARABLE_FITNESS: {
                'sampling_rate_hz': 1,
                'battery_optimization': True,
                'data_compression': True,
                'alert_thresholds': {
                    'heart_rate_low': 50,
                    'heart_rate_high': 120
                }
            },
            DeviceType.GLUCOSE_MONITOR: {
                'measurement_interval_minutes': 15,
                'auto_calibration': True,
                'critical_glucose_alerts': True
            }
        }
        
        default_config = default_configs.get(device.device_type, {})
        device.configuration.update(default_config)
    
    async def _initialize_mqtt(self):
        """Initialize MQTT client for IoT communication."""
        
        if not MQTT_AVAILABLE:
            return
        
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            
            broker_host = self.config.get('mqtt_broker_host', 'localhost')
            broker_port = self.config.get('mqtt_broker_port', 1883)
            
            self.mqtt_client.connect(broker_host, broker_port, 60)
            self.mqtt_client.loop_start()
            
            self.logger.info("MQTT client initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MQTT: {e}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        
        if rc == 0:
            self.logger.info("Connected to MQTT broker")
            # Subscribe to device topics
            client.subscribe("healthcare/devices/+/data")
            client.subscribe("healthcare/devices/+/status")
        else:
            self.logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        
        try:
            topic_parts = msg.topic.split('/')
            device_id = topic_parts[2]
            message_type = topic_parts[3]
            
            payload = json.loads(msg.payload.decode())
            
            if message_type == 'data':
                # Process incoming sensor data
                asyncio.create_task(self._handle_mqtt_data(device_id, payload))
            elif message_type == 'status':
                # Update device status
                self._handle_device_status_update(device_id, payload)
                
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
    
    async def _handle_mqtt_data(self, device_id: str, payload: Dict[str, Any]):
        """Handle incoming MQTT data."""
        
        try:
            # Convert payload to sensor readings
            sensor_readings = self._convert_mqtt_payload_to_readings(device_id, payload)
            
            # Create processing request
            request = EdgeDataRequest(
                device_id=device_id,
                sensor_readings=sensor_readings,
                processing_mode=ProcessingMode.REAL_TIME,
                patient_id=payload.get('patient_id', 'unknown')
            )
            
            # Process data
            response = await self.process_device_data(request)
            
            # Send response back to device if needed
            if response.alerts:
                await self._send_alerts_to_device(device_id, response.alerts)
                
        except Exception as e:
            self.logger.error(f"Error handling MQTT data from {device_id}: {e}")
    
    def _convert_mqtt_payload_to_readings(self, device_id: str, payload: Dict[str, Any]) -> List[SensorReading]:
        """Convert MQTT payload to sensor readings."""
        
        readings = []
        timestamp = datetime.utcnow()
        
        # Extract sensor data from payload
        sensor_data = payload.get('sensors', {})
        
        for sensor_type, value in sensor_data.items():
            try:
                # Map sensor type to DataType enum
                data_type = self._map_sensor_type_to_data_type(sensor_type)
                if data_type:
                    reading = SensorReading(
                        device_id=device_id,
                        device_type=self._get_device_type(device_id),
                        data_type=data_type,
                        value=value,
                        unit=self._get_unit_for_data_type(data_type),
                        timestamp=timestamp,
                        metadata=payload.get('metadata', {})
                    )
                    readings.append(reading)
            except Exception as e:
                self.logger.warning(f"Failed to convert sensor data {sensor_type}: {e}")
        
        return readings
    
    def _map_sensor_type_to_data_type(self, sensor_type: str) -> Optional[DataType]:
        """Map MQTT sensor type to DataType enum."""
        
        mapping = {
            'heart_rate': DataType.HEART_RATE,
            'hr': DataType.HEART_RATE,
            'blood_pressure': DataType.BLOOD_PRESSURE,
            'bp': DataType.BLOOD_PRESSURE,
            'glucose': DataType.BLOOD_GLUCOSE,
            'steps': DataType.STEPS,
            'temperature': DataType.BODY_TEMPERATURE,
            'temp': DataType.BODY_TEMPERATURE,
            'spo2': DataType.OXYGEN_SATURATION,
            'oxygen_saturation': DataType.OXYGEN_SATURATION,
            'acceleration': DataType.ACCELERATION,
            'accel': DataType.ACCELERATION,
            'weight': DataType.WEIGHT
        }
        
        return mapping.get(sensor_type.lower())
    
    def _get_device_type(self, device_id: str) -> DeviceType:
        """Get device type for device ID."""
        
        device = self.registered_devices.get(device_id)
        return device.device_type if device else DeviceType.WEARABLE_FITNESS
    
    def _get_unit_for_data_type(self, data_type: DataType) -> str:
        """Get unit for data type."""
        
        units = {
            DataType.HEART_RATE: 'bpm',
            DataType.BLOOD_PRESSURE: 'mmHg',
            DataType.BLOOD_GLUCOSE: 'mg/dL',
            DataType.OXYGEN_SATURATION: '%',
            DataType.BODY_TEMPERATURE: '°F',
            DataType.WEIGHT: 'lbs',
            DataType.STEPS: 'count',
            DataType.CALORIES: 'kcal',
            DataType.ACCELERATION: 'g'
        }
        
        return units.get(data_type, 'unknown')
    
    async def _initialize_bluetooth(self):
        """Initialize Bluetooth connectivity."""
        
        if not BLUETOOTH_AVAILABLE:
            return
        
        try:
            # Initialize Bluetooth discovery and connections
            # This would be implemented based on specific Bluetooth libraries
            self.logger.info("Bluetooth connectivity initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Bluetooth: {e}")
    
    async def _start_processing_threads(self):
        """Start background processing threads."""
        
        # Real-time processing thread
        real_time_thread = threading.Thread(
            target=self._real_time_processing_worker,
            name="RealTimeProcessor"
        )
        real_time_thread.daemon = True
        real_time_thread.start()
        self.processing_threads['real_time'] = real_time_thread
        
        # Batch processing thread
        batch_thread = threading.Thread(
            target=self._batch_processing_worker,
            name="BatchProcessor"
        )
        batch_thread.daemon = True
        batch_thread.start()
        self.processing_threads['batch'] = batch_thread
    
    def _real_time_processing_worker(self):
        """Worker thread for real-time data processing."""
        
        while self.running:
            try:
                # Check for real-time processing requests
                if not self.real_time_queue.empty():
                    request = self.real_time_queue.get(timeout=1)
                    asyncio.run(self.process_device_data(request))
                else:
                    time.sleep(0.1)  # Brief pause
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Real-time processing error: {e}")
    
    def _batch_processing_worker(self):
        """Worker thread for batch data processing."""
        
        batch_requests = []
        
        while self.running:
            try:
                # Collect batch requests
                try:
                    request = self.batch_queue.get(timeout=5)
                    batch_requests.append(request)
                except queue.Empty:
                    pass
                
                # Process batch when we have enough requests or timeout
                if len(batch_requests) >= 10 or (batch_requests and time.time() % 60 < 1):
                    asyncio.run(self._process_batch_requests(batch_requests))
                    batch_requests.clear()
                    
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
    
    async def _process_batch_requests(self, requests: List[EdgeDataRequest]):
        """Process multiple requests in batch mode."""
        
        for request in requests:
            try:
                await self.process_device_data(request)
            except Exception as e:
                self.logger.error(f"Batch processing failed for device {request.device_id}: {e}")
    
    def _create_batch_response(self, request: EdgeDataRequest) -> EdgeProcessingResponse:
        """Create response for batch processing."""
        
        return EdgeProcessingResponse(
            device_id=request.device_id,
            patient_id=request.patient_id,
            processed_readings=request.sensor_readings,
            alerts=[],
            insights={'processing_mode': 'batch'},
            recommendations=['Data queued for batch processing'],
            sync_required=False,
            processing_latency_ms=0.0,
            battery_impact="minimal",
            next_sync_time=datetime.utcnow() + timedelta(hours=1),
            metadata={'queued_at': datetime.utcnow().isoformat()}
        )
    
    async def _sync_to_cloud(self, response: EdgeProcessingResponse):
        """Sync processed data to cloud."""
        
        try:
            # This would implement actual cloud synchronization
            # For now, just log the sync operation
            
            sync_data = {
                'device_id': response.device_id,
                'patient_id': response.patient_id,
                'alerts': [alert.__dict__ for alert in response.alerts],
                'insights': response.insights,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Simulate cloud API call
            await asyncio.sleep(0.1)  # Simulate network latency
            
            self.performance_metrics['sync_operations'] += 1
            
            self.logger.info(f"Synced data to cloud for device {response.device_id}")
            
        except Exception as e:
            self.logger.error(f"Cloud sync failed for device {response.device_id}: {e}")
    
    async def _send_alerts_to_device(self, device_id: str, alerts: List[EdgeAlert]):
        """Send alerts back to device."""
        
        try:
            # Format alerts for device
            alert_data = {
                'alerts': [
                    {
                        'level': alert.alert_level.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'requires_action': alert.requires_immediate_action
                    }
                    for alert in alerts
                ]
            }
            
            # Send via MQTT if available
            if self.mqtt_client and MQTT_AVAILABLE:
                topic = f"healthcare/devices/{device_id}/alerts"
                self.mqtt_client.publish(topic, json.dumps(alert_data))
            
            self.logger.info(f"Sent {len(alerts)} alerts to device {device_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alerts to device {device_id}: {e}")
    
    def _handle_device_status_update(self, device_id: str, status_data: Dict[str, Any]):
        """Handle device status update."""
        
        device = self.registered_devices.get(device_id)
        if device:
            device.battery_level = status_data.get('battery_level')
            device.connection_status = status_data.get('connection_status', 'unknown')
            device.last_seen = datetime.utcnow()
            
            # Log low battery warnings
            if device.battery_level and device.battery_level < 20:
                self.logger.warning(f"Device {device_id} has low battery: {device.battery_level}%")
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific device."""
        
        device = self.registered_devices.get(device_id)
        if not device:
            return None
        
        return {
            'device_id': device.device_id,
            'device_name': device.device_name,
            'device_type': device.device_type.value,
            'connection_status': device.connection_status,
            'battery_level': device.battery_level,
            'last_seen': device.last_seen.isoformat() if device.last_seen else None,
            'firmware_version': device.firmware_version,
            'capabilities': device.capabilities
        }
    
    def get_all_devices_status(self) -> List[Dict[str, Any]]:
        """Get status of all registered devices."""
        
        return [
            self.get_device_status(device_id) 
            for device_id in self.registered_devices.keys()
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        
        return {
            'devices_connected': len(self.registered_devices),
            'total_data_processed': self.performance_metrics['total_data_processed'],
            'alerts_generated': self.performance_metrics['alerts_generated'],
            'sync_operations': self.performance_metrics['sync_operations'],
            'average_latency_ms': self.performance_metrics['average_latency_ms'],
            'queue_sizes': {
                'real_time': self.real_time_queue.qsize(),
                'batch': self.batch_queue.qsize()
            },
            'uptime': 'Running' if self.running else 'Stopped'
        }
    
    async def shutdown(self):
        """Shutdown the IoT device manager."""
        
        self.running = False
        
        # Disconnect MQTT
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        # Wait for processing threads to finish
        for thread in self.processing_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.logger.info("IoT Device Manager shutdown complete")


# Factory function
def create_iot_device_manager(config: Dict[str, Any]) -> IoTDeviceManager:
    """Create IoT device manager with configuration."""
    return IoTDeviceManager(config)