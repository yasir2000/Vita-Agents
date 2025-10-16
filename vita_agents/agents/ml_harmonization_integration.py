"""
Integration module for ML-based data harmonization with existing Vita Agents framework
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from .ml_data_harmonization import (
    MLDataHarmonizationAgent,
    MLHarmonizationMethod,
    ConflictResolutionML,
    create_ml_harmonization_agent
)
from .data_harmonization_agent import DataHarmonizationAgent
from ..core.orchestrator import AgentOrchestrator
from ..core.config import Settings


class EnhancedDataHarmonizationOrchestrator:
    """
    Enhanced orchestrator that combines traditional and ML-based data harmonization
    """
    
    def __init__(self, settings: Settings, db_manager=None):
        self.settings = settings
        self.db_manager = db_manager
        
        # Initialize both traditional and ML agents
        self.traditional_agent = DataHarmonizationAgent(
            "traditional_harmonization", settings, db_manager
        )
        self.ml_agent = create_ml_harmonization_agent(
            "ml_harmonization", settings, db_manager
        )
        
        # Performance tracking
        self.performance_metrics = {
            'traditional_processing_time': 0.0,
            'ml_processing_time': 0.0,
            'hybrid_processing_time': 0.0,
            'accuracy_improvement': 0.0,
            'last_updated': datetime.utcnow()
        }
    
    async def initialize(self):
        """Initialize both harmonization agents"""
        
        await self.traditional_agent.initialize() if hasattr(self.traditional_agent, 'initialize') else None
        await self.ml_agent.initialize()
    
    async def harmonize_data(
        self,
        data_sources: List[Dict[str, Any]],
        user_id: str,
        user_permissions: List[str],
        access_reason: str,
        method: str = "hybrid",
        ml_method: MLHarmonizationMethod = MLHarmonizationMethod.HYBRID_APPROACH,
        conflict_resolution: ConflictResolutionML = ConflictResolutionML.ENSEMBLE_VOTING
    ) -> Dict[str, Any]:
        """
        Harmonize data using traditional, ML, or hybrid approach
        
        Args:
            data_sources: List of data sources to harmonize
            user_id: User requesting harmonization
            user_permissions: User permissions
            access_reason: Reason for data access
            method: "traditional", "ml", or "hybrid"
            ml_method: ML harmonization method
            conflict_resolution: ML conflict resolution method
        """
        
        start_time = datetime.utcnow()
        
        try:
            if method == "traditional":
                result = await self._traditional_harmonization(
                    data_sources, user_id, user_permissions, access_reason
                )
            elif method == "ml":
                result = await self._ml_harmonization(
                    data_sources, user_id, user_permissions, access_reason,
                    ml_method, conflict_resolution
                )
            else:  # hybrid
                result = await self._hybrid_harmonization(
                    data_sources, user_id, user_permissions, access_reason,
                    ml_method, conflict_resolution
                )
            
            # Update performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.performance_metrics[f'{method}_processing_time'] = processing_time
            self.performance_metrics['last_updated'] = datetime.utcnow()
            
            # Add performance metadata to result
            result['performance_metrics'] = self.performance_metrics.copy()
            result['method_used'] = method
            result['processing_time_seconds'] = processing_time
            
            return result
            
        except Exception as e:
            raise Exception(f"Data harmonization failed ({method} method): {e}")
    
    async def _traditional_harmonization(
        self,
        data_sources: List[Dict[str, Any]],
        user_id: str,
        user_permissions: List[str],
        access_reason: str
    ) -> Dict[str, Any]:
        """Perform traditional rule-based harmonization"""
        
        # Extract patient ID for traditional agent
        patient_id = self._extract_patient_id(data_sources)
        
        return await self.traditional_agent.harmonize_patient_data(
            data_sources=data_sources,
            patient_id=patient_id or "unknown",
            user_id=user_id,
            user_permissions=user_permissions,
            access_reason=access_reason
        )
    
    async def _ml_harmonization(
        self,
        data_sources: List[Dict[str, Any]],
        user_id: str,
        user_permissions: List[str],
        access_reason: str,
        ml_method: MLHarmonizationMethod,
        conflict_resolution: ConflictResolutionML
    ) -> Dict[str, Any]:
        """Perform ML-based harmonization"""
        
        # Convert data sources to records format for ML agent
        records = self._convert_sources_to_records(data_sources)
        
        return await self.ml_agent.harmonize_patient_records(
            records=records,
            user_id=user_id,
            user_permissions=user_permissions,
            access_reason=access_reason,
            harmonization_method=ml_method,
            conflict_resolution=conflict_resolution
        )
    
    async def _hybrid_harmonization(
        self,
        data_sources: List[Dict[str, Any]],
        user_id: str,
        user_permissions: List[str],
        access_reason: str,
        ml_method: MLHarmonizationMethod,
        conflict_resolution: ConflictResolutionML
    ) -> Dict[str, Any]:
        """Perform hybrid harmonization combining both approaches"""
        
        # Step 1: Run traditional harmonization for basic processing
        traditional_result = await self._traditional_harmonization(
            data_sources, user_id, user_permissions, access_reason
        )
        
        # Step 2: Run ML harmonization for advanced processing
        ml_result = await self._ml_harmonization(
            data_sources, user_id, user_permissions, access_reason,
            ml_method, conflict_resolution
        )
        
        # Step 3: Combine results intelligently
        hybrid_result = await self._combine_harmonization_results(
            traditional_result, ml_result
        )
        
        return hybrid_result
    
    async def _combine_harmonization_results(
        self,
        traditional_result: Dict[str, Any],
        ml_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Intelligently combine traditional and ML harmonization results"""
        
        # Use ML results as primary, traditional as fallback/validation
        combined_result = ml_result.copy()
        
        # Add traditional processing insights
        combined_result['traditional_insights'] = {
            'conflicts_detected': len(traditional_result.get('conflicts_detected', [])),
            'data_quality_score': traditional_result.get('quality_assessment', {}).get('overall_score', 0.0),
            'source_contribution': traditional_result.get('source_contribution', {})
        }
        
        # Calculate improvement metrics
        traditional_quality = traditional_result.get('quality_assessment', {}).get('overall_score', 0.0)
        ml_quality = ml_result.get('performance_metrics', {}).get('average_quality_score', 0.0) * 100
        
        if traditional_quality > 0:
            improvement = ((ml_quality - traditional_quality) / traditional_quality) * 100
            self.performance_metrics['accuracy_improvement'] = improvement
            combined_result['quality_improvement_percent'] = improvement
        
        # Combine metadata
        combined_result['harmonization_method'] = 'hybrid_traditional_ml'
        combined_result['traditional_metadata'] = traditional_result.get('metadata', {})
        combined_result['ml_metadata'] = ml_result.get('performance_metrics', {})
        
        return combined_result
    
    def _extract_patient_id(self, data_sources: List[Dict[str, Any]]) -> Optional[str]:
        """Extract patient ID from data sources"""
        
        for source in data_sources:
            data = source.get('data', {})
            patient_id = data.get('patient', {}).get('id')
            if patient_id:
                return patient_id
        
        return None
    
    def _convert_sources_to_records(self, data_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert data sources format to records format for ML agent"""
        
        records = []
        
        for i, source in enumerate(data_sources):
            source_info = source.get('source_info', {})
            data = source.get('data', {})
            
            record = {
                'id': f"record_{i}_{source_info.get('source_id', 'unknown')}",
                'source': source_info,
                **data  # Merge all data fields
            }
            
            records.append(record)
        
        return records
    
    async def get_harmonization_capabilities(self) -> Dict[str, Any]:
        """Get available harmonization capabilities"""
        
        return {
            'traditional_capabilities': self.traditional_agent.capabilities,
            'ml_capabilities': self.ml_agent.capabilities,
            'available_methods': [
                'traditional',
                'ml', 
                'hybrid'
            ],
            'ml_harmonization_methods': [method.value for method in MLHarmonizationMethod],
            'conflict_resolution_methods': [method.value for method in ConflictResolutionML],
            'performance_metrics': self.performance_metrics
        }
    
    async def benchmark_methods(
        self,
        test_data_sources: List[Dict[str, Any]],
        user_id: str,
        user_permissions: List[str]
    ) -> Dict[str, Any]:
        """Benchmark different harmonization methods"""
        
        benchmark_results = {}
        
        methods = ['traditional', 'ml', 'hybrid']
        
        for method in methods:
            try:
                start_time = datetime.utcnow()
                
                result = await self.harmonize_data(
                    data_sources=test_data_sources,
                    user_id=user_id,
                    user_permissions=user_permissions,
                    access_reason="performance_benchmarking",
                    method=method
                )
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                benchmark_results[method] = {
                    'processing_time': processing_time,
                    'records_processed': len(test_data_sources),
                    'conflicts_resolved': len(result.get('conflicts_resolved', [])),
                    'quality_score': result.get('quality_assessment', {}).get('overall_score', 0.0),
                    'success': True
                }
                
            except Exception as e:
                benchmark_results[method] = {
                    'error': str(e),
                    'success': False
                }
        
        # Calculate relative performance
        if all(r.get('success') for r in benchmark_results.values()):
            fastest_time = min(r['processing_time'] for r in benchmark_results.values())
            highest_quality = max(r['quality_score'] for r in benchmark_results.values())
            
            for method, result in benchmark_results.items():
                result['speed_ratio'] = fastest_time / result['processing_time']
                result['quality_ratio'] = result['quality_score'] / highest_quality if highest_quality > 0 else 0
        
        return {
            'benchmark_results': benchmark_results,
            'benchmark_timestamp': datetime.utcnow().isoformat(),
            'test_data_size': len(test_data_sources)
        }


# Integration with main orchestrator
class VitaAgentsMLIntegration:
    """Integration class for adding ML capabilities to Vita Agents"""
    
    @staticmethod
    async def register_ml_harmonization_agent(
        orchestrator: AgentOrchestrator,
        settings: Settings,
        db_manager=None
    ):
        """Register ML harmonization capabilities with the main orchestrator"""
        
        # Create enhanced harmonization orchestrator
        ml_harmonization = EnhancedDataHarmonizationOrchestrator(settings, db_manager)
        await ml_harmonization.initialize()
        
        # Register with main orchestrator (if it supports this pattern)
        if hasattr(orchestrator, 'register_enhanced_agent'):
            await orchestrator.register_enhanced_agent(
                'ml_data_harmonization',
                ml_harmonization
            )
        
        return ml_harmonization
    
    @staticmethod
    def get_ml_agent_config() -> Dict[str, Any]:
        """Get configuration for ML data harmonization agent"""
        
        return {
            'agent_type': 'ml_data_harmonization',
            'version': '2.0.0',
            'capabilities': [
                'ml_record_linkage',
                'smart_conflict_resolution',
                'data_quality_assessment',
                'semantic_mapping',
                'anomaly_detection',
                'predictive_harmonization',
                'adaptive_learning'
            ],
            'dependencies': [
                'scikit-learn',
                'numpy',
                'pandas',
                'spacy',
                'fuzzywuzzy',
                'networkx',
                'torch (optional)',
                'transformers (optional)'
            ],
            'configuration': {
                'similarity_thresholds': {
                    'exact': 0.95,
                    'probable': 0.85,
                    'possible': 0.70
                },
                'quality_thresholds': {
                    'completeness': 0.8,
                    'accuracy': 0.9,
                    'consistency': 0.85,
                    'timeliness': 0.7,
                    'validity': 0.95,
                    'uniqueness': 0.98
                },
                'ml_models': {
                    'record_linkage': 'ensemble_matching',
                    'conflict_resolution': 'ensemble_voting',
                    'quality_assessment': 'random_forest'
                }
            }
        }


# Factory functions for easy integration
def create_enhanced_harmonization_system(settings: Settings, db_manager=None) -> EnhancedDataHarmonizationOrchestrator:
    """Create complete enhanced harmonization system"""
    return EnhancedDataHarmonizationOrchestrator(settings, db_manager)


async def setup_ml_harmonization_capabilities(
    orchestrator: AgentOrchestrator,
    settings: Settings,
    db_manager=None
) -> EnhancedDataHarmonizationOrchestrator:
    """Setup ML harmonization capabilities in existing Vita Agents system"""
    
    return await VitaAgentsMLIntegration.register_ml_harmonization_agent(
        orchestrator, settings, db_manager
    )