"""
Configuration templates and examples for Enhanced FHIR Agent
"""

import os
from typing import Dict, Any, List

# Example configuration for the Enhanced FHIR Agent
ENHANCED_FHIR_AGENT_CONFIG = {
    "fhir_engines": {
        # Engines to auto-connect on startup
        "enabled_engines": [
            "hapi_fhir_test",
            "hapi_fhir_r4",
            # "ibm_fhir_local",
            # "medplum_demo"
        ],
        
        # Default engine for single-engine operations
        "default_engine": "hapi_fhir_r4",
        
        # Auto-connect to engines on startup
        "auto_connect": True,
        
        # Connection settings
        "connection_timeout": 30,
        "max_concurrent_operations": 10,
        "retry_failed_operations": True,
        "max_retries": 3,
        
        # Performance monitoring
        "enable_performance_tracking": True,
        "performance_sample_size": 10,
        
        # Validation settings
        "strict_validation": False,
        "validate_before_operations": True,
        
        # Migration settings
        "migration_batch_size": 100,
        "migration_retry_attempts": 3,
        "migration_timeout_seconds": 300
    },
    
    # Logging configuration
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_fhir_requests": False,  # Set to True for detailed HTTP logging
        "log_performance_metrics": True
    },
    
    # Security settings
    "security": {
        "require_authentication": False,
        "allowed_operations": ["read", "search", "create", "update", "delete", "validate"],
        "rate_limiting": {
            "enabled": False,
            "requests_per_minute": 100
        }
    }
}

# Environment-specific configurations
def get_development_config() -> Dict[str, Any]:
    """Development environment configuration"""
    config = ENHANCED_FHIR_AGENT_CONFIG.copy()
    config["fhir_engines"]["enabled_engines"] = [
        "hapi_fhir_test",  # Use test server for development
    ]
    config["logging"]["level"] = "DEBUG"
    config["logging"]["log_fhir_requests"] = True
    return config

def get_production_config() -> Dict[str, Any]:
    """Production environment configuration"""
    config = ENHANCED_FHIR_AGENT_CONFIG.copy()
    config["fhir_engines"]["enabled_engines"] = [
        "hapi_fhir_r4",
        "ibm_fhir_production",
    ]
    config["logging"]["level"] = "INFO"
    config["logging"]["log_fhir_requests"] = False
    config["security"]["require_authentication"] = True
    config["security"]["rate_limiting"]["enabled"] = True
    return config

def get_testing_config() -> Dict[str, Any]:
    """Testing environment configuration"""
    config = ENHANCED_FHIR_AGENT_CONFIG.copy()
    config["fhir_engines"]["enabled_engines"] = [
        "hapi_fhir_test",
    ]
    config["fhir_engines"]["auto_connect"] = False  # Manual control in tests
    config["logging"]["level"] = "WARNING"
    return config

# Custom server configurations (examples)
CUSTOM_SERVER_CONFIGS = {
    # Local HAPI FHIR server
    "hapi_fhir_local": {
        "server_id": "hapi_fhir_local",
        "name": "Local HAPI FHIR Server",
        "engine_type": "HAPI_FHIR",
        "base_url": "http://localhost:8080/fhir",
        "fhir_version": "R4",
        "authentication": {
            "type": "NONE"
        },
        "description": "Local HAPI FHIR server for development"
    },
    
    # IBM FHIR with basic auth
    "ibm_fhir_auth": {
        "server_id": "ibm_fhir_auth",
        "name": "IBM FHIR Server (Auth)",
        "engine_type": "IBM_FHIR",
        "base_url": "https://your-ibm-fhir-server.com/fhir-server/api/v4",
        "fhir_version": "R4",
        "authentication": {
            "type": "BASIC",
            "username": os.getenv("IBM_FHIR_USERNAME", ""),
            "password": os.getenv("IBM_FHIR_PASSWORD", "")
        },
        "headers": {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json"
        },
        "description": "IBM FHIR server with basic authentication"
    },
    
    # Medplum with OAuth2
    "medplum_oauth": {
        "server_id": "medplum_oauth",
        "name": "Medplum FHIR (OAuth2)",
        "engine_type": "MEDPLUM",
        "base_url": "https://api.medplum.com/fhir/R4",
        "fhir_version": "R4",
        "authentication": {
            "type": "OAUTH2",
            "token_url": "https://api.medplum.com/oauth2/token",
            "client_id": os.getenv("MEDPLUM_CLIENT_ID", ""),
            "client_secret": os.getenv("MEDPLUM_CLIENT_SECRET", ""),
            "scope": "fhir:read fhir:write"
        },
        "description": "Medplum FHIR server with OAuth2 authentication"
    },
    
    # SMART on FHIR configuration
    "smart_fhir_server": {
        "server_id": "smart_fhir_server",
        "name": "SMART on FHIR Server",
        "engine_type": "HAPI_FHIR",  # or other engine supporting SMART
        "base_url": "https://launch.smarthealthit.org/v/r4/fhir",
        "fhir_version": "R4",
        "authentication": {
            "type": "SMART_ON_FHIR",
            "authorize_url": "https://launch.smarthealthit.org/v/r4/auth/authorize",
            "token_url": "https://launch.smarthealthit.org/v/r4/auth/token",
            "client_id": os.getenv("SMART_CLIENT_ID", ""),
            "redirect_uri": "http://localhost:8000/callback",
            "scope": "launch/patient patient/read patient/write"
        },
        "description": "SMART on FHIR enabled server"
    }
}

# Performance testing configurations
PERFORMANCE_TEST_CONFIGS = {
    "light_load": {
        "sample_size": 10,
        "concurrent_operations": 2,
        "resource_types": ["Patient", "Observation"],
        "operations": ["search", "read", "capability"]
    },
    
    "medium_load": {
        "sample_size": 50,
        "concurrent_operations": 5,
        "resource_types": ["Patient", "Observation", "DiagnosticReport", "Medication"],
        "operations": ["search", "read", "create", "update", "capability"]
    },
    
    "heavy_load": {
        "sample_size": 100,
        "concurrent_operations": 10,
        "resource_types": ["Patient", "Observation", "DiagnosticReport", "Medication", "Encounter", "Procedure"],
        "operations": ["search", "read", "create", "update", "delete", "capability", "validate"]
    }
}

# Migration configurations
MIGRATION_CONFIGS = {
    "patient_data_migration": {
        "resource_types": ["Patient", "Observation", "DiagnosticReport"],
        "migration_strategy": "incremental",
        "batch_size": 50,
        "include_references": True,
        "validate_before_migration": True,
        "rollback_on_error": True
    },
    
    "full_database_migration": {
        "resource_types": ["*"],  # All resource types
        "migration_strategy": "full",
        "batch_size": 100,
        "include_references": True,
        "validate_before_migration": False,  # Skip validation for speed
        "rollback_on_error": False
    },
    
    "selective_migration": {
        "resource_types": ["Patient"],
        "migration_strategy": "selective",
        "batch_size": 25,
        "filters": {
            "Patient": {
                "active": "true",
                "_lastUpdated": "ge2023-01-01"
            }
        },
        "include_references": False,
        "validate_before_migration": True,
        "rollback_on_error": True
    }
}

# Engine-specific optimizations
ENGINE_OPTIMIZATIONS = {
    "HAPI_FHIR": {
        "search_parameters": {
            "_count": 100,  # Optimal batch size for HAPI
            "_total": "accurate"
        },
        "preferred_formats": ["application/fhir+json"],
        "bulk_operations_supported": True,
        "transaction_bundle_size": 50
    },
    
    "IBM_FHIR": {
        "search_parameters": {
            "_count": 50,  # Conservative batch size
            "_total": "none"  # IBM FHIR doesn't support total counts efficiently
        },
        "preferred_formats": ["application/fhir+json"],
        "bulk_operations_supported": True,
        "transaction_bundle_size": 25
    },
    
    "MEDPLUM": {
        "search_parameters": {
            "_count": 200,  # Medplum handles larger batches well
            "_total": "accurate"
        },
        "preferred_formats": ["application/fhir+json"],
        "bulk_operations_supported": True,
        "transaction_bundle_size": 100
    },
    
    "FIRELY": {
        "search_parameters": {
            "_count": 75,
            "_total": "estimate"
        },
        "preferred_formats": ["application/fhir+json", "application/fhir+xml"],
        "bulk_operations_supported": False,
        "transaction_bundle_size": 30
    }
}

def get_config_for_environment(env: str = None) -> Dict[str, Any]:
    """Get configuration for specified environment"""
    env = env or os.getenv("VITA_ENVIRONMENT", "development")
    
    if env == "production":
        return get_production_config()
    elif env == "testing":
        return get_testing_config()
    else:
        return get_development_config()

def get_custom_server_config(server_name: str) -> Dict[str, Any]:
    """Get custom server configuration by name"""
    return CUSTOM_SERVER_CONFIGS.get(server_name, {})

def get_performance_test_config(load_type: str = "light_load") -> Dict[str, Any]:
    """Get performance test configuration"""
    return PERFORMANCE_TEST_CONFIGS.get(load_type, PERFORMANCE_TEST_CONFIGS["light_load"])

def get_migration_config(migration_type: str = "patient_data_migration") -> Dict[str, Any]:
    """Get migration configuration"""
    return MIGRATION_CONFIGS.get(migration_type, MIGRATION_CONFIGS["patient_data_migration"])

def get_engine_optimizations(engine_type: str) -> Dict[str, Any]:
    """Get engine-specific optimizations"""
    return ENGINE_OPTIMIZATIONS.get(engine_type, {})

# Example usage configurations
USAGE_EXAMPLES = {
    "basic_multi_engine_setup": {
        "description": "Basic setup with HAPI FHIR and Medplum",
        "config": {
            "fhir_engines": {
                "enabled_engines": ["hapi_fhir_r4", "medplum_demo"],
                "auto_connect": True,
                "default_engine": "hapi_fhir_r4"
            }
        }
    },
    
    "performance_testing_setup": {
        "description": "Setup optimized for performance testing",
        "config": {
            "fhir_engines": {
                "enabled_engines": ["hapi_fhir_test", "ibm_fhir_local"],
                "max_concurrent_operations": 20,
                "connection_timeout": 60,
                "enable_performance_tracking": True
            }
        }
    },
    
    "migration_setup": {
        "description": "Setup for data migration between engines",
        "config": {
            "fhir_engines": {
                "enabled_engines": ["source_engine", "target_engine"],
                "migration_batch_size": 50,
                "migration_retry_attempts": 5,
                "validate_before_operations": True
            }
        }
    }
}