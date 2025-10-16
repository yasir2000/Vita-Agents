"""
Core configuration management for Vita Agents.
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    url: str = Field(default="postgresql://vita_user:vita_password@localhost:5432/vita_agents")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    echo: bool = Field(default=False)


class RedisSettings(BaseSettings):
    """Redis configuration."""
    url: str = Field(default="redis://localhost:6379/0")
    prefix: str = Field(default="vita_agents")
    max_connections: int = Field(default=20)


class SecuritySettings(BaseSettings):
    """Security and encryption configuration."""
    secret_key: str = Field(default="your-super-secret-key-change-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=30)
    encryption_key: str = Field(default="your-encryption-key-32-chars-long")
    encryption_salt: str = Field(default="your-encryption-salt-change-in-production")
    hipaa_compliance: bool = Field(default=True)
    audit_log_enabled: bool = Field(default=True)
    audit_log_retention_days: int = Field(default=2555)  # 7 years
    data_retention_years: int = Field(default=7)  # HIPAA requirement
    data_encryption_at_rest: bool = Field(default=True)
    
    # OAuth settings
    oauth_client_id: str = Field(default="")
    oauth_client_secret: str = Field(default="")
    oauth_redirect_uri: str = Field(default="http://localhost:8000/auth/callback")
    
    # Access control
    session_timeout_minutes: int = Field(default=15)
    max_failed_login_attempts: int = Field(default=3)
    account_lockout_minutes: int = Field(default=30)
    
    # JWT settings  
    jwt_secret: str = Field(default="your-jwt-secret-key-change-in-production")
    jwt_expiration_minutes: int = Field(default=30)


class APISettings(BaseSettings):
    """API server configuration."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])
    debug: bool = Field(default=False)


class AgentSettings(BaseSettings):
    """Agent configuration."""
    max_concurrent_agents: int = Field(default=10)
    agent_timeout_seconds: int = Field(default=300)
    agent_retry_attempts: int = Field(default=3)
    workflow_engine: str = Field(default="crewai")  # crewai, langchain, autogen
    workflow_timeout: int = Field(default=600)
    max_workflow_steps: int = Field(default=50)


class HealthcareSettings(BaseSettings):
    """Healthcare standards configuration."""
    fhir_server_url: str = Field(default="http://hapi.fhir.org/baseR4")
    fhir_version: str = Field(default="R4")
    hl7_version: str = Field(default="2.8")
    terminology_service_url: str = Field(default="https://tx.fhir.org")
    snomed_ct_edition: str = Field(default="us")
    icd10_version: str = Field(default="2024")


class EHRSettings(BaseSettings):
    """EHR integration configuration."""
    epic_client_id: Optional[str] = Field(default=None)
    epic_client_secret: Optional[str] = Field(default=None)
    cerner_client_id: Optional[str] = Field(default=None)
    cerner_client_secret: Optional[str] = Field(default=None)


class AISettings(BaseSettings):
    """AI/ML configuration."""
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    model_provider: str = Field(default="openai")
    default_model: str = Field(default="gpt-4-turbo-preview")
    max_tokens: int = Field(default=4000)
    temperature: float = Field(default=0.1)


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    health_check_interval: int = Field(default=30)
    log_level: str = Field(default="INFO")


class Settings(BaseSettings):
    """Main application settings."""
    
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    api: APISettings = Field(default_factory=APISettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    healthcare: HealthcareSettings = Field(default_factory=HealthcareSettings)
    ehr: EHRSettings = Field(default_factory=EHRSettings)
    ai: AISettings = Field(default_factory=AISettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "_"
        case_sensitive = False


def load_config(config_path: Optional[Path] = None) -> Settings:
    """Load configuration from environment and optional YAML file."""
    
    # Load from environment first
    settings = Settings()
    
    # Override with YAML config if provided
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        # Update settings with YAML values
        for key, value in yaml_config.items():
            if hasattr(settings, key):
                if isinstance(value, dict):
                    # Handle nested configuration
                    nested_setting = getattr(settings, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_setting, nested_key):
                            setattr(nested_setting, nested_key, nested_value)
                else:
                    setattr(settings, key, value)
    
    return settings


# Global settings instance
settings = load_config()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def update_settings(**kwargs) -> None:
    """Update global settings."""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)