"""
Enhanced EHR Vendor Connectors for Vita Agents.

This module provides advanced connectors for major EHR vendors with vendor-specific
optimizations, enhanced authentication, real-time synchronization, and FHIR-compliant
data exchange capabilities.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from urllib.parse import urljoin
import json
import aiohttp
from aiohttp import ClientSession, ClientTimeout
import jwt
from cryptography.fernet import Fernet

from ..core.config import Settings
from ..core.exceptions import VitaAgentsError

logger = logging.getLogger(__name__)


class EHRVendor(str, Enum):
    """Supported EHR vendors."""
    EPIC = "epic"
    CERNER = "cerner"
    ALLSCRIPTS = "allscripts"
    ATHENAHEALTH = "athenahealth"
    ECLINICALWORKS = "eclinicalworks"
    NEXTGEN = "nextgen"
    PRACTICE_FUSION = "practice_fusion"
    MEDITECH = "meditech"


class AuthenticationType(str, Enum):
    """Authentication types supported by EHR systems."""
    OAUTH2 = "oauth2"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    CLIENT_CREDENTIALS = "client_credentials"
    SMART_ON_FHIR = "smart_on_fhir"
    CUSTOM = "custom"


class SyncMode(str, Enum):
    """Data synchronization modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    POLLING = "polling"
    WEBHOOK = "webhook"
    STREAM = "stream"


@dataclass
class EHRConnectionConfig:
    """Configuration for EHR connections."""
    vendor: EHRVendor
    base_url: str
    client_id: str
    client_secret: str
    auth_type: AuthenticationType
    scope: str = "patient/*.read"
    
    # Connection settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    max_concurrent_requests: int = 10
    
    # FHIR settings
    fhir_version: str = "R4"
    fhir_base_path: str = "/fhir"
    
    # Vendor-specific settings
    vendor_specific: Dict[str, Any] = field(default_factory=dict)
    
    # Security settings
    encrypt_credentials: bool = True
    validate_ssl: bool = True
    
    # Sync settings
    sync_mode: SyncMode = SyncMode.POLLING
    sync_interval: int = 300  # seconds
    bulk_export_enabled: bool = True


@dataclass
class EHRResponse:
    """Response from EHR API calls."""
    status_code: int
    data: Dict[str, Any]
    headers: Dict[str, str]
    response_time: float
    vendor: EHRVendor
    request_id: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        """Check if response was successful."""
        return 200 <= self.status_code < 300
    
    @property
    def is_fhir_bundle(self) -> bool:
        """Check if response is a FHIR Bundle."""
        return (self.data.get("resourceType") == "Bundle" 
                if isinstance(self.data, dict) else False)


@dataclass
class SyncResult:
    """Result of a synchronization operation."""
    vendor: EHRVendor
    sync_mode: SyncMode
    start_time: datetime
    end_time: datetime
    resources_processed: int
    resources_created: int
    resources_updated: int
    resources_failed: int
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> timedelta:
        """Duration of the sync operation."""
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Success rate of the sync operation."""
        if self.resources_processed == 0:
            return 0.0
        return (self.resources_created + self.resources_updated) / self.resources_processed


class EHRConnectorError(VitaAgentsError):
    """Base exception for EHR connector errors."""
    
    def __init__(self, message: str, vendor: EHRVendor, status_code: Optional[int] = None):
        super().__init__(message)
        self.vendor = vendor
        self.status_code = status_code


class EHRAuthenticationError(EHRConnectorError):
    """Authentication error with EHR system."""
    pass


class EHRRateLimitError(EHRConnectorError):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str, vendor: EHRVendor, retry_after: Optional[int] = None):
        super().__init__(message, vendor)
        self.retry_after = retry_after


class EHRConnectionError(EHRConnectorError):
    """Connection error with EHR system."""
    pass


class BaseEHRConnector(ABC):
    """
    Abstract base class for EHR vendor connectors.
    
    Provides common functionality for all EHR connectors including:
    - Authentication management
    - Rate limiting and retry logic
    - FHIR resource operations
    - Error handling and logging
    - Connection pooling
    """
    
    def __init__(self, config: EHRConnectionConfig):
        """Initialize the EHR connector."""
        self.config = config
        self.vendor = config.vendor
        self._session: Optional[ClientSession] = None
        self._auth_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._encryption_key: Optional[Fernet] = None
        self._request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Initialize encryption if enabled
        if config.encrypt_credentials:
            self._encryption_key = Fernet(Fernet.generate_key())
        
        # Vendor-specific initialization
        self._initialize_vendor_specific()
    
    @abstractmethod
    def _initialize_vendor_specific(self) -> None:
        """Initialize vendor-specific configurations."""
        pass
    
    @abstractmethod
    async def authenticate(self) -> str:
        """Authenticate with the EHR system and return access token."""
        pass
    
    @abstractmethod
    async def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> EHRResponse:
        """Make an authenticated request to the EHR system."""
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if connector is connected and authenticated."""
        return (self._session is not None and 
                self._auth_token is not None and 
                self._token_expires_at is not None and 
                datetime.utcnow() < self._token_expires_at)
    
    async def connect(self) -> None:
        """Establish connection to the EHR system."""
        if self._session is None:
            timeout = ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(
                limit=self.config.max_concurrent_requests,
                limit_per_host=self.config.max_concurrent_requests,
                ssl=self.config.validate_ssl
            )
            self._session = ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": "Vita-Agents/1.0"}
            )
        
        # Authenticate if not already authenticated or token expired
        if not self.is_connected:
            await self.authenticate()
        
        logger.info(f"Connected to {self.vendor} EHR system")
    
    async def disconnect(self) -> None:
        """Disconnect from the EHR system."""
        if self._session:
            await self._session.close()
            self._session = None
        
        self._auth_token = None
        self._token_expires_at = None
        
        logger.info(f"Disconnected from {self.vendor} EHR system")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if self._encryption_key:
            return self._encryption_key.encrypt(data.encode()).decode()
        return data
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if self._encryption_key:
            return self._encryption_key.decrypt(encrypted_data.encode()).decode()
        return encrypted_data
    
    async def _retry_request(self, request_func, max_retries: Optional[int] = None) -> EHRResponse:
        """Retry a request with exponential backoff."""
        max_retries = max_retries or self.config.max_retries
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await request_func()
            except EHRRateLimitError as e:
                if attempt == max_retries:
                    raise
                
                # Wait for rate limit reset
                wait_time = e.retry_after or (2 ** attempt * self.config.retry_delay)
                logger.warning(f"Rate limited by {self.vendor}, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                last_exception = e
                
            except EHRConnectionError as e:
                if attempt == max_retries:
                    raise
                
                wait_time = 2 ** attempt * self.config.retry_delay
                logger.warning(f"Connection error with {self.vendor}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
                last_exception = e
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
    
    async def get_patient(self, patient_id: str) -> EHRResponse:
        """Get a patient resource by ID."""
        endpoint = f"Patient/{patient_id}"
        return await self._make_authenticated_request("GET", endpoint)
    
    async def search_patients(
        self,
        family_name: Optional[str] = None,
        given_name: Optional[str] = None,
        birthdate: Optional[str] = None,
        identifier: Optional[str] = None,
        **kwargs
    ) -> EHRResponse:
        """Search for patients using FHIR search parameters."""
        params = {}
        
        if family_name:
            params["family"] = family_name
        if given_name:
            params["given"] = given_name
        if birthdate:
            params["birthdate"] = birthdate
        if identifier:
            params["identifier"] = identifier
        
        # Add any additional search parameters
        params.update(kwargs)
        
        return await self._make_authenticated_request("GET", "Patient", params=params)
    
    async def get_observations(
        self,
        patient_id: str,
        code: Optional[str] = None,
        category: Optional[str] = None,
        date_range: Optional[tuple] = None
    ) -> EHRResponse:
        """Get observations for a patient."""
        params = {"patient": patient_id}
        
        if code:
            params["code"] = code
        if category:
            params["category"] = category
        if date_range:
            start_date, end_date = date_range
            params["date"] = f"ge{start_date}&date=le{end_date}"
        
        return await self._make_authenticated_request("GET", "Observation", params=params)
    
    async def get_medications(self, patient_id: str) -> EHRResponse:
        """Get medications for a patient."""
        params = {"patient": patient_id}
        return await self._make_authenticated_request("GET", "MedicationRequest", params=params)
    
    async def get_conditions(self, patient_id: str) -> EHRResponse:
        """Get conditions for a patient."""
        params = {"patient": patient_id}
        return await self._make_authenticated_request("GET", "Condition", params=params)
    
    async def get_encounters(
        self,
        patient_id: str,
        status: Optional[str] = None,
        date_range: Optional[tuple] = None
    ) -> EHRResponse:
        """Get encounters for a patient."""
        params = {"patient": patient_id}
        
        if status:
            params["status"] = status
        if date_range:
            start_date, end_date = date_range
            params["date"] = f"ge{start_date}&date=le{end_date}"
        
        return await self._make_authenticated_request("GET", "Encounter", params=params)
    
    async def create_resource(self, resource_type: str, resource_data: Dict[str, Any]) -> EHRResponse:
        """Create a new FHIR resource."""
        return await self._make_authenticated_request(
            "POST",
            resource_type,
            data=resource_data,
            headers={"Content-Type": "application/fhir+json"}
        )
    
    async def update_resource(
        self,
        resource_type: str,
        resource_id: str,
        resource_data: Dict[str, Any]
    ) -> EHRResponse:
        """Update an existing FHIR resource."""
        endpoint = f"{resource_type}/{resource_id}"
        return await self._make_authenticated_request(
            "PUT",
            endpoint,
            data=resource_data,
            headers={"Content-Type": "application/fhir+json"}
        )
    
    async def delete_resource(self, resource_type: str, resource_id: str) -> EHRResponse:
        """Delete a FHIR resource."""
        endpoint = f"{resource_type}/{resource_id}"
        return await self._make_authenticated_request("DELETE", endpoint)
    
    async def bulk_export(
        self,
        resource_types: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        type_filter: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Perform bulk export of FHIR resources.
        
        Returns an async generator that yields FHIR resources.
        """
        if not self.config.bulk_export_enabled:
            raise EHRConnectorError(
                "Bulk export is not enabled for this connector",
                self.vendor
            )
        
        # Build export parameters
        params = {}
        if resource_types:
            params["_type"] = ",".join(resource_types)
        if since:
            params["_since"] = since.isoformat()
        if type_filter:
            params["_typeFilter"] = type_filter
        
        # Start bulk export operation
        export_response = await self._make_authenticated_request(
            "GET",
            "$export",
            params=params,
            headers={"Accept": "application/fhir+json", "Prefer": "respond-async"}
        )
        
        if export_response.status_code != 202:
            raise EHRConnectorError(
                f"Bulk export failed to start: {export_response.status_code}",
                self.vendor,
                export_response.status_code
            )
        
        # Get the content location for polling
        content_location = export_response.headers.get("Content-Location")
        if not content_location:
            raise EHRConnectorError(
                "No Content-Location header in bulk export response",
                self.vendor
            )
        
        # Poll for completion
        while True:
            status_response = await self._make_authenticated_request("GET", content_location)
            
            if status_response.status_code == 200:
                # Export completed, process the results
                export_manifest = status_response.data
                
                for output in export_manifest.get("output", []):
                    file_url = output.get("url")
                    if file_url:
                        # Download and yield resources from each file
                        async for resource in self._download_bulk_file(file_url):
                            yield resource
                break
                
            elif status_response.status_code == 202:
                # Still processing, wait and poll again
                retry_after = int(status_response.headers.get("Retry-After", "30"))
                await asyncio.sleep(retry_after)
                
            else:
                raise EHRConnectorError(
                    f"Bulk export status check failed: {status_response.status_code}",
                    self.vendor,
                    status_response.status_code
                )
    
    async def _download_bulk_file(self, file_url: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Download and parse NDJSON file from bulk export."""
        async with self._request_semaphore:
            async with self._session.get(file_url) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line.strip():
                            try:
                                resource = json.loads(line.decode('utf-8'))
                                yield resource
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse JSON line: {e}")
                else:
                    raise EHRConnectionError(
                        f"Failed to download bulk file: {response.status}",
                        self.vendor,
                        response.status
                    )
    
    async def sync_patient_data(
        self,
        patient_id: str,
        resource_types: Optional[List[str]] = None
    ) -> SyncResult:
        """
        Synchronize all data for a specific patient.
        
        Args:
            patient_id: The patient ID to sync
            resource_types: Optional list of resource types to sync
            
        Returns:
            SyncResult with details of the sync operation
        """
        start_time = datetime.utcnow()
        resources_processed = 0
        resources_created = 0
        resources_updated = 0
        resources_failed = 0
        errors = []
        
        # Default resource types to sync
        if resource_types is None:
            resource_types = [
                "Patient", "Observation", "MedicationRequest",
                "Condition", "Encounter", "AllergyIntolerance",
                "Procedure", "DiagnosticReport", "Immunization"
            ]
        
        try:
            for resource_type in resource_types:
                try:
                    if resource_type == "Patient":
                        response = await self.get_patient(patient_id)
                    else:
                        # Search for resources related to the patient
                        response = await self._make_authenticated_request(
                            "GET",
                            resource_type,
                            params={"patient": patient_id}
                        )
                    
                    if response.is_success and response.is_fhir_bundle:
                        bundle = response.data
                        entries = bundle.get("entry", [])
                        
                        for entry in entries:
                            resources_processed += 1
                            resource = entry.get("resource", {})
                            
                            # Here you would typically save the resource to your database
                            # For now, we'll just count it as processed
                            if resource:
                                resources_created += 1
                            else:
                                resources_failed += 1
                                
                    elif response.is_success and not response.is_fhir_bundle:
                        # Single resource response
                        resources_processed += 1
                        resources_created += 1
                        
                except Exception as e:
                    error_msg = f"Failed to sync {resource_type}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    resources_failed += 1
        
        except Exception as e:
            error_msg = f"Critical error during patient sync: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        end_time = datetime.utcnow()
        
        return SyncResult(
            vendor=self.vendor,
            sync_mode=self.config.sync_mode,
            start_time=start_time,
            end_time=end_time,
            resources_processed=resources_processed,
            resources_created=resources_created,
            resources_updated=resources_updated,
            resources_failed=resources_failed,
            errors=errors
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the EHR connection.
        
        Returns:
            Dictionary with health check results
        """
        health_info = {
            "vendor": self.vendor.value,
            "connected": False,
            "authenticated": False,
            "response_time": None,
            "fhir_version": self.config.fhir_version,
            "errors": []
        }
        
        try:
            # Check connection
            if self._session is None:
                await self.connect()
            
            health_info["connected"] = True
            
            # Check authentication
            health_info["authenticated"] = self.is_connected
            
            # Test basic FHIR operation
            start_time = datetime.utcnow()
            response = await self._make_authenticated_request("GET", "metadata")
            end_time = datetime.utcnow()
            
            health_info["response_time"] = (end_time - start_time).total_seconds()
            
            if response.is_success:
                capability_statement = response.data
                health_info["server_version"] = capability_statement.get("software", {}).get("version")
                health_info["supported_resources"] = [
                    res.get("type") for res in 
                    capability_statement.get("rest", [{}])[0].get("resource", [])
                ]
            else:
                health_info["errors"].append(f"Metadata request failed: {response.status_code}")
                
        except Exception as e:
            health_info["errors"].append(str(e))
            logger.error(f"Health check failed for {self.vendor}: {e}")
        
        return health_info