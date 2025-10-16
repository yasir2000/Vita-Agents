"""
Allscripts EHR Connector for Vita Agents.

This module provides enhanced connectivity to Allscripts EHR systems using
Allscripts FHIR APIs, including Sunrise Clinical Manager, Professional EHR,
and FollowMyHealth patient portal integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlencode
import json
import base64

from .base import (
    BaseEHRConnector,
    EHRConnectionConfig,
    EHRResponse,
    EHRVendor,
    AuthenticationType,
    EHRAuthenticationError,
    EHRRateLimitError,
    EHRConnectionError,
)

logger = logging.getLogger(__name__)


class AllscriptsConnector(BaseEHRConnector):
    """
    Enhanced Allscripts EHR connector with Allscripts-specific optimizations.
    
    Features:
    - Allscripts FHIR R4 support
    - Sunrise Clinical Manager integration
    - Professional EHR connectivity
    - FollowMyHealth patient portal support
    - Unity API integration
    - TouchWorks EHR support
    - Allscripts-specific clinical workflows
    """
    
    def __init__(self, config: EHRConnectionConfig):
        """Initialize Allscripts connector."""
        if config.vendor != EHRVendor.ALLSCRIPTS:
            raise ValueError("Allscripts connector requires Allscripts vendor configuration")
        
        super().__init__(config)
        
        # Allscripts-specific configuration
        self.allscripts_config = config.vendor_specific
        self.unity_enabled = self.allscripts_config.get("unity_enabled", True)
        self.touchworks_mode = self.allscripts_config.get("touchworks_mode", False)
        self.sunrise_mode = self.allscripts_config.get("sunrise_mode", False)
        self.app_name = self.allscripts_config.get("app_name", "VitaAgents")
        self.app_version = self.allscripts_config.get("app_version", "1.0")
        
        # Allscripts endpoints
        self.auth_url = self.allscripts_config.get("auth_url", f"{config.base_url}/authorization/connect/authorize")
        self.token_url = self.allscripts_config.get("token_url", f"{config.base_url}/authorization/connect/token")
        self.unity_url = self.allscripts_config.get("unity_url", f"{config.base_url}/Unity/UnityService.svc")
        self.fhir_base_url = urljoin(config.base_url, config.fhir_base_path)
        
        # Rate limiting (Allscripts is more restrictive)
        self.requests_per_minute = self.allscripts_config.get("requests_per_minute", 30)
        self.concurrent_limit = self.allscripts_config.get("concurrent_limit", 5)
        
        # Request tracking
        self._request_times = []
        self._magic_token = None  # Unity API magic token
    
    def _initialize_vendor_specific(self) -> None:
        """Initialize Allscripts-specific configurations."""
        # Set Allscripts-specific headers
        self.default_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"{self.app_name}/{self.app_version} (Vita-Agents)",
        }
        
        # Add product-specific headers
        if self.touchworks_mode:
            self.default_headers["X-Allscripts-Product"] = "TouchWorks"
        elif self.sunrise_mode:
            self.default_headers["X-Allscripts-Product"] = "Sunrise"
        else:
            self.default_headers["X-Allscripts-Product"] = "Professional"
        
        logger.info(f"Initialized Allscripts connector (Unity: {self.unity_enabled}, TouchWorks: {self.touchworks_mode})")
    
    async def authenticate(self) -> str:
        """
        Authenticate with Allscripts using OAuth 2.0 or Unity API.
        
        Allscripts supports:
        1. OAuth 2.0 Client Credentials
        2. Unity API with GetToken
        3. Basic Authentication (legacy)
        """
        if self.unity_enabled:
            return await self._authenticate_unity()
        elif self.config.auth_type == AuthenticationType.CLIENT_CREDENTIALS:
            return await self._authenticate_client_credentials()
        elif self.config.auth_type == AuthenticationType.BASIC_AUTH:
            return await self._authenticate_basic()
        else:
            raise EHRAuthenticationError(
                f"Unsupported authentication type for Allscripts: {self.config.auth_type}",
                EHRVendor.ALLSCRIPTS
            )
    
    async def _authenticate_unity(self) -> str:
        """Authenticate using Allscripts Unity API."""
        # Unity API GetToken method
        unity_request = {
            "Action": "GetToken",
            "AppUserID": self.config.client_id,
            "Appname": self.app_name,
            "PatientID": "",
            "Parameter1": self.config.client_secret,
            "Parameter2": "",
            "Parameter3": "",
            "Parameter4": "",
            "Parameter5": "",
            "Parameter6": "",
            "Data": ""
        }
        
        headers = {
            "Content-Type": "application/json",
            "SOAPAction": "http://tempuri.org/IUnityService/GetToken"
        }
        
        try:
            async with self._session.post(
                self.unity_url + "/GetToken",
                json=unity_request,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get("Success"):
                        self._magic_token = result.get("Token")
                        self._auth_token = f"Unity-{self._magic_token}"
                        self._token_expires_at = datetime.utcnow() + timedelta(hours=8)  # Unity tokens typically last 8 hours
                        
                        logger.info("Successfully authenticated with Allscripts Unity API")
                        return self._auth_token
                    else:
                        error_msg = result.get("Error", "Unknown Unity authentication error")
                        raise EHRAuthenticationError(
                            f"Unity authentication failed: {error_msg}",
                            EHRVendor.ALLSCRIPTS
                        )
                else:
                    error_text = await response.text()
                    raise EHRAuthenticationError(
                        f"Unity authentication request failed: {response.status} - {error_text}",
                        EHRVendor.ALLSCRIPTS,
                        response.status
                    )
        except Exception as e:
            raise EHRAuthenticationError(
                f"Unity authentication error: {str(e)}",
                EHRVendor.ALLSCRIPTS
            )
    
    async def _authenticate_client_credentials(self) -> str:
        """Authenticate using OAuth 2.0 client credentials."""
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "scope": self.config.scope
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        try:
            async with self._session.post(
                self.token_url,
                data=urlencode(auth_data),
                headers=headers
            ) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self._auth_token = token_data["access_token"]
                    
                    expires_in = token_data.get("expires_in", 3600)
                    self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 60)
                    
                    logger.info("Successfully authenticated with Allscripts OAuth")
                    return self._auth_token
                else:
                    error_text = await response.text()
                    raise EHRAuthenticationError(
                        f"Allscripts OAuth authentication failed: {response.status} - {error_text}",
                        EHRVendor.ALLSCRIPTS,
                        response.status
                    )
        except Exception as e:
            raise EHRAuthenticationError(
                f"Allscripts OAuth authentication error: {str(e)}",
                EHRVendor.ALLSCRIPTS
            )
    
    async def _authenticate_basic(self) -> str:
        """Authenticate using basic authentication (legacy)."""
        credentials = f"{self.config.client_id}:{self.config.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        self._auth_token = f"Basic {encoded_credentials}"
        self._token_expires_at = datetime.utcnow() + timedelta(hours=24)  # Basic auth doesn't expire
        
        logger.info("Using Allscripts basic authentication")
        return self._auth_token
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce Allscripts-specific rate limits."""
        now = datetime.utcnow()
        
        # Remove old request times (older than 1 minute)
        self._request_times = [
            req_time for req_time in self._request_times
            if (now - req_time).total_seconds() < 60
        ]
        
        # Check per-minute rate limit (Allscripts is more restrictive)
        if len(self._request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self._request_times[0]).total_seconds()
            if wait_time > 0:
                logger.warning(f"Allscripts rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self._request_times.append(now)
    
    async def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> EHRResponse:
        """Make an authenticated request to Allscripts API."""
        # Ensure we're connected and authenticated
        if not self.is_connected:
            await self.authenticate()
        
        # Check rate limits
        await self._check_rate_limit()
        
        # Build request headers
        request_headers = self.default_headers.copy()
        if headers:
            request_headers.update(headers)
        
        # Add authentication header
        if self.unity_enabled and self._magic_token:
            request_headers["Authorization"] = f"Bearer {self._auth_token}"
            request_headers["X-Unity-Token"] = self._magic_token
        else:
            request_headers["Authorization"] = self._auth_token
        
        # Build full URL
        if endpoint.startswith("http"):
            url = endpoint
        else:
            url = urljoin(self.fhir_base_url + "/", endpoint)
        
        # Add Allscripts-specific parameters
        if params is None:
            params = {}
        
        start_time = datetime.utcnow()
        
        async with self._request_semaphore:
            try:
                async with self._session.request(
                    method,
                    url,
                    headers=request_headers,
                    params=params,
                    json=data if data else None
                ) as response:
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", "120"))
                        raise EHRRateLimitError(
                            "Allscripts rate limit exceeded",
                            EHRVendor.ALLSCRIPTS,
                            retry_after
                        )
                    
                    # Handle authentication errors
                    if response.status == 401:
                        await self.authenticate()
                        raise EHRAuthenticationError(
                            "Allscripts authentication expired",
                            EHRVendor.ALLSCRIPTS,
                            response.status
                        )
                    
                    # Parse response
                    try:
                        response_data = await response.json()
                    except:
                        response_data = {"text": await response.text()}
                    
                    return EHRResponse(
                        status_code=response.status,
                        data=response_data,
                        headers=dict(response.headers),
                        response_time=response_time,
                        vendor=EHRVendor.ALLSCRIPTS,
                        request_id=response.headers.get("X-Request-ID")
                    )
                    
            except asyncio.TimeoutError:
                raise EHRConnectionError(
                    "Request to Allscripts timed out",
                    EHRVendor.ALLSCRIPTS
                )
            except Exception as e:
                raise EHRConnectionError(
                    f"Request to Allscripts failed: {str(e)}",
                    EHRVendor.ALLSCRIPTS
                )
    
    async def unity_api_call(
        self,
        action: str,
        patient_id: str = "",
        parameters: Optional[Dict[str, str]] = None,
        data: str = ""
    ) -> Dict[str, Any]:
        """
        Make a call to Allscripts Unity API.
        
        Unity API provides access to Allscripts-specific functionality.
        """
        if not self._magic_token:
            await self.authenticate()
        
        # Build Unity request
        unity_request = {
            "Action": action,
            "AppUserID": self.config.client_id,
            "Appname": self.app_name,
            "PatientID": patient_id,
            "Token": self._magic_token,
            "Parameter1": parameters.get("param1", "") if parameters else "",
            "Parameter2": parameters.get("param2", "") if parameters else "",
            "Parameter3": parameters.get("param3", "") if parameters else "",
            "Parameter4": parameters.get("param4", "") if parameters else "",
            "Parameter5": parameters.get("param5", "") if parameters else "",
            "Parameter6": parameters.get("param6", "") if parameters else "",
            "Data": data
        }
        
        headers = {
            "Content-Type": "application/json",
            "SOAPAction": f"http://tempuri.org/IUnityService/{action}"
        }
        
        async with self._session.post(
            f"{self.unity_url}/{action}",
            json=unity_request,
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise EHRConnectionError(
                    f"Unity API call failed: {response.status}",
                    EHRVendor.ALLSCRIPTS,
                    response.status
                )
    
    async def get_patient_demographics_unity(self, patient_id: str) -> Dict[str, Any]:
        """Get patient demographics using Unity API."""
        return await self.unity_api_call("GetPatient", patient_id)
    
    async def get_patient_problems_unity(self, patient_id: str) -> Dict[str, Any]:
        """Get patient problems/conditions using Unity API."""
        return await self.unity_api_call("GetProblemList", patient_id)
    
    async def get_patient_medications_unity(self, patient_id: str) -> Dict[str, Any]:
        """Get patient medications using Unity API."""
        return await self.unity_api_call("GetMedications", patient_id)
    
    async def get_patient_allergies_unity(self, patient_id: str) -> Dict[str, Any]:
        """Get patient allergies using Unity API."""
        return await self.unity_api_call("GetAllergies", patient_id)
    
    async def get_patient_vitals_unity(
        self,
        patient_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get patient vital signs using Unity API."""
        parameters = {}
        if start_date:
            parameters["param1"] = start_date
        if end_date:
            parameters["param2"] = end_date
        
        return await self.unity_api_call("GetVitals", patient_id, parameters)
    
    async def get_touchworks_documents(self, patient_id: str) -> EHRResponse:
        """
        Get TouchWorks clinical documents.
        
        TouchWorks is Allscripts' EHR platform.
        """
        params = {
            "patient": patient_id,
            "category": "clinical-note"
        }
        
        return await self._make_authenticated_request("GET", "DocumentReference", params=params)
    
    async def get_sunrise_data(self, patient_id: str) -> EHRResponse:
        """
        Get Sunrise Clinical Manager data.
        
        Sunrise is Allscripts' hospital information system.
        """
        if not self.sunrise_mode:
            raise EHRConnectionError(
                "Sunrise mode not enabled",
                EHRVendor.ALLSCRIPTS
            )
        
        params = {
            "patient": patient_id,
            "source": "sunrise"
        }
        
        return await self._make_authenticated_request("GET", "Patient", params=params)
    
    async def get_followmyhealth_data(self, patient_id: str) -> EHRResponse:
        """
        Get FollowMyHealth patient portal data.
        
        FollowMyHealth is Allscripts' patient engagement platform.
        """
        params = {
            "patient": patient_id,
            "category": "patient-portal"
        }
        
        return await self._make_authenticated_request("GET", "Communication", params=params)
    
    async def search_allscripts_encounters(
        self,
        patient_id: str,
        encounter_type: Optional[str] = None,
        provider_id: Optional[str] = None
    ) -> EHRResponse:
        """Search Allscripts encounters with enhanced filtering."""
        params = {"patient": patient_id}
        
        if encounter_type:
            params["type"] = encounter_type
        
        if provider_id:
            params["practitioner"] = provider_id
        
        return await self._make_authenticated_request("GET", "Encounter", params=params)
    
    async def get_allscripts_lab_results(
        self,
        patient_id: str,
        test_name: Optional[str] = None,
        date_range: Optional[tuple] = None
    ) -> EHRResponse:
        """Get Allscripts laboratory results."""
        params = {
            "patient": patient_id,
            "category": "laboratory"
        }
        
        if test_name:
            params["code"] = test_name
        
        if date_range:
            start_date, end_date = date_range
            params["date"] = f"ge{start_date}&date=le{end_date}"
        
        return await self._make_authenticated_request("GET", "Observation", params=params)
    
    async def create_allscripts_encounter_note(
        self,
        patient_id: str,
        encounter_id: str,
        note_content: str,
        provider_id: str,
        note_type: str = "progress-note"
    ) -> EHRResponse:
        """Create a clinical note in Allscripts."""
        document_data = {
            "resourceType": "DocumentReference",
            "status": "current",
            "type": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "11506-3",
                    "display": note_type
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "context": {
                "encounter": [{"reference": f"Encounter/{encounter_id}"}]
            },
            "author": [{"reference": f"Practitioner/{provider_id}"}],
            "content": [{
                "attachment": {
                    "contentType": "text/plain",
                    "data": base64.b64encode(note_content.encode()).decode()
                }
            }]
        }
        
        return await self.create_resource("DocumentReference", document_data)
    
    async def get_allscripts_clinical_summaries(self, patient_id: str) -> Dict[str, Any]:
        """
        Get comprehensive clinical summaries from Allscripts.
        
        Combines data from multiple sources for a complete patient view.
        """
        try:
            # Gather data from multiple endpoints
            patient_response = await self.get_patient(patient_id)
            conditions_response = await self.get_conditions(patient_id)
            medications_response = await self.get_medications(patient_id)
            
            # If Unity is enabled, get additional data
            unity_data = {}
            if self.unity_enabled:
                try:
                    unity_data = {
                        "demographics": await self.get_patient_demographics_unity(patient_id),
                        "problems": await self.get_patient_problems_unity(patient_id),
                        "medications": await self.get_patient_medications_unity(patient_id),
                        "allergies": await self.get_patient_allergies_unity(patient_id),
                        "vitals": await self.get_patient_vitals_unity(patient_id)
                    }
                except Exception as e:
                    logger.warning(f"Failed to get Unity data: {e}")
            
            # Compile comprehensive summary
            summary = {
                "patient_id": patient_id,
                "fhir_data": {
                    "patient": patient_response.data if patient_response.is_success else None,
                    "conditions": conditions_response.data if conditions_response.is_success else None,
                    "medications": medications_response.data if medications_response.is_success else None
                },
                "unity_data": unity_data,
                "generated_at": datetime.utcnow().isoformat(),
                "data_sources": ["FHIR"] + (["Unity"] if unity_data else [])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating clinical summary: {e}")
            raise EHRConnectionError(
                f"Failed to generate clinical summary: {str(e)}",
                EHRVendor.ALLSCRIPTS
            )
    
    async def validate_allscripts_data_integrity(self, patient_id: str) -> Dict[str, Any]:
        """
        Validate data integrity across Allscripts systems.
        
        Checks for consistency between FHIR and Unity API data.
        """
        integrity_report = {
            "patient_id": patient_id,
            "validation_time": datetime.utcnow().isoformat(),
            "inconsistencies": [],
            "warnings": [],
            "status": "valid"
        }
        
        try:
            # Get data from both FHIR and Unity
            fhir_patient = await self.get_patient(patient_id)
            
            if self.unity_enabled:
                unity_patient = await self.get_patient_demographics_unity(patient_id)
                
                # Compare basic demographics
                if fhir_patient.is_success and unity_patient.get("Success"):
                    fhir_data = fhir_patient.data
                    unity_data = unity_patient.get("GetPatientResult", {})
                    
                    # Check name consistency
                    fhir_name = fhir_data.get("name", [{}])[0]
                    unity_name = unity_data.get("Name", {})
                    
                    if (fhir_name.get("family") != unity_name.get("LastName") or
                        fhir_name.get("given", [{}])[0] != unity_name.get("FirstName")):
                        integrity_report["inconsistencies"].append({
                            "field": "patient_name",
                            "fhir_value": f"{fhir_name.get('given', [''])[0]} {fhir_name.get('family', '')}",
                            "unity_value": f"{unity_name.get('FirstName', '')} {unity_name.get('LastName', '')}"
                        })
                        integrity_report["status"] = "inconsistent"
            
            return integrity_report
            
        except Exception as e:
            integrity_report["status"] = "error"
            integrity_report["error"] = str(e)
            return integrity_report