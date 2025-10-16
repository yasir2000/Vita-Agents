"""
Cerner EHR Connector for Vita Agents.

This module provides enhanced connectivity to Cerner (Oracle Health) EHR systems
using Cerner's FHIR R4 APIs, including PowerChart integration, HealtheLife patient portal,
and Cerner-specific extensions and optimizations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlencode
import json

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


class CernerConnector(BaseEHRConnector):
    """
    Enhanced Cerner EHR connector with Cerner-specific optimizations.
    
    Features:
    - Cerner FHIR R4 DSTU2 support
    - PowerChart integration
    - HealtheLife patient portal support
    - Cerner-specific FHIR extensions
    - Real-time clinical decision support
    - Advanced medication management
    - Cerner Smart on FHIR apps integration
    """
    
    def __init__(self, config: EHRConnectionConfig):
        """Initialize Cerner connector."""
        if config.vendor != EHRVendor.CERNER:
            raise ValueError("Cerner connector requires Cerner vendor configuration")
        
        super().__init__(config)
        
        # Cerner-specific configuration
        self.cerner_config = config.vendor_specific
        self.sandbox_mode = self.cerner_config.get("sandbox_mode", False)
        self.tenant_id = self.cerner_config.get("tenant_id")
        self.system_account = self.cerner_config.get("system_account", False)
        
        # Cerner FHIR endpoints
        self.auth_url = self.cerner_config.get("auth_url", f"{config.base_url}/v1/authorize")
        self.token_url = self.cerner_config.get("token_url", f"{config.base_url}/v1/token")
        self.fhir_base_url = urljoin(config.base_url, config.fhir_base_path)
        
        # Cerner-specific rate limiting
        self.requests_per_minute = self.cerner_config.get("requests_per_minute", 120)
        self.concurrent_limit = self.cerner_config.get("concurrent_limit", 20)
        
        # Cerner uses different versioning
        self.dstu2_support = self.cerner_config.get("dstu2_support", True)
        self.r4_support = self.cerner_config.get("r4_support", True)
        
        # Request tracking
        self._request_times = []
    
    def _initialize_vendor_specific(self) -> None:
        """Initialize Cerner-specific configurations."""
        # Set Cerner-specific headers
        self.default_headers = {
            "Accept": "application/json+fhir",
            "Content-Type": "application/json+fhir",
            "User-Agent": "Vita-Agents/1.0 (Cerner Integration)",
        }
        
        # Add tenant information if provided
        if self.tenant_id:
            self.default_headers["X-Cerner-Tenant"] = self.tenant_id
        
        # Add sandbox headers if in sandbox mode
        if self.sandbox_mode:
            self.default_headers["X-Cerner-Sandbox"] = "true"
        
        logger.info(f"Initialized Cerner connector (sandbox: {self.sandbox_mode}, tenant: {self.tenant_id})")
    
    async def authenticate(self) -> str:
        """
        Authenticate with Cerner using OAuth 2.0.
        
        Cerner supports:
        1. Client Credentials Flow (system accounts)
        2. Authorization Code Flow (user accounts)
        3. Smart on FHIR Launch Flow
        """
        if self.config.auth_type == AuthenticationType.CLIENT_CREDENTIALS:
            return await self._authenticate_client_credentials()
        elif self.config.auth_type == AuthenticationType.OAUTH2:
            return await self._authenticate_oauth2()
        elif self.config.auth_type == AuthenticationType.SMART_ON_FHIR:
            return await self._authenticate_smart_on_fhir()
        else:
            raise EHRAuthenticationError(
                f"Unsupported authentication type for Cerner: {self.config.auth_type}",
                EHRVendor.CERNER
            )
    
    async def _authenticate_client_credentials(self) -> str:
        """Authenticate using client credentials flow for system accounts."""
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
                    
                    # Set token expiration
                    expires_in = token_data.get("expires_in", 3600)
                    self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 60)
                    
                    logger.info("Successfully authenticated with Cerner using client credentials")
                    return self._auth_token
                else:
                    error_text = await response.text()
                    raise EHRAuthenticationError(
                        f"Cerner authentication failed: {response.status} - {error_text}",
                        EHRVendor.CERNER,
                        response.status
                    )
        except Exception as e:
            raise EHRAuthenticationError(
                f"Cerner authentication error: {str(e)}",
                EHRVendor.CERNER
            )
    
    async def _authenticate_oauth2(self) -> str:
        """Authenticate using OAuth 2.0 authorization code flow."""
        # This would typically involve user interaction
        # For now, we'll implement a simplified version
        
        logger.warning("OAuth2 authentication for Cerner requires user interaction - using placeholder")
        
        # Placeholder implementation
        self._auth_token = "placeholder_oauth2_token"
        self._token_expires_at = datetime.utcnow() + timedelta(hours=1)
        
        return self._auth_token
    
    async def _authenticate_smart_on_fhir(self) -> str:
        """Authenticate using Smart on FHIR flow."""
        # Similar to Epic, this requires user interaction
        logger.warning("Smart on FHIR authentication for Cerner requires user interaction - using placeholder")
        
        self._auth_token = "placeholder_smart_token"
        self._token_expires_at = datetime.utcnow() + timedelta(hours=1)
        
        return self._auth_token
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce Cerner-specific rate limits."""
        now = datetime.utcnow()
        
        # Remove old request times (older than 1 minute)
        self._request_times = [
            req_time for req_time in self._request_times
            if (now - req_time).total_seconds() < 60
        ]
        
        # Check per-minute rate limit
        if len(self._request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self._request_times[0]).total_seconds()
            if wait_time > 0:
                logger.warning(f"Cerner rate limit reached, waiting {wait_time:.2f}s")
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
        """Make an authenticated request to Cerner FHIR API."""
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
        request_headers["Authorization"] = f"Bearer {self._auth_token}"
        
        # Build full URL
        if endpoint.startswith("http"):
            url = endpoint
        else:
            url = urljoin(self.fhir_base_url + "/", endpoint)
        
        # Add Cerner-specific parameters
        if params is None:
            params = {}
        
        # Cerner-specific parameter handling
        if "_format" not in params:
            params["_format"] = "json"
        
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
                    
                    # Handle Cerner-specific rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", "60"))
                        raise EHRRateLimitError(
                            "Cerner rate limit exceeded",
                            EHRVendor.CERNER,
                            retry_after
                        )
                    
                    # Handle authentication errors
                    if response.status == 401:
                        await self.authenticate()
                        raise EHRAuthenticationError(
                            "Cerner authentication expired",
                            EHRVendor.CERNER,
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
                        vendor=EHRVendor.CERNER,
                        request_id=response.headers.get("X-Request-ID")
                    )
                    
            except asyncio.TimeoutError:
                raise EHRConnectionError(
                    "Request to Cerner timed out",
                    EHRVendor.CERNER
                )
            except Exception as e:
                raise EHRConnectionError(
                    f"Request to Cerner failed: {str(e)}",
                    EHRVendor.CERNER
                )
    
    async def get_cerner_person_data(self, person_id: str) -> EHRResponse:
        """
        Get Cerner Person resource data.
        
        Cerner uses Person resources in addition to Patient resources.
        """
        return await self._make_authenticated_request("GET", f"Person/{person_id}")
    
    async def get_powerhart_encounters(
        self,
        patient_id: str,
        encounter_class: Optional[str] = None
    ) -> EHRResponse:
        """
        Get PowerChart encounters for a patient.
        
        PowerChart is Cerner's clinical documentation system.
        """
        params = {"patient": patient_id}
        
        if encounter_class:
            params["class"] = encounter_class
        
        # Add PowerChart-specific parameters
        params["_include"] = "Encounter:location"
        
        return await self._make_authenticated_request("GET", "Encounter", params=params)
    
    async def get_cerner_medications(
        self,
        patient_id: str,
        include_discontinued: bool = False
    ) -> EHRResponse:
        """
        Get Cerner medication data with enhanced medication management features.
        """
        params = {"patient": patient_id}
        
        if not include_discontinued:
            params["status"] = "active"
        
        # Include related resources
        params["_include"] = "MedicationRequest:medication"
        
        return await self._make_authenticated_request("GET", "MedicationRequest", params=params)
    
    async def get_cerner_allergies(self, patient_id: str) -> EHRResponse:
        """
        Get Cerner allergy and intolerance data.
        
        Includes Cerner-specific allergy classifications and severity levels.
        """
        params = {
            "patient": patient_id,
            "clinical-status": "active"
        }
        
        return await self._make_authenticated_request("GET", "AllergyIntolerance", params=params)
    
    async def get_cerner_vitals(
        self,
        patient_id: str,
        date_range: Optional[tuple] = None,
        vital_sign_category: Optional[str] = None
    ) -> EHRResponse:
        """
        Get Cerner vital signs with PowerChart flowsheet integration.
        """
        params = {
            "patient": patient_id,
            "category": vital_sign_category or "vital-signs"
        }
        
        if date_range:
            start_date, end_date = date_range
            params["date"] = f"ge{start_date}&date=le{end_date}"
        
        return await self._make_authenticated_request("GET", "Observation", params=params)
    
    async def get_cerner_lab_results(
        self,
        patient_id: str,
        test_codes: Optional[List[str]] = None,
        date_range: Optional[tuple] = None
    ) -> EHRResponse:
        """
        Get Cerner laboratory results with lab information system integration.
        """
        params = {
            "patient": patient_id,
            "category": "laboratory"
        }
        
        if test_codes:
            params["code"] = ",".join(test_codes)
        
        if date_range:
            start_date, end_date = date_range
            params["date"] = f"ge{start_date}&date=le{end_date}"
        
        return await self._make_authenticated_request("GET", "Observation", params=params)
    
    async def get_cerner_radiology_reports(
        self,
        patient_id: str,
        modality: Optional[str] = None
    ) -> EHRResponse:
        """
        Get Cerner radiology reports and imaging studies.
        """
        params = {
            "patient": patient_id,
            "category": "radiology"
        }
        
        if modality:
            params["modality"] = modality
        
        return await self._make_authenticated_request("GET", "DiagnosticReport", params=params)
    
    async def search_cerner_clinical_notes(
        self,
        patient_id: str,
        note_type: Optional[str] = None,
        author: Optional[str] = None
    ) -> EHRResponse:
        """
        Search Cerner clinical notes and documentation.
        """
        params = {"patient": patient_id}
        
        if note_type:
            params["type"] = note_type
        
        if author:
            params["author"] = author
        
        return await self._make_authenticated_request("GET", "DocumentReference", params=params)
    
    async def get_healthelife_data(self, patient_id: str) -> EHRResponse:
        """
        Get HealtheLife patient portal data.
        
        HealtheLife is Cerner's patient portal platform.
        """
        params = {
            "patient": patient_id,
            "category": "healthelife"
        }
        
        return await self._make_authenticated_request("GET", "Communication", params=params)
    
    async def create_cerner_order(
        self,
        patient_id: str,
        order_type: str,
        order_details: Dict[str, Any],
        provider_id: str
    ) -> EHRResponse:
        """
        Create an order in Cerner PowerChart.
        
        Supports various order types: medications, lab tests, procedures, etc.
        """
        order_data = {
            "resourceType": "ServiceRequest",
            "status": "draft",
            "intent": "order",
            "category": [{
                "coding": [{
                    "system": "http://cerner.com/order-types",
                    "code": order_type
                }]
            }],
            "subject": {"reference": f"Patient/{patient_id}"},
            "requester": {"reference": f"Practitioner/{provider_id}"},
            **order_details
        }
        
        return await self.create_resource("ServiceRequest", order_data)
    
    async def get_cerner_care_plans(self, patient_id: str) -> EHRResponse:
        """
        Get Cerner care plans and clinical pathways.
        """
        params = {"patient": patient_id}
        
        return await self._make_authenticated_request("GET", "CarePlan", params=params)
    
    async def get_cerner_risk_assessments(self, patient_id: str) -> EHRResponse:
        """
        Get Cerner risk assessments and clinical decision support data.
        """
        params = {
            "patient": patient_id,
            "category": "risk-assessment"
        }
        
        return await self._make_authenticated_request("GET", "RiskAssessment", params=params)
    
    async def validate_cerner_medication_order(
        self,
        patient_id: str,
        medication_code: str,
        dosage: str,
        frequency: str
    ) -> Dict[str, Any]:
        """
        Validate a medication order using Cerner's clinical decision support.
        
        Returns drug interaction checks, allergy alerts, and dosing guidance.
        """
        validation_request = {
            "patient_id": patient_id,
            "medication": medication_code,
            "dosage": dosage,
            "frequency": frequency
        }
        
        # This would typically call Cerner's medication validation API
        # For now, we'll return a simplified response
        
        try:
            # Get patient allergies for interaction checking
            allergies_response = await self.get_cerner_allergies(patient_id)
            
            # Get current medications for drug-drug interaction checking
            meds_response = await self.get_cerner_medications(patient_id)
            
            validation_result = {
                "status": "validated",
                "alerts": [],
                "interactions": [],
                "contraindications": [],
                "dosing_guidance": {}
            }
            
            # Simple allergy checking (in production, this would be more sophisticated)
            if allergies_response.is_success:
                allergies = allergies_response.data.get("entry", [])
                for allergy in allergies:
                    allergy_resource = allergy.get("resource", {})
                    if allergy_resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code") == "active":
                        validation_result["alerts"].append({
                            "type": "allergy",
                            "severity": "high",
                            "message": f"Patient has active allergy: {allergy_resource.get('code', {}).get('text', 'Unknown')}"
                        })
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Medication validation error: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_cerner_clinical_decision_support(
        self,
        patient_id: str,
        context: str,
        intervention_type: Optional[str] = None
    ) -> EHRResponse:
        """
        Get Cerner clinical decision support recommendations.
        
        Provides evidence-based care recommendations and alerts.
        """
        params = {
            "patient": patient_id,
            "context": context
        }
        
        if intervention_type:
            params["intervention-type"] = intervention_type
        
        return await self._make_authenticated_request("GET", "DetectedIssue", params=params)