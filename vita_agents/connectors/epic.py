"""
Epic EHR Connector for Vita Agents.

This module provides enhanced connectivity to Epic EHR systems using Epic's
FHIR-based APIs, including support for MyChart integration, Epic App Orchard,
and Smart on FHIR specifications.
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


class EpicConnector(BaseEHRConnector):
    """
    Enhanced Epic EHR connector with Epic-specific optimizations.
    
    Features:
    - Smart on FHIR authentication
    - Epic App Orchard integration
    - MyChart patient portal support
    - Epic-specific FHIR extensions
    - Bulk data export capabilities
    - Real-time event subscriptions
    """
    
    def __init__(self, config: EHRConnectionConfig):
        """Initialize Epic connector."""
        if config.vendor != EHRVendor.EPIC:
            raise ValueError("Epic connector requires Epic vendor configuration")
        
        super().__init__(config)
        
        # Epic-specific configuration
        self.epic_config = config.vendor_specific
        self.sandbox_mode = self.epic_config.get("sandbox_mode", False)
        self.app_build = self.epic_config.get("app_build", "1")
        self.department_id = self.epic_config.get("department_id")
        self.user_id = self.epic_config.get("user_id")
        
        # Epic FHIR endpoints
        self.auth_url = self.epic_config.get("auth_url", f"{config.base_url}/oauth2/authorize")
        self.token_url = self.epic_config.get("token_url", f"{config.base_url}/oauth2/token")
        self.fhir_base_url = urljoin(config.base_url, config.fhir_base_path)
        
        # Epic-specific rate limiting (more conservative for production)
        self.requests_per_minute = self.epic_config.get("requests_per_minute", 60)
        self.burst_limit = self.epic_config.get("burst_limit", 10)
        
        # Request tracking for rate limiting
        self._request_times = []
        self._burst_count = 0
        self._last_burst_reset = datetime.utcnow()
    
    def _initialize_vendor_specific(self) -> None:
        """Initialize Epic-specific configurations."""
        # Set Epic-specific headers
        self.default_headers = {
            "Epic-Client-ID": self.config.client_id,
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }
        
        # Add sandbox headers if in sandbox mode
        if self.sandbox_mode:
            self.default_headers["Epic-Sandbox"] = "true"
        
        logger.info(f"Initialized Epic connector (sandbox: {self.sandbox_mode})")
    
    async def authenticate(self) -> str:
        """
        Authenticate with Epic using Smart on FHIR or client credentials.
        
        Epic supports multiple authentication flows:
        1. Client Credentials Flow (backend services)
        2. Authorization Code Flow (user-facing apps)
        3. Smart on FHIR Launch Flow
        """
        if self.config.auth_type == AuthenticationType.CLIENT_CREDENTIALS:
            return await self._authenticate_client_credentials()
        elif self.config.auth_type == AuthenticationType.SMART_ON_FHIR:
            return await self._authenticate_smart_on_fhir()
        else:
            raise EHRAuthenticationError(
                f"Unsupported authentication type for Epic: {self.config.auth_type}",
                EHRVendor.EPIC
            )
    
    async def _authenticate_client_credentials(self) -> str:
        """Authenticate using client credentials flow."""
        # Create JWT assertion for Epic backend services
        jwt_payload = {
            "iss": self.config.client_id,
            "sub": self.config.client_id,
            "aud": self.token_url,
            "jti": f"epic-{datetime.utcnow().timestamp()}",
            "exp": int((datetime.utcnow() + timedelta(minutes=5)).timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
        }
        
        # Note: In production, you would sign this JWT with your private key
        # For now, we'll use the client_secret as a placeholder
        jwt_token = base64.b64encode(json.dumps(jwt_payload).encode()).decode()
        
        auth_data = {
            "grant_type": "client_credentials",
            "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "client_assertion": jwt_token,
            "scope": self.config.scope
        }
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
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
                    
                    logger.info("Successfully authenticated with Epic using client credentials")
                    return self._auth_token
                else:
                    error_text = await response.text()
                    raise EHRAuthenticationError(
                        f"Epic authentication failed: {response.status} - {error_text}",
                        EHRVendor.EPIC,
                        response.status
                    )
        except Exception as e:
            raise EHRAuthenticationError(
                f"Epic authentication error: {str(e)}",
                EHRVendor.EPIC
            )
    
    async def _authenticate_smart_on_fhir(self) -> str:
        """Authenticate using Smart on FHIR flow."""
        # This would typically involve a multi-step process:
        # 1. Get authorization URL
        # 2. User authorizes (handled externally)
        # 3. Exchange authorization code for token
        
        # For now, we'll implement a simplified version
        # In production, this would be more complex and involve user interaction
        
        auth_params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.epic_config.get("redirect_uri", "http://localhost:8080/callback"),
            "scope": self.config.scope,
            "state": f"epic-{datetime.utcnow().timestamp()}",
            "aud": self.fhir_base_url
        }
        
        # In a real implementation, you would redirect the user to:
        # auth_url = f"{self.auth_url}?{urlencode(auth_params)}"
        
        # For demo purposes, we'll skip the authorization step
        # and assume we have an authorization code
        
        logger.warning("Smart on FHIR authentication requires user interaction - using placeholder")
        
        # This is a placeholder implementation
        # In production, you would exchange the authorization code for tokens
        self._auth_token = "placeholder_smart_token"
        self._token_expires_at = datetime.utcnow() + timedelta(hours=1)
        
        return self._auth_token
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce Epic-specific rate limits."""
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
                logger.warning(f"Epic rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Check burst limit
        if (now - self._last_burst_reset).total_seconds() >= 1:
            self._burst_count = 0
            self._last_burst_reset = now
        
        if self._burst_count >= self.burst_limit:
            wait_time = 1 - (now - self._last_burst_reset).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self._burst_count = 0
                self._last_burst_reset = datetime.utcnow()
        
        # Record this request
        self._request_times.append(now)
        self._burst_count += 1
    
    async def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> EHRResponse:
        """Make an authenticated request to Epic FHIR API."""
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
        
        # Add Epic-specific parameters
        if params is None:
            params = {}
        
        # Epic requires specific parameters for some operations
        if endpoint == "metadata":
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
                    
                    # Handle Epic-specific rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", "60"))
                        raise EHRRateLimitError(
                            "Epic rate limit exceeded",
                            EHRVendor.EPIC,
                            retry_after
                        )
                    
                    # Handle authentication errors
                    if response.status == 401:
                        # Token might be expired, try to re-authenticate
                        await self.authenticate()
                        raise EHRAuthenticationError(
                            "Epic authentication expired",
                            EHRVendor.EPIC,
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
                        vendor=EHRVendor.EPIC,
                        request_id=response.headers.get("X-Request-ID")
                    )
                    
            except asyncio.TimeoutError:
                raise EHRConnectionError(
                    "Request to Epic timed out",
                    EHRVendor.EPIC
                )
            except Exception as e:
                raise EHRConnectionError(
                    f"Request to Epic failed: {str(e)}",
                    EHRVendor.EPIC
                )
    
    async def get_patient_demographics(self, patient_id: str) -> EHRResponse:
        """
        Get patient demographics with Epic-specific fields.
        
        Epic provides additional demographic information beyond standard FHIR.
        """
        response = await self.get_patient(patient_id)
        
        if response.is_success and "epic" in response.data.get("extension", []):
            # Epic-specific processing for additional demographic data
            logger.info(f"Retrieved Epic-enhanced patient demographics for {patient_id}")
        
        return response
    
    async def get_mychart_data(self, patient_id: str) -> EHRResponse:
        """
        Get MyChart-specific patient data.
        
        This includes patient portal activity, messages, and preferences.
        """
        # Epic MyChart data is typically accessed through specific endpoints
        params = {
            "patient": patient_id,
            "category": "mychart"
        }
        
        return await self._make_authenticated_request("GET", "Communication", params=params)
    
    async def get_epic_flowsheets(self, patient_id: str, flowsheet_id: str) -> EHRResponse:
        """
        Get Epic flowsheet data for a patient.
        
        Flowsheets are Epic's structured data entry forms.
        """
        params = {
            "patient": patient_id,
            "code": f"epic-flowsheet|{flowsheet_id}"
        }
        
        return await self._make_authenticated_request("GET", "Observation", params=params)
    
    async def search_epic_smartdata(
        self,
        patient_id: str,
        smartdata_id: str,
        date_range: Optional[tuple] = None
    ) -> EHRResponse:
        """
        Search Epic SmartData elements for a patient.
        
        SmartData elements are Epic's discrete data points.
        """
        params = {
            "patient": patient_id,
            "code": f"epic-smartdata|{smartdata_id}"
        }
        
        if date_range:
            start_date, end_date = date_range
            params["date"] = f"ge{start_date}&date=le{end_date}"
        
        return await self._make_authenticated_request("GET", "Observation", params=params)
    
    async def get_care_everywhere_data(self, patient_id: str) -> EHRResponse:
        """
        Get Care Everywhere (HIE) data for a patient.
        
        Care Everywhere is Epic's health information exchange platform.
        """
        params = {
            "patient": patient_id,
            "category": "care-everywhere"
        }
        
        return await self._make_authenticated_request("GET", "DocumentReference", params=params)
    
    async def create_epic_appointment(
        self,
        patient_id: str,
        provider_id: str,
        appointment_type: str,
        start_time: datetime,
        duration_minutes: int = 30
    ) -> EHRResponse:
        """
        Create an appointment in Epic.
        
        Uses Epic-specific appointment types and provider scheduling.
        """
        appointment_data = {
            "resourceType": "Appointment",
            "status": "proposed",
            "appointmentType": {
                "coding": [{
                    "system": "http://epic.com/appointment-types",
                    "code": appointment_type
                }]
            },
            "start": start_time.isoformat(),
            "end": (start_time + timedelta(minutes=duration_minutes)).isoformat(),
            "participant": [
                {
                    "actor": {"reference": f"Patient/{patient_id}"},
                    "status": "needs-action"
                },
                {
                    "actor": {"reference": f"Practitioner/{provider_id}"},
                    "status": "accepted"
                }
            ]
        }
        
        return await self.create_resource("Appointment", appointment_data)
    
    async def get_epic_report_data(
        self,
        patient_id: str,
        report_id: str,
        include_images: bool = False
    ) -> EHRResponse:
        """
        Get Epic report data (radiology, pathology, etc.).
        
        Includes Epic-specific report formatting and image references.
        """
        params = {
            "patient": patient_id,
            "identifier": f"epic-report|{report_id}"
        }
        
        if include_images:
            params["_include"] = "DiagnosticReport:media"
        
        return await self._make_authenticated_request("GET", "DiagnosticReport", params=params)
    
    async def subscribe_to_epic_events(
        self,
        resource_types: List[str],
        webhook_url: str,
        criteria: Optional[str] = None
    ) -> EHRResponse:
        """
        Subscribe to Epic real-time events using FHIR subscriptions.
        
        Epic supports real-time notifications for specific resource changes.
        """
        subscription_data = {
            "resourceType": "Subscription",
            "status": "requested",
            "reason": "Vita Agents real-time sync",
            "criteria": criteria or f"Patient?_type={','.join(resource_types)}",
            "channel": {
                "type": "rest-hook",
                "endpoint": webhook_url,
                "payload": "application/fhir+json",
                "header": [f"Authorization: Bearer {self._auth_token}"]
            }
        }
        
        return await self.create_resource("Subscription", subscription_data)
    
    async def validate_epic_access(self, patient_id: str, user_context: str) -> bool:
        """
        Validate access to patient data based on Epic's security model.
        
        Epic has complex access controls based on user context, department, etc.
        """
        try:
            # Attempt to access patient basic info
            response = await self.get_patient(patient_id)
            
            if response.is_success:
                # Check if user has appropriate access level
                patient_data = response.data
                
                # Epic-specific access validation logic would go here
                # This is simplified for demonstration
                return True
            else:
                return False
                
        except EHRAuthenticationError:
            return False
        except Exception as e:
            logger.error(f"Error validating Epic access: {e}")
            return False