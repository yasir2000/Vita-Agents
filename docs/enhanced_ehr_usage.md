# Enhanced EHR Connector System Usage Guide

## Overview

The Enhanced EHR Connector System provides enterprise-grade connectivity to major Electronic Health Record (EHR) systems including Epic, Cerner, and Allscripts. This system is part of Phase 2 of the Vita Agents roadmap and includes advanced features like connection pooling, health monitoring, and vendor-specific optimizations.

## Features

### 🏥 Multi-Vendor Support
- **Epic**: Smart on FHIR, MyChart integration, Epic App Orchard
- **Cerner**: PowerChart integration, HealtheLife portal  
- **Allscripts**: Unity API, TouchWorks EHR, Sunrise Clinical Manager

### 🔗 Connection Management
- Connection pooling for optimal resource usage
- Automatic failover and load balancing
- Health monitoring with background checks
- Intelligent routing across multiple systems

### 🔐 Security & Authentication
- OAuth 2.0 and JWT token support
- Smart on FHIR authentication
- Client credentials and basic auth
- Secure credential management

### ⚡ Performance & Reliability
- Async/await operations for high performance
- Rate limiting and throttling
- Automatic retry logic with exponential backoff
- Error handling and recovery mechanisms

## Quick Start

### 1. Basic Usage

```python
import asyncio
from vita_agents.agents.enhanced_ehr_agent import EnhancedEHRAgent
from vita_agents.connectors import EHRVendor, EHRConnectionConfig, AuthenticationType

async def main():
    # Create enhanced EHR agent
    agent = EnhancedEHRAgent()
    await agent.initialize()
    
    # Add EHR system configuration
    config = EnhancedEHRConnectionConfig(
        config_id="epic_sandbox",
        vendor=EHRVendor.EPIC,
        base_url="https://fhir.epic.com/interconnect-fhir-oauth",
        client_id="your_client_id",
        client_secret="your_client_secret",
        auth_type=AuthenticationType.CLIENT_CREDENTIALS,
        scope="patient/*.read"
    )
    
    await agent.add_ehr_system(config)
    
    # Get patient data
    patient_data = await agent.get_patient_data(
        config_id="epic_sandbox",
        patient_id="12345",
        resource_types=["Patient", "Observation", "MedicationRequest"]
    )
    
    print(f"Retrieved patient data: {patient_data}")
    
    await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Multi-System Synchronization

```python
from vita_agents.agents.enhanced_ehr_agent import MultiSystemSyncRequest

async def sync_patient_across_systems():
    agent = EnhancedEHRAgent()
    await agent.initialize()
    
    # Configure multiple systems
    # ... add configurations for epic_prod, cerner_prod, allscripts_prod
    
    # Sync patient across all systems
    sync_request = MultiSystemSyncRequest(
        patient_identifier="12345",
        identifier_type="MRN",
        system_configs=["epic_prod", "cerner_prod", "allscripts_prod"],
        resource_types=["Patient", "Observation", "MedicationRequest"],
        include_historical=True,
        resolve_conflicts=True
    )
    
    summary = await agent.sync_patient_across_systems(sync_request)
    
    print(f"Sync completed:")
    print(f"  Successful systems: {summary.successful_systems}")
    print(f"  Failed systems: {summary.failed_systems}")
    print(f"  Conflicts detected: {len(summary.conflicts_detected)}")
    
    if summary.harmonized_data:
        print("  Harmonized data available")
    
    await agent.shutdown()
```

### 3. Health Monitoring

```python
async def monitor_system_health():
    agent = EnhancedEHRAgent()
    await agent.initialize()
    
    # Get health status for all systems
    health_status = await agent.get_system_health_status()
    
    for status in health_status:
        print(f"System: {status.config_id}")
        print(f"  Vendor: {status.vendor}")
        print(f"  Connected: {'✅' if status.is_connected else '❌'}")
        print(f"  Healthy: {'✅' if status.is_healthy else '❌'}")
        print(f"  Response Time: {status.response_time}ms")
        print(f"  Uptime: {status.uptime_percentage}%")
        print(f"  Error Count: {status.error_count}")
        print()
    
    await agent.shutdown()
```

### 4. Bulk Data Export

```python
async def bulk_export_example():
    agent = EnhancedEHRAgent()
    await agent.initialize()
    
    # Perform bulk export
    async for resource in agent.bulk_export_patient_data(
        config_id="epic_prod",
        resource_types=["Patient", "Observation"],
        since=datetime(2024, 1, 1),
        patient_filter="active=true"
    ):
        # Process each resource
        print(f"Exported resource: {resource.get('resourceType')}")
    
    await agent.shutdown()
```

## Configuration

### Environment Variables

Set these environment variables for secure credential management:

```bash
# Epic Configuration
EPIC_SANDBOX_CLIENT_ID=your_epic_client_id
EPIC_SANDBOX_CLIENT_SECRET=your_epic_client_secret
EPIC_SANDBOX_BASE_URL=https://fhir.epic.com/interconnect-fhir-oauth

# Cerner Configuration  
CERNER_SANDBOX_CLIENT_ID=your_cerner_client_id
CERNER_SANDBOX_CLIENT_SECRET=your_cerner_client_secret
CERNER_SANDBOX_BASE_URL=https://fhir-open.cerner.com/r4

# Allscripts Configuration
ALLSCRIPTS_CLIENT_ID=your_allscripts_client_id
ALLSCRIPTS_CLIENT_SECRET=your_allscripts_client_secret
ALLSCRIPTS_BASE_URL=https://your-allscripts-instance.com
```

### Settings File Configuration

Add to your settings.py or configuration file:

```python
EHR_SYSTEMS = {
    "epic_sandbox": {
        "vendor": "epic",
        "base_url": "https://fhir.epic.com/interconnect-fhir-oauth",
        "client_id": os.getenv("EPIC_SANDBOX_CLIENT_ID"),
        "client_secret": os.getenv("EPIC_SANDBOX_CLIENT_SECRET"),
        "auth_type": "client_credentials",
        "scope": "patient/*.read",
        "fhir_version": "R4",
        "timeout": 30,
        "max_retries": 3,
        "vendor_specific": {
            "enable_smart_on_fhir": True,
            "mychart_integration": True
        }
    },
    "cerner_prod": {
        "vendor": "cerner",
        "base_url": "https://fhir-open.cerner.com/r4",
        "client_id": os.getenv("CERNER_PROD_CLIENT_ID"),
        "client_secret": os.getenv("CERNER_PROD_CLIENT_SECRET"),
        "auth_type": "client_credentials",
        "scope": "patient/*.read",
        "vendor_specific": {
            "enable_powerchart": True,
            "healthelife_portal": True
        }
    }
}
```

## Vendor-Specific Features

### Epic Connector
- **Smart on FHIR**: OAuth 2.0 and JWT authentication
- **MyChart Integration**: Patient portal data access
- **Epic App Orchard**: Certified app support
- **Epic-specific FHIR Extensions**: Enhanced data access

### Cerner Connector  
- **PowerChart Integration**: Clinical workflow support
- **HealtheLife Portal**: Patient engagement platform
- **Clinical Decision Support**: Real-time alerts and recommendations
- **Medication Management**: Drug interaction checking

### Allscripts Connector
- **Unity API**: Comprehensive EHR access
- **TouchWorks EHR**: Clinical documentation system
- **Sunrise Clinical Manager**: Hospital information system
- **FollowMyHealth Portal**: Patient health records

## Error Handling

The system provides comprehensive error handling:

```python
from vita_agents.connectors import EHRConnectorError

try:
    patient_data = await agent.get_patient_data("epic_prod", "12345")
except EHRConnectorError as e:
    print(f"EHR Error: {e.message}")
    print(f"Vendor: {e.vendor}")
    print(f"Status Code: {e.status_code}")
    
    # Handle specific error types
    if e.is_authentication_error():
        # Refresh tokens or re-authenticate
        pass
    elif e.is_rate_limit_error():
        # Wait and retry
        pass
    elif e.is_timeout_error():
        # Use alternate system or cached data
        pass
```

## Performance Optimization

### Connection Pooling

The system automatically manages connection pools:

```python
# Connection pool is managed automatically
# No manual connection management needed
async with ehr_factory.get_connector("epic_prod") as connector:
    # Connection is reused from pool
    response = await connector.get_patient("12345")
    # Connection returned to pool automatically
```

### Rate Limiting

Built-in rate limiting prevents API quota exhaustion:

```python
# Rate limiting is automatic
# Requests are throttled based on vendor limits
# Epic: 600 requests/hour per client
# Cerner: 10,000 requests/hour per client  
# Allscripts: Varies by agreement
```

### Caching

Enable caching for improved performance:

```python
from vita_agents.core.config import get_settings

settings = get_settings()
settings.EHR_CACHE_ENABLED = True
settings.EHR_CACHE_TTL = 300  # 5 minutes
```

## Testing

Run the test suite to verify functionality:

```bash
python test_enhanced_ehr.py
```

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify client credentials
   - Check scope permissions
   - Ensure valid JWT tokens

2. **Connection Timeouts**
   - Increase timeout settings
   - Check network connectivity
   - Verify EHR system availability

3. **Rate Limiting**
   - Monitor request rates
   - Implement exponential backoff
   - Use connection pooling

4. **Data Conflicts**
   - Enable conflict resolution
   - Review harmonization rules
   - Check data sources

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enhanced EHR operations will now log detailed information
```

## Support

For additional support:
- Review the connector source code in `vita_agents/connectors/`
- Check the agent implementation in `vita_agents/agents/enhanced_ehr_agent.py`  
- Refer to vendor-specific documentation for API details

## Next Steps

This Enhanced EHR Connector System provides the foundation for:
- **Phase 2**: Advanced clinical decision support algorithms
- **Phase 2**: ML-based data harmonization  
- **Phase 3**: Predictive analytics integration
- **Phase 3**: Real-time clinical monitoring