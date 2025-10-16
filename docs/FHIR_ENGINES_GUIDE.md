# Enhanced FHIR Agent with Open Source FHIR Engines Support

## Overview

The Enhanced FHIR Agent is a comprehensive solution for working with multiple open source FHIR engines simultaneously. It provides seamless integration with popular FHIR servers like HAPI FHIR, IBM FHIR Server, Medplum, Firely, and many more.

## Supported FHIR Engines

### Primary Engines
- **HAPI FHIR Server** (hapifhir.io) - Most popular open source FHIR server
- **IBM FHIR Server** (github.com/IBM/FHIR) - Enterprise-grade FHIR server
- **Medplum FHIR Server** (medplum.com) - Modern cloud-native FHIR platform
- **Firely .NET SDK** (fire.ly) - Comprehensive .NET FHIR implementation

### Additional Supported Engines
- **Spark FHIR Server** - Open source FHIR server by Firely
- **LinuxForHealth FHIR Server** - IBM's healthcare integration platform
- **Aidbox FHIR Platform** - Cloud-native FHIR platform
- **Microsoft FHIR Server** - Azure-based FHIR service
- **Google Cloud Healthcare API** - Google's managed FHIR service
- **Amazon HealthLake** - AWS FHIR data lake
- **Smile CDR** - Commercial FHIR server with open source components

## Key Features

### 🚀 Multi-Engine Operations
- Execute FHIR operations across multiple engines simultaneously
- Parallel processing for improved performance
- Automatic failover and load balancing
- Consolidated results from multiple sources

### 🔧 Comprehensive FHIR Support
- Full CRUD operations (Create, Read, Update, Delete)
- Advanced search with multiple parameters
- Bundle operations and transactions
- Resource validation across engines
- Capability statement analysis

### 📊 Performance Analysis
- Benchmark response times across engines
- Identify fastest engines for specific operations
- Resource utilization monitoring
- Performance recommendations

### 🔄 Data Migration
- Migrate data between different FHIR engines
- Support for incremental and full migrations
- Validation during migration process
- Rollback capabilities for failed migrations

### 🔐 Security & Authentication
- Multiple authentication methods:
  - No authentication (open servers)
  - Basic authentication
  - OAuth2 / Bearer tokens
  - SMART on FHIR
- Secure credential management
- SSL/TLS support

### 🛠️ Developer Tools
- Rich CLI interface for engine management
- Configuration templates for common setups
- Comprehensive logging and monitoring
- Easy integration with existing applications

## Installation

```bash
# Install the Enhanced FHIR Agent
pip install vita-agents[fhir-engines]

# Or install with all dependencies
pip install vita-agents[all]
```

## Quick Start

### 1. Using Pre-configured Templates

The easiest way to get started is using our pre-configured server templates:

```python
from vita_agents.agents.enhanced_fhir_agent import EnhancedFHIRAgent
from vita_agents.core.agent import TaskRequest

# Initialize the agent
agent = EnhancedFHIRAgent()
await agent.start()

# Connect to HAPI FHIR test server
connect_task = TaskRequest(
    task_id="connect_hapi",
    agent_id="enhanced-fhir-agent",
    task_type="connect_fhir_engine",
    parameters={
        "template_name": "hapi_fhir_r4"
    }
)

response = await agent.process_task(connect_task)
print(f"Connected: {response.success}")
```

### 2. CLI Quick Start

```bash
# List available FHIR server templates
fhir-engines list-templates

# Connect to HAPI FHIR R4 server
fhir-engines connect hapi_fhir_r4

# List connected engines
fhir-engines list-engines

# Search for patients across all engines
fhir-engines search Patient --parameters '{"family": "Smith"}' --count 10

# Run performance tests
fhir-engines performance-test --load-type medium_load
```

## Configuration

### Environment Configuration

Create different configurations for different environments:

```python
from vita_agents.config.fhir_engines_config import get_config_for_environment

# Development configuration
dev_config = get_config_for_environment("development")

# Production configuration  
prod_config = get_config_for_environment("production")

# Initialize agent with config
agent = EnhancedFHIRAgent(config=dev_config)
```

### Custom Server Configuration

```python
from vita_agents.fhir_engines.open_source_clients import FHIRServerConfiguration, FHIREngineType, FHIRVersion

# Configure custom HAPI FHIR server
custom_server = FHIRServerConfiguration(
    server_id="my_hapi_server",
    name="My HAPI FHIR Server",
    engine_type=FHIREngineType.HAPI_FHIR,
    base_url="http://localhost:8080/fhir",
    fhir_version=FHIRVersion.R4,
    description="Local HAPI FHIR development server"
)

# Connect using custom configuration
connect_task = TaskRequest(
    task_id="connect_custom",
    agent_id="enhanced-fhir-agent", 
    task_type="connect_fhir_engine",
    parameters={
        "server_config": custom_server.dict()
    }
)
```

## Usage Examples

### Multi-Engine Search

Search for resources across multiple FHIR engines simultaneously:

```python
# Search for patients across all connected engines
search_task = TaskRequest(
    task_id="multi_search",
    agent_id="enhanced-fhir-agent",
    task_type="multi_engine_search",
    parameters={
        "resource_type": "Patient",
        "search_parameters": {
            "family": "Smith",
            "given": "John",
            "active": "true"
        },
        "count": 50
    }
)

response = await agent.process_task(search_task)

if response.success:
    data = response.data
    print(f"Searched {data['total_engines']} engines")
    print(f"Found {data['total_resources_found']} resources")
    print(f"Execution time: {data['execution_time_ms']}ms")
    
    # Results per engine
    for engine_id, result in data['results'].items():
        if result['success']:
            resource_count = len(result['data'].get('entry', []))
            print(f"{engine_id}: {resource_count} resources")
```

### Cross-Engine Validation

Validate FHIR resources against multiple engines to ensure compliance:

```python
# Sample patient resource
patient = {
    "resourceType": "Patient",
    "id": "example",
    "active": True,
    "name": [{
        "use": "official",
        "family": "Chalmers",
        "given": ["Peter", "James"]
    }],
    "gender": "male",
    "birthDate": "1974-12-25"
}

# Validate across engines
validate_task = TaskRequest(
    task_id="validate_patient",
    agent_id="enhanced-fhir-agent",
    task_type="validate_across_engines",  
    parameters={
        "resource": patient,
        "resource_type": "Patient"
    }
)

response = await agent.process_task(validate_task)

if response.success:
    data = response.data
    if data['consensus_valid']:
        print("✅ Resource is valid across all engines")
    else:
        print("⚠️ Validation differs between engines")
        
        # Show differences
        for diff in data.get('engine_differences', []):
            print(f"{diff['engine']}: {len(diff['issues'])} issues")
```

### Performance Analysis

Compare performance across different FHIR engines:

```python
# Run performance analysis
perf_task = TaskRequest(
    task_id="performance_analysis",
    agent_id="enhanced-fhir-agent",
    task_type="engine_performance_analysis",
    parameters={
        "operation_type": "search",
        "resource_type": "Patient", 
        "sample_size": 20
    }
)

response = await agent.process_task(perf_task)

if response.success:
    data = response.data
    fastest_engine = data['fastest_engine']
    print(f"🏆 Fastest engine: {fastest_engine}")
    
    # Performance details
    for engine_id, metrics in data['performance_results'].items():
        avg_time = metrics['avg_response_time']
        success_rate = metrics['success_count'] / 20 * 100
        print(f"{engine_id}: {avg_time:.2f}ms avg, {success_rate:.1f}% success")
```

### Data Migration

Migrate data between different FHIR engines:

```python
# Migrate patient data from HAPI to IBM FHIR
migration_task = TaskRequest(
    task_id="migrate_patients",
    agent_id="enhanced-fhir-agent",
    task_type="fhir_engine_migration",
    parameters={
        "source_engine": "hapi_fhir_r4",
        "target_engine": "ibm_fhir_local",
        "resource_types": ["Patient", "Observation"],
        "migration_strategy": "selective"
    }
)

response = await agent.process_task(migration_task)

if response.success:
    results = response.data['migration_results']
    print(f"Migrated {results['migrated_resources']} resources")
    print(f"Failed: {results['failed_resources']} resources")
    
    # Per resource type results
    for resource_type, type_results in results['resource_type_results'].items():
        print(f"{resource_type}: {type_results['migrated']}/{type_results['total']}")
```

## CLI Reference

### Available Commands

```bash
# Connection Management
fhir-engines list-templates          # List server templates
fhir-engines show-template <id>      # Show template details  
fhir-engines connect <template-id>   # Connect to server
fhir-engines list-engines            # List connected engines

# FHIR Operations
fhir-engines search <resource-type>  # Search across engines
fhir-engines validate <file>         # Validate resource
fhir-engines performance-test        # Run performance tests

# Configuration  
fhir-engines config                  # Show current config
fhir-engines info                    # Show agent information
```

### CLI Examples

```bash
# Connect to multiple engines
fhir-engines connect hapi_fhir_r4
fhir-engines connect medplum_demo

# Search with parameters
fhir-engines search Patient \
  --parameters '{"family": "Smith", "active": "true"}' \
  --engines "hapi_fhir_r4,medplum_demo" \
  --count 25

# Validate a resource file
fhir-engines validate patient.json --engines "hapi_fhir_r4"

# Run comprehensive performance test
fhir-engines performance-test --load-type heavy_load
```

## Authentication Configuration

### Basic Authentication

```python
from vita_agents.fhir_engines.open_source_clients import FHIRServerConfiguration, AuthenticationType

server_config = FHIRServerConfiguration(
    server_id="secure_server",
    name="Secure FHIR Server",
    engine_type=FHIREngineType.HAPI_FHIR,
    base_url="https://secure-fhir.example.com/fhir",
    fhir_version=FHIRVersion.R4,
    authentication=AuthenticationConfig(
        type=AuthenticationType.BASIC,
        username="your_username",
        password="your_password"
    )
)
```

### OAuth2 Authentication

```python
oauth_config = FHIRServerConfiguration(
    server_id="oauth_server", 
    name="OAuth2 FHIR Server",
    engine_type=FHIREngineType.MEDPLUM,
    base_url="https://api.medplum.com/fhir/R4",
    fhir_version=FHIRVersion.R4,
    authentication=AuthenticationConfig(
        type=AuthenticationType.OAUTH2,
        token_url="https://api.medplum.com/oauth2/token",
        client_id="your_client_id",
        client_secret="your_client_secret",
        scope="fhir:read fhir:write"
    )
)
```

### SMART on FHIR

```python
smart_config = FHIRServerConfiguration(
    server_id="smart_server",
    name="SMART on FHIR Server", 
    engine_type=FHIREngineType.HAPI_FHIR,
    base_url="https://launch.smarthealthit.org/v/r4/fhir",
    fhir_version=FHIRVersion.R4,
    authentication=AuthenticationConfig(
        type=AuthenticationType.SMART_ON_FHIR,
        authorize_url="https://launch.smarthealthit.org/v/r4/auth/authorize",
        token_url="https://launch.smarthealthit.org/v/r4/auth/token",
        client_id="your_client_id",
        redirect_uri="http://localhost:8000/callback",
        scope="launch/patient patient/read patient/write"
    )
)
```

## Advanced Configuration

### Engine-Specific Optimizations

```python
from vita_agents.config.fhir_engines_config import get_engine_optimizations

# Get optimized settings for HAPI FHIR
hapi_opts = get_engine_optimizations("HAPI_FHIR")
print(f"Optimal batch size: {hapi_opts['transaction_bundle_size']}")
print(f"Preferred format: {hapi_opts['preferred_formats'][0]}")

# Get optimized settings for IBM FHIR  
ibm_opts = get_engine_optimizations("IBM_FHIR")
print(f"Conservative batch size: {ibm_opts['transaction_bundle_size']}")
```

### Custom Performance Testing

```python
from vita_agents.config.fhir_engines_config import get_performance_test_config

# Light load test
light_config = get_performance_test_config("light_load")
# Sample size: 10, Concurrent ops: 2

# Heavy load test  
heavy_config = get_performance_test_config("heavy_load")
# Sample size: 100, Concurrent ops: 10
```

### Migration Strategies

```python
from vita_agents.config.fhir_engines_config import get_migration_config

# Patient data migration
patient_migration = get_migration_config("patient_data_migration")

# Full database migration
full_migration = get_migration_config("full_database_migration") 

# Selective migration with filters
selective_migration = get_migration_config("selective_migration")
```

## Troubleshooting

### Common Issues

**Connection Timeouts**
```python
# Increase connection timeout
config = {
    "fhir_engines": {
        "connection_timeout": 60,  # Increase to 60 seconds
        "max_retries": 5
    }
}
```

**Authentication Errors**
```bash
# Test connection to verify credentials
fhir-engines connect your_server_template

# Check server capabilities
fhir-engines search Patient --count 1 --engines "your_server"
```

**Performance Issues**
```python
# Reduce concurrent operations
config = {
    "fhir_engines": {
        "max_concurrent_operations": 5,  # Reduce from default 10
    }
}
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure structured logging
config = {
    "logging": {
        "level": "DEBUG",
        "log_fhir_requests": True,
        "log_performance_metrics": True
    }
}
```

### Health Checks

```python
# Test engine connections
test_task = TaskRequest(
    task_id="health_check",
    agent_id="enhanced-fhir-agent",
    task_type="test_engine_connection",
    parameters={}
)

response = await agent.process_task(test_task)

for engine_id, result in response.data['connection_tests'].items():
    if result['connected']:
        print(f"✅ {engine_id}: {result['response_time_ms']}ms")
    else:
        print(f"❌ {engine_id}: {result['error']}")
```

## Best Practices

### 1. Server Selection
- Use **HAPI FHIR** for development/testing (excellent tool support)
- Choose **IBM FHIR** for enterprise deployments (robust, scalable)
- Consider **Medplum** for cloud-native applications (modern APIs)
- Use **Firely** for .NET integration (comprehensive SDK)

### 2. Performance Optimization
- Start with smaller batch sizes and increase gradually
- Use engine-specific optimizations from the configuration
- Monitor response times and adjust concurrent operations
- Enable performance tracking for production monitoring

### 3. Authentication
- Use environment variables for credentials
- Implement proper credential rotation
- Test authentication in development environment first
- Consider using SMART on FHIR for EHR integration

### 4. Error Handling
- Implement retry logic for transient failures
- Monitor and log authentication errors
- Use validation before operations to catch issues early
- Set up health checks for production deployments

### 5. Migration Planning
- Always test migrations in development first
- Use incremental migration for large datasets
- Validate data integrity after migration
- Have rollback procedures ready

## API Reference

### Core Classes

- `EnhancedFHIRAgent` - Main agent class with multi-engine support
- `FHIREngineManager` - Manages connections to multiple FHIR engines
- `FHIRServerConfiguration` - Configuration for FHIR server connections
- `FHIRSearchParameters` - Search parameter configuration
- `FHIROperationResult` - Result from FHIR operations

### Task Types

- `connect_fhir_engine` - Connect to a FHIR engine
- `multi_engine_search` - Search across multiple engines
- `validate_across_engines` - Validate resources across engines  
- `engine_performance_analysis` - Analyze engine performance
- `fhir_engine_migration` - Migrate data between engines
- `multi_engine_create` - Create resources across engines
- `multi_engine_read` - Read resources from engines
- `multi_engine_update` - Update resources across engines
- `multi_engine_delete` - Delete resources from engines

## Contributing

We welcome contributions to the Enhanced FHIR Agent! Here's how you can help:

### Adding New FHIR Engines

1. Extend the `FHIREngineType` enum
2. Implement a new client class inheriting from `BaseFHIRClient`
3. Add engine-specific optimizations
4. Create server templates
5. Add tests and documentation

### Reporting Issues

Please report issues on our GitHub repository with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- FHIR engine and version information
- Relevant logs (with sensitive data removed)

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

- 📖 Documentation: [Full documentation](https://vita-agents.readthedocs.io)
- 🐛 Issues: [GitHub Issues](https://github.com/vita-agents/vita-agents/issues)  
- 💬 Discussions: [GitHub Discussions](https://github.com/vita-agents/vita-agents/discussions)
- 📧 Email: support@vita-agents.dev

---

## Acknowledgments

Special thanks to the open source FHIR community:
- HAPI FHIR team for their excellent server implementation
- IBM FHIR team for their enterprise-grade server
- Firely team for comprehensive FHIR tooling
- Medplum team for modern FHIR platform innovation
- HL7 FHIR Working Group for the FHIR standard

The Enhanced FHIR Agent builds upon the foundation provided by these amazing projects to enable seamless multi-engine FHIR operations.