# 🏥 Vita Agents: Multi-Agent AI for Healthcare Interoperability

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![HIPAA Compliant](https://img.shields.io/badge/HIPAA-Compliant-green.svg)](https://www.hhs.gov/hipaa/)
[![FHIR R4/R5](https://img.shields.io/badge/FHIR-R4%2FR5-orange.svg)](https://hl7.org/fhir/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![GitHub release](https://img.shields.io/github/release/yasir2000/vita-agents.svg)](https://GitHub.com/yasir2000/vita-agents/releases/)
[![GitHub stars](https://img.shields.io/github/stars/yasir2000/vita-agents.svg?style=social&label=Star)](https://GitHub.com/yasir2000/vita-agents/stargazers/)

> **🚀 Latest Release v2.1.0**: Multi-Engine FHIR Support with 11+ Open Source FHIR Servers!

An enterprise-grade, open-source framework that leverages multi-agent AI systems to streamline healthcare data interoperability. Supporting FHIR (Fast Healthcare Interoperability Resources), HL7 standards, and Electronic Health Records (EHR) integration with **11+ open source FHIR engines** including HAPI FHIR, IBM FHIR, Medplum, and more.

## 🌟 What's New in v2.1.0

### 🔥 **Multi-Engine FHIR Support** 
Connect to **11+ open source FHIR servers simultaneously**:
- **HAPI FHIR Server** (hapifhir.io) - Most popular implementation
- **IBM FHIR Server** (github.com/IBM/FHIR) - Enterprise-grade
- **Medplum FHIR Server** (medplum.com) - Modern cloud-native  
- **Firely .NET SDK** (fire.ly) - Comprehensive .NET implementation
- **Spark FHIR Server** - Open source by Firely
- **LinuxForHealth FHIR Server** - IBM's healthcare platform
- **Aidbox FHIR Platform** - Cloud-native FHIR
- And 4+ more engines with extensible architecture

### 🎯 **Key Features**
- ⚡ **Parallel Operations**: Execute across multiple FHIR engines simultaneously  
- 📊 **Performance Benchmarking**: Compare engines and identify optimal performance
- 🔄 **Cross-Engine Validation**: Ensure FHIR compliance across implementations
- 🚀 **Data Migration**: Seamlessly migrate between different FHIR engines
- 🔐 **Multi-Auth Support**: OAuth2, SMART on FHIR, Basic Auth, Bearer tokens
- 🎨 **Professional CLI**: Beautiful command-line interface with rich features
- 🏥 **Production Ready**: Enterprise-grade with Docker, monitoring, and security

## 🐳 Docker Integration (New!)

**Production-Ready Containerized Setup with Real Infrastructure Components:**

### Quick Start with Docker
```bash
# Start full stack with Docker services
python vita_agents_launcher.py

# Or use Docker Compose directly
docker-compose up -d

# Test all services
python test_docker_integration.py
```

### Docker Services Included
- **PostgreSQL 15**: Primary database with connection pooling
- **Redis 7**: Caching and session management  
- **Elasticsearch 8**: Advanced search and analytics
- **RabbitMQ 3.12**: Message queue for background tasks
- **MinIO**: Object storage for files and documents
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Beautiful dashboards and visualization
- **MailHog**: Email testing and development
- **Nginx**: Reverse proxy and load balancing

### Access Points
| Service | URL | Credentials |
|---------|-----|-------------|
| **Main Application** | http://localhost:8083 | admin / admin123 |
| **Grafana Dashboard** | http://localhost:3000 | admin / admin |
| **MailHog Interface** | http://localhost:8025 | - |
| **MinIO Console** | http://localhost:9001 | vita_admin / vita_minio_pass_2024 |

## 🚀 Quick Start

### One-Command Startup

```bash
# Start the healthcare portal (auto-detects available port)
python start_portal.py

# Or use platform-specific scripts:
# Windows: double-click start_portal.bat
# Unix/Linux/macOS: ./start_portal.sh
```

**Access the portal at:** http://localhost:8080

### Advanced Options

```bash
# Development mode with hot reload
python start_portal.py --dev

# Specific port
python start_portal.py --port 8081

# Clean start (kills existing processes)
python start_portal.py --clean

# Auto-find available port
python start_portal.py --find-port
```

## 🎯 Project Overview

Vita Agents creates specialized AI agents that work collaboratively to handle different aspects of healthcare data processing, each with specific expertise in healthcare standards and workflows.

## 🤖 Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Agent Orchestrator                      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ FHIR Parser │  │ HL7 Translator│  │ EHR Integration │  │
│  │   Agent     │  │    Agent      │  │     Agent       │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Clinical    │  │ Data        │  │ Compliance &    │  │
│  │ Decision    │  │ Harmonization│  │ Security Agent  │  │
│  │ Support     │  │   Agent     │  │                 │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         Natural Language Processing Agent           │  │
│  └─────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│              Shared Knowledge Base                      │
│          (Medical Ontologies, Standards)               │
└─────────────────────────────────────────────────────────┘
```

## 🎖️ Core Agent Types

### 1. **FHIR Parser Agent** (`fhir_agent.py`)
- ✅ Validates and parses FHIR resources (Patient, Observation, Medication, etc.)
- ✅ Handles FHIR R4/R5 versions with backward compatibility
- ✅ Performs comprehensive data quality checks and validation
- ✅ FHIR resource transformation and normalization

### 2. **HL7 Translation Agent** (`hl7_agent.py`)
- ✅ Converts between HL7 v2.x messages and FHIR resources
- ✅ Handles CDA (Clinical Document Architecture) processing
- ✅ Manages terminology mapping (SNOMED CT, ICD-10, LOINC)
- ✅ Real-time message validation and error reporting

### 3. **EHR Integration Agent** (`ehr_agent.py`)
- ✅ Connects with major EHR systems (Epic, Cerner, Allscripts)
- ✅ Handles API authentication and rate limiting
- ✅ Manages bulk data operations and FHIR bulk export
- ✅ Real-time and batch data synchronization

### 4. **Clinical Decision Support Agent** (`clinical_decision_agent.py`)
- ✅ Analyzes patient data for clinical insights and recommendations
- ✅ Identifies potential drug interactions and allergies
- ✅ Suggests evidence-based care recommendations
- ✅ Clinical risk assessment and alerts
- ✅ Integration with clinical guidelines and protocols

### 5. **Data Harmonization Agent** (`data_harmonization_agent.py`)
- ✅ Normalizes data from multiple healthcare sources
- ✅ Resolves conflicts between overlapping records
- ✅ Ensures data consistency across systems
- ✅ Patient identity resolution and record linkage
- ✅ Quality assessment and data completeness scoring

### 6. **Compliance & Security Agent** (`compliance_security_agent.py`)
- ✅ Enforces HIPAA compliance and healthcare regulations
- ✅ Manages patient consent and data privacy
- ✅ Handles comprehensive audit logging and security monitoring
- ✅ PHI access validation and minimum necessary enforcement
- ✅ Security incident detection and response

### 7. **Natural Language Processing Agent** (`nlp_agent.py`)
- ✅ Clinical note analysis and entity extraction
- ✅ Medical terminology standardization
- ✅ PHI identification and anonymization
- ✅ Clinical sentiment analysis and quality assessment
- ✅ Automated clinical documentation insights

## 🚀 Key Features

### 🤖 **Multi-Agent Coordination**
- Agents communicate through standardized protocols
- Workflow orchestration for complex healthcare tasks
- Load balancing and parallel processing capabilities
- Real-time agent status monitoring and health checks

### 📊 **Healthcare Standards Support**
- Full FHIR R4/R5 compliance with backward compatibility
- HL7 v2.x and CDA document processing
- Support for major medical coding systems (SNOMED CT, ICD-10, LOINC, CPT)
- DICOM integration for medical imaging workflows

### 🔌 **EHR Integration**
- Pre-built connectors for major EHR vendors
- RESTful APIs and webhook support
- Real-time and batch data synchronization
- OAuth 2.0 and API key authentication

### 🛡️ **Security & Compliance**
- Built-in HIPAA compliance tools and validation
- End-to-end encryption (AES-256) for PHI
- Comprehensive audit trails and logging
- Role-based access control (RBAC)
- Patient consent management

### 🎯 **Clinical Intelligence**
- Natural language processing for clinical notes
- Clinical decision support algorithms
- Population health analytics
- Real-time alerts and notifications

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.9+ |
| **AI Framework** | CrewAI, LangChain |
| **Healthcare Libraries** | fhirclient, hl7apy, pydicom |
| **Database** | PostgreSQL with JSONB for FHIR resources |
| **API Framework** | FastAPI with OpenAPI documentation |
| **Security** | OAuth 2.0, JWT, AES-256 encryption |
| **Deployment** | Docker, Kubernetes, cloud-native |
| **Monitoring** | Prometheus, structured logging |

<img width="1349" height="608" alt="image" src="https://github.com/user-attachments/assets/84c57e64-098f-472b-b08f-1bf2ddf3f0e8" />

<img width="1134" height="641" alt="image" src="https://github.com/user-attachments/assets/744efe00-0351-462f-a551-afe761cc4fd9" />

Courtesy of @https://medium.com/@alexglee/building-framework-for-ai-agents-in-healthcare-e6b2c0935c93

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yasir2000/vita-agents.git
cd vita-agents

# Install dependencies
pip install -r requirements.txt

# Or install using setup.py
python setup.py install

# Or install using pip
pip install vita-agents
```

### Basic Usage

```python
from vita_agents import AgentOrchestrator
from vita_agents.agents import FHIRAgent, HL7Agent, ClinicalDecisionSupportAgent

# Initialize the multi-agent system
orchestrator = AgentOrchestrator()

# Add specialized agents
fhir_agent = FHIRAgent("fhir-agent-1")
hl7_agent = HL7Agent("hl7-agent-1") 
clinical_agent = ClinicalDecisionSupportAgent("clinical-agent-1")

# Register agents with orchestrator
await orchestrator.register_agent(fhir_agent)
await orchestrator.register_agent(hl7_agent)
await orchestrator.register_agent(clinical_agent)

# Process healthcare data workflow
result = await orchestrator.execute_workflow(
    workflow_type="patient_data_integration",
    input_data={
        "source": "hl7_message",
        "data": "MSH|^~\\&|GHH LAB|ELAB..."
    },
    agents=["fhir-agent-1", "hl7-agent-1", "clinical-agent-1"]
)

print(f"Workflow completed: {result.status}")
print(f"FHIR resources created: {len(result.fhir_resources)}")
```

### CLI Usage

```bash
# Start the orchestrator with all agents
python -m vita_agents.orchestrator start

# Start with custom configuration
python -m vita_agents.orchestrator start --config config.yml --port 8080

# Check agent status
vita-agents status

# Run specific workflow
vita-agents workflow execute patient_data_integration --input data.hl7
```

### API Server

```bash
# Start the API server
uvicorn vita_agents.api.main:app --host 0.0.0.0 --port 8000

# Or use the CLI
vita-agents server start --port 8000
```

## 📖 Configuration

### YAML Configuration (`config.yml`)

```yaml
# Agent Configuration
agents:
  fhir_parser:
    enabled: true
    version: "R4"
    validation_level: "strict"
    
  hl7_translator:
    enabled: true
    supported_versions: ["2.5", "2.6", "2.8"]
    
  ehr_integration:
    enabled: true
    vendors: ["epic", "cerner", "allscripts"]
    
  clinical_decision_support:
    enabled: true
    drug_interaction_checking: true
    allergy_screening: true
    
  data_harmonization:
    enabled: true
    conflict_resolution: "priority_based"
    
  compliance_security:
    enabled: true
    hipaa_compliance: true
    audit_level: "comprehensive"
    
  nlp:
    enabled: true
    phi_detection: true
    anonymization: true

# Security Configuration
security:
  encryption: true
  hipaa_compliance: true
  audit_logging: true
  access_control: "rbac"
  
# Database Configuration
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  name: "vita_agents"
  
# Monitoring
monitoring:
  prometheus_enabled: true
  log_level: "info"
  health_checks: true
```

### Environment Variables

```bash
# Database
VITA_DB_URL="postgresql://user:pass@localhost:5432/vita_agents"

# Security
VITA_SECRET_KEY="your-secret-key"
VITA_ENCRYPTION_KEY="your-encryption-key"

# API Configuration
VITA_API_HOST="0.0.0.0"
VITA_API_PORT="8000"

# FHIR Server
VITA_FHIR_SERVER_URL="https://your-fhir-server.com"

# Logging
VITA_LOG_LEVEL="INFO"
```

## 🌐 API Documentation

### Core Endpoints

#### Agent Management

```http
GET /api/v1/agents/status
```
Get status of all agents
```json
{
  "agents": [
    {
      "name": "fhir_parser",
      "status": "active",
      "last_activity": "2025-10-16T10:05:00Z"
    }
  ]
}
```

#### Workflow Execution

```http
POST /api/v1/workflows/execute
```
Execute a multi-agent workflow
```json
{
  "workflow_type": "patient_data_integration",
  "input_data": {
    "source": "hl7_message",
    "data": "MSH|^~\\&|GHH LAB|ELAB..."
  },
  "agents": ["fhir_parser", "hl7_translator", "clinical_decision_support"]
}
```

#### FHIR Operations

```http
POST /api/v1/fhir/validate
POST /api/v1/fhir/quality-check
POST /api/v1/fhir/transform
```

#### HL7 Operations

```http
POST /api/v1/hl7/validate
POST /api/v1/hl7/to-fhir
POST /api/v1/hl7/from-fhir
```

#### Clinical Decision Support

```http
POST /api/v1/clinical/analyze
POST /api/v1/clinical/drug-interactions
POST /api/v1/clinical/recommendations
```

#### Compliance & Security

```http
POST /api/v1/compliance/validate-access
POST /api/v1/compliance/audit-log
POST /api/v1/security/encrypt
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run compliance tests
pytest tests/compliance/

# Generate test coverage report
pytest --cov=vita_agents tests/ --cov-report=html
```

### Test Structure

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── test_fhir_agent.py
│   ├── test_hl7_agent.py
│   ├── test_clinical_decision_agent.py
│   ├── test_compliance_agent.py
│   └── test_nlp_agent.py
├── integration/          # Integration tests for workflows
│   ├── test_patient_workflow.py
│   ├── test_clinical_workflow.py
│   └── test_compliance_workflow.py
└── compliance/          # HIPAA and security compliance tests
    ├── test_hipaa_compliance.py
    ├── test_security_standards.py
    └── test_audit_trails.py
```

## 🎯 Use Cases

### 1. **Hospital System Integration**
- Connect disparate EHR systems across departments
- Enable seamless patient data sharing between providers
- Reduce manual data entry and transcription errors
- Real-time clinical decision support integration

### 2. **Research Data Aggregation**
- Collect and harmonize research datasets from multiple sites
- Enable multi-site clinical studies with standardized data
- Support real-world evidence generation and analysis
- De-identification and anonymization for research use

### 3. **Telehealth Platform Support**
- Integrate remote monitoring data with EHR systems
- Support virtual care workflows and documentation
- Enable care coordination across providers and platforms
- Real-time clinical alerts and monitoring

### 4. **Public Health Reporting**
- Automate reporting to health departments and registries
- Support disease surveillance and outbreak detection
- Enable population health monitoring and analytics
- Compliance with reporting mandates and regulations

### 5. **Clinical Quality Improvement**
- Automated quality measure calculation and reporting
- Clinical documentation improvement and optimization
- Performance monitoring and benchmarking
- Evidence-based care recommendations

## 🤝 Contributing

We welcome contributions from the healthcare IT community!

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/vita-agents.git
cd vita-agents

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Code Standards

- Follow PEP 8 for Python code style
- Use type hints throughout the codebase
- Maintain 90%+ test coverage for all new code
- Document all public APIs with docstrings
- Follow FHIR and HL7 naming conventions
- Ensure HIPAA compliance in all healthcare data handling

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass and coverage requirements are met
4. Update documentation as needed
5. Submit a pull request with a clear description

## 📚 Documentation

- **[User Guide](docs/user-guide.md)** - Complete user documentation
- **[API Reference](docs/api-reference.md)** - Detailed API documentation  
- **[Agent Development Guide](docs/agent-development.md)** - Creating custom agents
- **[Healthcare Standards Guide](docs/healthcare-standards.md)** - FHIR, HL7, and EHR integration
- **[Deployment Guide](docs/deployment.md)** - Production deployment instructions
- **[Security Guide](docs/security.md)** - HIPAA compliance and security best practices

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🌟 Community

- **[GitHub Discussions](https://github.com/yasir2000/vita-agents/discussions)** - General questions and community support
- **[Issues](https://github.com/yasir2000/vita-agents/issues)** - Bug reports and feature requests
- **[Discord](https://discord.gg/vita-agents)** - Join our healthcare AI community server
- **[Monthly Calls](https://github.com/yasir2000/vita-agents/wiki/Community-Calls)** - Community calls every first Thursday of the month

## 🗺️ Roadmap

### Phase 1 (Q1 2026) - Core Foundation ✅
- [x] Core agent framework architecture
- [x] Basic FHIR R4 support and validation
- [x] HL7 v2.x parsing and conversion
- [x] RESTful API with OpenAPI documentation
- [x] Basic security and compliance features

### Phase 2 (Q2 2026) - Advanced Features ✅
- [x] Enhanced EHR vendor connectors (Epic, Cerner, Allscripts)
- [x] Advanced clinical decision support algorithms
- [x] Machine learning-based data harmonization
- [ ] Performance optimization and caching
- [ ] Advanced security features and penetration testing

### Phase 3 (Q3 2026) - Intelligence & Scale 📋
- [ ] Advanced machine learning capabilities
- [ ] Real-time data streaming and processing
- [ ] Advanced population health analytics
- [ ] Multi-language support (Spanish, French, etc.)
- [ ] Mobile SDKs and edge computing support

### Phase 4 (Q4 2026) - Enterprise Ready 📋
- [ ] Enterprise deployment tools
- [ ] Advanced monitoring and observability
- [ ] Certified EHR module capabilities
- [ ] International healthcare standards support
- [ ] Regulatory certifications and validations

## 🙏 Acknowledgments

- **FHIR Community** for healthcare interoperability standards
- **HL7 International** for messaging standards and protocols
- **Healthcare IT Professionals** who inspire and guide this work
- **Open Source AI Frameworks** (CrewAI, LangChain) that make this possible
- **Contributors and Community Members** who help improve this project

## 📞 Contact

- **Project Lead**: Yasir ([@yasir2000](https://github.com/yasir2000))
- **Email**: contact@vita-agents.org
- **Website**: https://vita-agents.org
- **LinkedIn**: [Vita Agents Project](https://linkedin.com/company/vita-agents)

---

<div align="center">

**Built with ❤️ for the healthcare community**

[⭐ Star this repo](https://github.com/yasir2000/vita-agents) | [📖 Documentation](https://vita-agents.org/docs) | [🤝 Contribute](CONTRIBUTING.md) | [💬 Discussions](https://github.com/yasir2000/vita-agents/discussions)

</div>
