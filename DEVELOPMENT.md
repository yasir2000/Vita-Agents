# Vita Agents Development Guide

## 🚀 Getting Started

Vita Agents is a comprehensive AI multi-agent framework for healthcare interoperability, supporting FHIR, HL7, and EHR standards with HIPAA compliance built-in.

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+

### Quick Start

1. **Clone and Setup**
```bash
git clone <repository-url>
cd Vita-Agents
pip install -e ".[dev]"
```

2. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Start with Docker**
```bash
docker-compose up -d
```

4. **Or Manual Setup**
```bash
# Backend
python -m vita_agents.api.main

# Frontend
cd web
npm install
npm run dev
```

## 🏗️ Architecture Overview

### Core Components

- **Agent Framework**: Multi-agent orchestration with healthcare specializations
- **Security Layer**: HIPAA-compliant encryption, authentication, and audit logging
- **Healthcare Agents**: Specialized agents for FHIR, HL7, and EHR processing
- **API Layer**: RESTful API with OpenAPI documentation
- **Web Interface**: Next.js dashboard for agent management
- **CLI Tool**: Command-line interface for agent operations

### Security & Compliance

- **HIPAA Compliance**: Built-in PHI protection and audit trails
- **Encryption**: AES-256 encryption for sensitive data at rest and in transit
- **Authentication**: JWT-based auth with role-based access control
- **Audit Logging**: Comprehensive audit trails for all healthcare data access

## 🤖 Healthcare Agents

### FHIR Agent (`FHIRAgent`)
- FHIR R4/R5 resource validation
- Data quality assessment
- Clinical data extraction
- PHI encryption and compliance

```python
from vita_agents import FHIRAgent

agent = FHIRAgent("fhir-agent-1", settings)
result = await agent.validate_fhir_resource(fhir_data, user_id, permissions, "Clinical validation")
```

### HL7 Agent (`HL7Agent`)
- HL7 v2.x message parsing
- FHIR conversion
- Message validation
- Terminology mapping

```python
from vita_agents import HL7Agent

agent = HL7Agent("hl7-agent-1", settings)
result = await agent.process_hl7_message(hl7_message, user_id, permissions, "Message processing")
```

### EHR Agent (`EHRAgent`)
- EHR system integration
- OAuth 2.0 authentication
- Bulk data operations
- Patient data retrieval

```python
from vita_agents import EHRAgent

agent = EHRAgent("ehr-agent-1", settings)
result = await agent.connect_ehr_system(ehr_config, user_id, permissions, "EHR connection")
```

## 🔧 Development

### Project Structure

```
vita_agents/
├── core/                 # Core framework
│   ├── agent.py         # Base agent classes
│   ├── orchestrator.py  # Multi-agent coordination
│   ├── config.py        # Configuration management
│   └── security.py      # HIPAA compliance & security
├── agents/              # Specialized healthcare agents
│   ├── fhir_agent.py    # FHIR processing
│   ├── hl7_agent.py     # HL7 message handling
│   └── ehr_agent.py     # EHR system integration
├── api/                 # REST API
│   └── main.py          # FastAPI application
├── cli/                 # Command-line interface
│   └── main.py          # CLI commands
└── web/                 # Frontend dashboard
    ├── pages/           # Next.js pages
    ├── components/      # React components
    └── lib/             # Utilities
```

### Adding New Agents

1. **Create Agent Class**
```python
from vita_agents.core.security import HIPAACompliantAgent
from vita_agents.core.agent import HealthcareAgent

class CustomAgent(HIPAACompliantAgent, HealthcareAgent):
    def __init__(self, agent_id: str, settings: Settings):
        super().__init__(agent_id, settings)
        self.capabilities = ["custom_capability"]
    
    async def _process_healthcare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Implement your processing logic
        return {"processed": True}
```

2. **Register Agent**
```python
# In orchestrator
orchestrator.register_agent(CustomAgent("custom-1", settings))
```

3. **Add API Endpoints**
```python
# In api/main.py
@app.post("/agents/custom/process")
async def process_custom_data(data: dict):
    agent = orchestrator.get_agent("custom-1")
    return await agent.secure_process_data(...)
```

### Testing

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/test_security.py -v
python -m pytest tests/test_agents.py -v

# Run with coverage
python -m pytest --cov=vita_agents
```

### Code Quality

```bash
# Linting
python -m flake8 vita_agents/
python -m black vita_agents/

# Type checking
python -m mypy vita_agents/
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build images
docker build -f Dockerfile.backend -t vita-agents/backend .
docker build -f web/Dockerfile.frontend -t vita-agents/frontend web/

# Deploy with compose
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/vita-agents.yaml

# Check status
kubectl get pods -n vita-agents
```

### Production Configuration

1. **Environment Variables**
```bash
# Security
VITA_SECURITY__JWT_SECRET=your-production-jwt-secret
VITA_SECURITY__ENCRYPTION_KEY=your-32-char-encryption-key
VITA_SECURITY__ENCRYPTION_SALT=your-encryption-salt

# Database
VITA_DATABASE__URL=postgresql://user:pass@host:5432/vita_agents

# Healthcare APIs
VITA_HEALTHCARE__FHIR_SERVER_URL=https://your-fhir-server.com
```

2. **SSL/TLS Configuration**
- Use cert-manager for automatic certificate management
- Configure ingress with TLS termination
- Enable HTTPS redirect in production

3. **Monitoring & Observability**
- Prometheus metrics collection
- Grafana dashboards
- Structured logging with ELK stack
- Health check endpoints

## 🔐 Security Best Practices

### HIPAA Compliance Checklist

- [ ] PHI encryption at rest and in transit
- [ ] Audit logging for all data access
- [ ] Role-based access control
- [ ] Data retention policies
- [ ] Minimum necessary access principle
- [ ] Business associate agreements
- [ ] Security incident response plan

### Security Configuration

```python
# Enable all security features
settings = Settings(
    security={
        "hipaa_compliance": True,
        "audit_log_enabled": True,
        "data_encryption_at_rest": True,
        "session_timeout_minutes": 15,
        "max_failed_login_attempts": 3
    }
)
```

### Data Encryption

```python
from vita_agents.core.security import EncryptionManager, ComplianceLevel

encryption = EncryptionManager(settings)

# Encrypt PHI
encrypted_phi = encryption.encrypt_sensitive_data(
    "Patient SSN: 123-45-6789",
    ComplianceLevel.RESTRICTED
)

# Decrypt when authorized
decrypted_phi = encryption.decrypt_sensitive_data(
    encrypted_phi,
    ComplianceLevel.RESTRICTED
)
```

## 📊 Monitoring & Metrics

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Agent status
curl http://localhost:8000/agents/status
```

### Metrics Collection

- Agent performance metrics
- Healthcare data processing stats
- Security audit metrics
- System resource utilization

### Alerting

Configure alerts for:
- Failed authentication attempts
- Data processing errors
- System resource exhaustion
- Security violations

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Run security and compliance checks
5. Submit pull request

### Code Standards

- Follow PEP 8 for Python code
- Use TypeScript for frontend code
- Comprehensive test coverage (>90%)
- Security review for all PRs
- HIPAA compliance verification

### Security Review Process

All code changes must pass:
- Automated security scanning
- HIPAA compliance review
- Penetration testing (for security features)
- Code review by security team

## 📚 API Documentation

### REST API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### CLI Commands

```bash
# Start agents
vita-agents start --config config.yaml

# Check status
vita-agents status

# Process FHIR data
vita-agents task fhir validate --file patient.json

# Run workflow
vita-agents workflow run patient-intake --input data.json
```

## 🆘 Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Check JWT secret configuration
   - Verify token expiration settings
   - Review user permissions

2. **Database Connection Issues**
   - Verify database URL
   - Check network connectivity
   - Review connection pool settings

3. **FHIR Validation Errors**
   - Validate FHIR resource structure
   - Check FHIR version compatibility
   - Review required fields

4. **Performance Issues**
   - Monitor resource utilization
   - Check database query performance
   - Review agent load balancing

### Debug Mode

```bash
# Enable debug logging
export VITA_LOG_LEVEL=DEBUG

# Run with profiling
python -m cProfile -o profile.stats -m vita_agents.api.main
```

### Support

- **Documentation**: [docs.vita-agents.com]
- **Issues**: [GitHub Issues]
- **Security**: security@vita-agents.com
- **Community**: [Discord/Slack Channel]

## 📋 Changelog

### v1.0.0 (Current)
- Initial release with FHIR, HL7, and EHR agents
- HIPAA compliance framework
- Web dashboard and CLI tools
- Docker and Kubernetes deployment
- Comprehensive test suite

### Roadmap

- [ ] Real-time data streaming
- [ ] Advanced ML/AI capabilities
- [ ] Additional EHR vendor support
- [ ] Multi-tenant architecture
- [ ] Advanced analytics dashboard

---

For more detailed information, see the [Technical Documentation](docs/) and [API Reference](api-docs/).