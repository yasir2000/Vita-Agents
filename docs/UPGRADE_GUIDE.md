# Upgrade Guide - Vita Agents v2.1.0

## 🚀 Upgrading from v2.0.x to v2.1.0

This guide helps you upgrade your Vita Agents installation to take advantage of the new Multi-Engine FHIR Support and HMCP (Healthcare Model Context Protocol) features.

## 📋 What's New in v2.1.0

### Major Features Added
- **Multi-Engine FHIR Support**: Connect to 11+ open source FHIR servers simultaneously
- **HMCP Protocol**: Healthcare Model Context Protocol for clinical workflows
- **Emergency Response**: Automated emergency protocols and care team coordination
- **Enhanced Security**: HIPAA-compliant messaging with comprehensive audit trails
- **Healthcare Workflows**: 5+ pre-built clinical workflows for immediate use

### New Components
- `vita_agents/protocols/hmcp.py` - HMCP protocol implementation
- `vita_agents/agents/hmcp_agent.py` - Healthcare communication agent
- `vita_agents/cli/hmcp_cli.py` - Interactive CLI for healthcare agents
- `examples/hmcp_workflows.py` - Complete healthcare workflow examples
- `docs/HMCP_INTEGRATION.md` - Comprehensive HMCP documentation

## 🔧 Installation & Setup

### Option 1: Fresh Installation (Recommended)

```bash
# Clone the latest version
git clone https://github.com/yasir2000/vita-agents.git
cd vita-agents

# Checkout the latest release
git checkout v2.1.0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Vita Agents
pip install -e .
```

### Option 2: Upgrade Existing Installation

```bash
# Navigate to your existing installation
cd /path/to/vita-agents

# Backup your configuration
cp config.yml config.yml.backup
cp -r data/ data_backup/

# Pull latest changes
git fetch origin
git checkout v2.1.0

# Update dependencies
pip install -r requirements.txt --upgrade

# Reinstall Vita Agents
pip install -e . --upgrade
```

## 🆕 New Configuration Options

### HMCP Configuration
Add the following to your `config.yml`:

```yaml
# HMCP Healthcare Communication
hmcp:
  enabled: true
  port: 8080
  encryption: true
  audit_logging: true
  
  # Healthcare roles and capabilities
  agents:
    diagnostic_copilot:
      role: "ai_agent"
      capabilities: ["differential_diagnosis", "clinical_reasoning"]
      emergency_capable: true
      
    medical_knowledge:
      role: "ai_agent" 
      capabilities: ["drug_interactions", "treatment_guidelines"]
      emergency_capable: false
      
    patient_data:
      role: "ai_agent"
      capabilities: ["ehr_integration", "lab_results_processing"]
      emergency_capable: true

# Enhanced FHIR engines configuration
fhir_engines:
  enabled: true
  default_engine: "hapi_fhir_r4"
  
  engines:
    hapi_fhir_r4:
      url: "http://hapi.fhir.org/baseR4"
      auth_type: "none"
      
    medplum_demo:
      url: "https://api.medplum.com/fhir/R4"
      auth_type: "bearer"
      token: "your_token_here"
```

### Security Configuration Updates
```yaml
# Enhanced security for healthcare data
security:
  hipaa_compliance: true
  phi_protection: true
  audit_level: "comprehensive"
  encryption:
    enabled: true
    algorithm: "AES-256"
  
  roles:
    - "physician"
    - "nurse" 
    - "pharmacist"
    - "ai_agent"
```

## 🚀 Quick Start with New Features

### 1. Test Multi-Engine FHIR Support

```bash
# List available FHIR engines
python -m vita_agents.cli.fhir_engines_cli list

# Test connections to all engines
python -m vita_agents.cli.fhir_engines_cli test-connections

# Search patients across multiple engines
python -m vita_agents.cli.fhir_engines_cli search Patient --parameters '{"family": "Smith"}' --count 5
```

### 2. Try HMCP Healthcare Communication

```bash
# Create healthcare agents
python -m vita_agents.cli.hmcp_cli create diagnostic_copilot --role ai_agent --capabilities differential_diagnosis --emergency-capable

python -m vita_agents.cli.hmcp_cli create medical_knowledge --role ai_agent --capabilities drug_interactions treatment_guidelines

# List created agents
python -m vita_agents.cli.hmcp_cli list

# Send clinical message
python -m vita_agents.cli.hmcp_cli send medical_knowledge request '{"action": "medication_check", "drugs": ["warfarin", "aspirin"]}' --patient-id PATIENT_001 --urgency urgent
```

### 3. Run Healthcare Workflow Examples

```bash
# Run comprehensive healthcare workflow examples
python examples/hmcp_workflows.py

# This will demonstrate:
# - Chest pain diagnosis workflow
# - Medication interaction checking
# - Emergency cardiac arrest response
# - Discharge planning coordination
# - Critical lab value notifications
```

### 4. Start Enhanced Web Portal

```bash
# Start the enhanced web portal with new features
python start_portal.py

# Access at: http://localhost:8080
# New features include:
# - HMCP agent management
# - Multi-engine FHIR operations
# - Healthcare workflow monitoring
# - Emergency response dashboards
```

## 🔄 Migration Guide

### Database Schema Updates
No database schema changes are required for v2.1.0. All new features use the existing database structure.

### Configuration Migration
Your existing `config.yml` will continue to work. The new HMCP and multi-engine features are optional and can be enabled gradually.

### API Compatibility
All existing APIs remain fully compatible. New HMCP endpoints have been added without affecting existing functionality.

## 🧪 Testing Your Upgrade

### 1. Verify Core Functionality
```bash
# Test basic agent functionality
python -c "from vita_agents.agents import FHIRAgent; print('✅ Core agents working')"

# Test orchestrator
python -m vita_agents.orchestrator status
```

### 2. Test New HMCP Features
```bash
# Run HMCP tests
python test_implemented_features.py

# Verify HMCP protocol
python -c "from vita_agents.protocols.hmcp import HMCPProtocol; print('✅ HMCP protocol loaded')"
```

### 3. Test Multi-Engine FHIR
```bash
# Test FHIR engines
python -m vita_agents.cli.fhir_engines_cli test-connections

# Verify enhanced FHIR agent
python -c "from vita_agents.agents.enhanced_fhir_agent import EnhancedFHIRAgent; print('✅ Enhanced FHIR agent working')"
```

## 🔧 Troubleshooting

### Common Issues

#### HMCP Import Errors
```python
# If you see HMCP import errors, verify installation:
python -c "import vita_agents.protocols.hmcp; print('HMCP available')"

# Reinstall if needed:
pip install -e . --force-reinstall
```

#### FHIR Engine Connection Issues
```bash
# Test individual engine connections
python -m vita_agents.cli.fhir_engines_cli test-connection hapi_fhir_r4

# Check configuration
python -m vita_agents.cli.fhir_engines_cli show-config hapi_fhir_r4
```

#### Port Conflicts
```bash
# If default ports are occupied, use alternatives:
python start_portal.py --port 8081

# Or use auto port finding:
python start_portal.py --find-port
```

### Getting Help

If you encounter issues during the upgrade:

1. **Check the logs**: Look for error messages in the console output
2. **Verify dependencies**: Run `pip install -r requirements.txt --upgrade`
3. **Reset configuration**: Try with a fresh `config.yml` based on the template
4. **Community support**: Visit our [GitHub Discussions](https://github.com/yasir2000/vita-agents/discussions)
5. **Report bugs**: Create an issue on [GitHub Issues](https://github.com/yasir2000/vita-agents/issues)

## 📚 Next Steps

After upgrading, explore these new capabilities:

1. **Healthcare Workflows**: Review `examples/hmcp_workflows.py` for clinical use cases
2. **HMCP Documentation**: Read `docs/HMCP_INTEGRATION.md` for detailed guidance
3. **Multi-Engine FHIR**: Test with your preferred FHIR servers
4. **Emergency Protocols**: Set up emergency response workflows for your organization
5. **Care Coordination**: Implement multi-disciplinary care team communication

## 🎯 Benefits of Upgrading

### Immediate Benefits
- **11+ FHIR Engines**: Connect to your preferred FHIR implementation
- **Clinical Workflows**: Ready-to-use healthcare communication patterns
- **Emergency Response**: Automated emergency protocols save critical time
- **HIPAA Compliance**: Built-in security and audit capabilities

### Long-term Benefits
- **Interoperability**: Seamlessly work with any FHIR-compliant system
- **Scalability**: Multi-engine support allows for better load distribution
- **Compliance**: Comprehensive audit trails and security monitoring
- **Efficiency**: Automated workflows reduce manual coordination overhead

## 🚀 Welcome to Vita Agents v2.1.0!

You're now ready to leverage the most advanced healthcare interoperability platform available. The combination of multi-engine FHIR support and HMCP healthcare communication opens up new possibilities for clinical AI workflows, emergency response coordination, and seamless healthcare data integration.

Happy coding, and welcome to the future of healthcare AI! 🏥✨