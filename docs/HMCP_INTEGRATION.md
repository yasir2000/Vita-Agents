# HMCP Integration Documentation

## Healthcare Multi-agent Communication Protocol (HMCP) for Vita Agents

This documentation covers the comprehensive integration of HMCP (Healthcare Multi-agent Communication Protocol) into the Vita Agents platform, enabling advanced healthcare multi-agent workflows with clinical context awareness, security, and interoperability.

## Overview

HMCP is a specialized communication protocol designed for healthcare AI agents that provides:
- **Clinical Context Awareness**: Patient data, clinical urgency levels, healthcare roles
- **Security & Compliance**: HIPAA-compliant communication, encryption, audit trails
- **Healthcare Interoperability**: FHIR, HL7, and clinical workflow integration
- **Emergency Response**: Specialized protocols for medical emergencies
- **Care Coordination**: Multi-disciplinary team communication and workflow orchestration

## Architecture

### Core Components

```
vita_agents/
├── protocols/
│   └── hmcp.py              # Core HMCP protocol implementation
├── agents/
│   └── hmcp_agent.py        # HMCP-enabled healthcare agents
├── cli/
│   └── hmcp_cli.py          # Command-line interface for HMCP
└── examples/
    └── hmcp_workflows.py    # Healthcare workflow examples
```

### Key Classes

#### HMCPMessage
Represents a healthcare message with clinical context:
```python
message = HMCPMessage(
    type=HMCPMessageType.REQUEST,
    sender_id="diagnostic_copilot",
    receiver_id="medical_knowledge",
    content={"action": "differential_diagnosis", "symptoms": [...]},
    urgency=ClinicalUrgency.URGENT,
    patient_context=PatientContext(patient_id="PATIENT_001"),
    clinical_context=ClinicalContext(specialty="cardiology"),
    security_context=SecurityContext(phi_flag=True)
)
```

#### HMCPAgent
Healthcare agent with HMCP communication capabilities:
```python
agent = HMCPAgent('diagnostic_copilot', {
    'role': 'ai_agent',
    'capabilities': ['differential_diagnosis', 'clinical_reasoning'],
    'emergency_capable': True
})
```

## Message Types

### 1. REQUEST
Used for requesting healthcare information or actions:
```python
# Example: Request differential diagnosis
content = {
    "action": "differential_diagnosis",
    "chief_complaint": "chest_pain",
    "symptoms": ["chest_pain", "shortness_of_breath"],
    "vital_signs": {"bp": "160/95", "hr": 110}
}
```

### 2. RESPONSE  
Response to a previous request:
```python
# Example: Diagnostic response
content = {
    "differential_diagnosis": [
        "acute_coronary_syndrome",
        "pulmonary_embolism",
        "aortic_dissection"
    ],
    "recommended_tests": ["ECG", "troponin", "chest_xray"]
}
```

### 3. NOTIFICATION
Clinical notifications and alerts:
```python
# Example: Critical lab value notification
content = {
    "type": "critical_lab_value",
    "lab_test": "troponin_i",
    "value": 15.2,
    "critical_threshold": 0.04
}
```

### 4. EMERGENCY
Emergency medical situations:
```python
# Example: Cardiac arrest emergency
content = {
    "emergency_type": "cardiac_arrest",
    "location": "room_305_icu",
    "vital_signs": {"hr": 0, "bp": "undetectable"},
    "witnessed": True
}
```

### 5. COORDINATION
Care team coordination:
```python
# Example: Multidisciplinary care coordination
content = {
    "coordination_type": "multidisciplinary_care",
    "care_plan": {...},
    "team_members": ["cardiology", "pharmacy", "nursing"]
}
```

### 6. EVENT
Healthcare events and updates:
```python
# Example: Patient admission event
content = {
    "event_type": "patient_admission",
    "admission_type": "emergency",
    "admitting_service": "cardiology"
}
```

## Clinical Urgency Levels

HMCP supports healthcare-appropriate urgency levels:

- **ROUTINE**: Standard non-urgent communications
- **URGENT**: Requires prompt attention (within hours)
- **EMERGENCY**: Immediate attention required (within minutes)

## Healthcare Roles

Supported healthcare roles for proper authorization and routing:

- **PHYSICIAN**: Licensed medical doctor
- **NURSE**: Registered or licensed practical nurse  
- **PHARMACIST**: Licensed pharmacist
- **AI_AGENT**: Artificial intelligence healthcare agent

## Security & Compliance

### HIPAA Compliance
- **PHI Protection**: Automatic identification and protection of Protected Health Information
- **Audit Trails**: Comprehensive logging of all healthcare communications
- **Access Controls**: Role-based access to patient information
- **Encryption**: End-to-end encryption of all messages containing PHI

### Security Context
```python
SecurityContext(
    user_id="agent_id",
    role=HealthcareRole.AI_AGENT,
    phi_flag=True,  # Contains Protected Health Information
    access_level="standard",
    audit_required=True
)
```

## Usage Examples

### Basic Agent Communication
```python
# Create HMCP agent
agent = HMCPAgent('diagnostic_agent', {
    'role': 'ai_agent',
    'capabilities': ['differential_diagnosis'],
    'emergency_capable': True
})

# Send clinical message
await agent.send_clinical_message(
    receiver_id='medical_knowledge_agent',
    message_type=HMCPMessageType.REQUEST,
    content={
        "action": "clinical_decision_support",
        "patient_data": {...}
    },
    patient_id="PATIENT_001",
    urgency=ClinicalUrgency.URGENT
)
```

### Emergency Response
```python
# Initiate emergency response
emergency_id = await agent.initiate_emergency_response(
    patient_id="PATIENT_001",
    emergency_type="cardiac_arrest",
    location="room_305_icu",
    details={
        "vital_signs": {"hr": 0, "bp": "undetectable"},
        "witnessed": True,
        "time_detected": datetime.now().isoformat()
    }
)
```

### Care Coordination
```python
# Coordinate care workflow
workflow_id = await agent.coordinate_care_workflow(
    patient_id="PATIENT_001",
    workflow_type="discharge_planning",
    participants=['pharmacy_agent', 'scheduling_agent'],
    care_plan={
        "discharge_date": "2024-12-17",
        "medications": [...],
        "follow_up": [...]
    }
)
```

## Command Line Interface

### Creating Agents
```bash
# Create diagnostic agent
python -m vita_agents.cli.hmcp_cli create diagnostic_copilot --role ai_agent --capabilities differential_diagnosis clinical_reasoning --emergency-capable

# Create medical knowledge agent  
python -m vita_agents.cli.hmcp_cli create medical_knowledge --role ai_agent --capabilities drug_interactions treatment_guidelines
```

### Sending Messages
```bash
# Send clinical request
python -m vita_agents.cli.hmcp_cli send medical_knowledge request '{"action": "medication_check", "drugs": ["warfarin", "aspirin"]}' --patient-id PATIENT_001 --urgency urgent

# Initiate emergency
python -m vita_agents.cli.hmcp_cli emergency PATIENT_001 cardiac_arrest "room_305_icu" --details '{"witnessed": true}'
```

### Interactive Mode
```bash
# Start interactive CLI
python -m vita_agents.cli.hmcp_cli interactive

# Interactive commands
hmcp> create diagnostic_agent
hmcp> list
hmcp> switch diagnostic_agent
hmcp(diagnostic_agent)> send medical_knowledge request {"action": "diagnosis"}
hmcp(diagnostic_agent)> metrics
hmcp(diagnostic_agent)> exit
```

## Healthcare Workflow Examples

### 1. Chest Pain Diagnosis
Complete workflow for chest pain evaluation involving:
- Patient data processing
- Differential diagnosis
- Clinical decision support
- Emergency procedure scheduling

### 2. Medication Interaction Check
Workflow for checking drug interactions:
- New prescription processing
- Interaction analysis
- Alert generation
- Alternative suggestions

### 3. Emergency Cardiac Arrest
Emergency response coordination:
- Immediate emergency detection
- Care team assembly
- ACLS protocol guidance
- Real-time coordination

### 4. Discharge Planning
Comprehensive discharge coordination:
- Assessment coordination
- Medication reconciliation
- Education material preparation
- Follow-up scheduling

### 5. Critical Lab Values
Critical lab value notification:
- Immediate clinical assessment
- Treatment protocol activation
- Urgent procedure scheduling
- Provider notification

## Integration with Existing Systems

### FHIR Integration
```python
# Validate FHIR resources
response = await agent.send_clinical_message(
    receiver_id='fhir_validator',
    message_type=HMCPMessageType.REQUEST,
    content={
        "action": "validate_fhir",
        "resource": fhir_patient_resource
    }
)
```

### HL7 Message Processing
```python
# Process HL7 messages
response = await agent.send_clinical_message(
    receiver_id='hl7_processor',
    message_type=HMCPMessageType.REQUEST,
    content={
        "action": "process_hl7",
        "message": hl7_adt_message
    }
)
```

## Performance Monitoring

### Agent Metrics
```python
metrics = agent.get_agent_metrics()
# Returns:
# {
#     "messages_sent": 45,
#     "messages_received": 38,
#     "emergency_responses": 3,
#     "workflow_completions": 12,
#     "average_response_time": 0.245,
#     "error_count": 1,
#     "active_conversations": 5,
#     "uptime": 3600.0
# }
```

### Health Status
```python
health = agent.get_health_status()
# Returns:
# {
#     "status": "healthy",
#     "hmcp_client_connected": true,
#     "hmcp_server_running": true,
#     "message_queue_size": 0,
#     "error_rate": 0.026,
#     "average_response_time": 0.245
# }
```

## Best Practices

### 1. Clinical Context
- Always include appropriate patient context for PHI-containing messages
- Use correct clinical urgency levels
- Specify healthcare roles for proper authorization

### 2. Security
- Mark messages containing PHI with appropriate security context
- Use encrypted communication for sensitive data
- Maintain comprehensive audit trails

### 3. Error Handling
- Implement robust error handling for clinical scenarios
- Provide fallback mechanisms for emergency situations
- Log all errors with appropriate clinical context

### 4. Performance
- Monitor response times for time-critical scenarios
- Implement efficient message routing for emergency situations
- Use appropriate message priorities

## Testing

### Unit Tests
```python
# Test HMCP message creation
def test_hmcp_message_creation():
    message = HMCPMessage(
        type=HMCPMessageType.REQUEST,
        sender_id="test_agent",
        receiver_id="target_agent",
        content={"test": "data"}
    )
    assert message.type == HMCPMessageType.REQUEST
    assert message.sender_id == "test_agent"
```

### Integration Tests
```python
# Test agent communication
async def test_agent_communication():
    agent1 = HMCPAgent('agent1', test_config)
    agent2 = HMCPAgent('agent2', test_config)
    
    success = await agent1.send_clinical_message(
        receiver_id='agent2',
        message_type=HMCPMessageType.REQUEST,
        content={"test": "message"}
    )
    
    assert success == True
```

## Future Enhancements

### Planned Features
1. **Advanced Clinical Workflows**: More specialized healthcare protocols
2. **AI/ML Integration**: Enhanced clinical decision support
3. **Real-time Monitoring**: Live dashboard for healthcare operations
4. **Mobile Support**: Mobile agent interfaces for healthcare providers
5. **Cloud Deployment**: Scalable cloud-based HMCP infrastructure

### Extensibility
The HMCP implementation is designed for extensibility:
- Custom message types for specialized workflows
- Pluggable authentication mechanisms
- Configurable routing strategies
- Custom clinical context providers

## Troubleshooting

### Common Issues

#### Agent Connection Problems
```bash
# Check agent health
python -m vita_agents.cli.hmcp_cli metrics --agent-id diagnostic_agent

# Check router status
python -m vita_agents.cli.hmcp_cli router
```

#### Message Delivery Failures
- Verify receiver agent is running and reachable
- Check network connectivity and firewall settings
- Validate message format and required fields
- Review security context and permissions

#### Performance Issues
- Monitor response times and queue sizes
- Check for resource constraints
- Review error logs for patterns
- Consider load balancing for high-volume scenarios

## Support and Documentation

For additional support and detailed API documentation:
- Review the inline code documentation
- Run the provided workflow examples
- Check the CLI help: `python -m vita_agents.cli.hmcp_cli --help`
- Monitor agent metrics and health status

## Conclusion

The HMCP integration provides Vita Agents with comprehensive healthcare multi-agent communication capabilities, enabling secure, compliant, and clinically-aware AI agent interactions. The protocol supports the full spectrum of healthcare workflows from routine communications to emergency response coordination, making it a powerful foundation for healthcare AI systems.