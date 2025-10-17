<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/2d04d9da-cf5b-4fe7-8b1f-8e6a861c529f" />

# 🏥 Vita Agents: Advanced Multi-Agent Healthcare AI Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![HIPAA Compliant](https://img.shields.io/badge/HIPAA-Compliant-green.svg)](https://www.hhs.gov/hipaa/)
[![FHIR R4/R5](https://img.shields.io/badge/FHIR-R4%2FR5-orange.svg)](https://hl7.org/fhir/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![GitHub release](https://img.shields.io/github/release/yasir2000/vita-agents.svg)](https://GitHub.com/yasir2000/vita-agents/releases/)
[![GitHub stars](https://img.shields.io/github/stars/yasir2000/vita-agents.svg?style=social&label=Star)](https://GitHub.com/yasir2000/vita-agents/stargazers/)

> **🚀 Latest Release v3.0.0**: Revolutionary Multi-Agent Healthcare Framework with Team Coordination!

An enterprise-grade, open-source framework that leverages advanced multi-agent AI systems to revolutionize healthcare workflows. Featuring specialized healthcare agents, intelligent team coordination, emergency response protocols, and seamless integration with 11+ FHIR engines. Built with inspiration from cutting-edge collaborative AI architectures for real-world healthcare applications.

## 🌟 What's New in v3.0.0

### 🤖 **Advanced Healthcare Agent Framework** (MAJOR UPDATE!)
Complete reimplementation with sophisticated multi-agent architecture:
- **3 Specialized Healthcare Agents**: AI Diagnostician, Pharmacist, and Care Coordinator with distinct roles and capabilities
- **Intelligent Task Assignment**: Automatic task routing based on agent expertise, workload, and patient severity
- **Collaborative Decision Making**: Complex cases handled through multi-agent collaboration and consensus
- **Performance Tracking**: Real-time monitoring of agent success rates, response times, and collaboration effectiveness
- **Clinical Reasoning**: Advanced diagnostic workflows with evidence-based decision support

### 👥 **Healthcare Team Management System** (NEW!)
Revolutionary team-based coordination inspired by advanced collaborative AI frameworks:
- **10+ Specialized Teams**: Emergency Response, ICU, Cardiac, Stroke, Primary Care, Surgical, and more
- **Emergency Response Protocols**: Standardized protocols for cardiac arrest, stroke, sepsis with defined response times
- **Dynamic Team Assembly**: Automatic team formation based on case requirements and agent availability
- **Multi-Team Coordination**: Complex cases managed across multiple specialized teams
- **Protocol Enforcement**: Automated adherence to clinical protocols and care pathways

### 🚨 **Emergency Response System** (NEW!)
Mission-critical emergency management capabilities:
- **Rapid Response Teams**: Sub-15-minute activation for cardiac arrest scenarios
- **Stroke Alert Protocols**: Automated stroke team coordination with thrombolytic preparation
- **Sepsis Management**: Multi-disciplinary sepsis response with early intervention protocols
- **Real-Time Coordination**: Live team coordination with role-based task assignment
- **Performance Analytics**: Emergency response time tracking and outcome analysis

### 🧠 **Advanced LLM Integration** (ENHANCED)
Seamless integration with multiple AI providers:
- **Multi-Provider Support**: Ollama (local), OpenAI, Anthropic, HuggingFace models
- **Healthcare-Optimized Models**: Specialized clinical reasoning and medical analysis models
- **Dynamic Model Selection**: Automatic model selection based on task type and requirements
- **Cost Optimization**: Intelligent routing to balance performance and cost
- **Clinical Prompting**: Specialized prompt templates for diagnostic and treatment workflows

### 🎯 **Key Features**
- 🤖 **Multi-Agent Collaboration**: Specialized AI agents working together on complex healthcare cases
- 👥 **Team-Based Coordination**: Dynamic team formation and emergency response protocols
- 🚨 **Emergency Management**: Rapid response systems for cardiac arrest, stroke, sepsis scenarios
- ⚡ **Parallel Operations**: Execute across multiple FHIR engines simultaneously  
- 🧠 **Advanced AI Integration**: Support for 10+ AI models with healthcare-optimized prompting
- 📊 **Performance Analytics**: Real-time monitoring of agent performance and team effectiveness
- 🔄 **Cross-Engine Validation**: Ensure FHIR compliance across implementations
- 🚀 **Data Migration**: Seamlessly migrate between different FHIR engines
- 🔐 **Multi-Auth Support**: OAuth2, SMART on FHIR, Basic Auth, Bearer tokens
- 🎨 **Professional CLI**: Beautiful command-line interface with 15+ healthcare commands
- 🏥 **Clinical Workflows**: Complete diagnosis, medication review, and care coordination workflows
- 🔒 **Security & Compliance**: HIPAA-compliant with comprehensive audit trails

## 🏗️ System Architecture

### Multi-Agent Healthcare Framework
```
🏥 VITA Healthcare AI Platform
├── 🤖 Healthcare Agents
│   ├── AI Diagnostician (Clinical reasoning, differential diagnosis)
│   ├── AI Pharmacist (Drug interactions, dosing, adverse effects)
│   └── Care Coordinator (Workflow management, patient navigation)
├── 👥 Healthcare Teams
│   ├── Emergency Response Team (Cardiac arrest, stroke protocols)
│   ├── ICU Team (Critical care management)
│   ├── Primary Care Team (Routine care coordination)
│   ├── Cardiac Team (Specialized cardiac care)
│   └── 6+ Additional Teams (Stroke, Surgical, Oncology, etc.)
├── 🧠 LLM Integration
│   ├── Local Models (Ollama, DeepSeek, Llama)
│   ├── Cloud Providers (OpenAI, Anthropic)
│   └── Healthcare Models (Clinical BERT, Med-Llama)
└── 📊 Management System
    ├── Task Assignment & Routing
    ├── Performance Monitoring
    ├── Emergency Protocols
    └── Team Coordination
```

## � Quick Start Guide

### Option 1: Complete System Demo
```bash
# Clone the repository
git clone https://github.com/yasir2000/vita-agents.git
cd vita-agents

# Install dependencies
pip install -r requirements.txt

# Run comprehensive system demonstration
python enhanced_cli.py demo
```

### Option 2: Interactive Healthcare AI Platform
```bash
# Start the enhanced CLI interface
python enhanced_cli.py dashboard

# Initialize healthcare agents
python enhanced_cli.py agents init

# Initialize healthcare teams
python enhanced_cli.py teams init

# Run emergency response simulation
python enhanced_cli.py teams emergency-response --type cardiac_arrest
```

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

### Option 3: Healthcare Portal
```bash
# Start the healthcare portal (auto-detects available port)
python start_portal.py

# Or use platform-specific scripts:
# Windows: double-click start_portal.bat
# Unix/Linux/macOS: ./start_portal.sh
```

**Access the portal at:** http://localhost:8080

## 💻 CLI Commands Reference

### 🏥 System Overview
```bash
# Complete system status dashboard
python enhanced_cli.py dashboard

# System version and components
python enhanced_cli.py version

# Comprehensive system demonstration
python enhanced_cli.py demo
```

### 🤖 Healthcare Agent Management
```bash
# Initialize default healthcare agents
python enhanced_cli.py agents init

# List all healthcare agents
python enhanced_cli.py agents list

# Show agent capabilities and specialties
python enhanced_cli.py agents capabilities

# Agent performance status
python enhanced_cli.py agents status

# Multi-agent workflow demonstration
python enhanced_cli.py agents workflow
```

### 🩺 Clinical AI Operations
```bash
# Collaborative AI diagnosis
python enhanced_cli.py agents diagnose \
  --age 65 \
  --gender female \
  --complaint "chest pain and shortness of breath" \
  --severity urgent

# Comprehensive medication review
python enhanced_cli.py agents medication-review \
  --age 72 \
  --gender male \
  --medications "metformin,lisinopril,atorvastatin" \
  --new-med "warfarin"
```

### 👥 Healthcare Team Management
```bash
# Initialize default healthcare teams
python enhanced_cli.py teams init

# List all healthcare teams
python enhanced_cli.py teams list

# Show detailed team information
python enhanced_cli.py teams details --name "Emergency Response Team"

# Team performance status
python enhanced_cli.py teams status

# Create new specialized team
python enhanced_cli.py teams create emergency_team --name "Trauma Team"
```

### 🚨 Emergency Response System
```bash
# Cardiac arrest emergency simulation
python enhanced_cli.py teams emergency-response \
  --type cardiac_arrest \
  --age 65 \
  --gender male

# Stroke alert protocol
python enhanced_cli.py teams emergency-response \
  --type stroke \
  --age 58 \
  --gender female

# Sepsis management protocol
python enhanced_cli.py teams emergency-response \
  --type sepsis \
  --age 72 \
  --gender male

# Multi-team coordination demonstration
python enhanced_cli.py teams workflow
```

### 🧠 LLM Integration
```bash
# List available AI models
python enhanced_cli.py llm list-models

# Show only healthcare-optimized models
python enhanced_cli.py llm list-models --healthcare

# Set active AI model
python enhanced_cli.py llm set-model "ollama:deepseek-coder:latest"

# Test AI model with clinical prompt
python enhanced_cli.py llm test \
  --prompt "What are the contraindications for warfarin therapy?" \
  --temp 0.3

# AI-powered clinical diagnosis
python enhanced_cli.py llm diagnose \
  --age 45 \
  --gender female \
  --complaint "severe headache with vision changes" \
  --vitals "BP: 180/110, HR: 88"

# Drug interaction analysis
python enhanced_cli.py llm drug-check \
  --current "warfarin,metformin,lisinopril" \
  --new "amoxicillin"
```

### 📊 Sample Data Management
```bash
# Generate sample healthcare data
python enhanced_cli.py data generate \
  --patients 100 \
  --scenarios 20 \
  --output healthcare_sample_data.json

# List sample patients
python enhanced_cli.py data list-patients --limit 10

# View clinical scenarios
python enhanced_cli.py data list-scenarios

# Show detailed scenario information
python enhanced_cli.py data scenario-details 0
```

## � Healthcare Use Cases & Examples

### 🚨 Emergency Response Scenario
```bash
# Critical cardiac arrest case
python enhanced_cli.py teams emergency-response \
  --type cardiac_arrest \
  --age 68 \
  --gender male

# Expected Output:
# 🚨 EMERGENCY RESPONSE SIMULATION
# Emergency: Cardiac Arrest
# 👤 Patient: 68-year-old male
# 🏥 Emergency Response Team activated
# 📋 Emergency Response Plan:
#   • Protocol: Cardiac Arrest Response
#   • Team Members: Dr. DiagnosisBot, PharmBot, CareCoordBot
#   • Response Time: <2 minutes
# ✅ Emergency response protocol activated successfully!
```

### 🤖 Multi-Agent Diagnosis
```bash
# Complex clinical case requiring collaboration
python enhanced_cli.py agents diagnose \
  --age 72 \
  --gender female \
  --complaint "chest pain, shortness of breath, leg swelling" \
  --history "diabetes,hypertension,heart_failure" \
  --medications "metformin,lisinopril,furosemide" \
  --severity urgent

# AI agents collaborate to provide:
# • Differential diagnosis considerations
# • Medication interaction analysis  
# • Care coordination recommendations
# • Risk stratification assessment
```

### 💊 Comprehensive Medication Review
```bash
# Multi-medication safety analysis
python enhanced_cli.py agents medication-review \
  --age 65 \
  --gender male \
  --medications "warfarin,metformin,atorvastatin,amlodipine" \
  --allergies "penicillin,sulfa" \
  --new-med "ciprofloxacin"

# PharmBot analyzes:
# • Drug-drug interactions
# • Dosing appropriateness
# • Allergy considerations
# • New medication safety
```

### 👥 Team Workflow Demonstration
```bash
# Complex case requiring multiple teams
python enhanced_cli.py teams workflow

# Demonstrates:
# • Emergency team initial response
# • ICU team coordination
# • Multi-team communication
# • Care transition planning
```

## �🎯 Project Overview

Vita Agents creates specialized AI agents that work collaboratively to handle different aspects of healthcare data processing, each with specific expertise in healthcare standards and workflows.

## 🤖 Sophisticated Agent Architecture

```mermaid
graph TB
    %% User Interfaces Layer
    subgraph UI["🎨 User Interfaces"]
        WEB["🌐 Web Portal<br/>Healthcare Dashboard"]
        CLI["💻 Enhanced CLI<br/>15+ Healthcare Commands"]
        API["🔌 REST API<br/>OpenAPI Docs"]
    end

    %% Orchestration Layer
    subgraph ORCH["🎼 Orchestration & Communication Layer"]
        ORCHESTRATOR["🎯 Agent Orchestrator<br/>Task Assignment & Routing"]
        TEAM_MGR["👥 Team Manager<br/>Emergency Protocols"]
        ROUTER["🔀 Message Router<br/>Load Balancing"]
        MONITOR["📊 Performance Monitor<br/>Real-time Analytics"]
    end

    %% Specialized Agents Layer
    subgraph AGENTS["🤖 Specialized Healthcare Agents"]
        direction TB
        
        subgraph CORE["🏥 Core Healthcare Agents"]
            DIAGNOSTICIAN["🩺 AI Diagnostician<br/>Clinical Reasoning<br/>Differential Diagnosis<br/>Pattern Recognition"]
            PHARMACIST["💊 AI Pharmacist<br/>Drug Interactions<br/>Dosing Optimization<br/>Adverse Effects"]
            COORDINATOR["📋 Care Coordinator<br/>Workflow Management<br/>Patient Navigation<br/>Care Planning"]
        end
        
        subgraph LEGACY["🔄 Legacy Integration Agents"]
            FHIR["📋 FHIR Agent<br/>Multi-Engine Support<br/>11+ FHIR Servers"]
            HL7["📨 HL7 Translation<br/>v2.x ↔ FHIR Conversion"]
            EHR["🏥 EHR Integration<br/>Epic, Cerner, Allscripts"]
        end
        
        subgraph INTELLIGENCE["🧠 AI Intelligence Layer"]
            LLM_MGR["� LLM Manager<br/>Ollama, OpenAI, Anthropic<br/>Healthcare Models"]
            REASONING["🧠 Clinical Reasoning<br/>Evidence-Based Analysis<br/>Decision Support"]
            NLP["📝 Medical NLP<br/>Clinical Notes Processing"]
        end
    end

    %% Healthcare Teams Layer
    subgraph TEAMS["👥 Healthcare Teams"]
        direction LR
        EMERGENCY["� Emergency Response<br/>Cardiac Arrest, Stroke<br/>Sepsis Management"]
        ICU["🏥 ICU Team<br/>Critical Care<br/>Monitoring"]
        PRIMARY["🩺 Primary Care<br/>Routine Cases<br/>Preventive Care"]
        SPECIALTY["⚕️ Specialty Teams<br/>Cardiac, Stroke, Surgical<br/>Oncology, Pediatric"]
    end

    %% Multi-Engine FHIR Support
    subgraph FHIR_ENGINES["🔄 Multi-Engine FHIR Support"]
        HAPI["HAPI FHIR<br/>hapifhir.io"]
        IBM["IBM FHIR<br/>Enterprise"]
        MEDPLUM["Medplum<br/>Cloud Native"]
        FIRELY["Firely .NET<br/>SDK"]
        SPARK["Spark FHIR<br/>Lightweight"]
        OTHERS["+ 6 More Engines<br/>LinuxForHealth, Aidbox<br/>Microsoft, Google, AWS"]
    end

    %% Healthcare Systems Integration
    subgraph SYSTEMS["🏥 Healthcare Systems"]
        EPIC["Epic<br/>MyChart API"]
        CERNER["Cerner<br/>PowerChart"]
        ALLSCRIPTS["Allscripts<br/>Developer API"]
        HL7_SYS["HL7 Systems<br/>v2.x, CDA"]
        DICOM["DICOM<br/>Medical Imaging"]
    end

    %% Emergency & Workflow Protocols
    subgraph EMERGENCY["🚨 Emergency & Care Protocols"]
        CARDIAC["💔 Cardiac Arrest<br/>ACLS Protocol"]
        STROKE["🧠 Stroke Alert<br/>Neuro Response"]
        SEPSIS["🦠 Sepsis Protocol<br/>Bundle Care"]
        RESPIRATORY["🫁 Respiratory Failure<br/>Ventilator Prep"]
        DISCHARGE["🏠 Discharge Planning<br/>Multi-disciplinary"]
    end

    %% Knowledge & Standards Layer
    subgraph KNOWLEDGE["📚 Healthcare Knowledge & Standards"]
        ONTOLOGIES["Medical Ontologies<br/>SNOMED CT, ICD-10"]
        LOINC["LOINC<br/>Lab Codes"]
        RXNORM["RxNorm<br/>Medications"]
        GUIDELINES["Clinical Guidelines<br/>Evidence-Based Care"]
        REGULATIONS["Regulations<br/>HIPAA, FDA"]
    end

    %% Infrastructure Layer
    subgraph INFRA["🏗️ Infrastructure & Security"]
        DATABASE["🗄️ PostgreSQL<br/>FHIR Resources"]
        REDIS["⚡ Redis<br/>Caching"]
        ELASTICSEARCH["🔍 Elasticsearch<br/>Search & Analytics"]
        ENCRYPTION["🔒 AES-256<br/>PHI Encryption"]
        AUDIT["📋 Audit Trails<br/>Compliance Logs"]
    end

    %% Connections - User Interfaces
    UI --> ORCH
    WEB --> ORCHESTRATOR
    CLI --> HMCP
    API --> ROUTER

    %% Connections - Orchestration
    ORCHESTRATOR --> AGENTS
    HMCP --> HMCP_AGENT
    %% Connections - Orchestration to Agents
    ORCHESTRATOR --> AGENTS
    TEAM_MGR --> TEAMS
    ROUTER --> CORE
    MONITOR --> AGENTS

    %% Connections - Core Healthcare Agents
    DIAGNOSTICIAN --> LLM_MGR
    PHARMACIST --> LLM_MGR
    COORDINATOR --> LLM_MGR
    LLM_MGR --> REASONING

    %% Connections - Teams to Agents
    EMERGENCY --> DIAGNOSTICIAN
    EMERGENCY --> PHARMACIST
    EMERGENCY --> COORDINATOR
    ICU --> DIAGNOSTICIAN
    PRIMARY --> COORDINATOR
    SPECIALTY --> DIAGNOSTICIAN

    %% Connections - Emergency Protocols
    EMERGENCY -.->|Activates| CARDIAC
    EMERGENCY -.->|Activates| STROKE
    EMERGENCY -.->|Activates| SEPSIS
    ICU -.->|Critical Care| CARDIAC
    SPECIALTY -.->|Specialized| STROKE

    %% Connections - Legacy Integration
    LEGACY --> FHIR_ENGINES
    FHIR --> SYSTEMS
    HL7 --> SYSTEMS
    EHR --> SYSTEMS
    
    %% Connections - Knowledge Integration
    AGENTS --> KNOWLEDGE
    TEAMS --> KNOWLEDGE
    REASONING --> KNOWLEDGE

    %% Connections - Infrastructure
    AGENTS --> INFRA
    TEAMS --> INFRA
    ORCH --> INFRA

    %% Styling
    classDef uiClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef orchClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef agentClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef teamClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef emergencyClass fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    classDef fhirClass fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    classDef infraClass fill:#fafafa,stroke:#424242,stroke-width:2px
    classDef knowledgeClass fill:#f1f8e9,stroke:#33691e,stroke-width:2px

    class UI,WEB,CLI,API uiClass
    class ORCH,ORCHESTRATOR,TEAM_MGR,ROUTER,MONITOR orchClass
    class AGENTS,CORE,LEGACY,INTELLIGENCE,DIAGNOSTICIAN,PHARMACIST,COORDINATOR,LLM_MGR,REASONING,NLP agentClass
    class TEAMS,EMERGENCY,ICU,PRIMARY,SPECIALTY teamClass
    class CARDIAC,STROKE,SEPSIS,TRAUMA emergencyClass
    class FHIR_ENGINES,HAPI,IBM,MEDPLUM,FIRELY,SPARK,OTHERS fhirClass
    class INFRA,DATABASE,REDIS,ELASTICSEARCH,ENCRYPTION,AUDIT infraClass
    class KNOWLEDGE,ONTOLOGIES,LOINC,RXNORM,GUIDELINES,REGULATIONS knowledgeClass
    class KNOWLEDGE,ONTOLOGIES,LOINC,RXNORM,GUIDELINES,REGULATIONS knowledgeClass
```

### 🏗️ Architecture Layers Explained

#### 🎨 **User Interface Layer**
- **Web Portal**: Healthcare dashboard with real-time monitoring and agent management
- **Interactive CLI**: HMCP-enabled command line for healthcare professionals
- **REST API**: OpenAPI-documented endpoints for system integration

#### 🎼 **Orchestration & Communication Layer**
- **Agent Orchestrator**: Central workflow management and agent coordination
- **HMCP Protocol**: Healthcare Model Context Protocol for clinical context
- **Message Router**: Intelligent load balancing and message routing
- **Health Monitor**: Real-time performance analytics and system health

#### 🤖 **Specialized Healthcare Agents**
- **Core Processing**: FHIR (multi-engine), HL7 translation, EHR integration
- **Clinical Intelligence**: Decision support, HMCP communication, NLP processing
- **Data & Security**: ML harmonization, HIPAA compliance, intelligent triage

#### 🔄 **Multi-Engine FHIR Support**
- **11+ FHIR Engines**: HAPI, IBM, Medplum, Firely, Spark, and cloud providers
- **Parallel Operations**: Simultaneous operations across multiple engines
- **Performance Benchmarking**: Engine comparison and optimization

#### 🚨 **Emergency & Care Protocols**
- **Emergency Response**: Cardiac arrest, stroke, sepsis, respiratory failure
- **Care Coordination**: Multi-disciplinary discharge planning and handoffs
- **Clinical Workflows**: Automated protocol execution and team notification

#### 📚 **Healthcare Knowledge & Standards**
- **Medical Ontologies**: SNOMED CT, ICD-10, LOINC, RxNorm
- **Clinical Guidelines**: Evidence-based care recommendations
- **Regulatory Compliance**: HIPAA, FDA, and international standards

#### 🏗️ **Infrastructure & Security**
- **Data Storage**: PostgreSQL with FHIR resource optimization
- **Performance**: Redis caching and Elasticsearch analytics
- **Security**: AES-256 encryption and comprehensive audit trails

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

### 8. **HMCP Agent** (`hmcp_agent.py`) - **NEW!**
- ✅ Healthcare Model Context Protocol implementation
- ✅ Clinical context awareness with patient data and urgency levels
- ✅ Emergency response protocols (cardiac arrest, stroke, sepsis, respiratory failure)
- ✅ Care coordination and multi-disciplinary team communication
- ✅ HIPAA-compliant secure messaging with audit trails
- ✅ Healthcare workflow orchestration and real-time clinical guidance

### 🔄 HMCP Communication Flow

```mermaid
sequenceDiagram
    participant Provider as 👨‍⚕️ Healthcare Provider
    participant HMCP as 🏥 HMCP Agent
    participant Diagnostic as 🧠 Diagnostic Agent
    participant Knowledge as 📚 Medical Knowledge
    participant Emergency as 🚨 Emergency System
    participant EHR as 🏥 EHR System

    Note over Provider,EHR: Emergency Chest Pain Scenario

    Provider->>HMCP: Patient presents with chest pain
    HMCP->>Diagnostic: Clinical assessment request
    Note right of HMCP: Patient context:<br/>ID, urgency level,<br/>clinical context
    
    Diagnostic->>Knowledge: Request differential diagnosis
    Knowledge-->>Diagnostic: Acute coronary syndrome likely
    Diagnostic->>HMCP: Urgent intervention needed
    
    HMCP->>Emergency: Initiate cardiac protocol
    Note right of Emergency: ACLS protocol:<br/>- Cardiology team<br/>- Cath lab prep<br/>- Medications
    
    HMCP->>EHR: Update patient record
    Emergency-->>Provider: Team assembled, ready for intervention
    
    Note over Provider,EHR: Care coordination complete in <5 minutes
```

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

### 🏥 **HMCP - Healthcare Multi-agent Communication** (NEW!)
- **Clinical Workflows**: 5+ pre-built healthcare workflows (chest pain diagnosis, medication interactions, emergency response)
- **Emergency Protocols**: Automated cardiac arrest, stroke, sepsis, and respiratory failure responses
- **Care Coordination**: Multi-disciplinary team communication and workflow orchestration
- **HIPAA Compliance**: Secure PHI handling with comprehensive audit trails
- **Healthcare Roles**: Support for physician, nurse, pharmacist, and AI agent roles
- **Clinical Urgency**: Routine, urgent, and emergency message prioritization
- **Interactive CLI**: Full-featured command line interface for healthcare agent management

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

### HMCP Healthcare Agent Communication (NEW!)

```bash
# Create healthcare agents with HMCP
python -m vita_agents.cli.hmcp_cli create diagnostic_copilot --role ai_agent --capabilities differential_diagnosis --emergency-capable

# Send clinical messages between agents
python -m vita_agents.cli.hmcp_cli send medical_knowledge request '{"action": "medication_check", "drugs": ["warfarin", "aspirin"]}' --patient-id PATIENT_001 --urgency urgent

# Initiate emergency response
python -m vita_agents.cli.hmcp_cli emergency PATIENT_001 cardiac_arrest "room_305_icu" --details '{"witnessed": true}'

# Coordinate care workflows
python -m vita_agents.cli.hmcp_cli coordinate PATIENT_001 discharge_planning diagnostic_copilot medical_knowledge --care-plan '{"discharge_date": "2024-12-17"}'

# Interactive healthcare agent management
python -m vita_agents.cli.hmcp_cli interactive

# Run healthcare workflow examples
python examples/hmcp_workflows.py
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

### 🚀 Getting Started
- **[User Guide](docs/user-guide.md)** - Complete user documentation
- **[Upgrade Guide](docs/UPGRADE_GUIDE.md)** - Upgrading from previous versions
- **[Features Overview](docs/FEATURES_OVERVIEW.md)** - Comprehensive feature matrix

### 🏥 HMCP Healthcare Communication (NEW!)
- **[HMCP Integration Guide](docs/HMCP_INTEGRATION.md)** - Complete HMCP protocol documentation
- **[Healthcare Workflow Examples](examples/hmcp_workflows.py)** - 5 complete clinical workflows
- **[HMCP CLI Reference](vita_agents/cli/hmcp_cli.py)** - Interactive CLI documentation

### 🔧 Technical Documentation
- **[API Reference](docs/api-reference.md)** - Detailed API documentation  
- **[Agent Development Guide](docs/agent-development.md)** - Creating custom agents
- **[Healthcare Standards Guide](docs/healthcare-standards.md)** - FHIR, HL7, and EHR integration
- **[Deployment Guide](docs/deployment.md)** - Production deployment instructions
- **[Security Guide](docs/security.md)** - HIPAA compliance and security best practices

### 📋 Release Information
- **[Release Notes](RELEASE_NOTES.md)** - Detailed v2.1.0 release information
- **[Changelog](CHANGELOG.md)** - Complete change history
- **[Version History](docs/VERSION_HISTORY.md)** - Comprehensive version timeline

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

## ⚡ Quick Reference

### Essential Commands
```bash
# Start healthcare portal
python start_portal.py

# Create HMCP healthcare agents
python -m vita_agents.cli.hmcp_cli create diagnostic_copilot --role ai_agent --emergency-capable

# Test multi-engine FHIR operations
python -m vita_agents.cli.fhir_engines_cli test-connections

# Run healthcare workflow examples
python examples/hmcp_workflows.py

# Interactive healthcare agent management
python -m vita_agents.cli.hmcp_cli interactive
```

### Key URLs (Default Setup)
- **Healthcare Portal**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **HMCP Agent Management**: CLI-based interface
- **Multi-Engine FHIR**: Integrated in portal and CLI

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
