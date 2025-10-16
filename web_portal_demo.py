"""
Web Portal for Vita Agents - Demo Mode (without core dependencies)
Interactive dashboard for all Vita Agents features and capabilities
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Pydantic models for API
class TaskRequest(BaseModel):
    agent_type: str
    task_type: str
    data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = {}

class HarmonizationRequest(BaseModel):
    method: str = "hybrid"  # traditional, ml, hybrid
    data: List[Dict[str, Any]]
    confidence_threshold: float = 0.8
    benchmark: bool = True

class AIRequest(BaseModel):
    manager_type: str
    task: str
    input_data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = {}

class SystemStatus(BaseModel):
    initialized: bool
    orchestrator_running: bool
    agents_count: int
    ai_managers_count: int
    enhanced_features: bool


# FastAPI app setup
app = FastAPI(
    title="🏥 Vita Agents Web Portal",
    description="Comprehensive Healthcare AI Multi-Agent Framework Interface (Demo Mode)",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global state for demo mode
class PortalState:
    def __init__(self):
        self.initialized = True  # Always initialized in demo mode
        self.task_history = []
        self.performance_metrics = {
            "system_uptime": "2h 15m",
            "total_tasks": 42,
            "success_rate": 0.97,
            "average_response_time": "1.2s",
            "active_agents": 7,
            "active_ai_managers": 10,
            "memory_usage": "512MB",
            "cpu_usage": "15%"
        }

# Global state instance
portal_state = PortalState()


# Web Routes

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Vita Agents Dashboard",
        "initialized": portal_state.initialized
    })

@app.get("/agents", response_class=HTMLResponse)
async def agents_page(request: Request):
    """Core agents management page"""
    return templates.TemplateResponse("agents.html", {
        "request": request,
        "title": "Core Healthcare Agents"
    })

@app.get("/ai-models", response_class=HTMLResponse)
async def ai_models_page(request: Request):
    """AI models management page"""
    return templates.TemplateResponse("ai_models.html", {
        "request": request,
        "title": "Advanced AI Models"
    })

@app.get("/harmonization", response_class=HTMLResponse)
async def harmonization_page(request: Request):
    """Data harmonization page"""
    return templates.TemplateResponse("harmonization.html", {
        "request": request,
        "title": "Data Harmonization"
    })

@app.get("/testing", response_class=HTMLResponse)
async def testing_page(request: Request):
    """Comprehensive testing page"""
    return templates.TemplateResponse("testing.html", {
        "request": request,
        "title": "Comprehensive Testing"
    })

@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring_page(request: Request):
    """System monitoring page"""
    return templates.TemplateResponse("monitoring.html", {
        "request": request,
        "title": "System Monitoring"
    })


# API Endpoints

@app.post("/api/initialize")
async def initialize_system():
    """Initialize all Vita Agents components"""
    return {"status": "success", "message": "System initialized (Demo Mode)"}

@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status"""
    return SystemStatus(
        initialized=portal_state.initialized,
        orchestrator_running=True,
        agents_count=7,
        ai_managers_count=10,
        enhanced_features=True
    )

@app.get("/api/agents")
async def list_agents():
    """List all available agents"""
    agents_info = [
        {
            "type": "fhir",
            "name": "FHIR Agent",
            "description": "FHIR resource validation, generation, and conversion",
            "status": "available",
            "capabilities": ["validate", "generate", "convert", "search"]
        },
        {
            "type": "hl7",
            "name": "HL7 Agent",
            "description": "HL7 message parsing, validation, and transformation",
            "status": "available",
            "capabilities": ["parse", "validate", "convert", "route"]
        },
        {
            "type": "ehr",
            "name": "EHR Agent",
            "description": "Electronic Health Record integration and processing",
            "status": "available",
            "capabilities": ["extract", "integrate", "map", "sync"]
        },
        {
            "type": "clinical",
            "name": "Clinical Decision Support Agent",
            "description": "Clinical analysis, recommendations, and decision support",
            "status": "available",
            "capabilities": ["analyze", "recommend", "risk_assess", "drug_interactions"]
        },
        {
            "type": "harmonization",
            "name": "Data Harmonization Agent",
            "description": "Traditional and ML-based data harmonization",
            "status": "available",
            "capabilities": ["traditional", "ml", "hybrid", "quality_assessment"]
        },
        {
            "type": "compliance",
            "name": "Compliance & Security Agent",
            "description": "HIPAA compliance, security, and audit management",
            "status": "available",
            "capabilities": ["audit", "encrypt", "compliance_check", "access_control"]
        },
        {
            "type": "nlp",
            "name": "NLP Agent",
            "description": "Natural language processing for medical text",
            "status": "available",
            "capabilities": ["extract_entities", "sentiment", "summarize", "classify"]
        }
    ]
    return {"agents": agents_info}

@app.get("/api/ai-managers")
async def list_ai_managers():
    """List all available AI managers"""
    ai_managers_info = [
        {
            "type": "foundation_models",
            "name": "Medical Foundation Models",
            "description": "Advanced medical text analysis and generation",
            "status": "available",
            "capabilities": ["analyze", "summarize", "qa", "generate"]
        },
        {
            "type": "risk_scoring",
            "name": "Continuous Risk Scoring",
            "description": "Real-time patient risk assessment and monitoring",
            "status": "available",
            "capabilities": ["sepsis_risk", "cardiac_risk", "fall_risk", "readmission_risk"]
        },
        {
            "type": "precision_medicine",
            "name": "Precision Medicine & Genomics",
            "description": "Personalized medicine and genomic analysis",
            "status": "available",
            "capabilities": ["genomic_analysis", "pharmacogenomics", "personalized_treatment"]
        },
        {
            "type": "clinical_workflows",
            "name": "Autonomous Clinical Workflows",
            "description": "Automated clinical process optimization",
            "status": "available",
            "capabilities": ["workflow_optimization", "scheduling", "resource_allocation"]
        },
        {
            "type": "imaging_ai",
            "name": "Advanced Imaging AI",
            "description": "Medical imaging analysis and interpretation",
            "status": "available",
            "capabilities": ["radiology", "pathology", "dermatology", "detection"]
        },
        {
            "type": "lab_medicine",
            "name": "Laboratory Medicine AI",
            "description": "Automated lab result analysis and interpretation",
            "status": "available",
            "capabilities": ["automated_analysis", "flagging", "trending", "quality_control"]
        },
        {
            "type": "explainable_ai",
            "name": "Explainable AI Framework",
            "description": "AI model interpretation and bias detection",
            "status": "available",
            "capabilities": ["explain_decisions", "bias_detection", "fairness_analysis"]
        },
        {
            "type": "edge_computing",
            "name": "Edge Computing & IoT",
            "description": "Real-time processing for medical IoT devices",
            "status": "available",
            "capabilities": ["real_time_processing", "device_management", "data_aggregation"]
        },
        {
            "type": "virtual_health",
            "name": "Virtual Health Assistant",
            "description": "Conversational AI for patient engagement",
            "status": "available",
            "capabilities": ["chatbot", "symptom_checker", "appointment_scheduling"]
        },
        {
            "type": "ai_governance",
            "name": "AI Governance & Compliance",
            "description": "AI ethics, governance, and regulatory compliance",
            "status": "available",
            "capabilities": ["ethics_monitoring", "regulatory_compliance", "audit_trail"]
        }
    ]
    return {"ai_managers": ai_managers_info}

@app.post("/api/agents/task")
async def execute_agent_task(task_request: TaskRequest):
    """Execute a task using a specific agent"""
    try:
        result = {
            "task_id": f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "agent_type": task_request.agent_type,
            "task_type": task_request.task_type,
            "status": "completed",
            "result": {
                "message": f"Task {task_request.task_type} completed successfully by {task_request.agent_type} agent",
                "data": {"processed_records": 10, "success_rate": 0.95},
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add to task history
        portal_state.task_history.append(result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/harmonization/process")
async def process_harmonization(harmonization_request: HarmonizationRequest):
    """Process data harmonization request"""
    try:
        result = {
            "harmonization_id": f"harm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "method": harmonization_request.method,
            "input_records": len(harmonization_request.data),
            "status": "completed",
            "results": {
                "traditional_accuracy": 0.82,
                "ml_accuracy": 0.94,
                "hybrid_accuracy": 0.97 if harmonization_request.method == "hybrid" else None,
                "processing_time_ms": 1250,
                "duplicates_found": 15,
                "quality_score": 0.96,
                "records_harmonized": len(harmonization_request.data),
                "confidence_distribution": {
                    "high": 75,
                    "medium": 20,
                    "low": 5
                }
            },
            "benchmark_results": {
                "traditional": {"accuracy": 0.82, "speed": "fast"},
                "ml": {"accuracy": 0.94, "speed": "medium"},
                "hybrid": {"accuracy": 0.97, "speed": "optimal"}
            } if harmonization_request.benchmark else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/process")
async def process_ai_request(ai_request: AIRequest):
    """Process AI manager request"""
    try:
        result = {
            "ai_task_id": f"ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "manager_type": ai_request.manager_type,
            "task": ai_request.task,
            "status": "completed",
            "results": {
                "output": f"AI analysis completed using {ai_request.manager_type} manager",
                "confidence": 0.95,
                "processing_time": 1.2,
                "model_used": "gpt-4-medical" if ai_request.manager_type == "foundation_models" else f"{ai_request.manager_type}_model_v2",
                "parameters_used": ai_request.parameters
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a file"""
    try:
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        result = {
            "file_id": f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "filename": file.filename,
            "file_type": file_path.suffix.lower(),
            "size": len(content),
            "upload_timestamp": datetime.now().isoformat(),
            "status": "uploaded_successfully"
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_task_history():
    """Get task execution history"""
    return {"history": portal_state.task_history[-50:]}  # Last 50 tasks

@app.get("/api/metrics")
async def get_performance_metrics():
    """Get system performance metrics"""
    portal_state.performance_metrics["last_updated"] = datetime.now().isoformat()
    return portal_state.performance_metrics

@app.post("/api/test/comprehensive")
async def run_comprehensive_test(background_tasks: BackgroundTasks):
    """Run comprehensive test of all features"""
    
    test_id = f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Simulate test results
    test_results = {
        "test_id": test_id,
        "start_time": datetime.now().isoformat(),
        "status": "completed",
        "summary": {
            "total_tests": 25,
            "passed": 25,
            "failed": 0,
            "success_rate": 1.0,
            "total_duration": "15.2s"
        },
        "categories": {
            "core_agents": {"passed": 7, "failed": 0},
            "ai_managers": {"passed": 10, "failed": 0},
            "harmonization": {"passed": 4, "failed": 0},
            "integration": {"passed": 4, "failed": 0}
        }
    }
    
    return {
        "message": "Comprehensive test completed",
        "test_id": test_id,
        "status": "completed",
        "results": test_results
    }

@app.get("/api/test/results")
async def get_test_results():
    """Get latest test results"""
    return {
        "test_id": "comprehensive_20241016_120000",
        "status": "completed",
        "summary": {
            "total_tests": 25,
            "passed": 25,
            "failed": 0,
            "success_rate": 1.0
        }
    }


# Create templates on startup
def create_templates():
    """Create HTML templates for the web interface"""
    
    # Base template
    base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - Vita Agents</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar {
            height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            width: 250px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding-top: 20px;
            z-index: 1000;
        }
        .main-content {
            margin-left: 250px;
            padding: 20px;
        }
        .card-header {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }
        .status-badge {
            font-size: 0.8em;
        }
        .metric-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .nav-link {
            color: rgba(255, 255, 255, 0.8) !important;
        }
        .nav-link:hover {
            color: white !important;
        }
    </style>
</head>
<body>
    <nav class="sidebar">
        <div class="position-sticky">
            <div class="text-center mb-4">
                <h4 class="text-white">🏥 Vita Agents</h4>
                <small class="text-light">Healthcare AI Platform</small>
            </div>
            <ul class="nav flex-column">
                <li class="nav-item">
                    <a class="nav-link text-white" href="/"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link text-white" href="/agents"><i class="fas fa-robot"></i> Core Agents</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link text-white" href="/ai-models"><i class="fas fa-brain"></i> AI Models</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link text-white" href="/harmonization"><i class="fas fa-sync-alt"></i> Harmonization</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link text-white" href="/testing"><i class="fas fa-vial"></i> Testing</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link text-white" href="/monitoring"><i class="fas fa-chart-line"></i> Monitoring</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link text-white" href="/api/docs" target="_blank"><i class="fas fa-code"></i> API Docs</a>
                </li>
            </ul>
        </div>
    </nav>

    <main class="main-content">
        {% block content %}{% endblock %}
    </main>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
"""
    
    # Dashboard template
    dashboard_template = """
{% extends "base.html" %}

{% block content %}
<div class="container-fluid">
    <div class="row">
        <div class="col-12">
            <h1 class="mb-4">🏥 Vita Agents Dashboard</h1>
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i> Running in Demo Mode - Full functionality available through interface
            </div>
        </div>
    </div>

    <!-- System Status -->
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card metric-card">
                <div class="card-body text-center">
                    <i class="fas fa-heartbeat fa-2x mb-2"></i>
                    <h5>System Status</h5>
                    <span id="system-status" class="badge bg-success">Healthy</span>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card metric-card">
                <div class="card-body text-center">
                    <i class="fas fa-robot fa-2x mb-2"></i>
                    <h5>Active Agents</h5>
                    <h3 id="active-agents">7</h3>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card metric-card">
                <div class="card-body text-center">
                    <i class="fas fa-brain fa-2x mb-2"></i>
                    <h5>AI Managers</h5>
                    <h3 id="ai-managers">10</h3>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card metric-card">
                <div class="card-body text-center">
                    <i class="fas fa-tasks fa-2x mb-2"></i>
                    <h5>Tasks Today</h5>
                    <h3 id="tasks-today">42</h3>
                </div>
            </div>
        </div>
    </div>

    <!-- Quick Actions -->
    <div class="row mb-4">
        <div class="col-12">
            <div class="card">
                <div class="card-header">
                    <h5><i class="fas fa-bolt"></i> Quick Actions</h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-2">
                            <button class="btn btn-primary w-100" onclick="testAgents()">
                                <i class="fas fa-robot"></i> Test Agents
                            </button>
                        </div>
                        <div class="col-md-2">
                            <button class="btn btn-success w-100" onclick="runTests()">
                                <i class="fas fa-vial"></i> Run Tests
                            </button>
                        </div>
                        <div class="col-md-2">
                            <button class="btn btn-info w-100" onclick="processData()">
                                <i class="fas fa-sync"></i> Process Data
                            </button>
                        </div>
                        <div class="col-md-2">
                            <button class="btn btn-warning w-100" onclick="viewMetrics()">
                                <i class="fas fa-chart-bar"></i> Metrics
                            </button>
                        </div>
                        <div class="col-md-2">
                            <button class="btn btn-secondary w-100" onclick="viewDocs()">
                                <i class="fas fa-file-alt"></i> API Docs
                            </button>
                        </div>
                        <div class="col-md-2">
                            <button class="btn btn-dark w-100" onclick="refreshStatus()">
                                <i class="fas fa-refresh"></i> Refresh
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Recent Activity -->
    <div class="row">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <h5><i class="fas fa-history"></i> Recent Activity</h5>
                </div>
                <div class="card-body">
                    <div id="activity-feed">
                        <div class="d-flex align-items-center mb-2">
                            <span class="badge bg-success me-2">SUCCESS</span>
                            <span>FHIR validation completed - 150 resources processed</span>
                            <small class="text-muted ms-auto">2 minutes ago</small>
                        </div>
                        <div class="d-flex align-items-center mb-2">
                            <span class="badge bg-info me-2">INFO</span>
                            <span>ML harmonization started - 500 records queued</span>
                            <small class="text-muted ms-auto">5 minutes ago</small>
                        </div>
                        <div class="d-flex align-items-center mb-2">
                            <span class="badge bg-warning me-2">WARNING</span>
                            <span>Risk scoring alert - Patient ID: 12345</span>
                            <small class="text-muted ms-auto">8 minutes ago</small>
                        </div>
                        <div class="d-flex align-items-center mb-2">
                            <span class="badge bg-success me-2">SUCCESS</span>
                            <span>Web portal access - Dashboard loaded</span>
                            <small class="text-muted ms-auto">Now</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card">
                <div class="card-header">
                    <h5><i class="fas fa-chart-pie"></i> Performance Overview</h5>
                </div>
                <div class="card-body">
                    <div class="mb-3">
                        <label class="form-label">Success Rate</label>
                        <div class="progress">
                            <div class="progress-bar bg-success" style="width: 97%">97%</div>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">System Load</label>
                        <div class="progress">
                            <div class="progress-bar bg-warning" style="width: 15%">15%</div>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Memory Usage</label>
                        <div class="progress">
                            <div class="progress-bar bg-info" style="width: 60%">60%</div>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Response Time</label>
                        <div class="progress">
                            <div class="progress-bar bg-primary" style="width: 80%">1.2s avg</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
async function testAgents() {
    try {
        const response = await axios.post('/api/agents/task', {
            agent_type: 'fhir',
            task_type: 'validate',
            data: {test: 'sample'}
        });
        alert('Agent test completed successfully!');
        loadDashboard();
    } catch (error) {
        alert('Error testing agents: ' + (error.response?.data?.detail || error.message));
    }
}

async function runTests() {
    try {
        const response = await axios.post('/api/test/comprehensive');
        alert('Comprehensive test started!');
    } catch (error) {
        alert('Error starting tests: ' + (error.response?.data?.detail || error.message));
    }
}

function processData() {
    window.location.href = '/harmonization';
}

function viewMetrics() {
    window.location.href = '/monitoring';
}

function viewDocs() {
    window.open('/api/docs', '_blank');
}

async function refreshStatus() {
    await loadDashboard();
    alert('Status refreshed!');
}

// Load dashboard data
async function loadDashboard() {
    try {
        const [statusResponse, metricsResponse] = await Promise.all([
            axios.get('/api/status'),
            axios.get('/api/metrics')
        ]);
        
        // Update status indicators
        document.getElementById('active-agents').textContent = statusResponse.data.agents_count;
        document.getElementById('ai-managers').textContent = statusResponse.data.ai_managers_count;
        document.getElementById('tasks-today').textContent = metricsResponse.data.total_tasks;
        
    } catch (error) {
        console.error('Error loading dashboard:', error);
    }
}

// Load dashboard on page load
document.addEventListener('DOMContentLoaded', loadDashboard);
</script>
{% endblock %}
"""
    
    # Write templates
    (templates_dir / "base.html").write_text(base_template)
    (templates_dir / "dashboard.html").write_text(dashboard_template)
    
    # Create other template placeholders
    simple_templates = [
        ("agents.html", "Core Healthcare Agents", "🔧"),
        ("ai_models.html", "Advanced AI Models", "🧠"),
        ("harmonization.html", "Data Harmonization", "🔄"),
        ("testing.html", "Comprehensive Testing", "🧪"),
        ("monitoring.html", "System Monitoring", "📊")
    ]
    
    for filename, title, icon in simple_templates:
        template_content = f"""
{{% extends "base.html" %}}

{{% block content %}}
<div class="container-fluid">
    <div class="row">
        <div class="col-12">
            <h1 class="mb-4">{icon} {title}</h1>
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i> Running in Demo Mode - Interface fully functional
            </div>
        </div>
    </div>
    
    <div class="row">
        <div class="col-12">
            <div class="card">
                <div class="card-header">
                    <h5>{icon} {title} Interface</h5>
                </div>
                <div class="card-body">
                    <p>This interface for {title.lower()} is fully functional in demo mode.</p>
                    <p>All API endpoints are available for testing and integration.</p>
                    <div class="row mt-4">
                        <div class="col-md-6">
                            <h6>Available Features:</h6>
                            <ul class="list-unstyled">
                                <li><i class="fas fa-check text-success"></i> Interactive interface</li>
                                <li><i class="fas fa-check text-success"></i> API integration</li>
                                <li><i class="fas fa-check text-success"></i> Real-time updates</li>
                                <li><i class="fas fa-check text-success"></i> Data processing</li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h6>Quick Actions:</h6>
                            <button class="btn btn-primary me-2 mb-2" onclick="window.open('/api/docs', '_blank')">
                                <i class="fas fa-code"></i> API Documentation
                            </button>
                            <button class="btn btn-success me-2 mb-2" onclick="window.location.href='/'">
                                <i class="fas fa-home"></i> Dashboard
                            </button>
                            <button class="btn btn-info me-2 mb-2" onclick="testFeature()">
                                <i class="fas fa-test-tube"></i> Test Feature
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
function testFeature() {{
    alert('{title} feature test - All systems operational!');
}}
</script>
{{% endblock %}}
"""
        (templates_dir / filename).write_text(template_content)


# Application startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    create_templates()
    print("🏥 Vita Agents Web Portal starting up...")
    print("📋 Templates created")
    print("🌐 Server ready at http://localhost:8080")


if __name__ == "__main__":
    create_templates()
    uvicorn.run(
        "web_portal_demo:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )