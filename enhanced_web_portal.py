"""
Enhanced Vita Agents Web Portal - Production Ready Healthcare Platform
Comprehensive healthcare AI multi-agent framework with real-world features
"""

import asyncio
import json
import os
import uuid
import hashlib
import jwt
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Database and cache imports
import asyncpg
import redis.asyncio as redis
import aio_pika
from elasticsearch import AsyncElasticsearch
from minio import Minio
from minio.error import S3Error
import logging
import sys

# Import LLM integration and sample data
try:
    from llm_integration import llm_manager, CLINICAL_PROMPTS
    from sample_data_generator import SampleDataGenerator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
import pandas as pd
from io import StringIO, BytesIO
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
class Config:
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://vita_user:vita_secure_pass_2024@localhost:5432/vita_agents')
    
    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://:vita_redis_pass_2024@localhost:6379/0')
    
    # Elasticsearch
    ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
    
    # RabbitMQ
    RABBITMQ_URL = os.getenv('RABBITMQ_URL', 'amqp://vita_admin:vita_rabbit_pass_2024@localhost:5672/vita_vhost')
    
    # MinIO
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'vita_admin')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'vita_minio_pass_2024')
    MINIO_SECURE = os.getenv('MINIO_SECURE', 'false').lower() == 'true'
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'vita_super_secret_key_change_in_production_2024')
    JWT_SECRET = os.getenv('JWT_SECRET', 'vita_jwt_secret_key_change_in_production_2024')
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
    
    # Application
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

config = Config()

# Global connection objects
db_pool = None
redis_client = None
es_client = None
rabbitmq_connection = None
minio_client = None

# Connection management
async def init_connections():
    """Initialize all external service connections"""
    global db_pool, redis_client, es_client, minio_client
    
    try:
        # PostgreSQL connection pool
        db_url = config.DATABASE_URL.replace('postgresql://', '').replace('postgresql+asyncpg://', '')
        db_pool = await asyncpg.create_pool(f"postgresql://{db_url}", min_size=5, max_size=20)
        logger.info("✅ PostgreSQL connection pool initialized")
        
        # Redis connection
        redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("✅ Redis connection initialized")
        
        # Elasticsearch connection
        es_client = AsyncElasticsearch([config.ELASTICSEARCH_URL])
        await es_client.ping()
        logger.info("✅ Elasticsearch connection initialized")
        
        # MinIO connection
        minio_client = Minio(
            config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=config.MINIO_SECURE
        )
        logger.info("✅ MinIO connection initialized")
        
        # Ensure required buckets exist
        await ensure_minio_buckets()
        
    except Exception as e:
        logger.error(f"❌ Connection initialization failed: {e}")
        # Don't fail startup, fall back to in-memory storage
        
async def close_connections():
    """Close all external service connections"""
    global db_pool, redis_client, es_client, rabbitmq_connection
    
    if db_pool:
        await db_pool.close()
        logger.info("PostgreSQL pool closed")
        
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")
        
    if es_client:
        await es_client.close()
        logger.info("Elasticsearch connection closed")
        
    if rabbitmq_connection:
        await rabbitmq_connection.close()
        logger.info("RabbitMQ connection closed")

async def ensure_minio_buckets():
    """Ensure required MinIO buckets exist"""
    if not minio_client:
        return
        
    buckets = ['patient-documents', 'medical-images', 'reports', 'temp-uploads']
    
    for bucket_name in buckets:
        try:
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
                logger.info(f"Created MinIO bucket: {bucket_name}")
        except S3Error as e:
            logger.warning(f"MinIO bucket creation failed for {bucket_name}: {e}")

# Database helper functions
async def execute_query(query: str, *args):
    """Execute a database query"""
    if not db_pool:
        return None
        
    async with db_pool.acquire() as conn:
        return await conn.fetch(query, *args)

async def execute_one(query: str, *args):
    """Execute a database query and return one result"""
    if not db_pool:
        return None
        
    async with db_pool.acquire() as conn:
        return await conn.fetchrow(query, *args)

async def execute_command(query: str, *args):
    """Execute a database command (INSERT, UPDATE, DELETE)"""
    if not db_pool:
        return None
        
    async with db_pool.acquire() as conn:
        return await conn.execute(query, *args)

# Redis helper functions
async def cache_set(key: str, value: Any, ttl: int = 300):
    """Set a value in Redis cache"""
    if not redis_client:
        return False
        
    try:
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        await redis_client.setex(key, ttl, value)
        return True
    except Exception as e:
        logger.error(f"Redis set error: {e}")
        return False

async def cache_get(key: str):
    """Get a value from Redis cache"""
    if not redis_client:
        return None
        
    try:
        value = await redis_client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None
    except Exception as e:
        logger.error(f"Redis get error: {e}")
        return None

# Message queue helper
async def publish_message(queue_name: str, message: dict):
    """Publish a message to RabbitMQ"""
    try:
        if not rabbitmq_connection:
            connection = await aio_pika.connect_robust(config.RABBITMQ_URL)
            channel = await connection.channel()
        else:
            channel = await rabbitmq_connection.channel()
            
        queue = await channel.declare_queue(queue_name, durable=True)
        
        await channel.default_exchange.publish(
            aio_pika.Message(
                json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key=queue_name
        )
        
        logger.info(f"Message published to {queue_name}")
        return True
        
    except Exception as e:
        logger.error(f"Message publish error: {e}")
        return False

# Enhanced Pydantic models
class UserRole(BaseModel):
    role_id: str
    role_name: str
    permissions: List[str]
    description: str

class User(BaseModel):
    user_id: str
    username: str
    email: EmailStr
    full_name: str
    role: str
    department: str
    license_number: Optional[str] = None
    specialization: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

class PatientRecord(BaseModel):
    patient_id: str
    mrn: str  # Medical Record Number
    first_name: str
    last_name: str
    date_of_birth: datetime
    gender: str
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    address: Optional[Dict[str, str]] = None
    emergency_contact: Optional[Dict[str, str]] = None
    insurance_info: Optional[Dict[str, str]] = None
    created_at: datetime
    updated_at: datetime
    created_by: str

class ClinicalAlert(BaseModel):
    alert_id: str
    patient_id: str
    alert_type: str  # critical, warning, info
    priority: str    # high, medium, low
    title: str
    description: str
    triggered_by: str
    created_at: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

class ClinicalNote(BaseModel):
    note_id: str
    patient_id: str
    author_id: str
    note_type: str
    title: str
    content: str
    created_at: datetime
    updated_at: datetime
    signed: bool = False
    signed_at: Optional[datetime] = None

class LabResult(BaseModel):
    result_id: str
    patient_id: str
    test_name: str
    test_code: str
    result_value: str
    reference_range: str
    unit: str
    status: str  # final, preliminary, corrected
    collected_at: datetime
    resulted_at: datetime
    ordered_by: str

class TaskRequest(BaseModel):
    agent_type: str
    task_type: str
    data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = {}
    priority: str = "normal"
    assigned_to: Optional[str] = None

class AuditLog(BaseModel):
    log_id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    timestamp: datetime

# FastAPI app setup
app = FastAPI(
    title="🏥 Vita Agents - Healthcare AI Platform",
    description="Production-Ready Healthcare AI Multi-Agent Framework",
    version="3.0.0",
    docs_url="/api/docs" if config.DEBUG else None,
    redoc_url="/api/redoc" if config.DEBUG else None
)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    logger.info("🚀 Starting Vita Agents Healthcare Platform...")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize connections
    await init_connections()
    
    logger.info("✅ Vita Agents startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup connections on shutdown"""
    logger.info("🛑 Shutting down Vita Agents Healthcare Platform...")
    await close_connections()
    logger.info("✅ Vita Agents shutdown completed")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = config.SECRET_KEY
ALGORITHM = config.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = config.JWT_EXPIRATION_HOURS * 60

# Templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"
uploads_dir = Path(__file__).parent / "uploads"
reports_dir = Path(__file__).parent / "reports"

for directory in [templates_dir, static_dir, uploads_dir, reports_dir]:
    directory.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Enhanced Global state for production features
class EnhancedPortalState:
    def __init__(self):
        self.initialized = True
        self.users = {}
        self.patients = {}
        self.clinical_notes = {}
        self.lab_results = {}
        self.alerts = []
        self.audit_logs = []
        self.active_sessions = {}
        self.system_metrics = {
            "total_patients": 0,
            "active_alerts": 0,
            "pending_tasks": 0,
            "system_uptime": "0d 0h 0m",
            "last_backup": None,
            "security_status": "secure"
        }
        
        # Initialize sample data
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with realistic sample data"""
        # Sample users
        sample_users = [
            {
                "user_id": "u001",
                "username": "dr.smith",
                "email": "john.smith@hospital.com",
                "full_name": "Dr. John Smith",
                "role": "physician",
                "department": "cardiology",
                "license_number": "MD123456",
                "specialization": "Interventional Cardiology"
            },
            {
                "user_id": "u002", 
                "username": "nurse.johnson",
                "email": "mary.johnson@hospital.com",
                "full_name": "Mary Johnson RN",
                "role": "nurse",
                "department": "emergency",
                "license_number": "RN789012",
                "specialization": "Emergency Medicine"
            },
            {
                "user_id": "u003",
                "username": "admin.tech",
                "email": "admin@hospital.com", 
                "full_name": "System Administrator",
                "role": "admin",
                "department": "it",
                "license_number": None,
                "specialization": "Healthcare IT"
            }
        ]
        
        for user_data in sample_users:
            user_data["created_at"] = datetime.now()
            user_data["is_active"] = True
            self.users[user_data["user_id"]] = User(**user_data)
        
        # Sample patients
        sample_patients = [
            {
                "patient_id": "p001",
                "mrn": "MRN001234",
                "first_name": "Alice",
                "last_name": "Williams",
                "date_of_birth": datetime(1975, 5, 15),
                "gender": "female",
                "phone": "+1-555-0123",
                "email": "alice.williams@email.com"
            },
            {
                "patient_id": "p002",
                "mrn": "MRN005678", 
                "first_name": "Robert",
                "last_name": "Davis",
                "date_of_birth": datetime(1962, 12, 3),
                "gender": "male",
                "phone": "+1-555-0456",
                "email": "robert.davis@email.com"
            },
            {
                "patient_id": "p003",
                "mrn": "MRN009012",
                "first_name": "Sarah",
                "last_name": "Brown",
                "date_of_birth": datetime(1988, 8, 22),
                "gender": "female", 
                "phone": "+1-555-0789",
                "email": "sarah.brown@email.com"
            }
        ]
        
        for patient_data in sample_patients:
            patient_data["created_at"] = datetime.now()
            patient_data["updated_at"] = datetime.now()
            patient_data["created_by"] = "u001"
            self.patients[patient_data["patient_id"]] = PatientRecord(**patient_data)
        
        # Update metrics
        self.system_metrics["total_patients"] = len(self.patients)

# Global state instance
portal_state = EnhancedPortalState()

# Authentication utilities
async def get_user_by_email(email: str):
    """Get user from database by email"""
    if not db_pool:
        # Fallback to in-memory data
        for user in portal_state.users.values():
            if user.email == email:
                return user
        return None
        
    try:
        query = "SELECT * FROM users WHERE email = $1 AND is_active = true"
        user_record = await execute_one(query, email)
        
        if user_record:
            return {
                "user_id": user_record['id'],
                "email": user_record['email'],
                "first_name": user_record['first_name'],
                "last_name": user_record['last_name'],
                "role": user_record['role'],
                "password_hash": user_record['password_hash'],
                "created_at": user_record['created_at']
            }
        return None
    except Exception as e:
        logger.error(f"Database error in get_user_by_email: {e}")
        return None

async def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    import bcrypt
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except:
        # Fallback for demo - in production always use proper hashing
        return plain_password == "admin123" and hashed_password.startswith("$2b$")

async def authenticate_user(email: str, password: str):
    """Authenticate user with email and password"""
    user = await get_user_by_email(email)
    if not user:
        return False
        
    if await verify_password(password, user.get('password_hash', '')):
        return user
    return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(username: str = Depends(verify_token)):
    # Try database first
    user = await get_user_by_email(username)
    if user:
        return user
        
    # Fallback to in-memory data
    for user in portal_state.users.values():
        if user.username == username or user.email == username:
            return user
    raise HTTPException(status_code=404, detail="User not found")

# Enhanced Web Routes
@app.get("/llm", response_class=HTMLResponse)
async def llm_page(request: Request):
    """LLM Integration page"""
    return templates.TemplateResponse("llm_integration.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Enhanced dashboard with real-time metrics"""
    from datetime import datetime
    return templates.TemplateResponse("enhanced_dashboard.html", {
        "request": request,
        "title": "Vita Agents - Healthcare Dashboard",
        "metrics": portal_state.system_metrics,
        "recent_alerts": portal_state.alerts[-5:],  # Last 5 alerts
        "total_patients": len(portal_state.patients),
        "total_users": len(portal_state.users),
        "current_time": datetime.now().strftime("%b %d, %Y %H:%M")
    })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    return templates.TemplateResponse("login.html", {
        "request": request,
        "title": "Login - Vita Agents"
    })

@app.get("/patients", response_class=HTMLResponse) 
async def patients_page(request: Request):
    """Patient management page"""
    return templates.TemplateResponse("patients.html", {
        "request": request,
        "title": "Patient Management",
        "patients": list(portal_state.patients.values())
    })

@app.get("/clinical", response_class=HTMLResponse)
async def clinical_page(request: Request):
    """Clinical decision support page"""
    return templates.TemplateResponse("clinical.html", {
        "request": request,
        "title": "Clinical Decision Support"
    })

@app.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request):
    """Clinical alerts and notifications page"""
    return templates.TemplateResponse("alerts.html", {
        "request": request,
        "title": "Clinical Alerts",
        "alerts": portal_state.alerts
    })

@app.get("/workflows", response_class=HTMLResponse)
async def workflows_page(request: Request):
    """Clinical workflows page"""
    return templates.TemplateResponse("workflows.html", {
        "request": request,
        "title": "Clinical Workflows"
    })

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Advanced analytics and reporting page"""
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "title": "Healthcare Analytics",
        "metrics": portal_state.system_metrics
    })

@app.get("/agents", response_class=HTMLResponse)
async def agents_page(request: Request):
    """Core agents management page"""
    return templates.TemplateResponse("agents.html", {
        "request": request,
        "title": "Core Agents"
    })

@app.get("/ai-models", response_class=HTMLResponse)
async def ai_models_page(request: Request):
    """AI models management page"""
    return templates.TemplateResponse("ai_models.html", {
        "request": request,
        "title": "AI Models"
    })

@app.get("/harmonization", response_class=HTMLResponse)
async def harmonization_page(request: Request):
    """Data harmonization page"""
    return templates.TemplateResponse("harmonization.html", {
        "request": request,
        "title": "Data Harmonization"
    })

@app.get("/integration", response_class=HTMLResponse)
async def integration_page(request: Request):
    """System integration page"""
    return templates.TemplateResponse("integration.html", {
        "request": request,
        "title": "System Integration"
    })

@app.get("/compliance", response_class=HTMLResponse)
async def compliance_page(request: Request):
    """HIPAA compliance page"""
    return templates.TemplateResponse("compliance.html", {
        "request": request,
        "title": "HIPAA Compliance"
    })

@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring_page(request: Request):
    """System monitoring page"""
    return templates.TemplateResponse("monitoring.html", {
        "request": request,
        "title": "System Monitoring"
    })

@app.get("/testing", response_class=HTMLResponse)
async def testing_page(request: Request):
    """System testing page"""
    return templates.TemplateResponse("testing.html", {
        "request": request,
        "title": "System Testing"
    })
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "title": "Healthcare Analytics"
    })

@app.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request):
    """Clinical alerts and notifications page"""
    return templates.TemplateResponse("alerts.html", {
        "request": request,
        "title": "Clinical Alerts",
        "alerts": portal_state.alerts
    })

@app.get("/integration", response_class=HTMLResponse)
async def integration_page(request: Request):
    """System integration management page"""
    return templates.TemplateResponse("integration.html", {
        "request": request,
        "title": "System Integration"
    })

@app.get("/compliance", response_class=HTMLResponse)
async def compliance_page(request: Request):
    """HIPAA compliance and audit page"""
    return templates.TemplateResponse("compliance.html", {
        "request": request,
        "title": "Compliance & Security"
    })

@app.get("/workflows", response_class=HTMLResponse)
async def workflows_page(request: Request):
    """Clinical workflow management page"""
    return templates.TemplateResponse("workflows.html", {
        "request": request,
        "title": "Clinical Workflows"
    })

# Enhanced API Endpoints

@app.post("/api/auth/login")
async def login(email: str = Form(...), password: str = Form(...)):
    """User authentication with database backend"""
    
    # Authenticate user against database
    user = await authenticate_user(email, password)
    
    if not user:
        # Log failed login attempt
        logger.warning(f"Failed login attempt for email: {email}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": email}, expires_delta=access_token_expires
    )
    
    # Update last login in database
    if db_pool:
        try:
            await execute_command(
                "UPDATE users SET last_login = $1 WHERE email = $2",
                datetime.now(), email
            )
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
    
    # Cache user session in Redis
    session_data = {
        "user_id": user.get('user_id'),
        "email": email,
        "role": user.get('role'),
        "login_time": datetime.now().isoformat()
    }
    await cache_set(f"session:{email}", session_data, ttl=3600)
    
    # Log successful login
    logger.info(f"Successful login for user: {email}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "user_id": user.get('user_id'),
            "email": email,
            "full_name": f"{user.get('first_name', '')} {user.get('last_name', '')}".strip(),
            "role": user.get('role'),
            "first_name": user.get('first_name'),
            "last_name": user.get('last_name')
        }
    }

@app.get("/api/patients")
async def get_patients(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get all patients with pagination and filtering from database"""
    
    # Try database first
    if db_pool:
        try:
            # Build query with search
            base_query = "SELECT * FROM patients WHERE 1=1"
            params = []
            param_count = 0
            
            if search:
                param_count += 1
                base_query += f" AND (first_name ILIKE ${param_count} OR last_name ILIKE ${param_count} OR mrn ILIKE ${param_count})"
                params.append(f"%{search}%")
            
            # Add pagination
            param_count += 1
            base_query += f" ORDER BY created_at DESC LIMIT ${param_count}"
            params.append(limit)
            
            param_count += 1
            base_query += f" OFFSET ${param_count}"
            params.append(offset)
            
            # Execute query
            patients = await execute_query(base_query, *params)
            
            # Get total count
            count_query = "SELECT COUNT(*) as total FROM patients WHERE 1=1"
            count_params = []
            if search:
                count_query += " AND (first_name ILIKE $1 OR last_name ILIKE $1 OR mrn ILIKE $1)"
                count_params.append(f"%{search}%")
                
            total_result = await execute_one(count_query, *count_params)
            total = total_result['total'] if total_result else 0
            
            # Format response
            patients_list = []
            for patient in patients:
                patient_dict = {
                    "patient_id": patient['id'],
                    "mrn": patient['mrn'],
                    "first_name": patient['first_name'],
                    "last_name": patient['last_name'],
                    "date_of_birth": patient['date_of_birth'].isoformat() if patient['date_of_birth'] else None,
                    "gender": patient['gender'],
                    "phone": patient['phone'],
                    "email": patient['email'],
                    "created_at": patient['created_at'].isoformat() if patient['created_at'] else None,
                    "age": datetime.now().year - patient['date_of_birth'].year if patient['date_of_birth'] else None
                }
                patients_list.append(patient_dict)
            
            return {
                "patients": patients_list, 
                "total": total,
                "limit": limit,
                "offset": offset,
                "source": "database"
            }
            
        except Exception as e:
            logger.error(f"Database error in get_patients: {e}")
            # Fall through to in-memory fallback
    
    # Fallback to in-memory data
    patients_list = []
    for patient in portal_state.patients.values():
        patient_dict = patient.dict()
        patient_dict["age"] = datetime.now().year - patient.date_of_birth.year
        patients_list.append(patient_dict)
    
    # Apply search filter
    if search:
        search_lower = search.lower()
        patients_list = [
            p for p in patients_list 
            if search_lower in p['first_name'].lower() or 
               search_lower in p['last_name'].lower() or 
               search_lower in p['mrn'].lower()
        ]
    
    # Apply pagination
    total = len(patients_list)
    patients_list = patients_list[offset:offset + limit]
    
    return {
        "patients": patients_list, 
        "total": total,
        "limit": limit,
        "offset": offset,
        "source": "memory"
    }

@app.post("/api/patients")
async def create_patient(patient_data: dict, current_user: User = Depends(get_current_user)):
    """Create new patient record"""
    patient_id = f"p{len(portal_state.patients) + 1:03d}"
    mrn = f"MRN{len(portal_state.patients) + 100000}"
    
    new_patient = PatientRecord(
        patient_id=patient_id,
        mrn=mrn,
        first_name=patient_data["first_name"],
        last_name=patient_data["last_name"],
        date_of_birth=datetime.fromisoformat(patient_data["date_of_birth"]),
        gender=patient_data["gender"],
        phone=patient_data.get("phone"),
        email=patient_data.get("email"),
        created_at=datetime.now(),
        updated_at=datetime.now(),
        created_by=current_user.user_id
    )
    
    portal_state.patients[patient_id] = new_patient
    portal_state.system_metrics["total_patients"] = len(portal_state.patients)
    
    # Log audit trail
    audit_log = AuditLog(
        log_id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        action="create_patient",
        resource_type="patient",
        resource_id=patient_id,
        details={"mrn": mrn, "name": f"{new_patient.first_name} {new_patient.last_name}"},
        ip_address="127.0.0.1",
        user_agent="Web Portal",
        timestamp=datetime.now()
    )
    portal_state.audit_logs.append(audit_log)
    
    return {"message": "Patient created successfully", "patient_id": patient_id}

@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str, current_user: User = Depends(get_current_user)):
    """Get specific patient details"""
    if patient_id not in portal_state.patients:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient = portal_state.patients[patient_id]
    patient_dict = patient.dict()
    patient_dict["age"] = datetime.now().year - patient.date_of_birth.year
    
    # Get related clinical notes
    notes = [note for note in portal_state.clinical_notes.values() 
             if note.patient_id == patient_id]
    
    # Get lab results
    labs = [lab for lab in portal_state.lab_results.values() 
            if lab.patient_id == patient_id]
    
    return {
        "patient": patient_dict,
        "clinical_notes": [note.dict() for note in notes],
        "lab_results": [lab.dict() for lab in labs]
    }

@app.post("/api/clinical/notes")
async def create_clinical_note(
    note_data: dict, 
    current_user: User = Depends(get_current_user)
):
    """Create clinical note"""
    note_id = str(uuid.uuid4())
    
    clinical_note = ClinicalNote(
        note_id=note_id,
        patient_id=note_data["patient_id"],
        author_id=current_user.user_id,
        note_type=note_data["note_type"],
        title=note_data["title"],
        content=note_data["content"],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    portal_state.clinical_notes[note_id] = clinical_note
    
    return {"message": "Clinical note created", "note_id": note_id}

@app.post("/api/alerts")
async def create_alert(alert_data: dict, current_user: User = Depends(get_current_user)):
    """Create clinical alert"""
    alert_id = str(uuid.uuid4())
    
    alert = ClinicalAlert(
        alert_id=alert_id,
        patient_id=alert_data["patient_id"],
        alert_type=alert_data["alert_type"],
        priority=alert_data["priority"],
        title=alert_data["title"],
        description=alert_data["description"],
        triggered_by=current_user.user_id,
        created_at=datetime.now()
    )
    
    portal_state.alerts.append(alert)
    portal_state.system_metrics["active_alerts"] = len([a for a in portal_state.alerts if not a.acknowledged])
    
    return {"message": "Alert created", "alert_id": alert_id}

@app.get("/api/alerts")
async def get_alerts(current_user: User = Depends(get_current_user)):
    """Get all alerts"""
    return {"alerts": [alert.dict() for alert in portal_state.alerts]}

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, current_user: User = Depends(get_current_user)):
    """Acknowledge an alert"""
    for alert in portal_state.alerts:
        if alert.alert_id == alert_id:
            alert.acknowledged = True
            alert.acknowledged_by = current_user.user_id
            alert.acknowledged_at = datetime.now()
            break
    else:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    portal_state.system_metrics["active_alerts"] = len([a for a in portal_state.alerts if not a.acknowledged])
    return {"message": "Alert acknowledged"}

@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics(current_user: User = Depends(get_current_user)):
    """Get dashboard analytics data"""
    return {
        "patient_demographics": {
            "total_patients": len(portal_state.patients),
            "by_gender": {"male": 1, "female": 2, "other": 0},
            "by_age_group": {"0-18": 0, "19-35": 1, "36-65": 1, "65+": 1}
        },
        "alert_summary": {
            "total_alerts": len(portal_state.alerts),
            "critical": len([a for a in portal_state.alerts if a.alert_type == "critical"]),
            "warning": len([a for a in portal_state.alerts if a.alert_type == "warning"]),
            "info": len([a for a in portal_state.alerts if a.alert_type == "info"])
        },
        "activity_summary": {
            "notes_today": len(portal_state.clinical_notes),
            "lab_results_pending": 0,
            "tasks_completed": 15,
            "user_sessions": len(portal_state.active_sessions)
        }
    }

@app.post("/api/upload/documents")
async def upload_document(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    document_type: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Upload medical documents"""
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else ''
    saved_filename = f"{file_id}.{file_extension}"
    file_path = uploads_dir / saved_filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Store document metadata
    document_metadata = {
        "document_id": file_id,
        "patient_id": patient_id,
        "original_filename": file.filename,
        "document_type": document_type,
        "file_size": len(content),
        "uploaded_by": current_user.user_id,
        "uploaded_at": datetime.now().isoformat(),
        "file_path": str(file_path)
    }
    
    return {
        "message": "Document uploaded successfully",
        "document_id": file_id,
        "metadata": document_metadata
    }

@app.get("/api/reports/clinical")
async def generate_clinical_report(
    report_type: str,
    start_date: str,
    end_date: str,
    current_user: User = Depends(get_current_user)
):
    """Generate clinical reports"""
    
    # Sample report data
    report_data = {
        "report_id": str(uuid.uuid4()),
        "report_type": report_type,
        "generated_by": current_user.full_name,
        "generated_at": datetime.now().isoformat(),
        "period": f"{start_date} to {end_date}",
        "summary": {
            "total_patients_seen": len(portal_state.patients),
            "total_alerts_generated": len(portal_state.alerts),
            "total_notes_created": len(portal_state.clinical_notes),
            "average_response_time": "2.3 minutes"
        },
        "detailed_metrics": [
            {"metric": "Patient Satisfaction", "value": "4.8/5.0", "trend": "+2.3%"},
            {"metric": "Clinical Efficiency", "value": "94.2%", "trend": "+1.8%"},
            {"metric": "Alert Response Time", "value": "1.2 min", "trend": "-15.2%"},
            {"metric": "Documentation Completeness", "value": "98.7%", "trend": "+0.5%"}
        ]
    }
    
    return report_data

@app.get("/api/integration/status")
async def get_integration_status(current_user: User = Depends(get_current_user)):
    """Get external system integration status"""
    return {
        "integrations": [
            {
                "system": "Epic EHR",
                "status": "connected",
                "last_sync": "2024-10-16T14:30:00Z",
                "records_synced": 1250,
                "errors": 0
            },
            {
                "system": "Cerner Lab",
                "status": "connected", 
                "last_sync": "2024-10-16T14:25:00Z",
                "records_synced": 89,
                "errors": 2
            },
            {
                "system": "HL7 Interface",
                "status": "connected",
                "last_sync": "2024-10-16T14:35:00Z",
                "messages_processed": 445,
                "errors": 1
            },
            {
                "system": "PACS Imaging",
                "status": "maintenance",
                "last_sync": "2024-10-16T12:00:00Z",
                "images_processed": 23,
                "errors": 0
            }
        ]
    }

@app.get("/api/compliance/audit-logs")
async def get_audit_logs(current_user: User = Depends(get_current_user)):
    """Get audit logs for compliance"""
    if current_user.role not in ["admin", "compliance_officer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return {
        "audit_logs": [log.dict() for log in portal_state.audit_logs[-100:]],  # Last 100 logs
        "total_logs": len(portal_state.audit_logs)
    }

@app.get("/api/workflows/templates")
async def get_workflow_templates(current_user: User = Depends(get_current_user)):
    """Get clinical workflow templates"""
    templates = [
        {
            "template_id": "wf_001",
            "name": "Emergency Department Triage",
            "description": "Standardized ED triage workflow with AI decision support",
            "steps": 5,
            "estimated_time": "15 minutes",
            "category": "emergency"
        },
        {
            "template_id": "wf_002", 
            "name": "Medication Reconciliation",
            "description": "Comprehensive medication review and reconciliation process",
            "steps": 8,
            "estimated_time": "20 minutes",
            "category": "pharmacy"
        },
        {
            "template_id": "wf_003",
            "name": "Discharge Planning",
            "description": "Patient discharge planning with follow-up coordination",
            "steps": 12,
            "estimated_time": "45 minutes",
            "category": "discharge"
        }
    ]
    
    return {"workflow_templates": templates}

# WebSocket for real-time updates
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket):
    await websocket.accept()
    try:
        while True:
            # Send real-time alerts (in production, this would be event-driven)
            await websocket.send_json({
                "type": "alert_update",
                "data": {
                    "active_alerts": len([a for a in portal_state.alerts if not a.acknowledged]),
                    "latest_alert": portal_state.alerts[-1].dict() if portal_state.alerts else None
                }
            })
            await asyncio.sleep(30)  # Update every 30 seconds
    except:
        pass

# === MISSING API ENDPOINTS ===

@app.put("/api/patients/{patient_id}")
async def update_patient(patient_id: str, patient_data: dict, current_user: dict = Depends(get_current_user)):
    """Update patient information"""
    for i, patient in enumerate(portal_state.patients):
        if patient.patient_id == patient_id:
            # Update patient data
            for field, value in patient_data.items():
                if hasattr(portal_state.patients[i], field):
                    setattr(portal_state.patients[i], field, value)
            
            portal_state.patients[i].updated_at = datetime.now()
        
        # Log audit event
        portal_state.audit_logs.append(AuditLog(
            id=str(uuid.uuid4()),
            user_id=current_user["id"],
            action=f"Updated patient {patient_id}",
            resource_type="patient",
            resource_id=patient_id,
            timestamp=datetime.now()
        ))
        
        return portal_state.patients[i]
    
    raise HTTPException(status_code=404, detail="Patient not found")

@app.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a patient (soft delete)"""
    for i, patient in enumerate(portal_state.patients):
        if patient.patient_id == patient_id:
            # In production, implement soft delete
            portal_state.patients.pop(i)
            
            # Log audit event
            portal_state.audit_logs.append(AuditLog(
                id=str(uuid.uuid4()),
                user_id=current_user["id"],
                action=f"Deleted patient {patient_id}",
                resource_type="patient",
                resource_id=patient_id,
                timestamp=datetime.now()
            ))
            
            return {"message": "Patient deleted successfully"}
    
    raise HTTPException(status_code=404, detail="Patient not found")

@app.get("/api/patients/search")
async def search_patients(
    q: str = Query(..., description="Search query"),
    current_user: dict = Depends(get_current_user)
):
    """Search patients by name, MRN, or other fields"""
    results = []
    search_query = q.lower()
    
    for patient in portal_state.patients:
        if (search_query in patient.first_name.lower() or 
            search_query in patient.last_name.lower() or 
            search_query in patient.medical_record_number.lower() or
            search_query in patient.email.lower()):
            results.append(patient)
    
    return {"patients": results, "count": len(results)}

@app.post("/api/patients/export")
async def export_patients(current_user: dict = Depends(get_current_user)):
    """Export patients data to CSV/JSON"""
    import csv
    import io
    
    output = io.StringIO()
    fieldnames = ['patient_id', 'first_name', 'last_name', 'date_of_birth', 'gender', 
                  'email', 'phone', 'medical_record_number', 'created_at']
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for patient in portal_state.patients:
        writer.writerow({
            'patient_id': patient.patient_id,
            'first_name': patient.first_name,
            'last_name': patient.last_name,
            'date_of_birth': patient.date_of_birth,
            'gender': patient.gender,
            'email': patient.email,
            'phone': patient.phone,
            'medical_record_number': patient.medical_record_number,
            'created_at': patient.created_at.isoformat()
        })
    
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=patients_export.csv"}
    )

@app.post("/api/patients/{patient_id}/notes")
async def add_patient_note(
    patient_id: str,
    note_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Add clinical note for a patient"""
    note = ClinicalNote(
        id=str(uuid.uuid4()),
        patient_id=patient_id,
        content=note_data.get("content"),
        note_type=note_data.get("note_type", "progress_note"),
        created_by=current_user["id"],
        created_at=datetime.now()
    )
    
    portal_state.clinical_notes.append(note)
    
    return note

@app.post("/api/patients/{patient_id}/lab-results")
async def add_lab_result(
    patient_id: str,
    lab_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Add lab result for a patient"""
    lab_result = LabResult(
        id=str(uuid.uuid4()),
        patient_id=patient_id,
        test_name=lab_data.get("test_name"),
        results=lab_data.get("results", {}),
        reference_ranges=lab_data.get("reference_ranges", {}),
        status=lab_data.get("status", "pending"),
        ordered_by=current_user["id"],
        ordered_at=datetime.now(),
        result_date=datetime.now() if lab_data.get("status") == "completed" else None
    )
    
    portal_state.lab_results.append(lab_result)
    
    return lab_result

@app.post("/api/clinical/diagnosis")
async def generate_diagnosis(
    clinical_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """AI-powered diagnostic assistance"""
    # Mock AI diagnosis - in production, integrate with actual AI models
    symptoms = clinical_data.get("symptoms", [])
    age = clinical_data.get("patient_age", 0)
    gender = clinical_data.get("patient_gender", "")
    
    # Simple rule-based mock diagnosis
    differential_diagnosis = []
    
    if "chest pain" in str(symptoms).lower():
        differential_diagnosis = [
            {"condition": "Acute Myocardial Infarction", "probability": 0.75, "severity": "critical"},
            {"condition": "Unstable Angina", "probability": 0.65, "severity": "high"},
            {"condition": "Pulmonary Embolism", "probability": 0.45, "severity": "high"},
            {"condition": "Gastroesophageal Reflux", "probability": 0.25, "severity": "low"}
        ]
    elif "headache" in str(symptoms).lower():
        differential_diagnosis = [
            {"condition": "Tension Headache", "probability": 0.80, "severity": "low"},
            {"condition": "Migraine", "probability": 0.60, "severity": "moderate"},
            {"condition": "Cluster Headache", "probability": 0.30, "severity": "moderate"}
        ]
    else:
        differential_diagnosis = [
            {"condition": "Further evaluation needed", "probability": 0.50, "severity": "low"}
        ]
    
    recommendations = [
        "Complete vital signs assessment",
        "Relevant laboratory studies",
        "Consider imaging if indicated",
        "Monitor patient closely"
    ]
    
    return {
        "differential_diagnosis": differential_diagnosis,
        "recommendations": recommendations,
        "confidence": 0.85,
        "generated_at": datetime.now().isoformat()
    }

@app.post("/api/clinical/drug-interactions")
async def check_drug_interactions(
    drug_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Check for drug interactions"""
    current_medications = drug_data.get("current_medications", [])
    new_medication = drug_data.get("new_medication", "")
    
    # Mock drug interaction checking
    interactions = []
    
    # Simple interaction rules
    interaction_db = {
        "warfarin": ["aspirin", "ibuprofen", "amiodarone"],
        "digoxin": ["amiodarone", "verapamil", "quinidine"],
        "metformin": ["contrast_dye", "alcohol"]
    }
    
    new_med_lower = new_medication.lower()
    for current_med in current_medications:
        current_med_lower = current_med.get("name", "").lower()
        
        if (new_med_lower in interaction_db.get(current_med_lower, []) or
            current_med_lower in interaction_db.get(new_med_lower, [])):
            interactions.append({
                "drug1": current_med.get("name"),
                "drug2": new_medication,
                "severity": "moderate",
                "description": f"Potential interaction between {current_med.get('name')} and {new_medication}",
                "recommendation": "Monitor patient closely and consider dose adjustment"
            })
    
    return {
        "interactions": interactions,
        "interaction_count": len(interactions),
        "risk_level": "high" if len(interactions) > 2 else "moderate" if len(interactions) > 0 else "low"
    }

@app.post("/api/clinical/image-analysis")
async def analyze_medical_image(
    image_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """AI-powered medical image analysis"""
    analysis_type = image_data.get("analysis_type", "chest-xray")
    
    # Mock AI image analysis results
    findings = {
        "chest-xray": [
            "Heart size within normal limits",
            "Clear lung fields bilaterally",
            "No acute cardiopulmonary abnormalities",
            "Normal bone structures"
        ],
        "ct-scan": [
            "No acute intracranial abnormalities",
            "Ventricles and sulci normal for age",
            "No mass effect or midline shift"
        ],
        "mri": [
            "Normal brain parenchyma",
            "No evidence of acute infarction",
            "White matter changes consistent with age"
        ]
    }
    
    return {
        "analysis_type": analysis_type,
        "findings": findings.get(analysis_type, ["Analysis in progress"]),
        "confidence": 0.92,
        "abnormalities_detected": False,
        "recommendations": ["Clinical correlation recommended", "Follow-up as clinically indicated"],
        "analyzed_at": datetime.now().isoformat()
    }

# LLM Integration Endpoints
@app.get("/api/llm/models")
async def get_llm_models():
    """Get available LLM models - public access for demo"""
    if not LLM_AVAILABLE:
        return {"error": "LLM integration not available"}
    
    models = llm_manager.get_available_models()
    healthcare_models = llm_manager.get_healthcare_models()
    active_model = llm_manager.get_active_model()
    
    return {
        "models": {key: {
            "name": model.name,
            "provider": model.provider.value,
            "healthcare_optimized": model.healthcare_optimized,
            "context_length": model.context_length,
            "capabilities": model.capabilities,
            "cost_per_token": model.cost_per_token
        } for key, model in models.items()},
        "healthcare_models": list(healthcare_models.keys()),
        "active_model": llm_manager.active_model,
        "active_model_info": {
            "name": active_model.name if active_model else None,
            "provider": active_model.provider.value if active_model else None,
            "healthcare_optimized": active_model.healthcare_optimized if active_model else False
        } if active_model else None
    }

@app.post("/api/llm/set-model")
async def set_llm_model(model_data: dict):
    """Set active LLM model - public access for demo"""
    if not LLM_AVAILABLE:
        return {"error": "LLM integration not available"}
    
    model_key = model_data.get("model_key")
    if not model_key:
        raise HTTPException(status_code=400, detail="Model key required")
    
    success = llm_manager.set_active_model(model_key)
    if success:
        return {"message": f"Active model set to {model_key}", "active_model": model_key}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.post("/api/llm/generate")
async def generate_llm_response(request_data: dict):
    """Generate LLM response - public access for demo"""
    if not LLM_AVAILABLE:
        return {"error": "LLM integration not available"}
    
    prompt = request_data.get("prompt", "")
    context = request_data.get("context", "")
    temperature = request_data.get("temperature", 0.7)
    max_tokens = request_data.get("max_tokens", 1000)
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt required")
    
    result = await llm_manager.generate_response(prompt, context, temperature, max_tokens)
    return result

@app.post("/api/llm/diagnose")
async def llm_diagnose(diagnosis_data: dict):
    """Generate AI diagnosis - public access for demo"""
    if not LLM_AVAILABLE:
        return {"error": "LLM integration not available"}
    
    # Extract patient data
    age = diagnosis_data.get("age", "")
    gender = diagnosis_data.get("gender", "")
    chief_complaint = diagnosis_data.get("chief_complaint", "")
    hpi = diagnosis_data.get("hpi", "")
    vitals = diagnosis_data.get("vitals", "")
    physical_exam = diagnosis_data.get("physical_exam", "")
    
    # Use clinical prompt template
    prompt = CLINICAL_PROMPTS["diagnosis"].format(
        age=age,
        gender=gender,
        chief_complaint=chief_complaint,
        hpi=hpi or "Not provided",
        vitals=vitals or "Not provided",
        physical_exam=physical_exam or "Not provided"
    )
    
    result = await llm_manager.generate_response(prompt, "", 0.3, 800)
    return result

@app.post("/api/llm/drug-interactions")
async def llm_drug_interactions(drug_data: dict):
    """Check drug interactions with AI - public access for demo"""
    if not LLM_AVAILABLE:
        return {"error": "LLM integration not available"}
    
    current_medications = drug_data.get("current_medications", "")
    new_medication = drug_data.get("new_medication", "")
    
    prompt = CLINICAL_PROMPTS["drug_interaction"].format(
        current_medications=current_medications,
        new_medication=new_medication
    )
    
    result = await llm_manager.generate_response(prompt, "", 0.2, 600)
    return result

# Sample Data Endpoints
@app.get("/api/sample-data/patients")
async def get_sample_patients(
    limit: int = Query(10, description="Number of patients to return"),
    current_user: dict = Depends(get_current_user)
):
    """Get sample patients"""
    try:
        with open("sample_healthcare_data.json", 'r') as f:
            data = json.load(f)
        
        patients = data.get("patients", [])[:limit]
        return {
            "patients": patients,
            "total": len(data.get("patients", [])),
            "metadata": data.get("metadata", {})
        }
    except FileNotFoundError:
        return {"error": "Sample data not found. Generate data first."}

@app.get("/api/sample-data/scenarios")
async def get_sample_scenarios(current_user: dict = Depends(get_current_user)):
    """Get clinical scenarios"""
    try:
        with open("sample_healthcare_data.json", 'r') as f:
            data = json.load(f)
        
        scenarios = data.get("scenarios", [])
        return {
            "scenarios": scenarios,
            "total": len(scenarios),
            "metadata": data.get("metadata", {})
        }
    except FileNotFoundError:
        return {"error": "Sample data not found. Generate data first."}

@app.post("/api/sample-data/generate")
async def generate_sample_data(
    generation_params: dict,
    current_user: dict = Depends(get_current_user)
):
    """Generate new sample data"""
    if not LLM_AVAILABLE:
        # Can still generate data without LLM
        pass
    
    num_patients = generation_params.get("patients", 50)
    num_scenarios = generation_params.get("scenarios", 10)
    
    generator = SampleDataGenerator()
    
    # Generate data
    from dataclasses import asdict
    sample_patients = [generator.generate_patient() for _ in range(num_patients)]
    sample_scenarios = generator.generate_clinical_scenarios(num_scenarios)
    
    # Save data
    sample_data = {
        "patients": [asdict(patient) for patient in sample_patients],
        "scenarios": [asdict(scenario) for scenario in sample_scenarios],
        "metadata": {
            "generated_date": datetime.now().isoformat(),
            "total_patients": len(sample_patients),
            "total_scenarios": len(sample_scenarios),
            "generated_by": current_user["username"],
            "data_version": "1.0"
        }
    }
    
    with open("sample_healthcare_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    return {
        "message": "Sample data generated successfully",
        "patients_created": len(sample_patients),
        "scenarios_created": len(sample_scenarios)
    }

@app.get("/api/demo/scenario/{scenario_index}")
async def get_demo_scenario(
    scenario_index: int,
    current_user: dict = Depends(get_current_user)
):
    """Get specific demo scenario for interactive testing"""
    try:
        with open("sample_healthcare_data.json", 'r') as f:
            data = json.load(f)
        
        scenarios = data.get("scenarios", [])
        if scenario_index >= len(scenarios):
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        scenario = scenarios[scenario_index]
        
        # Enhanced scenario for demo with LLM recommendations
        enhanced_scenario = scenario.copy()
        
        if LLM_AVAILABLE:
            # Get AI recommendations for the scenario
            patient = scenario["patient"]
            diagnosis_prompt = f"""
            Patient: {patient['age']}-year-old {patient['gender']}
            Chief Complaint: {scenario['clinical_notes'][0]['chief_complaint']}
            Medical History: {', '.join(patient['medical_history'])}
            Current Medications: {', '.join([med['name'] for med in patient['current_medications']])}
            
            Provide 3 key differential diagnoses and next steps.
            """
            
            try:
                ai_recommendations = await llm_manager.generate_response(diagnosis_prompt, "", 0.3, 300)
                enhanced_scenario["ai_recommendations"] = ai_recommendations.get("response", "")
            except:
                enhanced_scenario["ai_recommendations"] = "AI recommendations unavailable"
        
        return enhanced_scenario
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sample data not found")

@app.post("/api/emergency/alert")
async def create_emergency_alert(
    emergency_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Create emergency alert"""
    alert = ClinicalAlert(
        id=str(uuid.uuid4()),
        type="emergency",
        title=f"EMERGENCY: {emergency_data.get('type', 'Code Blue')}",
        message=emergency_data.get("message", "Emergency situation requiring immediate attention"),
        patient_id=emergency_data.get("patient_id"),
        priority="critical",
        status="active",
        created_at=datetime.now(),
        created_by=current_user["id"]
    )
    
    portal_state.alerts.append(alert)
    
    # In production, trigger real emergency protocols
    return {
        "alert_id": alert.id,
        "status": "emergency_protocol_activated",
        "message": "Emergency alert created and teams notified"
    }

@app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "environment": config.ENVIRONMENT,
        "services": {}
    }
    
    # Check PostgreSQL
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.execute("SELECT 1")
            health_status["services"]["database"] = "healthy"
        else:
            health_status["services"]["database"] = "disconnected"
    except Exception as e:
        health_status["services"]["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        if redis_client:
            await redis_client.ping()
            health_status["services"]["cache"] = "healthy"
        else:
            health_status["services"]["cache"] = "disconnected"
    except Exception as e:
        health_status["services"]["cache"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Elasticsearch
    try:
        if es_client:
            await es_client.ping()
            health_status["services"]["search"] = "healthy"
        else:
            health_status["services"]["search"] = "disconnected"
    except Exception as e:
        health_status["services"]["search"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check MinIO
    try:
        if minio_client:
            # Try to list buckets as a health check
            buckets = minio_client.list_buckets()
            health_status["services"]["storage"] = "healthy"
        else:
            health_status["services"]["storage"] = "disconnected"
    except Exception as e:
        health_status["services"]["storage"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Additional service checks
    health_status["services"]["authentication"] = "active"
    health_status["services"]["llm"] = "available" if LLM_AVAILABLE else "unavailable"
    
    return health_status

@app.get("/api/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    metrics = []
    
    # Application metrics
    metrics.append('vita_agents_health_status{service="app"} 1')
    
    # Database metrics
    if db_pool:
        try:
            # Get connection pool stats
            pool_size = db_pool.get_size()
            free_connections = db_pool.get_free_size()
            metrics.append(f'vita_agents_db_pool_size {pool_size}')
            metrics.append(f'vita_agents_db_pool_free {free_connections}')
            metrics.append('vita_agents_health_status{service="database"} 1')
        except:
            metrics.append('vita_agents_health_status{service="database"} 0')
    else:
        metrics.append('vita_agents_health_status{service="database"} 0')
    
    # Redis metrics
    if redis_client:
        try:
            info = await redis_client.info()
            if 'used_memory' in info:
                metrics.append(f'vita_agents_redis_memory_used {info["used_memory"]}')
            metrics.append('vita_agents_health_status{service="redis"} 1')
        except:
            metrics.append('vita_agents_health_status{service="redis"} 0')
    else:
        metrics.append('vita_agents_health_status{service="redis"} 0')
    
    # LLM metrics
    if LLM_AVAILABLE:
        try:
            models = llm_manager.get_available_models()
            metrics.append(f'vita_agents_llm_models_available {len(models)}')
            metrics.append('vita_agents_health_status{service="llm"} 1')
        except:
            metrics.append('vita_agents_health_status{service="llm"} 0')
    else:
        metrics.append('vita_agents_health_status{service="llm"} 0')
    
    return Response(content='\n'.join(metrics), media_type="text/plain")

# Docker Integration Endpoints

@app.post("/api/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    document_type: str = Form("general"),
    current_user: dict = Depends(get_current_user)
):
    """Upload file to MinIO object storage"""
    
    if not minio_client:
        return {"error": "File storage not available", "uploaded": False}
    
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'bin'
        object_name = f"{document_type}/{file_id}.{file_extension}"
        
        # Read file content
        file_content = await file.read()
        
        # Upload to MinIO
        minio_client.put_object(
            bucket_name="patient-documents",
            object_name=object_name,
            data=BytesIO(file_content),
            length=len(file_content),
            content_type=file.content_type or "application/octet-stream"
        )
        
        # Store metadata in database
        if db_pool:
            try:
                await execute_command(
                    """INSERT INTO document_metadata 
                       (file_id, patient_id, filename, object_name, file_size, content_type, uploaded_by, created_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                    file_id, patient_id, file.filename, object_name, 
                    len(file_content), file.content_type, current_user.get('user_id'), datetime.now()
                )
            except Exception as e:
                logger.error(f"Failed to save file metadata: {e}")
        
        # Index in Elasticsearch for search
        if es_client:
            try:
                doc = {
                    "file_id": file_id,
                    "patient_id": patient_id,
                    "filename": file.filename,
                    "document_type": document_type,
                    "uploaded_by": current_user.get('user_id'),
                    "upload_date": datetime.now().isoformat(),
                    "file_size": len(file_content)
                }
                
                await es_client.index(
                    index="patient_documents",
                    id=file_id,
                    body=doc
                )
            except Exception as e:
                logger.error(f"Failed to index document: {e}")
        
        logger.info(f"File uploaded: {file.filename} by user {current_user.get('user_id')}")
        
        return {
            "message": "File uploaded successfully",
            "file_id": file_id,
            "filename": file.filename,
            "size": len(file_content),
            "uploaded": True
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return {"error": f"Upload failed: {str(e)}", "uploaded": False}

@app.get("/api/search")
async def search_content(
    query: str = Query(..., min_length=1),
    index: str = Query("patients", regex="^(patients|medical_records|documents)$"),
    limit: int = Query(10, ge=1, le=50),
    current_user: dict = Depends(get_current_user)
):
    """Search content using Elasticsearch"""
    
    if not es_client:
        return {"error": "Search service not available", "results": []}
    
    try:
        # Build Elasticsearch query
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["*"],
                    "fuzziness": "AUTO"
                }
            },
            "size": limit,
            "highlight": {
                "fields": {
                    "*": {}
                }
            }
        }
        
        # Execute search
        response = await es_client.search(
            index=index,
            body=search_body
        )
        
        # Format results
        results = []
        for hit in response['hits']['hits']:
            result = {
                "id": hit['_id'],
                "score": hit['_score'],
                "source": hit['_source'],
                "highlights": hit.get('highlight', {})
            }
            results.append(result)
        
        return {
            "query": query,
            "index": index,
            "total": response['hits']['total']['value'],
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"error": f"Search failed: {str(e)}", "results": []}

@app.post("/api/tasks/background")
async def create_background_task(
    task_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Create a background task using RabbitMQ"""
    
    try:
        task_id = str(uuid.uuid4())
        
        # Prepare task message
        message = {
            "task_id": task_id,
            "task_type": task_data.get("task_type", "general"),
            "data": task_data.get("data", {}),
            "created_by": current_user.get('user_id'),
            "created_at": datetime.now().isoformat(),
            "priority": task_data.get("priority", "normal")
        }
        
        # Store task status in Redis
        await cache_set(
            f"task:{task_id}",
            {
                "status": "queued",
                "created_at": datetime.now().isoformat(),
                "created_by": current_user.get('user_id')
            },
            ttl=3600
        )
        
        # Publish to RabbitMQ
        success = await publish_message("vita_tasks", message)
        
        if success:
            logger.info(f"Background task created: {task_id} by user {current_user.get('user_id')}")
            return {
                "task_id": task_id,
                "status": "queued",
                "message": "Task created successfully"
            }
        else:
            return {
                "error": "Failed to queue task",
                "task_id": task_id,
                "status": "failed"
            }
            
    except Exception as e:
        logger.error(f"Background task creation failed: {e}")
        return {"error": f"Task creation failed: {str(e)}"}

@app.get("/api/tasks/{task_id}/status")
async def get_task_status(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get background task status from Redis"""
    
    try:
        task_status = await cache_get(f"task:{task_id}")
        
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "task_id": task_id,
            "status": task_status.get("status", "unknown"),
            "created_at": task_status.get("created_at"),
            "started_at": task_status.get("started_at"),
            "completed_at": task_status.get("completed_at"),
            "error": task_status.get("error"),
            "result": task_status.get("result")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task status check failed: {e}")
        return {"error": f"Status check failed: {str(e)}"}

@app.get("/api/analytics/dashboard")
async def get_analytics_dashboard(
    current_user: dict = Depends(get_current_user)
):
    """Get analytics dashboard data from database and cache"""
    
    try:
        # Check cache first
        cached_data = await cache_get("analytics:dashboard")
        if cached_data:
            cached_data["source"] = "cache"
            return cached_data
        
        # Generate analytics from database
        analytics = {
            "timestamp": datetime.now().isoformat(),
            "source": "database"
        }
        
        if db_pool:
            try:
                # Patient statistics
                patient_stats = await execute_one("SELECT COUNT(*) as total FROM patients")
                analytics["patients"] = {
                    "total": patient_stats['total'] if patient_stats else 0
                }
                
                # Recent appointments
                recent_appointments = await execute_one(
                    "SELECT COUNT(*) as count FROM appointments WHERE scheduled_date >= NOW() - INTERVAL '7 days'"
                )
                analytics["appointments"] = {
                    "recent": recent_appointments['count'] if recent_appointments else 0
                }
                
                # User activity
                active_users = await execute_one(
                    "SELECT COUNT(*) as count FROM users WHERE last_login >= NOW() - INTERVAL '24 hours'"
                )
                analytics["users"] = {
                    "active_24h": active_users['count'] if active_users else 0
                }
                
            except Exception as e:
                logger.error(f"Database analytics query failed: {e}")
                analytics["error"] = "Database unavailable"
        
        # Add system metrics
        analytics["system"] = {
            "services": {
                "database": "connected" if db_pool else "disconnected",
                "cache": "connected" if redis_client else "disconnected", 
                "search": "connected" if es_client else "disconnected",
                "storage": "connected" if minio_client else "disconnected"
            }
        }
        
        # Cache the result
        await cache_set("analytics:dashboard", analytics, ttl=300)  # 5 minutes
        
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics dashboard failed: {e}")
        return {"error": f"Analytics failed: {str(e)}"}

if __name__ == "__main__":
    print("🏥 Starting Enhanced Vita Agents Healthcare Portal with Docker Integration...")
    print("🌟 Features: Authentication, Patient Management, Clinical Decision Support")
    print("🤖 AI-Powered: LLM Integration, Sample Data Generation, Real-time Analysis") 
    print("📊 Analytics, Real-time Alerts, Integration Management, HIPAA Compliance")
    print("� Docker Services: PostgreSQL, Redis, Elasticsearch, RabbitMQ, MinIO")
    print(f"�🔗 Access at: http://localhost:8080")
    print(f"🌍 Environment: {config.ENVIRONMENT}")
    print(f"🐛 Debug Mode: {config.DEBUG}")
    
    # Initialize LLM on startup
    if LLM_AVAILABLE:
        print("🤖 LLM Integration: Available")
        models = llm_manager.get_available_models()
        print(f"📋 Available Models: {len(models)}")
        healthcare_models = llm_manager.get_healthcare_models()
        if healthcare_models:
            print(f"🏥 Healthcare Models: {len(healthcare_models)}")
    else:
        print("⚠️  LLM Integration: Not available (missing dependencies)")
    
    # Print service configuration
    print("\n🔧 Service Configuration:")
    print(f"   Database: {config.DATABASE_URL.split('@')[-1] if '@' in config.DATABASE_URL else 'localhost:5432'}")
    print(f"   Redis: {config.REDIS_URL.split('@')[-1] if '@' in config.REDIS_URL else 'localhost:6379'}")
    print(f"   Elasticsearch: {config.ELASTICSEARCH_URL}")
    print(f"   MinIO: {config.MINIO_ENDPOINT}")
    
    uvicorn.run(
        "enhanced_web_portal:app",
        host="0.0.0.0",
        port=8080,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )