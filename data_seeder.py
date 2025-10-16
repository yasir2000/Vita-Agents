#!/usr/bin/env python3
"""
Data Seeder for Vita Agents
Populates the system with initial data for demo/testing purposes
"""

import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any

import asyncpg
import redis.asyncio as redis
from faker import Faker
from elasticsearch import AsyncElasticsearch
from minio import Minio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

fake = Faker()

class DataSeeder:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', '')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.elasticsearch_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
        self.minio_endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        self.minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        
        self.db_pool = None
        self.redis_client = None
        self.es_client = None
        self.minio_client = None
        
    async def setup_connections(self):
        """Initialize all connections"""
        try:
            # Database connection
            if self.database_url:
                db_url = self.database_url.replace('postgresql://', '').replace('postgresql+asyncpg://', '')
                self.db_pool = await asyncpg.create_pool(f"postgresql://{db_url}")
                
            # Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            
            # Elasticsearch connection
            self.es_client = AsyncElasticsearch([self.elasticsearch_url])
            
            # MinIO connection
            self.minio_client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=False
            )
            
            logger.info("All connections established")
            
        except Exception as e:
            logger.error(f"Connection setup failed: {e}")
            raise
            
    async def create_database_schema(self):
        """Create database tables"""
        if not self.db_pool:
            return
            
        schema_sql = """
        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            role VARCHAR(50) DEFAULT 'user',
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Patients table
        CREATE TABLE IF NOT EXISTS patients (
            id SERIAL PRIMARY KEY,
            mrn VARCHAR(50) UNIQUE NOT NULL,
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            date_of_birth DATE,
            gender VARCHAR(10),
            phone VARCHAR(20),
            email VARCHAR(255),
            address JSONB,
            emergency_contact JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Medical records table
        CREATE TABLE IF NOT EXISTS medical_records (
            id SERIAL PRIMARY KEY,
            patient_id INTEGER REFERENCES patients(id),
            record_type VARCHAR(50),
            chief_complaint TEXT,
            diagnosis TEXT,
            treatment_plan TEXT,
            medications JSONB,
            lab_results JSONB,
            notes TEXT,
            provider_id INTEGER,
            visit_date TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Appointments table
        CREATE TABLE IF NOT EXISTS appointments (
            id SERIAL PRIMARY KEY,
            patient_id INTEGER REFERENCES patients(id),
            provider_id INTEGER,
            appointment_type VARCHAR(50),
            scheduled_date TIMESTAMP,
            duration_minutes INTEGER DEFAULT 30,
            status VARCHAR(20) DEFAULT 'scheduled',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Audit logs table
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            action VARCHAR(100),
            resource_type VARCHAR(50),
            resource_id INTEGER,
            old_values JSONB,
            new_values JSONB,
            ip_address INET,
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- System metrics table
        CREATE TABLE IF NOT EXISTS system_metrics (
            id SERIAL PRIMARY KEY,
            metric_name VARCHAR(100),
            metric_value NUMERIC,
            metric_unit VARCHAR(20),
            tags JSONB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            logger.info("Database schema created")
        except Exception as e:
            logger.error(f"Schema creation failed: {e}")
            
    async def seed_users(self, count: int = 20):
        """Seed users table"""
        if not self.db_pool:
            return
            
        users_data = []
        roles = ['admin', 'doctor', 'nurse', 'technician', 'user']
        
        # Add default admin user
        users_data.append((
            'admin@vita-agents.com',
            '$2b$12$LQv3c1yqBwVHdkuLM4tZjOZB8V3m8K9L2Xz1Y3Q4R5T6U7V8W9X0Y1',  # hashed 'admin123'
            'System',
            'Administrator',
            'admin',
            True
        ))
        
        for _ in range(count - 1):
            users_data.append((
                fake.email(),
                '$2b$12$LQv3c1yqBwVHdkuLM4tZjOZB8V3m8K9L2Xz1Y3Q4R5T6U7V8W9X0Y1',  # hashed 'password123'
                fake.first_name(),
                fake.last_name(),
                random.choice(roles),
                True
            ))
            
        try:
            async with self.db_pool.acquire() as conn:
                await conn.executemany(
                    """INSERT INTO users (email, password_hash, first_name, last_name, role, is_active)
                       VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (email) DO NOTHING""",
                    users_data
                )
            logger.info(f"Seeded {count} users")
        except Exception as e:
            logger.error(f"User seeding failed: {e}")
            
    async def seed_patients(self, count: int = 100):
        """Seed patients table"""
        if not self.db_pool:
            return
            
        patients_data = []
        
        for i in range(count):
            mrn = f"MRN{str(i+1).zfill(6)}"
            
            # Generate address
            address = {
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "zip": fake.zipcode(),
                "country": "USA"
            }
            
            # Generate emergency contact
            emergency_contact = {
                "name": fake.name(),
                "relationship": random.choice(["Spouse", "Parent", "Sibling", "Child", "Friend"]),
                "phone": fake.phone_number()
            }
            
            patients_data.append((
                mrn,
                fake.first_name(),
                fake.last_name(),
                fake.date_of_birth(minimum_age=0, maximum_age=100),
                random.choice(['M', 'F', 'O']),
                fake.phone_number(),
                fake.email(),
                json.dumps(address),
                json.dumps(emergency_contact)
            ))
            
        try:
            async with self.db_pool.acquire() as conn:
                await conn.executemany(
                    """INSERT INTO patients (mrn, first_name, last_name, date_of_birth, gender, phone, email, address, emergency_contact)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) ON CONFLICT (mrn) DO NOTHING""",
                    patients_data
                )
            logger.info(f"Seeded {count} patients")
        except Exception as e:
            logger.error(f"Patient seeding failed: {e}")
            
    async def seed_medical_records(self, count: int = 500):
        """Seed medical records table"""
        if not self.db_pool:
            return
            
        # Get patient IDs
        async with self.db_pool.acquire() as conn:
            patient_ids = await conn.fetch("SELECT id FROM patients")
            
        if not patient_ids:
            logger.warning("No patients found, skipping medical records seeding")
            return
            
        record_types = ['consultation', 'emergency', 'routine_checkup', 'follow_up', 'surgery', 'lab_test']
        diagnoses = [
            'Hypertension', 'Diabetes Type 2', 'Asthma', 'Depression', 'Anxiety',
            'Arthritis', 'Migraine', 'COPD', 'Heart Disease', 'Obesity'
        ]
        
        records_data = []
        
        for _ in range(count):
            patient_id = random.choice(patient_ids)['id']
            
            # Generate medications
            medications = [
                {
                    "name": random.choice(["Lisinopril", "Metformin", "Albuterol", "Sertraline", "Ibuprofen"]),
                    "dosage": f"{random.randint(5, 100)}mg",
                    "frequency": random.choice(["Once daily", "Twice daily", "As needed"])
                }
                for _ in range(random.randint(0, 3))
            ]
            
            # Generate lab results
            lab_results = {
                "glucose": random.randint(70, 200),
                "cholesterol": random.randint(150, 300),
                "blood_pressure": f"{random.randint(90, 180)}/{random.randint(60, 110)}",
                "hemoglobin": round(random.uniform(10.0, 16.0), 1)
            }
            
            visit_date = fake.date_time_between(start_date='-2y', end_date='now')
            
            records_data.append((
                patient_id,
                random.choice(record_types),
                fake.text(max_nb_chars=100),
                random.choice(diagnoses),
                fake.text(max_nb_chars=200),
                json.dumps(medications),
                json.dumps(lab_results),
                fake.text(max_nb_chars=500),
                random.randint(1, 10),  # provider_id
                visit_date
            ))
            
        try:
            async with self.db_pool.acquire() as conn:
                await conn.executemany(
                    """INSERT INTO medical_records (patient_id, record_type, chief_complaint, diagnosis, 
                       treatment_plan, medications, lab_results, notes, provider_id, visit_date)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                    records_data
                )
            logger.info(f"Seeded {count} medical records")
        except Exception as e:
            logger.error(f"Medical records seeding failed: {e}")
            
    async def seed_appointments(self, count: int = 200):
        """Seed appointments table"""
        if not self.db_pool:
            return
            
        # Get patient IDs
        async with self.db_pool.acquire() as conn:
            patient_ids = await conn.fetch("SELECT id FROM patients")
            
        if not patient_ids:
            return
            
        appointment_types = ['consultation', 'follow_up', 'emergency', 'routine_checkup', 'specialist']
        statuses = ['scheduled', 'completed', 'cancelled', 'no_show']
        
        appointments_data = []
        
        for _ in range(count):
            patient_id = random.choice(patient_ids)['id']
            scheduled_date = fake.date_time_between(start_date='-1m', end_date='+3m')
            
            appointments_data.append((
                patient_id,
                random.randint(1, 10),  # provider_id
                random.choice(appointment_types),
                scheduled_date,
                random.choice([15, 30, 45, 60]),
                random.choice(statuses),
                fake.text(max_nb_chars=200)
            ))
            
        try:
            async with self.db_pool.acquire() as conn:
                await conn.executemany(
                    """INSERT INTO appointments (patient_id, provider_id, appointment_type, 
                       scheduled_date, duration_minutes, status, notes)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    appointments_data
                )
            logger.info(f"Seeded {count} appointments")
        except Exception as e:
            logger.error(f"Appointments seeding failed: {e}")
            
    async def seed_elasticsearch_data(self):
        """Seed Elasticsearch with sample healthcare data"""
        if not self.es_client:
            return
            
        # Create indices
        indices = [
            {
                "name": "patients",
                "mapping": {
                    "mappings": {
                        "properties": {
                            "mrn": {"type": "keyword"},
                            "full_name": {"type": "text", "analyzer": "standard"},
                            "date_of_birth": {"type": "date"},
                            "gender": {"type": "keyword"},
                            "address": {"type": "text"},
                            "created_at": {"type": "date"}
                        }
                    }
                }
            },
            {
                "name": "medical_records",
                "mapping": {
                    "mappings": {
                        "properties": {
                            "patient_mrn": {"type": "keyword"},
                            "record_type": {"type": "keyword"},
                            "diagnosis": {"type": "text", "analyzer": "standard"},
                            "notes": {"type": "text", "analyzer": "standard"},
                            "visit_date": {"type": "date"},
                            "provider_id": {"type": "integer"}
                        }
                    }
                }
            }
        ]
        
        try:
            for index_config in indices:
                await self.es_client.indices.create(
                    index=index_config["name"],
                    body=index_config["mapping"],
                    ignore=400  # Ignore if index already exists
                )
                
            # Get data from database and index it
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    # Index patients
                    patients = await conn.fetch("""
                        SELECT mrn, first_name, last_name, date_of_birth, gender, address, created_at
                        FROM patients LIMIT 100
                    """)
                    
                    for patient in patients:
                        doc = {
                            "mrn": patient['mrn'],
                            "full_name": f"{patient['first_name']} {patient['last_name']}",
                            "date_of_birth": patient['date_of_birth'].isoformat() if patient['date_of_birth'] else None,
                            "gender": patient['gender'],
                            "address": patient['address'],
                            "created_at": patient['created_at'].isoformat()
                        }
                        
                        await self.es_client.index(
                            index="patients",
                            id=patient['mrn'],
                            body=doc
                        )
                        
                    # Index medical records
                    records = await conn.fetch("""
                        SELECT mr.*, p.mrn as patient_mrn
                        FROM medical_records mr
                        JOIN patients p ON mr.patient_id = p.id
                        LIMIT 500
                    """)
                    
                    for record in records:
                        doc = {
                            "patient_mrn": record['patient_mrn'],
                            "record_type": record['record_type'],
                            "diagnosis": record['diagnosis'],
                            "notes": record['notes'],
                            "visit_date": record['visit_date'].isoformat() if record['visit_date'] else None,
                            "provider_id": record['provider_id']
                        }
                        
                        await self.es_client.index(
                            index="medical_records",
                            body=doc
                        )
                        
            logger.info("Elasticsearch data seeded successfully")
            
        except Exception as e:
            logger.error(f"Elasticsearch seeding failed: {e}")
            
    async def seed_redis_cache(self):
        """Seed Redis with sample cache data"""
        if not self.redis_client:
            return
            
        try:
            # Sample session data
            session_data = {
                "user_id": "1",
                "email": "admin@vita-agents.com",
                "role": "admin",
                "login_time": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(
                "session:sample_admin",
                3600,  # 1 hour
                json.dumps(session_data)
            )
            
            # Sample metrics
            metrics = {
                "active_users": 25,
                "total_patients": 100,
                "pending_appointments": 15,
                "system_load": 0.75
            }
            
            for key, value in metrics.items():
                await self.redis_client.setex(f"metric:{key}", 300, str(value))
                
            # Sample configuration
            config = {
                "max_upload_size": "10MB",
                "session_timeout": "3600",
                "maintenance_mode": "false"
            }
            
            for key, value in config.items():
                await self.redis_client.set(f"config:{key}", value)
                
            logger.info("Redis cache seeded successfully")
            
        except Exception as e:
            logger.error(f"Redis seeding failed: {e}")
            
    async def setup_minio_buckets(self):
        """Setup MinIO buckets and sample files"""
        try:
            buckets = [
                "patient-documents",
                "medical-images",
                "reports",
                "backups",
                "temp-uploads"
            ]
            
            for bucket_name in buckets:
                if not self.minio_client.bucket_exists(bucket_name):
                    self.minio_client.make_bucket(bucket_name)
                    logger.info(f"Created bucket: {bucket_name}")
                    
            # Upload sample files
            sample_content = "This is a sample medical document for demonstration purposes."
            
            for i in range(5):
                file_name = f"sample_document_{i+1}.txt"
                self.minio_client.put_object(
                    "patient-documents",
                    file_name,
                    data=sample_content.encode(),
                    length=len(sample_content.encode()),
                    content_type="text/plain"
                )
                
            logger.info("MinIO buckets and sample files created")
            
        except Exception as e:
            logger.error(f"MinIO setup failed: {e}")
            
    async def run_seeding(self):
        """Run all seeding operations"""
        logger.info("Starting data seeding process...")
        
        await self.setup_connections()
        
        # Database seeding
        await self.create_database_schema()
        await self.seed_users(20)
        await self.seed_patients(100)
        await self.seed_medical_records(500)
        await self.seed_appointments(200)
        
        # Elasticsearch seeding
        await self.seed_elasticsearch_data()
        
        # Redis seeding
        await self.seed_redis_cache()
        
        # MinIO setup
        self.setup_minio_buckets()
        
        logger.info("Data seeding completed successfully!")
        
    async def cleanup(self):
        """Cleanup connections"""
        if self.db_pool:
            await self.db_pool.close()
            
        if self.redis_client:
            await self.redis_client.close()
            
        if self.es_client:
            await self.es_client.close()

async def main():
    seeder = DataSeeder()
    try:
        await seeder.run_seeding()
    finally:
        await seeder.cleanup()

if __name__ == "__main__":
    asyncio.run(main())