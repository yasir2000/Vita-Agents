#!/usr/bin/env python3
"""
Background Task Worker for Vita Agents
Processes async tasks from RabbitMQ queue
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Any

import aio_pika
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/worker.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TaskWorker:
    def __init__(self):
        self.rabbitmq_url = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672/')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.database_url = os.getenv('DATABASE_URL', '').replace('postgresql://', 'postgresql+asyncpg://')
        
        self.connection = None
        self.channel = None
        self.redis_client = None
        self.db_engine = None
        self.running = True
        
    async def setup(self):
        """Initialize connections"""
        try:
            # RabbitMQ connection
            self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
            self.channel = await self.connection.channel()
            
            # Declare queues
            await self.channel.declare_queue('vita_tasks', durable=True)
            await self.channel.declare_queue('vita_notifications', durable=True)
            await self.channel.declare_queue('vita_analytics', durable=True)
            
            # Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            
            # Database connection
            if self.database_url:
                self.db_engine = create_async_engine(self.database_url)
                
            logger.info("Worker setup completed")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
            
    async def process_task(self, message: aio_pika.IncomingMessage):
        """Process a single task"""
        async with message.process():
            try:
                task_data = json.loads(message.body.decode())
                task_id = task_data.get('task_id')
                task_type = task_data.get('task_type')
                
                logger.info(f"Processing task {task_id} of type {task_type}")
                
                # Update task status to processing
                if self.redis_client:
                    await self.redis_client.hset(
                        f"task:{task_id}",
                        mapping={
                            'status': 'processing',
                            'started_at': datetime.utcnow().isoformat(),
                            'worker_id': os.getpid()
                        }
                    )
                
                # Route task to appropriate handler
                result = await self.route_task(task_type, task_data)
                
                # Update task status to completed
                if self.redis_client:
                    await self.redis_client.hset(
                        f"task:{task_id}",
                        mapping={
                            'status': 'completed',
                            'completed_at': datetime.utcnow().isoformat(),
                            'result': json.dumps(result)
                        }
                    )
                
                logger.info(f"Task {task_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Task processing failed: {e}")
                if 'task_id' in locals() and self.redis_client:
                    await self.redis_client.hset(
                        f"task:{task_id}",
                        mapping={
                            'status': 'failed',
                            'error': str(e),
                            'failed_at': datetime.utcnow().isoformat()
                        }
                    )
                
    async def route_task(self, task_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Route task to appropriate handler"""
        handlers = {
            'fhir_validation': self.handle_fhir_validation,
            'hl7_processing': self.handle_hl7_processing,
            'data_analysis': self.handle_data_analysis,
            'notification': self.handle_notification,
            'report_generation': self.handle_report_generation,
            'ml_inference': self.handle_ml_inference,
        }
        
        handler = handlers.get(task_type, self.handle_unknown_task)
        return await handler(task_data)
        
    async def handle_fhir_validation(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle FHIR resource validation"""
        logger.info("Processing FHIR validation task")
        # Simulate FHIR validation
        await asyncio.sleep(1)
        return {
            'status': 'validated',
            'resource_type': task_data.get('resource_type', 'Patient'),
            'validation_errors': [],
            'processed_at': datetime.utcnow().isoformat()
        }
        
    async def handle_hl7_processing(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle HL7 message processing"""
        logger.info("Processing HL7 message")
        await asyncio.sleep(2)
        return {
            'status': 'processed',
            'message_type': task_data.get('message_type', 'ADT'),
            'segments_parsed': 5,
            'processed_at': datetime.utcnow().isoformat()
        }
        
    async def handle_data_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data analysis task"""
        logger.info("Processing data analysis")
        await asyncio.sleep(3)
        return {
            'status': 'analyzed',
            'metrics': {
                'total_records': 1000,
                'processed_records': 995,
                'error_rate': 0.005
            },
            'processed_at': datetime.utcnow().isoformat()
        }
        
    async def handle_notification(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle notification sending"""
        logger.info("Processing notification")
        await asyncio.sleep(1)
        return {
            'status': 'sent',
            'notification_type': task_data.get('type', 'email'),
            'recipient': task_data.get('recipient', 'unknown'),
            'sent_at': datetime.utcnow().isoformat()
        }
        
    async def handle_report_generation(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle report generation"""
        logger.info("Processing report generation")
        await asyncio.sleep(5)
        return {
            'status': 'generated',
            'report_type': task_data.get('report_type', 'summary'),
            'file_path': f"/app/data/reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            'generated_at': datetime.utcnow().isoformat()
        }
        
    async def handle_ml_inference(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ML model inference"""
        logger.info("Processing ML inference")
        await asyncio.sleep(2)
        return {
            'status': 'completed',
            'model': task_data.get('model', 'default'),
            'predictions': [0.85, 0.12, 0.03],
            'confidence': 0.85,
            'processed_at': datetime.utcnow().isoformat()
        }
        
    async def handle_unknown_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unknown task types"""
        logger.warning(f"Unknown task type: {task_data.get('task_type')}")
        return {
            'status': 'skipped',
            'reason': 'unknown_task_type',
            'processed_at': datetime.utcnow().isoformat()
        }
        
    async def start(self):
        """Start the worker"""
        logger.info("Starting Vita Agents Task Worker")
        
        await self.setup()
        
        # Set up signal handlers
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal, stopping worker...")
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start consuming tasks
        queue = await self.channel.declare_queue('vita_tasks', durable=True)
        await queue.consume(self.process_task, no_ack=False)
        
        logger.info("Worker started, waiting for tasks...")
        
        # Keep running until shutdown
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Worker interrupted")
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up worker resources")
        
        if self.connection:
            await self.connection.close()
            
        if self.redis_client:
            await self.redis_client.close()
            
        if self.db_engine:
            await self.db_engine.dispose()

async def main():
    worker = TaskWorker()
    await worker.start()

if __name__ == "__main__":
    asyncio.run(main())