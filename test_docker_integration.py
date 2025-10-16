#!/usr/bin/env python3
"""
Test Docker Integration for Vita Agents
Tests connectivity to all Docker services
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Import the same libraries used in the main app
try:
    import asyncpg
    import redis.asyncio as redis
    import aio_pika
    from elasticsearch import AsyncElasticsearch
    from minio import Minio
    from minio.error import S3Error
    
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Run: pip install -r requirements.txt")
    DEPENDENCIES_AVAILABLE = False

class DockerServiceTester:
    def __init__(self):
        # Use the same configuration as the main app
        self.database_url = os.getenv('DATABASE_URL', 'postgresql://vita_user:vita_secure_pass_2024@localhost:5432/vita_agents')
        self.redis_url = os.getenv('REDIS_URL', 'redis://:vita_redis_pass_2024@localhost:6379/0')
        self.elasticsearch_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
        self.rabbitmq_url = os.getenv('RABBITMQ_URL', 'amqp://vita_admin:vita_rabbit_pass_2024@localhost:5672/vita_vhost')
        self.minio_endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        self.minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'vita_admin')
        self.minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'vita_minio_pass_2024')
        
        self.results = {}
        
    async def test_postgresql(self):
        """Test PostgreSQL connection and basic operations"""
        print("🗄️  Testing PostgreSQL...")
        
        try:
            # Parse connection URL
            db_url = self.database_url.replace('postgresql://', '').replace('postgresql+asyncpg://', '')
            pool = await asyncpg.create_pool(f"postgresql://{db_url}", min_size=1, max_size=2)
            
            async with pool.acquire() as conn:
                # Test basic query
                result = await conn.fetchval("SELECT version()")
                print(f"   ✅ Connected to PostgreSQL")
                print(f"   📋 Version: {result.split(',')[0]}")
                
                # Test if vita_agents database exists
                db_exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = 'vita_agents')"
                )
                print(f"   📂 vita_agents database: {'exists' if db_exists else 'missing'}")
                
                # Test user table existence
                try:
                    user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
                    print(f"   👥 Users table: {user_count} records")
                except:
                    print(f"   ⚠️  Users table: not found (run seeder)")
                
            await pool.close()
            self.results['postgresql'] = {'status': 'success', 'message': 'Connected successfully'}
            
        except Exception as e:
            print(f"   ❌ PostgreSQL connection failed: {e}")
            self.results['postgresql'] = {'status': 'error', 'message': str(e)}
            
    async def test_redis(self):
        """Test Redis connection and basic operations"""
        print("\n🔴 Testing Redis...")
        
        try:
            redis_client = redis.from_url(self.redis_url, decode_responses=True)
            
            # Test basic operations
            await redis_client.ping()
            print(f"   ✅ Connected to Redis")
            
            # Test set/get
            test_key = "vita_test_key"
            test_value = f"test_{datetime.now().isoformat()}"
            await redis_client.setex(test_key, 30, test_value)
            retrieved = await redis_client.get(test_key)
            
            if retrieved == test_value:
                print(f"   📝 Read/Write: working")
            else:
                print(f"   ⚠️  Read/Write: inconsistent")
                
            # Get Redis info
            info = await redis_client.info()
            print(f"   💾 Memory used: {info.get('used_memory_human', 'unknown')}")
            
            await redis_client.delete(test_key)
            await redis_client.close()
            
            self.results['redis'] = {'status': 'success', 'message': 'Connected successfully'}
            
        except Exception as e:
            print(f"   ❌ Redis connection failed: {e}")
            self.results['redis'] = {'status': 'error', 'message': str(e)}
            
    async def test_elasticsearch(self):
        """Test Elasticsearch connection and basic operations"""
        print("\n🔍 Testing Elasticsearch...")
        
        try:
            es_client = AsyncElasticsearch([self.elasticsearch_url])
            
            # Test connection
            health = await es_client.cluster.health()
            print(f"   ✅ Connected to Elasticsearch")
            print(f"   🟢 Cluster status: {health['status']}")
            print(f"   📊 Nodes: {health['number_of_nodes']}")
            
            # Test index operations
            test_index = "vita_test_index"
            test_doc = {
                "message": "Test document",
                "timestamp": datetime.now().isoformat()
            }
            
            # Create test document
            await es_client.index(
                index=test_index,
                id="test_doc_1",
                body=test_doc
            )
            print(f"   📝 Document indexing: working")
            
            # Search test
            await asyncio.sleep(1)  # Wait for indexing
            search_result = await es_client.search(
                index=test_index,
                body={"query": {"match": {"message": "Test"}}}
            )
            
            if search_result['hits']['total']['value'] > 0:
                print(f"   🔎 Search functionality: working")
            else:
                print(f"   ⚠️  Search functionality: no results")
                
            # Cleanup
            await es_client.indices.delete(index=test_index, ignore=404)
            await es_client.close()
            
            self.results['elasticsearch'] = {'status': 'success', 'message': 'Connected successfully'}
            
        except Exception as e:
            print(f"   ❌ Elasticsearch connection failed: {e}")
            self.results['elasticsearch'] = {'status': 'error', 'message': str(e)}
            
    async def test_rabbitmq(self):
        """Test RabbitMQ connection and basic operations"""
        print("\n🐰 Testing RabbitMQ...")
        
        try:
            connection = await aio_pika.connect_robust(self.rabbitmq_url)
            channel = await connection.channel()
            
            print(f"   ✅ Connected to RabbitMQ")
            
            # Declare test queue
            test_queue = await channel.declare_queue("vita_test_queue", durable=False, auto_delete=True)
            
            # Send test message
            test_message = {"test": "message", "timestamp": datetime.now().isoformat()}
            await channel.default_exchange.publish(
                aio_pika.Message(json.dumps(test_message).encode()),
                routing_key="vita_test_queue"
            )
            print(f"   📤 Message publishing: working")
            
            # Receive test message
            message = await test_queue.get(timeout=5)
            if message:
                received_data = json.loads(message.body.decode())
                if received_data.get("test") == "message":
                    print(f"   📥 Message consuming: working")
                    await message.ack()
                else:
                    print(f"   ⚠️  Message consuming: data mismatch")
            else:
                print(f"   ⚠️  Message consuming: no message received")
                
            await connection.close()
            
            self.results['rabbitmq'] = {'status': 'success', 'message': 'Connected successfully'}
            
        except Exception as e:
            print(f"   ❌ RabbitMQ connection failed: {e}")
            self.results['rabbitmq'] = {'status': 'error', 'message': str(e)}
            
    def test_minio(self):
        """Test MinIO connection and basic operations"""
        print("\n📦 Testing MinIO...")
        
        try:
            minio_client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=False
            )
            
            # Test connection by listing buckets
            buckets = minio_client.list_buckets()
            print(f"   ✅ Connected to MinIO")
            print(f"   🪣 Buckets: {len(buckets)} found")
            
            for bucket in buckets:
                print(f"      - {bucket.name} (created: {bucket.creation_date})")
            
            # Test bucket operations
            test_bucket = "vita-test-bucket"
            
            try:
                # Create test bucket
                if not minio_client.bucket_exists(test_bucket):
                    minio_client.make_bucket(test_bucket)
                    print(f"   📂 Bucket creation: working")
                
                # Test file upload
                test_content = f"Test file content - {datetime.now().isoformat()}"
                test_file = "test-file.txt"
                
                minio_client.put_object(
                    test_bucket,
                    test_file,
                    data=test_content.encode(),
                    length=len(test_content.encode()),
                    content_type="text/plain"
                )
                print(f"   📤 File upload: working")
                
                # Test file download
                response = minio_client.get_object(test_bucket, test_file)
                downloaded_content = response.read().decode()
                
                if downloaded_content == test_content:
                    print(f"   📥 File download: working")
                else:
                    print(f"   ⚠️  File download: content mismatch")
                
                # Cleanup
                minio_client.remove_object(test_bucket, test_file)
                minio_client.remove_bucket(test_bucket)
                
            except S3Error as e:
                print(f"   ⚠️  MinIO operations failed: {e}")
                
            self.results['minio'] = {'status': 'success', 'message': 'Connected successfully'}
            
        except Exception as e:
            print(f"   ❌ MinIO connection failed: {e}")
            self.results['minio'] = {'status': 'error', 'message': str(e)}
            
    async def run_all_tests(self):
        """Run all service tests"""
        print("🧪 Vita Agents Docker Services Integration Test")
        print("=" * 60)
        
        if not DEPENDENCIES_AVAILABLE:
            print("❌ Required dependencies not available")
            return False
            
        # Test all services
        await self.test_postgresql()
        await self.test_redis()
        await self.test_elasticsearch()
        await self.test_rabbitmq()
        self.test_minio()  # Synchronous
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        successful = 0
        failed = 0
        
        for service, result in self.results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_icon} {service.capitalize()}: {result['message']}")
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
                
        print(f"\n📈 Results: {successful} successful, {failed} failed")
        
        if failed == 0:
            print("🎉 All Docker services are working correctly!")
            print("🚀 Ready to start Vita Agents with full Docker integration")
            return True
        else:
            print("⚠️  Some services failed - check Docker containers")
            print("💡 Try: docker-compose up -d")
            return False

async def main():
    tester = DockerServiceTester()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())