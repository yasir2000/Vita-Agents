#!/usr/bin/env python3
"""
Vita Agents - Final Integration Verification
Comprehensive check to ensure Docker integration is complete
"""

import asyncio
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

class VitaAgentsVerifier:
    def __init__(self):
        self.results = {}
        self.project_root = Path(__file__).parent
        
    def check_file_exists(self, filepath, description):
        """Check if a file exists"""
        full_path = self.project_root / filepath
        exists = full_path.exists()
        
        status = "✅" if exists else "❌"
        print(f"{status} {description}: {filepath}")
        
        return exists
        
    def check_docker_files(self):
        """Verify all Docker-related files exist"""
        print("🐳 Checking Docker Configuration Files...")
        
        files_to_check = [
            ("docker-compose.yml", "Main Docker Compose file"),
            ("Dockerfile", "Application Dockerfile"),
            (".dockerignore", "Docker ignore file"),
            ("configs/postgresql.conf", "PostgreSQL configuration"),
            ("configs/redis.conf", "Redis configuration"),
            ("configs/rabbitmq.conf", "RabbitMQ configuration"),
            ("configs/nginx.conf", "Nginx configuration"),
            ("configs/prometheus.yml", "Prometheus configuration"),
            ("configs/grafana/provisioning/dashboards/vita-dashboard.json", "Grafana dashboard"),
            ("vita-docker.sh", "Unix Docker script"),
            ("vita-docker.bat", "Windows Docker script"),
        ]
        
        all_exist = True
        for filepath, description in files_to_check:
            if not self.check_file_exists(filepath, description):
                all_exist = False
                
        self.results['docker_files'] = all_exist
        return all_exist
        
    def check_application_files(self):
        """Verify application files exist and have Docker integration"""
        print("\n🏥 Checking Application Files...")
        
        files_to_check = [
            ("enhanced_web_portal.py", "Main application with Docker integration"),
            ("vita_agents_launcher.py", "Production launcher script"),
            ("test_docker_integration.py", "Docker integration tests"),
            ("data_seeder.py", "Database seeder"),
            ("task_worker.py", "Background task worker"),
            ("requirements.txt", "Python dependencies"),
        ]
        
        all_exist = True
        for filepath, description in files_to_check:
            if not self.check_file_exists(filepath, description):
                all_exist = False
                
        self.results['application_files'] = all_exist
        return all_exist
        
    def check_docker_integration_code(self):
        """Check if application code has Docker service integration"""
        print("\n🔧 Checking Docker Integration in Code...")
        
        try:
            with open(self.project_root / "enhanced_web_portal.py", 'r', encoding='utf-8') as f:
                content = f.read()
                
            checks = [
                ("import asyncpg", "PostgreSQL driver import"),
                ("import redis.asyncio", "Redis client import"),
                ("from elasticsearch import AsyncElasticsearch", "Elasticsearch client import"),
                ("import aio_pika", "RabbitMQ client import"),
                ("from minio import Minio", "MinIO client import"),
                ("class Config:", "Configuration class"),
                ("DATABASE_URL", "Database configuration"),
                ("REDIS_URL", "Redis configuration"),
                ("init_connections", "Connection initialization"),
                ("@app.on_event(\"startup\")", "FastAPI startup events"),
            ]
            
            all_found = True
            for check, description in checks:
                if check in content:
                    print(f"   ✅ {description}")
                else:
                    print(f"   ❌ {description}")
                    all_found = False
                    
            self.results['docker_integration'] = all_found
            return all_found
            
        except Exception as e:
            print(f"   ❌ Error checking application code: {e}")
            self.results['docker_integration'] = False
            return False
            
    def check_docker_availability(self):
        """Check if Docker and Docker Compose are available"""
        print("\n🐳 Checking Docker Availability...")
        
        try:
            # Check Docker
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"   ✅ Docker: {version}")
                docker_ok = True
            else:
                print(f"   ❌ Docker not available")
                docker_ok = False
                
            # Check Docker Compose
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"   ✅ Docker Compose: {version}")
                compose_ok = True
            else:
                print(f"   ❌ Docker Compose not available")
                compose_ok = False
                
            both_ok = docker_ok and compose_ok
            self.results['docker_availability'] = both_ok
            return both_ok
            
        except FileNotFoundError:
            print(f"   ❌ Docker not found in PATH")
            self.results['docker_availability'] = False
            return False
            
    def check_python_dependencies(self):
        """Check if required Python packages are installed"""
        print("\n🐍 Checking Python Dependencies...")
        
        required_packages = [
            'fastapi',
            'uvicorn',
            'asyncpg',
            'redis',
            'elasticsearch',
            'aio_pika',
            'minio',
            'prometheus_client'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} (missing)")
                missing_packages.append(package)
                
        if missing_packages:
            print(f"\n   💡 Install missing packages: pip install {' '.join(missing_packages)}")
            
        all_installed = len(missing_packages) == 0
        self.results['python_dependencies'] = all_installed
        return all_installed
        
    def generate_summary_report(self):
        """Generate a summary report"""
        print("\n" + "=" * 70)
        print("📊 VITA AGENTS DOCKER INTEGRATION VERIFICATION REPORT")
        print("=" * 70)
        
        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results.values() if result)
        
        for check_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            formatted_name = check_name.replace('_', ' ').title()
            print(f"{status} - {formatted_name}")
            
        print(f"\n📈 Overall Score: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            print("\n🎉 CONGRATULATIONS! 🎉")
            print("✅ All systems are ready for Docker integration!")
            print("🚀 You can now run: python vita_agents_launcher.py")
            print("\n🌐 Access points after startup:")
            print("   • Main Application: http://localhost:8083")
            print("   • Grafana Dashboard: http://localhost:3000")
            print("   • MailHog Interface: http://localhost:8025")
            print("   • MinIO Console: http://localhost:9001")
            return True
        else:
            print("\n⚠️  Some checks failed. Please address the issues above.")
            print("💡 Common solutions:")
            print("   • Install Docker: https://docs.docker.com/get-docker/")
            print("   • Install dependencies: pip install -r requirements.txt")
            print("   • Check file permissions and paths")
            return False
            
    def run_verification(self):
        """Run all verification checks"""
        print("🔍 Vita Agents Docker Integration Verification")
        print("=" * 70)
        
        # Run all checks
        checks = [
            self.check_docker_files,
            self.check_application_files,
            self.check_docker_integration_code,
            self.check_docker_availability,
            self.check_python_dependencies,
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                print(f"❌ Error during check: {e}")
                
        # Generate final report
        return self.generate_summary_report()

def main():
    """Main entry point"""
    verifier = VitaAgentsVerifier()
    success = verifier.run_verification()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()