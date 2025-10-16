#!/usr/bin/env python3
"""
Vita Agents - Enhanced Healthcare Platform with Docker Integration
Complete production-ready startup script
"""

import asyncio
import sys
import os
import signal
import subprocess
import time
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the enhanced web portal
try:
    from enhanced_web_portal import app, Config, init_connections, close_connections
    import uvicorn
    MAIN_APP_AVAILABLE = True
except ImportError as e:
    print(f"❌ Cannot import main application: {e}")
    MAIN_APP_AVAILABLE = False

class VitaAgentsLauncher:
    def __init__(self):
        self.config = Config()
        self.docker_running = False
        self.app_process = None
        
    def check_docker_status(self):
        """Check if Docker services are running"""
        print("🐳 Checking Docker services status...")
        
        try:
            result = subprocess.run(['docker-compose', 'ps'], 
                                  capture_output=True, text=True, cwd=Path(__file__).parent)
            
            if result.returncode == 0:
                output = result.stdout
                if 'vita-postgresql' in output and 'Up' in output:
                    print("   ✅ PostgreSQL is running")
                    self.docker_running = True
                else:
                    print("   ❌ Docker services not running")
                    return False
            else:
                print("   ❌ Docker Compose not available")
                return False
                
        except FileNotFoundError:
            print("   ❌ Docker or Docker Compose not found")
            return False
            
        return True
        
    def start_docker_services(self):
        """Start Docker services if not running"""
        if self.check_docker_status():
            print("✅ Docker services are already running")
            return True
            
        print("🚀 Starting Docker services...")
        
        try:
            # Start services in detached mode
            result = subprocess.run(['docker-compose', 'up', '-d'], 
                                  cwd=Path(__file__).parent)
            
            if result.returncode == 0:
                print("✅ Docker services started successfully")
                
                # Wait for services to be ready
                print("⏳ Waiting for services to initialize...")
                time.sleep(10)
                
                return self.check_docker_status()
            else:
                print("❌ Failed to start Docker services")
                return False
                
        except Exception as e:
            print(f"❌ Error starting Docker services: {e}")
            return False
            
    def run_integration_test(self):
        """Run integration tests to verify Docker connectivity"""
        print("\n🧪 Running Docker integration tests...")
        
        try:
            test_script = Path(__file__).parent / "test_docker_integration.py"
            if test_script.exists():
                result = subprocess.run([sys.executable, str(test_script)], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ All Docker services are working correctly")
                    return True
                else:
                    print("❌ Some Docker services failed tests:")
                    print(result.stdout)
                    print(result.stderr)
                    return False
            else:
                print("⚠️  Integration test script not found, skipping tests")
                return True
                
        except Exception as e:
            print(f"❌ Error running integration tests: {e}")
            return False
            
    def start_web_application(self):
        """Start the main web application"""
        if not MAIN_APP_AVAILABLE:
            print("❌ Main application not available")
            return False
            
        print("\n🌐 Starting Vita Agents Web Application...")
        print(f"🔧 Configuration:")
        print(f"   📊 Mode: {self.config.ENVIRONMENT}")
        print(f"   🗄️  Database: {self.config.DATABASE_URL.split('@')[1] if '@' in self.config.DATABASE_URL else 'Not configured'}")
        print(f"   🔴 Redis: {self.config.REDIS_URL.split('@')[1] if '@' in self.config.REDIS_URL else 'Not configured'}")
        print(f"   🔍 Elasticsearch: {self.config.ELASTICSEARCH_URL}")
        print(f"   🐰 RabbitMQ: {self.config.RABBITMQ_URL.split('@')[1] if '@' in self.config.RABBITMQ_URL else 'Not configured'}")
        print(f"   📦 MinIO: {self.config.MINIO_ENDPOINT}")
        
        try:
            # Start the FastAPI application
            uvicorn.run(
                "enhanced_web_portal:app",
                host="0.0.0.0",
                port=8083,
                log_level="info",
                reload=self.config.ENVIRONMENT == "development"
            )
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down application...")
            return True
        except Exception as e:
            print(f"❌ Failed to start web application: {e}")
            return False
            
    def cleanup(self):
        """Cleanup resources on shutdown"""
        print("\n🧹 Cleaning up resources...")
        
        if self.app_process:
            self.app_process.terminate()
            
        # Note: We don't stop Docker services here as they might be used by other processes
        print("✅ Cleanup completed")
        
    def run(self):
        """Main execution method"""
        print("🏥 Vita Agents - Healthcare Platform with Docker Integration")
        print("=" * 70)
        
        try:
            # Step 1: Start Docker services
            if not self.start_docker_services():
                print("❌ Failed to start Docker services")
                return False
                
            # Step 2: Run integration tests
            if not self.run_integration_test():
                print("⚠️  Some services may not be working properly")
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    return False
                    
            # Step 3: Start web application
            print("\n" + "=" * 70)
            print("🎯 All systems ready! Starting web application...")
            print(f"🌐 Application will be available at: http://localhost:8083")
            print(f"📊 Grafana dashboard: http://localhost:3000 (admin/admin)")
            print(f"📧 MailHog interface: http://localhost:8025")
            print(f"📦 MinIO console: http://localhost:9001")
            print("=" * 70)
            
            return self.start_web_application()
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            return True
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
        finally:
            self.cleanup()

def show_help():
    """Show help information"""
    print("""
🏥 Vita Agents - Healthcare Platform with Docker Integration

Usage:
    python vita_agents_launcher.py [command]

Commands:
    start       Start the full platform (default)
    test        Run Docker integration tests only
    docker      Start Docker services only
    help        Show this help message

Examples:
    python vita_agents_launcher.py          # Start everything
    python vita_agents_launcher.py start    # Start everything
    python vita_agents_launcher.py test     # Test Docker services
    python vita_agents_launcher.py docker   # Start Docker only

Prerequisites:
    - Docker and Docker Compose installed
    - Python dependencies: pip install -r requirements.txt

Access Points:
    🌐 Main Application:    http://localhost:8083
    📊 Grafana Dashboard:   http://localhost:3000 (admin/admin)
    📧 MailHog Interface:   http://localhost:8025
    📦 MinIO Console:       http://localhost:9001 (vita_admin/vita_minio_pass_2024)
    🔍 Elasticsearch:       http://localhost:9200
    🗄️  PostgreSQL:         localhost:5432 (vita_user/vita_secure_pass_2024)
    """)

def main():
    """Main entry point"""
    launcher = VitaAgentsLauncher()
    
    # Parse command line arguments
    command = sys.argv[1] if len(sys.argv) > 1 else "start"
    
    if command == "help":
        show_help()
        return
    elif command == "test":
        if launcher.start_docker_services():
            launcher.run_integration_test()
        return
    elif command == "docker":
        launcher.start_docker_services()
        print("🐳 Docker services started. Use 'docker-compose ps' to check status.")
        return
    elif command == "start":
        success = launcher.run()
        sys.exit(0 if success else 1)
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python vita_agents_launcher.py help' for usage information")
        sys.exit(1)

if __name__ == "__main__":
    main()