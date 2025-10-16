#!/usr/bin/env python3
"""
Vita Agents - Complete Application Status Checker
Comprehensive health check for all components
"""

import requests
import subprocess
import time
import sys
from datetime import datetime

def check_docker_services():
    """Check Docker infrastructure"""
    print("🐳 DOCKER INFRASTRUCTURE")
    print("=" * 50)
    
    try:
        result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            healthy_count = 0
            total_count = 0
            
            for line in lines[1:]:  # Skip header
                if 'vita-' in line:
                    total_count += 1
                    if 'healthy' in line or 'Up' in line:
                        healthy_count += 1
                        print(f"✅ {line}")
                    else:
                        print(f"❌ {line}")
            
            print(f"\n📊 Docker Status: {healthy_count}/{total_count} services healthy")
            return healthy_count > 0
        else:
            print("❌ Docker not available")
            return False
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return False

def check_web_application():
    """Check if web application is running"""
    print("\n🌐 WEB APPLICATION")
    print("=" * 50)
    
    ports_to_check = [8080, 8081, 8082, 8083, 5000]
    running_apps = []
    
    for port in ports_to_check:
        try:
            response = requests.get(f'http://localhost:{port}/api/status', timeout=2)
            if response.status_code == 200:
                print(f"✅ Web app running on port {port}")
                running_apps.append(port)
        except:
            try:
                response = requests.get(f'http://localhost:{port}/', timeout=2)
                if response.status_code == 200:
                    print(f"✅ Web server running on port {port}")
                    running_apps.append(port)
            except:
                print(f"❌ No service on port {port}")
    
    if running_apps:
        print(f"\n📊 Web Status: {len(running_apps)} application(s) running")
        return True
    else:
        print("\n📊 Web Status: No web applications running")
        return False

def check_python_processes():
    """Check Python processes"""
    print("\n🐍 PYTHON PROCESSES")
    print("=" * 50)
    
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            python_processes = []
            
            for line in lines:
                if 'python.exe' in line and 'vita' not in line.lower():
                    # Filter out VS Code extension processes
                    if 'sema4ai' not in line and 'vscode' not in line:
                        python_processes.append(line)
            
            if python_processes:
                print(f"✅ {len(python_processes)} Python process(es) running")
                for process in python_processes[:3]:  # Show first 3
                    print(f"   • {process}")
                return True
            else:
                print("❌ No Vita Agents Python processes running")
                return False
        else:
            print("❌ Cannot check Python processes")
            return False
    except Exception as e:
        print(f"❌ Python process check failed: {e}")
        return False

def check_database_connection():
    """Check database connectivity"""
    print("\n🗄️ DATABASE CONNECTION")
    print("=" * 50)
    
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='vita_agents',
            user='vita_user',
            password='vita_secure_pass_2024'
        )
        cursor = conn.cursor()
        cursor.execute('SELECT version();')
        version = cursor.fetchone()[0]
        print(f"✅ PostgreSQL connected: {version[:50]}...")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def check_redis_connection():
    """Check Redis connectivity"""
    print("\n🔴 REDIS CONNECTION")
    print("=" * 50)
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        info = r.info()
        print(f"✅ Redis connected: {info['redis_version']}")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def main():
    """Main status check function"""
    print(f"""
🏥 VITA AGENTS - COMPLETE STATUS CHECK
{'=' * 60}
⏰ Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    
    # Run all checks
    docker_ok = check_docker_services()
    web_ok = check_web_application()
    python_ok = check_python_processes()
    db_ok = check_database_connection()
    redis_ok = check_redis_connection()
    
    # Summary
    print(f"\n🎯 OVERALL STATUS SUMMARY")
    print("=" * 60)
    
    checks = [
        ("Docker Infrastructure", docker_ok),
        ("Web Application", web_ok),
        ("Python Processes", python_ok),
        ("Database Connection", db_ok),
        ("Redis Connection", redis_ok)
    ]
    
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
    
    print(f"\n📊 Health Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed >= 4:
        print("🎉 System Status: HEALTHY")
    elif passed >= 2:
        print("⚠️  System Status: PARTIAL")
    else:
        print("🚨 System Status: CRITICAL")
    
    # Recommendations
    print(f"\n🚀 QUICK START RECOMMENDATIONS")
    print("=" * 60)
    
    if not docker_ok:
        print("1. Start Docker services: docker-compose up -d")
    
    if not web_ok:
        print("2. Start web application: python vita_agents_launcher.py start")
    
    if not python_ok:
        print("3. Check application logs for errors")
    
    if docker_ok and not (db_ok or redis_ok):
        print("4. Wait for database containers to fully initialize")
    
    print(f"\n🌐 ACCESS POINTS (if running):")
    print("   • Main App:      http://localhost:8083")
    print("   • Grafana:       http://localhost:3000 (admin/admin)")
    print("   • MailHog:       http://localhost:8025")
    print("   • MinIO Console: http://localhost:9001")

if __name__ == "__main__":
    main()