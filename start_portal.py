#!/usr/bin/env python3
"""
Unified Startup Script for Vita Agents Healthcare Portal
========================================================

This script provides a single entry point to start the enhanced web portal
with automatic port detection and conflict resolution.

Usage:
    python start_portal.py              # Start on default port 8080
    python start_portal.py --port 8081  # Start on specific port
    python start_portal.py --dev        # Start in development mode with hot reload
"""

import sys
import socket
import subprocess
import argparse
import time
import requests
from pathlib import Path

# Default configuration
DEFAULT_PORT = 8080
HOST = "0.0.0.0"
PORTAL_MODULE = "enhanced_web_portal:app"

def check_port_available(port):
    """Check if a port is available for use"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            return result != 0
    except Exception:
        return False

def find_available_port(start_port=DEFAULT_PORT, max_attempts=10):
    """Find the next available port starting from start_port"""
    for i in range(max_attempts):
        port = start_port + i
        if check_port_available(port):
            return port
    return None

def kill_existing_processes():
    """Kill any existing Python processes that might be using the port"""
    try:
        if sys.platform == "win32":
            # Windows
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"], 
                         capture_output=True, text=True)
        else:
            # Unix/Linux/macOS
            subprocess.run(["pkill", "-f", "uvicorn"], 
                         capture_output=True, text=True)
        time.sleep(2)  # Wait for processes to terminate
    except Exception as e:
        print(f"Note: Could not kill existing processes: {e}")

def test_server(port, max_retries=30):
    """Test if the server is responding"""
    url = f"http://localhost:{port}"
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False

def start_server(port, dev_mode=False):
    """Start the Vita Agents portal server"""
    print(f"🏥 Starting Vita Agents Healthcare Portal")
    print(f"📍 Host: {HOST}")
    print(f"🚪 Port: {port}")
    print(f"🔗 URL: http://localhost:{port}")
    print(f"🌟 Mode: {'Development' if dev_mode else 'Production'}")
    print("=" * 60)

    # Build uvicorn command
    cmd = [
        sys.executable, "-m", "uvicorn", PORTAL_MODULE,
        "--host", HOST,
        "--port", str(port)
    ]
    
    if dev_mode:
        cmd.extend(["--reload", "--log-level", "debug"])
    
    try:
        # Start the server
        print("🚀 Starting server...")
        process = subprocess.Popen(cmd, cwd=Path(__file__).parent)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Test if server is responding
        print("🔍 Testing server connection...")
        if test_server(port, max_retries=10):
            print("✅ Server is running successfully!")
            print(f"🌐 Access the portal at: http://localhost:{port}")
            print("\n📊 Available Pages:")
            print(f"   • Dashboard:     http://localhost:{port}/")
            print(f"   • LLM AI:        http://localhost:{port}/llm")
            print(f"   • Patients:      http://localhost:{port}/patients")
            print(f"   • Clinical:      http://localhost:{port}/clinical")
            print(f"   • Analytics:     http://localhost:{port}/analytics")
            print(f"   • Agents:        http://localhost:{port}/agents")
            print(f"   • AI Models:     http://localhost:{port}/ai-models")
            print(f"   • Workflows:     http://localhost:{port}/workflows")
            print(f"   • Alerts:        http://localhost:{port}/alerts")
            print(f"   • Integration:   http://localhost:{port}/integration")
            print(f"   • Compliance:    http://localhost:{port}/compliance")
            print(f"   • Monitoring:    http://localhost:{port}/monitoring")
            print(f"   • Testing:       http://localhost:{port}/testing")
            print(f"   • API Docs:      http://localhost:{port}/api/docs")
            print("\n🛑 Press Ctrl+C to stop the server")
            
            # Wait for the process to complete
            process.wait()
        else:
            print("❌ Server failed to start properly")
            process.terminate()
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("✅ Server stopped successfully")
        return True
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Start Vita Agents Healthcare Portal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_portal.py              # Start on port 8080
  python start_portal.py --port 8081  # Start on port 8081
  python start_portal.py --dev        # Development mode with auto-reload
  python start_portal.py --clean      # Clean start (kill existing processes)
        """
    )
    
    parser.add_argument(
        "--port", "-p", 
        type=int, 
        default=DEFAULT_PORT,
        help=f"Port to run the server on (default: {DEFAULT_PORT})"
    )
    
    parser.add_argument(
        "--dev", 
        action="store_true",
        help="Run in development mode with hot reload"
    )
    
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Kill existing Python processes before starting"
    )
    
    parser.add_argument(
        "--find-port", 
        action="store_true",
        help="Automatically find an available port if specified port is busy"
    )

    args = parser.parse_args()

    # Clean existing processes if requested
    if args.clean:
        print("🧹 Cleaning existing processes...")
        kill_existing_processes()

    # Determine port to use
    port = args.port
    
    if not check_port_available(port):
        if args.find_port:
            print(f"⚠️  Port {port} is busy, finding available port...")
            available_port = find_available_port(port)
            if available_port:
                port = available_port
                print(f"✅ Found available port: {port}")
            else:
                print("❌ No available ports found in range")
                return 1
        else:
            print(f"❌ Port {port} is already in use!")
            print("💡 Options:")
            print("   1. Use --find-port to automatically find available port")
            print("   2. Use --clean to kill existing processes")
            print("   3. Specify different port with --port")
            return 1

    # Start the server
    success = start_server(port, dev_mode=args.dev)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())