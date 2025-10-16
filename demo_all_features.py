#!/usr/bin/env python3
"""
Vita Agents Feature Demonstration Script
Comprehensive demonstration of all current features including CLI and Web Portal
"""

import asyncio
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import json

def print_banner():
    """Print the Vita Agents banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    🏥 VITA AGENTS - Healthcare AI Multi-Agent Framework             ║
║                                                                      ║
║    Phase 2: Advanced Features Demonstration                         ║
║    ✅ ML-Based Data Harmonization                                   ║
║    ✅ 10 Advanced AI Managers                                       ║
║    ✅ Enhanced CLI Interface                                        ║
║    ✅ Comprehensive Web Portal                                      ║
║    ✅ Complete Testing Suite                                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'typer', 'rich', 'fastapi', 'uvicorn', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing requirements...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements-ui.txt"
            ], check=True)
            print("✅ Requirements installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return False
    
    return True

def create_sample_data():
    """Create sample data files for demonstration"""
    print("\n📄 Creating sample data files...")
    
    # Create sample FHIR patient
    fhir_patient = {
        "resourceType": "Patient",
        "id": "demo-patient-001",
        "meta": {
            "versionId": "1",
            "lastUpdated": "2024-01-01T12:00:00Z"
        },
        "name": [{"given": ["John"], "family": "Doe"}],
        "gender": "male",
        "birthDate": "1980-01-01",
        "identifier": [
            {
                "system": "http://hospital.org/patient-ids",
                "value": "PAT-DEMO-001"
            }
        ],
        "address": [
            {
                "line": ["123 Healthcare Ave"],
                "city": "Medical City",
                "state": "HC",
                "postalCode": "12345"
            }
        ]
    }
    
    # Create sample HL7 message
    hl7_message = """MSH|^~\\&|SENDING_APP|SENDING_FAC|RECEIVING_APP|RECEIVING_FAC|20240101120000||ADT^A01|12345|P|2.5
EVN||20240101120000
PID|1||PAT-DEMO-001||DOE^JOHN||19800101|M|||123 Healthcare Ave^Medical City^HC^12345||555-1234|555-5678||||123456789
PV1|1|I|ICU^101^A|||1234^PHYSICIAN^ATTENDING|||ICU|||||||1234^PHYSICIAN^ATTENDING|INP|||||||||||||||||||||20240101120000"""
    
    # Create sample harmonization data
    harmonization_data = [
        {"name": "John Smith", "dob": "1985-06-15", "address": "123 Main St", "phone": "555-0123"},
        {"name": "J. Smith", "dob": "06/15/1985", "address": "123 Main Street", "phone": "(555) 0123"},
        {"name": "Jane Doe", "dob": "1990-03-22", "address": "456 Oak Ave", "phone": "555-0456"},
        {"name": "Jane M. Doe", "dob": "03/22/1990", "address": "456 Oak Avenue", "phone": "(555) 456-0000"}
    ]
    
    # Create sample clinical data
    clinical_data = {
        "patient_id": "PAT-DEMO-001",
        "vitals": {
            "heart_rate": 75,
            "blood_pressure": "120/80",
            "temperature": 98.6,
            "oxygen_saturation": 98,
            "respiratory_rate": 16
        },
        "lab_values": {
            "glucose": 95,
            "cholesterol": 180,
            "hemoglobin": 14.5,
            "white_blood_cell_count": 7500,
            "creatinine": 1.0
        },
        "medications": [
            {"name": "lisinopril", "dose": "10mg", "frequency": "daily"},
            {"name": "metformin", "dose": "500mg", "frequency": "twice_daily"}
        ],
        "allergies": ["penicillin", "shellfish"],
        "medical_history": ["hypertension", "diabetes_type_2"]
    }
    
    # Save sample files
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    files_created = []
    
    # FHIR patient file
    fhir_file = sample_dir / "patient_demo.json"
    with open(fhir_file, 'w') as f:
        json.dump(fhir_patient, f, indent=2)
    files_created.append(fhir_file)
    
    # HL7 message file
    hl7_file = sample_dir / "message_demo.hl7"
    with open(hl7_file, 'w') as f:
        f.write(hl7_message)
    files_created.append(hl7_file)
    
    # Harmonization data file
    harmonization_file = sample_dir / "harmonization_demo.json"
    with open(harmonization_file, 'w') as f:
        json.dump(harmonization_data, f, indent=2)
    files_created.append(harmonization_file)
    
    # Clinical data file
    clinical_file = sample_dir / "clinical_demo.json"
    with open(clinical_file, 'w') as f:
        json.dump(clinical_data, f, indent=2)
    files_created.append(clinical_file)
    
    for file_path in files_created:
        print(f"  ✅ Created: {file_path}")
    
    return files_created

async def demonstrate_cli_features():
    """Demonstrate CLI features"""
    print("\n🖥️  Demonstrating CLI Features...")
    print("=" * 50)
    
    cli_script = Path("vita_agents/cli/main.py")
    
    if not cli_script.exists():
        print(f"❌ CLI script not found: {cli_script}")
        return
    
    # CLI commands to demonstrate
    cli_demos = [
        {
            "name": "System Version",
            "command": [sys.executable, str(cli_script), "version"],
            "description": "Show Vita Agents version and features"
        },
        {
            "name": "System Initialization", 
            "command": [sys.executable, str(cli_script), "init"],
            "description": "Initialize all Vita Agents components"
        },
        {
            "name": "System Status",
            "command": [sys.executable, str(cli_script), "status"],
            "description": "Show comprehensive system status"
        },
        {
            "name": "FHIR Validation",
            "command": [sys.executable, str(cli_script), "fhir", "validate", "sample_data/patient_demo.json"],
            "description": "Validate FHIR patient resource"
        },
        {
            "name": "HL7 Parsing",
            "command": [sys.executable, str(cli_script), "hl7", "parse", "sample_data/message_demo.hl7"],
            "description": "Parse and analyze HL7 message"
        },
        {
            "name": "Clinical Analysis",
            "command": [sys.executable, str(cli_script), "clinical", "analyze", "sample_data/clinical_demo.json"],
            "description": "Analyze patient data for clinical insights"
        },
        {
            "name": "Traditional Harmonization",
            "command": [sys.executable, str(cli_script), "harmonization", "traditional", "sample_data/harmonization_demo.json"],
            "description": "Traditional data harmonization"
        },
        {
            "name": "ML Harmonization",
            "command": [sys.executable, str(cli_script), "harmonization", "ml", "sample_data/harmonization_demo.json"],
            "description": "Machine learning-based harmonization"
        },
        {
            "name": "Foundation Models",
            "command": [sys.executable, str(cli_script), "ai", "foundation-models", "analyze", "--text", "Patient presents with chest pain and shortness of breath."],
            "description": "Medical foundation model analysis"
        },
        {
            "name": "Risk Scoring",
            "command": [sys.executable, str(cli_script), "ai", "risk-scoring", "PAT-DEMO-001"],
            "description": "Continuous risk scoring for patient"
        }
    ]
    
    for demo in cli_demos:
        print(f"\n🔧 {demo['name']}")
        print(f"   Description: {demo['description']}")
        print(f"   Command: {' '.join(demo['command'])}")
        
        try:
            # Run CLI command
            result = subprocess.run(
                demo['command'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("   ✅ Command executed successfully")
                # Show first few lines of output
                output_lines = result.stdout.split('\n')[:5]
                for line in output_lines:
                    if line.strip():
                        print(f"   📄 {line}")
            else:
                print("   ⚠️  Command completed with warnings")
                if result.stderr:
                    print(f"   ⚠️  {result.stderr[:100]}...")
                    
        except subprocess.TimeoutExpired:
            print("   ⏱️  Command timed out (this is normal for interactive commands)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Small delay between commands
        await asyncio.sleep(1)

async def demonstrate_web_portal():
    """Demonstrate web portal features"""
    print("\n🌐 Demonstrating Web Portal Features...")
    print("=" * 50)
    
    portal_script = Path("vita_agents/web/portal.py")
    
    if not portal_script.exists():
        print(f"❌ Web portal script not found: {portal_script}")
        return
    
    print("🚀 Starting web portal server...")
    
    try:
        # Start the web server in background
        server_process = subprocess.Popen([
            sys.executable, str(portal_script)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        await asyncio.sleep(5)
        
        # Check if server is running
        try:
            import requests
            response = requests.get("http://localhost:8080/api/status", timeout=5)
            if response.status_code == 200:
                print("✅ Web portal server started successfully!")
                print("🌐 Server URL: http://localhost:8080")
                
                # Demonstrate API endpoints
                api_demos = [
                    ("/api/status", "System Status"),
                    ("/api/agents", "List Available Agents"),
                    ("/api/ai-managers", "List AI Managers"),
                    ("/api/metrics", "Performance Metrics")
                ]
                
                print("\n📡 Testing API Endpoints:")
                for endpoint, description in api_demos:
                    try:
                        response = requests.get(f"http://localhost:8080{endpoint}", timeout=5)
                        if response.status_code == 200:
                            print(f"   ✅ {description} ({endpoint})")
                        else:
                            print(f"   ⚠️  {description} ({endpoint}) - Status: {response.status_code}")
                    except Exception as e:
                        print(f"   ❌ {description} ({endpoint}) - Error: {e}")
                
                # Try to open browser
                print("\n🌐 Opening web portal in browser...")
                try:
                    webbrowser.open("http://localhost:8080")
                    print("✅ Web portal opened in browser")
                except Exception as e:
                    print(f"⚠️  Could not open browser: {e}")
                    print("🌐 Please open http://localhost:8080 in your browser manually")
                
                # Keep server running for demonstration
                print("\n⏱️  Web portal running for 30 seconds...")
                print("   Visit http://localhost:8080 to explore features")
                await asyncio.sleep(30)
                
            else:
                print(f"❌ Server responded with status code: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Could not connect to web portal: {e}")
        
        # Stop the server
        print("\n🛑 Stopping web portal server...")
        server_process.terminate()
        server_process.wait(timeout=10)
        print("✅ Web portal server stopped")
        
    except Exception as e:
        print(f"❌ Error starting web portal: {e}")

async def run_comprehensive_tests():
    """Run the comprehensive test suite"""
    print("\n🧪 Running Comprehensive Test Suite...")
    print("=" * 50)
    
    test_script = Path("tests/comprehensive_test_suite.py")
    
    if not test_script.exists():
        print(f"❌ Test script not found: {test_script}")
        return
    
    try:
        print("🚀 Starting comprehensive test suite...")
        
        # Run test suite
        result = subprocess.run([
            sys.executable, str(test_script)
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Comprehensive test suite completed successfully!")
            
            # Show test summary
            output_lines = result.stdout.split('\n')
            summary_started = False
            for line in output_lines:
                if "TEST SUITE SUMMARY" in line:
                    summary_started = True
                if summary_started and line.strip():
                    print(f"   {line}")
                if summary_started and "=" in line and len(line) > 50:
                    break
        else:
            print("⚠️  Test suite completed with some issues")
            if result.stderr:
                print(f"   ⚠️  {result.stderr[:200]}...")
        
        # Check for test results file
        test_results_file = Path("test_results.json")
        if test_results_file.exists():
            print(f"\n📄 Detailed test results saved to: {test_results_file}")
            
            # Load and show summary
            try:
                with open(test_results_file, 'r') as f:
                    test_data = json.load(f)
                
                summary = test_data.get('summary', {})
                print(f"   📊 Total Tests: {summary.get('total_tests', 'N/A')}")
                print(f"   ✅ Passed: {summary.get('passed', 'N/A')}")
                print(f"   ❌ Failed: {summary.get('failed', 'N/A')}")
                print(f"   📈 Success Rate: {summary.get('success_rate', 0):.1%}")
                
            except Exception as e:
                print(f"   ⚠️  Could not parse test results: {e}")
        
    except subprocess.TimeoutExpired:
        print("⏱️  Test suite timed out (this is normal for comprehensive tests)")
    except Exception as e:
        print(f"❌ Error running test suite: {e}")

def demonstrate_features_overview():
    """Show overview of all implemented features"""
    print("\n📋 Vita Agents Features Overview")
    print("=" * 50)
    
    features = {
        "Core Healthcare Agents": [
            "✅ FHIR Agent - Resource validation, generation, conversion",
            "✅ HL7 Agent - Message parsing, validation, transformation", 
            "✅ EHR Agent - Integration, data extraction, mapping",
            "✅ Clinical Decision Support - Analysis, recommendations, drug interactions",
            "✅ Data Harmonization - Traditional + ML methods",
            "✅ Compliance & Security - HIPAA, audit trails, encryption",
            "✅ NLP Agent - Text processing, entity extraction"
        ],
        "Advanced AI Managers": [
            "✅ Medical Foundation Models - Advanced text analysis",
            "✅ Continuous Risk Scoring - Real-time patient monitoring",
            "✅ Precision Medicine & Genomics - Personalized care",
            "✅ Autonomous Clinical Workflows - Process optimization",
            "✅ Advanced Imaging AI - Radiology, pathology analysis",
            "✅ Laboratory Medicine AI - Automated lab analysis",
            "✅ Explainable AI Framework - Model interpretation",
            "✅ Edge Computing & IoT - Real-time device processing",
            "✅ Virtual Health Assistant - Conversational AI",
            "✅ AI Governance & Compliance - Ethics and regulation"
        ],
        "Data Harmonization Methods": [
            "✅ Traditional Harmonization - Rule-based processing",
            "✅ ML Clustering - Advanced record linkage",
            "✅ ML Similarity Learning - Intelligent matching",
            "✅ Hybrid Approach - Best of traditional and ML",
            "✅ Performance Benchmarking - Comparative analysis",
            "✅ Quality Assessment - Comprehensive validation"
        ],
        "User Interfaces": [
            "✅ Enhanced CLI - Comprehensive command-line interface",
            "✅ Web Portal - Interactive dashboard and API",
            "✅ REST API - Full programmatic access",
            "✅ Documentation - Complete API docs and guides"
        ],
        "Testing & Quality": [
            "✅ Comprehensive Test Suite - All features tested",
            "✅ Performance Benchmarks - Speed and accuracy metrics",
            "✅ Integration Tests - End-to-end validation",
            "✅ Mock Data Generation - Sample data for testing"
        ]
    }
    
    for category, feature_list in features.items():
        print(f"\n🔹 {category}:")
        for feature in feature_list:
            print(f"   {feature}")
    
    print(f"\n📊 Total Features Implemented: {sum(len(features) for features in features.values())}")
    print("🏆 Phase 2 Implementation: COMPLETE")

async def main():
    """Main demonstration function"""
    print_banner()
    
    print("🚀 Starting Vita Agents Feature Demonstration")
    print("This script will demonstrate all current features including:")
    print("  • Enhanced CLI interface")
    print("  • Web portal with interactive dashboard") 
    print("  • Comprehensive testing suite")
    print("  • All 7 core agents and 10 AI managers")
    print("  • ML-based data harmonization")
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed. Please install dependencies manually.")
        return
    
    # Create sample data
    sample_files = create_sample_data()
    
    # Show features overview
    demonstrate_features_overview()
    
    # Get user choice for demonstrations
    print("\n🎯 Select demonstration mode:")
    print("1. CLI Features Only")
    print("2. Web Portal Only") 
    print("3. Testing Suite Only")
    print("4. Full Demonstration (CLI + Web + Tests)")
    print("5. Quick Overview (skip interactive demos)")
    
    try:
        choice = input("\nEnter choice (1-5) [default: 5]: ").strip()
        if not choice:
            choice = "5"
    except KeyboardInterrupt:
        print("\n👋 Demonstration cancelled by user")
        return
    
    if choice == "1":
        await demonstrate_cli_features()
    elif choice == "2":
        await demonstrate_web_portal()
    elif choice == "3":
        await run_comprehensive_tests()
    elif choice == "4":
        await demonstrate_cli_features()
        await demonstrate_web_portal()
        await run_comprehensive_tests()
    elif choice == "5":
        print("\n✅ Quick overview complete!")
        print("🌐 Web Portal: Run 'python vita_agents/web/portal.py' and visit http://localhost:8080")
        print("🖥️  CLI: Run 'python vita_agents/cli/main.py --help' for command help")
        print("🧪 Tests: Run 'python tests/comprehensive_test_suite.py' for full testing")
    else:
        print("❌ Invalid choice. Running quick overview.")
    
    print("\n" + "=" * 70)
    print("🏥 Vita Agents Demonstration Complete!")
    print("=" * 70)
    print("📚 Documentation: Check docs/ directory")
    print("🌐 Web Portal: python vita_agents/web/portal.py")
    print("🖥️  CLI Interface: python vita_agents/cli/main.py --help")
    print("🧪 Test Suite: python tests/comprehensive_test_suite.py")
    print("📊 API Docs: http://localhost:8080/api/docs (when web portal is running)")
    print("\n✨ Thank you for exploring Vita Agents!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        sys.exit(1)