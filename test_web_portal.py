#!/usr/bin/env python3
"""
Web Portal API Test Suite
Tests all API endpoints of the Vita Agents web portal
"""

import requests
import json
import time
from datetime import datetime

# Base URL for the web portal
BASE_URL = "http://localhost:8080"

def test_api_endpoint(endpoint, method="GET", data=None, description=""):
    """Test a single API endpoint"""
    try:
        url = f"{BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        
        print(f"✅ {method} {endpoint} - {response.status_code} - {description}")
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict) and len(result) > 0:
                # Show a sample of the response
                key = list(result.keys())[0]
                print(f"   📄 Sample: {key}: {str(result[key])[:100]}...")
            return True
        else:
            print(f"   ❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {method} {endpoint} - ERROR - {str(e)}")
        return False

def run_comprehensive_api_tests():
    """Run comprehensive tests of all API endpoints"""
    print("🧪 Starting Vita Agents Web Portal API Tests")
    print("=" * 60)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    tests_passed = 0
    total_tests = 0
    
    # Test basic endpoints
    endpoints = [
        ("/api/status", "GET", None, "System status check"),
        ("/api/agents", "GET", None, "List all agents"),
        ("/api/ai-managers", "GET", None, "List AI managers"),
        ("/api/history", "GET", None, "Task history"),
        ("/api/metrics", "GET", None, "Performance metrics"),
        ("/api/test/results", "GET", None, "Test results")
    ]
    
    for endpoint, method, data, description in endpoints:
        if test_api_endpoint(endpoint, method, data, description):
            tests_passed += 1
        total_tests += 1
        time.sleep(0.5)  # Small delay between tests
    
    print("\n🔧 Testing POST endpoints...")
    
    # Test POST endpoints
    post_tests = [
        ("/api/initialize", "POST", {}, "Initialize system"),
        ("/api/agents/task", "POST", {
            "agent_type": "fhir",
            "task_type": "validate",
            "data": {"test": "sample"},
            "parameters": {}
        }, "Execute agent task"),
        ("/api/harmonization/process", "POST", {
            "method": "hybrid",
            "data": [{"record": "test"}],
            "confidence_threshold": 0.8,
            "benchmark": True
        }, "Process harmonization"),
        ("/api/ai/process", "POST", {
            "manager_type": "foundation_models",
            "task": "analyze",
            "input_data": {"text": "test medical text"},
            "parameters": {}
        }, "AI processing"),
        ("/api/test/comprehensive", "POST", {}, "Comprehensive test")
    ]
    
    for endpoint, method, data, description in post_tests:
        if test_api_endpoint(endpoint, method, data, description):
            tests_passed += 1
        total_tests += 1
        time.sleep(0.5)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"🏁 API Test Results")
    print(f"📊 Tests Passed: {tests_passed}/{total_tests}")
    print(f"✅ Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 All API tests passed! Web portal is fully functional.")
    else:
        print(f"⚠️  {total_tests - tests_passed} tests failed.")
    
    return tests_passed == total_tests

def test_web_pages():
    """Test web page endpoints"""
    print("\n🌐 Testing Web Pages...")
    
    pages = [
        ("/", "Dashboard"),
        ("/agents", "Core Agents"),
        ("/ai-models", "AI Models"),
        ("/harmonization", "Harmonization"),
        ("/testing", "Testing"),
        ("/monitoring", "Monitoring")
    ]
    
    pages_working = 0
    for endpoint, name in pages:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint} - {name} page loaded")
                pages_working += 1
            else:
                print(f"❌ {endpoint} - {name} page failed ({response.status_code})")
        except Exception as e:
            print(f"❌ {endpoint} - {name} page error: {str(e)}")
    
    print(f"📄 Pages Working: {pages_working}/{len(pages)}")
    return pages_working == len(pages)

if __name__ == "__main__":
    print("🏥 Vita Agents Web Portal - Comprehensive Test Suite")
    print(f"🕐 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Target URL: {BASE_URL}")
    
    # Test API endpoints
    api_success = run_comprehensive_api_tests()
    
    # Test web pages
    web_success = test_web_pages()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 FINAL TEST SUMMARY")
    print(f"🔧 API Tests: {'✅ PASSED' if api_success else '❌ FAILED'}")
    print(f"🌐 Web Tests: {'✅ PASSED' if web_success else '❌ FAILED'}")
    
    if api_success and web_success:
        print("🎉 ALL TESTS PASSED - Web portal is fully functional!")
        print("✅ CLI Testing: COMPLETED")
        print("✅ Web Portal Testing: COMPLETED")
        print("🏆 Vita Agents Enhanced Interfaces: VALIDATED")
    else:
        print("⚠️  Some tests failed - check logs above")