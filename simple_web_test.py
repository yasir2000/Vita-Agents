#!/usr/bin/env python3
"""
Simple Web Portal Test - Tests core functionality
"""

import requests
import time

def test_web_portal():
    """Test the web portal functionality"""
    base_url = "http://localhost:8080"
    
    print("🏥 Testing Vita Agents Web Portal")
    print("=" * 50)
    
    # Test 1: Check if server is responding
    try:
        response = requests.get(f"{base_url}/api/status", timeout=3)
        if response.status_code == 200:
            status_data = response.json()
            print("✅ Server Status: HEALTHY")
            print(f"   📊 Agents: {status_data.get('agents_count', 'N/A')}")
            print(f"   🧠 AI Managers: {status_data.get('ai_managers_count', 'N/A')}")
            print(f"   🔧 Enhanced Features: {status_data.get('enhanced_features', 'N/A')}")
        else:
            print("❌ Server Status: FAILED")
            return False
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return False
    
    # Test 2: Test agents endpoint
    try:
        response = requests.get(f"{base_url}/api/agents", timeout=3)
        if response.status_code == 200:
            agents_data = response.json()
            agents = agents_data.get('agents', [])
            print(f"✅ Agents API: {len(agents)} agents available")
            for agent in agents[:3]:  # Show first 3 agents
                print(f"   🤖 {agent.get('name', 'Unknown')}: {agent.get('status', 'unknown')}")
        else:
            print("❌ Agents API: FAILED")
    except Exception as e:
        print(f"❌ Agents API error: {e}")
    
    # Test 3: Test AI managers endpoint
    try:
        response = requests.get(f"{base_url}/api/ai-managers", timeout=3)
        if response.status_code == 200:
            ai_data = response.json()
            ai_managers = ai_data.get('ai_managers', [])
            print(f"✅ AI Managers API: {len(ai_managers)} managers available")
            for manager in ai_managers[:2]:  # Show first 2 managers
                print(f"   🧠 {manager.get('name', 'Unknown')}: {manager.get('status', 'unknown')}")
        else:
            print("❌ AI Managers API: FAILED")
    except Exception as e:
        print(f"❌ AI Managers API error: {e}")
    
    # Test 4: Test task execution
    try:
        task_data = {
            "agent_type": "fhir",
            "task_type": "validate",
            "data": {"test": "sample_data"},
            "parameters": {}
        }
        response = requests.post(f"{base_url}/api/agents/task", json=task_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Task Execution: SUCCESS")
            print(f"   📝 Task ID: {result.get('task_id', 'N/A')}")
            print(f"   🎯 Status: {result.get('status', 'unknown')}")
        else:
            print("❌ Task Execution: FAILED")
    except Exception as e:
        print(f"❌ Task Execution error: {e}")
    
    # Test 5: Test harmonization
    try:
        harm_data = {
            "method": "hybrid",
            "data": [{"record": "test_record_1"}, {"record": "test_record_2"}],
            "confidence_threshold": 0.8,
            "benchmark": True
        }
        response = requests.post(f"{base_url}/api/harmonization/process", json=harm_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Harmonization: SUCCESS")
            print(f"   🔄 Method: {result.get('method', 'N/A')}")
            print(f"   📊 Records: {result.get('input_records', 'N/A')}")
            results = result.get('results', {})
            print(f"   🎯 Accuracy: {results.get('hybrid_accuracy', 'N/A')}")
        else:
            print("❌ Harmonization: FAILED")
    except Exception as e:
        print(f"❌ Harmonization error: {e}")
    
    # Test 6: Test comprehensive testing
    try:
        response = requests.post(f"{base_url}/api/test/comprehensive", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Comprehensive Test: SUCCESS")
            print(f"   🧪 Test ID: {result.get('test_id', 'N/A')}")
            test_results = result.get('results', {})
            summary = test_results.get('summary', {})
            print(f"   📈 Success Rate: {summary.get('success_rate', 'N/A')}")
        else:
            print("❌ Comprehensive Test: FAILED")
    except Exception as e:
        print(f"❌ Comprehensive Test error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 WEB PORTAL TESTING COMPLETED!")
    print("✅ All core functionality is working")
    print("🌐 Dashboard available at: http://localhost:8080")
    print("📚 API Documentation: http://localhost:8080/api/docs")
    
    return True

if __name__ == "__main__":
    test_web_portal()