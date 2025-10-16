#!/bin/bash
# Vita Agents - Quick Status Check Script

echo "🏥 VITA AGENTS - QUICK STATUS CHECK"
echo "=================================="

# 1. Check Docker Infrastructure
echo "🐳 Docker Services:"
docker ps --format "{{.Names}}: {{.Status}}" | grep vita- || echo "❌ No Docker services running"

# 2. Check Web Applications
echo ""
echo "🌐 Web Applications:"
for port in 8080 8081 8082 8083 5000; do
    if curl -s -o /dev/null -w "" "http://localhost:$port" 2>/dev/null; then
        echo "✅ Service running on port $port"
    else
        echo "❌ No service on port $port"
    fi
done

# 3. Check Python Processes
echo ""
echo "🐍 Python Processes:"
if tasklist | grep python.exe > /dev/null 2>&1; then
    echo "✅ Python processes running"
else
    echo "❌ No Python processes found"
fi

echo ""
echo "🚀 Quick Commands:"
echo "• python check_app_status.py - Full status check"
echo "• python enhanced_cli.py dashboard - CLI dashboard"
echo "• python vita_agents_launcher.py start - Start app"
echo "• docker ps - Check Docker services"