#!/bin/bash
# Vita Agents Healthcare Portal - Unix/Linux/macOS Startup Script
# ================================================================

echo
echo "🏥 Vita Agents Healthcare Portal"
echo "================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Check if start_portal.py exists
if [ ! -f "start_portal.py" ]; then
    echo "❌ start_portal.py not found"
    echo "Please run this script from the Vita-Agents directory"
    exit 1
fi

echo "✅ Starting Vita Agents Healthcare Portal..."
echo

# Start the portal with auto port detection
$PYTHON_CMD start_portal.py --find-port

echo
echo "🛑 Server stopped"