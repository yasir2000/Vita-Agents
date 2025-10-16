# 🏥 Vita Agents - Healthcare AI Platform

## Quick Start

### 🚀 Single Command Startup

Use the unified startup script for the best experience:

```bash
# Start on default port 8080
python start_portal.py

# Start on specific port
python start_portal.py --port 8081

# Development mode with hot reload
python start_portal.py --dev

# Clean start (kills existing processes)
python start_portal.py --clean

# Auto-find available port
python start_portal.py --find-port
```

### 🌐 Access the Portal

Once started, access the healthcare portal at:
- **Main Portal**: http://localhost:8080
- **API Documentation**: http://localhost:8080/api/docs

### 📊 Available Features

| Feature | URL | Description |
|---------|-----|-------------|
| 🏠 Dashboard | `/` | Main healthcare dashboard |
| 🤖 LLM AI | `/llm` | AI language models integration |
| 👥 Patients | `/patients` | Patient management system |
| 🩺 Clinical | `/clinical` | Clinical decision support |
| 📈 Analytics | `/analytics` | Healthcare analytics |
| ⚙️ Agents | `/agents` | AI agent management |
| 🧠 AI Models | `/ai-models` | AI model configuration |
| 🔄 Workflows | `/workflows` | Clinical workflows |
| 🚨 Alerts | `/alerts` | Real-time alerts |
| 🔗 Integration | `/integration` | System integrations |
| 🛡️ Compliance | `/compliance` | HIPAA compliance |
| 📊 Monitoring | `/monitoring` | System monitoring |
| 🧪 Testing | `/testing` | Quality assurance |

### 🛠️ Alternative Startup Methods

If you prefer direct methods:

```bash
# Direct uvicorn (port 8080)
python -m uvicorn enhanced_web_portal:app --host 0.0.0.0 --port 8080

# Direct Python execution (port 8080)
python enhanced_web_portal.py
```

### 🔧 Troubleshooting

**Port Already in Use?**
```bash
# Kill existing processes and start clean
python start_portal.py --clean
```

**Want Different Port?**
```bash
# Use specific port
python start_portal.py --port 8081
```

**Port Conflicts?**
```bash
# Auto-find available port
python start_portal.py --find-port
```

### 🎯 Consistent Port Strategy

- **Default Port**: 8080 (standardized across all scripts)
- **Auto-detection**: Finds available ports automatically
- **Conflict Resolution**: Handles port conflicts gracefully
- **Clean Startup**: Kills existing processes when needed

### 🚦 Server Status

The startup script will show:
- ✅ Server health check
- 🌐 All available URLs
- 📊 Feature overview
- 🛑 Graceful shutdown instructions

---

**🎉 Enjoy your unified Vita Agents Healthcare AI Platform!**