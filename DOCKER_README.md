# Vita Agents Docker Deployment Guide

This guide provides a comprehensive semi-production ready Docker setup for the Vita Agents Healthcare Platform with real infrastructure components.

## 🏗️ Architecture Overview

The Docker Compose setup includes the following components:

### Core Services
- **Vita App**: Main FastAPI application (Port 8080)
- **Vita Worker**: Background task processor
- **Nginx**: Reverse proxy and load balancer (Ports 80, 443)

### Data Layer
- **PostgreSQL 15**: Primary database with multiple databases (Port 5432)
- **Redis 7**: Cache and session store with authentication (Port 6379)
- **Elasticsearch 8**: Search and analytics engine (Ports 9200, 9300)
- **MinIO**: S3-compatible object storage (Ports 9000, 9001)

### Message Queue
- **RabbitMQ 3.12**: Message broker with management UI (Ports 5672, 15672)

### Monitoring & Observability
- **Prometheus**: Metrics collection (Port 9090)
- **Grafana**: Monitoring dashboards (Port 3000)

### Development Tools
- **MailHog**: SMTP testing server (Ports 1025, 8025)

## 🚀 Quick Start

### Prerequisites

- Docker 24.0+
- Docker Compose 2.0+
- 8GB+ RAM recommended
- 20GB+ disk space

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Vita-Agents
```

### 2. Start the Platform

**Linux/macOS:**
```bash
chmod +x vita-docker.sh
./vita-docker.sh start
```

**Windows:**
```batch
vita-docker.bat start
```

### 3. Wait for Services

The startup process takes approximately 2-3 minutes. Monitor progress:

```bash
./vita-docker.sh status
```

### 4. Seed Sample Data

```bash
./vita-docker.sh seed
```

## 📋 Service Access

After successful startup, access services at:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Main Application** | http://localhost:80 | admin@vita-agents.com / admin123 |
| **HTTPS Application** | https://localhost:443 | (Self-signed cert) |
| **Grafana Dashboard** | http://localhost:3000 | admin / vita_grafana_admin_2024 |
| **Prometheus** | http://localhost:9090 | No auth |
| **RabbitMQ Management** | http://localhost:15672 | vita_admin / vita_rabbit_pass_2024 |
| **MinIO Console** | http://localhost:9001 | vita_admin / vita_minio_pass_2024 |
| **MailHog Web UI** | http://localhost:8025 | No auth |
| **Elasticsearch** | http://localhost:9200 | No auth |

## 🔧 Management Commands

### Linux/macOS (vita-docker.sh)

```bash
# Start all services
./vita-docker.sh start

# Start only infrastructure
./vita-docker.sh start-infra

# Start only application
./vita-docker.sh start-app

# Stop all services
./vita-docker.sh stop

# Restart all services
./vita-docker.sh restart

# View service status
./vita-docker.sh status

# View logs (all or specific service)
./vita-docker.sh logs
./vita-docker.sh logs vita-app

# Seed sample data
./vita-docker.sh seed

# Create data backup
./vita-docker.sh backup

# Health check
./vita-docker.sh health

# Clean up (removes all data)
./vita-docker.sh cleanup
```

### Windows (vita-docker.bat)

```batch
REM Start all services
vita-docker.bat start

REM Stop all services
vita-docker.bat stop

REM Restart services
vita-docker.bat restart

REM View status
vita-docker.bat status

REM View logs
vita-docker.bat logs
vita-docker.bat logs vita-app

REM Seed data
vita-docker.bat seed

REM Clean up
vita-docker.bat cleanup
```

## 🗃️ Database Schema

The system automatically creates three PostgreSQL databases:

1. **vita_agents** - Main application data
   - Users, roles, authentication
   - Patients and medical records
   - Appointments and schedules
   - System configuration

2. **vita_analytics** - Analytics and reporting data
   - Aggregated metrics
   - Report definitions
   - User behavior tracking

3. **vita_audit** - Audit and compliance data
   - User actions log
   - Data access tracking
   - Security events

## 📊 Sample Data

The seeder creates realistic healthcare data:

- **20 Users** with various roles (admin, doctor, nurse, technician)
- **100 Patients** with complete demographics
- **500 Medical Records** with diagnoses, treatments, lab results
- **200 Appointments** across different types and statuses
- **Sample Documents** in MinIO object storage
- **Search Indices** in Elasticsearch
- **Cache Data** in Redis

## 🔐 Security Features

### Development Security
- Self-signed SSL certificates for HTTPS
- Password-protected Redis
- RabbitMQ authentication
- JWT-based API authentication
- Rate limiting via Nginx

### Production Recommendations
⚠️ **IMPORTANT**: Change all default passwords before production use!

Required changes for production:
1. Update all passwords in `.env.production`
2. Use proper SSL certificates
3. Configure firewall rules
4. Enable PostgreSQL SSL
5. Set up proper backup strategies
6. Configure log aggregation
7. Implement proper secrets management

## 📈 Monitoring & Metrics

### Prometheus Metrics
The platform exposes metrics for:
- Application performance
- Database connections
- Redis operations
- Queue status
- HTTP request metrics
- Custom business metrics

### Grafana Dashboards
Pre-configured dashboards for:
- System Overview
- Database Performance
- Application Metrics
- Infrastructure Health
- User Activity

Access Grafana at http://localhost:3000 with credentials:
- Username: `admin`
- Password: `vita_grafana_admin_2024`

## 🔄 Background Tasks

The system includes a dedicated worker service for:

- **FHIR Validation**: Healthcare data validation
- **HL7 Processing**: Message processing
- **Data Analysis**: Batch analytics processing
- **Notifications**: Email and system notifications
- **Report Generation**: Automated report creation
- **ML Inference**: AI/ML model execution

## 💾 Data Persistence

All data is persisted in Docker volumes:

```bash
# View volumes
docker volume ls | grep vita-agents

# Backup volumes
docker run --rm -v vita-agents_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v vita-agents_postgres_data:/data -v $(pwd):/backup ubuntu tar xzf /backup/postgres_backup.tar.gz -C /data
```

## 🔍 Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check Docker daemon
docker info

# Check logs
./vita-docker.sh logs

# Verify ports are available
netstat -tulpn | grep -E "(80|443|5432|6379|9200|5672)"
```

**Out of memory:**
```bash
# Check memory usage
docker stats

# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Memory > 8GB+
```

**Port conflicts:**
```bash
# Find processes using ports
lsof -i :80
lsof -i :5432

# Stop conflicting services
sudo systemctl stop apache2  # Example
```

**Permission issues:**
```bash
# Fix permissions (Linux/macOS)
sudo chown -R $USER:$USER .
chmod +x vita-docker.sh
```

### Service Health Checks

Each service includes health checks:
- **Application**: HTTP health endpoint
- **PostgreSQL**: Connection test
- **Redis**: Ping command
- **Elasticsearch**: Cluster health
- **RabbitMQ**: Management API

Check overall health:
```bash
./vita-docker.sh health
```

### Logs Location

Service logs are available via:
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs vita-app
docker-compose logs postgres
docker-compose logs redis

# Follow logs in real-time
docker-compose logs -f vita-app
```

## 🧪 Development Workflow

### Making Changes

1. **Code Changes**: Modify application code
2. **Rebuild**: `docker-compose build vita-app`
3. **Restart**: `./vita-docker.sh restart`

### Adding New Services

1. Add service to `docker-compose.yml`
2. Update Nginx configuration if needed
3. Add health checks and monitoring
4. Update management scripts

### Database Migrations

```bash
# Run inside application container
docker-compose exec vita-app alembic upgrade head

# Create new migration
docker-compose exec vita-app alembic revision --autogenerate -m "description"
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation)
- [Elasticsearch Guide](https://www.elastic.co/guide/)
- [RabbitMQ Documentation](https://www.rabbitmq.com/documentation.html)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review service logs
3. Verify system requirements
4. Check Docker and Docker Compose versions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.