#!/bin/bash
# Vita Agents Docker Management Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.production"
PROJECT_NAME="vita-agents"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                     Vita Agents Healthcare Platform             ║"
    echo "║                        Docker Management                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed!"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed!"
        exit 1
    fi
    
    print_status "Dependencies check passed ✓"
}

generate_ssl_certs() {
    print_status "Generating SSL certificates for development..."
    
    mkdir -p docker/nginx/ssl
    
    if [ ! -f "docker/nginx/ssl/cert.pem" ] || [ ! -f "docker/nginx/ssl/key.pem" ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout docker/nginx/ssl/key.pem \
            -out docker/nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Vita-Agents/OU=IT/CN=vita-agents.local"
        
        print_status "SSL certificates generated ✓"
    else
        print_status "SSL certificates already exist ✓"
    fi
}

start_infrastructure() {
    print_status "Starting infrastructure services..."
    
    docker-compose --project-name ${PROJECT_NAME} up -d \
        postgres redis elasticsearch rabbitmq minio prometheus grafana
    
    print_status "Waiting for services to be ready..."
    sleep 30
}

start_application() {
    print_status "Starting application services..."
    
    docker-compose --project-name ${PROJECT_NAME} up -d \
        vita-app vita-worker nginx
}

seed_data() {
    print_status "Seeding initial data..."
    
    docker-compose --project-name ${PROJECT_NAME} \
        --profile seeder run --rm vita-seeder
}

show_status() {
    print_status "Service Status:"
    docker-compose --project-name ${PROJECT_NAME} ps
    
    echo ""
    print_status "Service Health:"
    docker-compose --project-name ${PROJECT_NAME} exec vita-app curl -s http://localhost:8080/api/health || echo "App not ready"
    
    echo ""
    print_status "Service URLs:"
    echo "🌐 Main Application: http://localhost:80 (HTTP) / https://localhost:443 (HTTPS)"
    echo "📊 Grafana Dashboard: http://localhost:3000 (admin/vita_grafana_admin_2024)"
    echo "📈 Prometheus: http://localhost:9090"
    echo "🐰 RabbitMQ Management: http://localhost:15672 (vita_admin/vita_rabbit_pass_2024)"
    echo "📦 MinIO Console: http://localhost:9001 (vita_admin/vita_minio_pass_2024)"
    echo "📧 MailHog: http://localhost:8025"
    echo "🔍 Elasticsearch: http://localhost:9200"
}

stop_services() {
    print_status "Stopping all services..."
    docker-compose --project-name ${PROJECT_NAME} down
}

cleanup() {
    print_warning "This will remove all containers, networks, and volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up..."
        docker-compose --project-name ${PROJECT_NAME} down -v --remove-orphans
        docker system prune -f
        print_status "Cleanup completed ✓"
    fi
}

view_logs() {
    if [ -z "$2" ]; then
        docker-compose --project-name ${PROJECT_NAME} logs -f
    else
        docker-compose --project-name ${PROJECT_NAME} logs -f "$2"
    fi
}

backup_data() {
    print_status "Creating backup..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup PostgreSQL
    docker-compose --project-name ${PROJECT_NAME} exec -T postgres \
        pg_dump -U vita_user vita_agents > "$BACKUP_DIR/postgres_backup.sql"
    
    # Backup Redis
    docker-compose --project-name ${PROJECT_NAME} exec -T redis \
        redis-cli --rdb /data/dump.rdb
    docker cp $(docker-compose --project-name ${PROJECT_NAME} ps -q redis):/data/dump.rdb \
        "$BACKUP_DIR/redis_backup.rdb"
    
    print_status "Backup created in $BACKUP_DIR ✓"
}

show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start         Start all services"
    echo "  start-infra   Start only infrastructure services"
    echo "  start-app     Start only application services"
    echo "  stop          Stop all services"
    echo "  restart       Restart all services"
    echo "  status        Show service status and URLs"
    echo "  logs [SERVICE] View logs (all services or specific service)"
    echo "  seed          Seed initial data"
    echo "  backup        Create data backup"
    echo "  cleanup       Remove all containers and volumes"
    echo "  health        Show detailed health information"
    echo "  help          Show this help message"
}

# Main script logic
case "${1:-help}" in
    start)
        print_header
        check_dependencies
        generate_ssl_certs
        start_infrastructure
        start_application
        sleep 10
        show_status
        echo ""
        print_status "🚀 Vita Agents is starting up!"
        print_status "Run '$0 seed' to populate with sample data"
        ;;
    start-infra)
        print_header
        check_dependencies
        start_infrastructure
        ;;
    start-app)
        print_header
        start_application
        ;;
    stop)
        print_header
        stop_services
        ;;
    restart)
        print_header
        stop_services
        sleep 5
        check_dependencies
        generate_ssl_certs
        start_infrastructure
        start_application
        ;;
    status)
        print_header
        show_status
        ;;
    logs)
        view_logs "$@"
        ;;
    seed)
        print_header
        seed_data
        ;;
    backup)
        print_header
        backup_data
        ;;
    cleanup)
        print_header
        cleanup
        ;;
    health)
        print_header
        show_status
        echo ""
        print_status "Detailed Health Check:"
        docker-compose --project-name ${PROJECT_NAME} exec vita-app python -c "
import asyncio
import aiohttp
import asyncpg
import redis.asyncio as redis

async def health_check():
    print('🔍 Testing application health...')
    
    # Test web app
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8080/api/health') as resp:
                if resp.status == 200:
                    print('✅ Web Application: Healthy')
                else:
                    print(f'❌ Web Application: Status {resp.status}')
    except Exception as e:
        print(f'❌ Web Application: {e}')
    
    # Test database
    try:
        conn = await asyncpg.connect('postgresql://vita_user:vita_secure_pass_2024@postgres:5432/vita_agents')
        await conn.execute('SELECT 1')
        await conn.close()
        print('✅ PostgreSQL: Connected')
    except Exception as e:
        print(f'❌ PostgreSQL: {e}')
    
    # Test Redis
    try:
        redis_client = redis.from_url('redis://:vita_redis_pass_2024@redis:6379/0')
        await redis_client.ping()
        await redis_client.close()
        print('✅ Redis: Connected')
    except Exception as e:
        print(f'❌ Redis: {e}')

asyncio.run(health_check())
"
        ;;
    help|*)
        print_header
        show_help
        ;;
esac