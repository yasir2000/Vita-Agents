@echo off
REM Vita Agents Docker Management Script for Windows

setlocal enabledelayedexpansion

set PROJECT_NAME=vita-agents
set COMPOSE_FILE=docker-compose.yml

echo.
echo ╔══════════════════════════════════════════════════════════════════╗
echo ║                     Vita Agents Healthcare Platform             ║
echo ║                        Docker Management                        ║
echo ╚══════════════════════════════════════════════════════════════════╝
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not in PATH!
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose is not installed or not in PATH!
    pause
    exit /b 1
)

REM Parse command line argument
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=help

if "%COMMAND%"=="start" goto start
if "%COMMAND%"=="stop" goto stop
if "%COMMAND%"=="restart" goto restart
if "%COMMAND%"=="status" goto status
if "%COMMAND%"=="logs" goto logs
if "%COMMAND%"=="seed" goto seed
if "%COMMAND%"=="cleanup" goto cleanup
if "%COMMAND%"=="help" goto help
goto help

:start
echo [INFO] Starting Vita Agents platform...
docker-compose --project-name %PROJECT_NAME% up -d
timeout /t 30 /nobreak >nul
goto status

:stop
echo [INFO] Stopping all services...
docker-compose --project-name %PROJECT_NAME% down
goto end

:restart
echo [INFO] Restarting services...
docker-compose --project-name %PROJECT_NAME% down
timeout /t 5 /nobreak >nul
docker-compose --project-name %PROJECT_NAME% up -d
timeout /t 30 /nobreak >nul
goto status

:status
echo [INFO] Service Status:
docker-compose --project-name %PROJECT_NAME% ps
echo.
echo [INFO] Service URLs:
echo 🌐 Main Application: http://localhost:80 (HTTP) / https://localhost:443 (HTTPS)
echo 📊 Grafana Dashboard: http://localhost:3000 (admin/vita_grafana_admin_2024)
echo 📈 Prometheus: http://localhost:9090
echo 🐰 RabbitMQ Management: http://localhost:15672 (vita_admin/vita_rabbit_pass_2024)
echo 📦 MinIO Console: http://localhost:9001 (vita_admin/vita_minio_pass_2024)
echo 📧 MailHog: http://localhost:8025
echo 🔍 Elasticsearch: http://localhost:9200
goto end

:logs
echo [INFO] Viewing logs...
if "%2"=="" (
    docker-compose --project-name %PROJECT_NAME% logs -f
) else (
    docker-compose --project-name %PROJECT_NAME% logs -f %2
)
goto end

:seed
echo [INFO] Seeding initial data...
docker-compose --project-name %PROJECT_NAME% --profile seeder run --rm vita-seeder
goto end

:cleanup
echo [WARN] This will remove all containers, networks, and volumes!
set /p CONFIRM=Are you sure? (y/N): 
if /i "%CONFIRM%"=="y" (
    echo [INFO] Cleaning up...
    docker-compose --project-name %PROJECT_NAME% down -v --remove-orphans
    docker system prune -f
    echo [INFO] Cleanup completed
)
goto end

:help
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   start         Start all services
echo   stop          Stop all services
echo   restart       Restart all services
echo   status        Show service status and URLs
echo   logs [SERVICE] View logs (all services or specific service)
echo   seed          Seed initial data
echo   cleanup       Remove all containers and volumes
echo   help          Show this help message
echo.
echo Example:
echo   %0 start       - Start the platform
echo   %0 logs vita-app - View application logs
echo   %0 seed        - Add sample data
goto end

:end
echo.
pause