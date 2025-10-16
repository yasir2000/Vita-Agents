# Multi-stage Dockerfile for Vita Agents Healthcare Platform
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN groupadd -r vita && useradd -r -g vita vita

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy application code
COPY --chown=vita:vita . /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/data && \
    chown -R vita:vita /app

# Switch to non-root user
USER vita

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8083/api/health || exit 1

# Expose port
EXPOSE 8083

# Command to run the application
CMD ["python", "enhanced_web_portal.py"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov black flake8 mypy

# Copy application code
COPY --chown=vita:vita . /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/data && \
    chown -R vita:vita /app

# Switch to non-root user
USER vita

# Expose port
EXPOSE 8083

# Command for development (with auto-reload)
CMD ["uvicorn", "enhanced_web_portal:app", "--host", "0.0.0.0", "--port", "8083", "--reload"]