# Multi-stage build for production efficiency
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r vita && useradd -r -g vita vita

# Copy Python packages from builder
COPY --from=builder /root/.local /home/vita/.local

# Set up application
WORKDIR /app
COPY --chown=vita:vita . .

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/data && \
    chown -R vita:vita /app

# Set environment variables
ENV PATH=/home/vita/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER vita

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Expose port
EXPOSE 8080

# Start command
CMD ["python", "enhanced_web_portal.py"]