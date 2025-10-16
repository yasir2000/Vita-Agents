FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r vita && useradd -r -g vita vita

# Set up application
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=vita:vita . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R vita:vita /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER vita

# Start worker
CMD ["python", "task_worker.py"]