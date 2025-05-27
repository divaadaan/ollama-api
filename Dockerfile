# Multi-stage Dockerfile supporting both full and lightweight builds

# Stage 1: Base image with common dependencies
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements*.txt ./

# Stage 2: Full build with telemetry dependencies
FROM base AS full-build

# Install all dependencies including OpenTelemetry
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8001

# Default environment for full build
ENV TELEMETRY_ENABLED=true
ENV TELEMETRY_METRICS_PORT=8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Lightweight build without telemetry dependencies
FROM base AS lite-build

# Install only core dependencies
RUN pip install --no-cache-dir -r requirements-lite.txt

# Copy application code
COPY . .

# Expose only main port
EXPOSE 8000

# Environment for lite build
ENV TELEMETRY_ENABLED=false

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Development build with additional tools
FROM full-build AS dev-build

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    mypy

# Enable hot reload and debug mode
ENV DEBUG=true
ENV RELOAD=true
ENV TELEMETRY_CONSOLE_EXPORT=true

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Default target is full build
FROM full-build AS final