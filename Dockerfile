# Multi-stage build: First stage builds the wheel package
FROM python:3.11-slim AS builder

# Set working directory for build stage
WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy project configuration and source code
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Create virtual environment and install build dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install build

# Build wheel package for distribution
RUN python -m build --wheel .

# Build wheels for all dependencies to avoid compilation in runtime
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels /app/dist/*.whl

# Second stage: Runtime image with minimal dependencies
FROM python:3.11-slim AS runtime

# Set working directory for runtime
WORKDIR /app

# Install minimal runtime dependencies (curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built wheels from builder stage (main package + dependencies)
COPY --from=builder /app/wheels /app/wheels
# Install the main package and all its dependencies from pre-built wheels
RUN pip install --no-cache-dir --no-index --find-links=/app/wheels relife-forecasting && \
    rm -rf /app/wheels

# Configure API server environment variables
ENV API_HOST=0.0.0.0
ENV API_PORT=9090

# Expose the API port
EXPOSE 9090

# Start the service API
CMD ["run-service"]