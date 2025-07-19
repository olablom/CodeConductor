# ─────────────────────────────────────────────────────────────
# CodeConductor v2.0 - Multi-Agent AI Code Generation System
# ─────────────────────────────────────────────────────────────

# Stage 1: Install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────
# Stage 2: Production image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/generated data/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV JWT_SECRET=supersecretkey
ENV CONFIG_PATH=/app/config/base.yaml

# Expose ports
EXPOSE 8000   # FastAPI
EXPOSE 8501   # Streamlit GUI

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start both API and GUI
CMD ["bash", "-c", "\
    echo '🚀 Starting CodeConductor v2.0...' && \
    echo '📊 API: http://localhost:8000' && \
    echo '🎨 GUI: http://localhost:8501' && \
    uvicorn generated_api:app --host 0.0.0.0 --port 8000 & \
    streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0"] 