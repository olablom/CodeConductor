# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data directories
RUN mkdir -p data/generated data/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash codeconductor \
    && chown -R codeconductor:codeconductor /app

# Switch to non-root user
USER codeconductor

# Expose port (if needed for future web interface)
EXPOSE 8000

# Set default command
CMD ["python", "pipeline.py", "--help"] 