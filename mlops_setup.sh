#!/bin/bash

echo "🚀 Setting up CodeConductor MLOps Foundation..."
echo "🎯 Real-time monitoring för din RTX 5090 AI stack!"

# Create monitoring directory structure
mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources},alertmanager}

echo "📁 Creating monitoring directory structure..."

# Create Prometheus configuration
cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # CodeConductor Services
  - job_name: 'codeconductor-services'
    static_configs:
      - targets:
        - 'host.docker.internal:9000'  # Gateway
        - 'host.docker.internal:9001'  # Agent
        - 'host.docker.internal:9002'  # Orchestrator
        - 'host.docker.internal:9005'  # Auth
        - 'host.docker.internal:9006'  # Data (CPU)
        - 'host.docker.internal:8007'  # Data (GPU)
    metrics_path: '/metrics'
    scrape_interval: 5s

  # System Metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['host.docker.internal:9100']
    scrape_interval: 5s
EOF

echo "✅ Created Prometheus configuration"

# Create Grafana datasource configuration
cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

echo "✅ Created Grafana datasource configuration"

# Create Grafana dashboard provisioning
cat > monitoring/grafana/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'codeconductor'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

echo "✅ Created Grafana dashboard provisioning"

# Create monitoring docker-compose
cat > monitoring/docker-compose.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: codeconductor-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"

  grafana:
    image: grafana/grafana:latest
    container_name: codeconductor-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=codeconductor
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
    depends_on:
      - prometheus

  node-exporter:
    image: prom/node-exporter:latest
    container_name: codeconductor-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: codeconductor-monitoring
EOF

echo "✅ Created monitoring docker-compose.yml"

# Add metrics to GPU service
echo "📊 Adding metrics endpoints to GPU service..."

# Create metrics integration script
cat > add_metrics_to_gpu_service.py << 'EOF'
import os
import sys

# Read the current GPU service main.py
gpu_service_path = "services/gpu_data_service/app/main.py"

if os.path.exists(gpu_service_path):
    with open(gpu_service_path, 'r') as f:
        content = f.read()
    
    # Check if metrics already exist
    if '/metrics' not in content:
        # Find the imports section and add our imports
        lines = content.split('\n')
        new_lines = []
        imports_added = False
        
        for line in lines:
            new_lines.append(line)
            # Add metrics imports after FastAPI import
            if 'from fastapi import' in line and not imports_added:
                new_lines.append('from fastapi.responses import Response')
                new_lines.append('from prometheus_client import Counter, Histogram, Gauge, generate_latest')
                new_lines.append('import time')
                new_lines.append('')
                new_lines.append('# Prometheus metrics')
                new_lines.append('REQUEST_COUNT = Counter("codeconductor_requests_total", "Total requests", ["service", "endpoint"])')
                new_lines.append('INFERENCE_DURATION = Histogram("codeconductor_inference_duration_seconds", "Inference duration", ["service", "algorithm"])')
                new_lines.append('GPU_MEMORY = Gauge("codeconductor_gpu_memory_bytes", "GPU memory usage", ["type"])')
                new_lines.append('BANDIT_SELECTIONS = Counter("codeconductor_bandit_selections_total", "Bandit arm selections", ["arm"])')
                new_lines.append('')
                imports_added = True
        
        # Add metrics endpoint before the end
        new_lines.append('')
        new_lines.append('@app.get("/metrics")')
        new_lines.append('async def metrics():')
        new_lines.append('    try:')
        new_lines.append('        import torch')
        new_lines.append('        if torch.cuda.is_available():')
        new_lines.append('            allocated = torch.cuda.memory_allocated()')
        new_lines.append('            reserved = torch.cuda.memory_reserved()')
        new_lines.append('            GPU_MEMORY.labels(type="allocated").set(allocated)')
        new_lines.append('            GPU_MEMORY.labels(type="reserved").set(reserved)')
        new_lines.append('    except:')
        new_lines.append('        pass')
        new_lines.append('    ')
        new_lines.append('    return Response(content=generate_latest(), media_type="text/plain")')
        
        # Write back
        with open(gpu_service_path, 'w') as f:
            f.write('\n'.join(new_lines))
        
        print("✅ Added metrics endpoint to GPU service")
    else:
        print("✅ Metrics endpoint already exists in GPU service")
else:
    print("❌ GPU service not found at", gpu_service_path)
EOF

python add_metrics_to_gpu_service.py

# Add prometheus-client to GPU service requirements
echo "📦 Adding prometheus-client to GPU service requirements..."
if [ -f "services/gpu_data_service/requirements.txt" ]; then
    if ! grep -q "prometheus-client" services/gpu_data_service/requirements.txt; then
        echo "prometheus-client==0.19.0" >> services/gpu_data_service/requirements.txt
        echo "✅ Added prometheus-client to requirements"
    else
        echo "✅ prometheus-client already in requirements"
    fi
fi

echo ""
echo "🎉 MLOps Foundation Setup Complete!"
echo ""
echo "📋 Next steps:"
echo "1. Start monitoring stack: cd monitoring && docker compose up -d"
echo "2. Access Grafana: http://localhost:3000 (admin/codeconductor)"
echo "3. Access Prometheus: http://localhost:9090"
echo "4. Start your AI services to see metrics flow in"
echo ""
echo "🚀 Ready for real-time AI performance monitoring!"
echo "🎯 Your RTX 5090 metrics will be visible in beautiful dashboards!" 