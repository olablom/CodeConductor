#!/bin/bash

echo "🚀 Setting up CodeConductor Monitoring System..."

# Check if health API is running
echo "📊 Checking if Health API is running..."
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✅ Health API is running on port 8080"
else
    echo "⚠️  Health API not running. Starting it now..."
    source venv/Scripts/activate
    python health_api.py &
    sleep 3
fi

# Start Docker containers
echo "🐳 Starting Docker containers..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service status..."

if curl -s http://localhost:9090/api/v1/status > /dev/null; then
    echo "✅ Prometheus is running on port 9090"
else
    echo "❌ Prometheus failed to start"
fi

if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana is running on port 3000"
else
    echo "❌ Grafana failed to start"
fi

if curl -s http://localhost:9093/api/v1/status > /dev/null; then
    echo "✅ Alertmanager is running on port 9093"
else
    echo "❌ Alertmanager failed to start"
fi

echo ""
echo "🎉 Monitoring setup complete!"
echo ""
echo "📊 Access URLs:"
echo "  - Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Alertmanager: http://localhost:9093"
echo "  - Health API: http://localhost:8080/health"
echo ""
echo "🔧 Next steps:"
echo "  1. Open Grafana at http://localhost:3000"
echo "  2. Login with admin/admin"
echo "  3. Add Prometheus data source: http://prometheus:9090"
echo "  4. Import dashboard or create your own"
echo ""
echo "📈 Test the system:"
echo "  curl http://localhost:8080/health"
echo "  curl http://localhost:8080/metrics" 