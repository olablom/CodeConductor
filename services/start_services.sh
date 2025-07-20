#!/bin/bash

# CodeConductor Microservices Startup Script
# Startar hela mikrotjänster-stacken med ett kommando

set -e

echo "🚀 Starting CodeConductor Microservices Stack..."
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p data/postgres
mkdir -p data/redis
mkdir -p data/rabbitmq

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🏥 Checking service health..."
services=(
    "gateway-service:8000"
    "agent-service:8001"
    "orchestrator-service:8002"
    "data-service:8003"
    "auth-service:8005"
)

for service in "${services[@]}"; do
    service_name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    echo -n "Checking $service_name... "
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo "✅ Healthy"
    else
        echo "❌ Unhealthy"
    fi
done

echo ""
echo "🎉 CodeConductor Microservices Stack is running!"
echo ""
echo "📊 Service URLs:"
echo "  Gateway API:      http://localhost:8000/docs"
echo "  Agent Service:    http://localhost:8001/docs"
echo "  Orchestrator:     http://localhost:8002/docs"
echo "  Data Service:     http://localhost:8003/docs"
echo "  Auth Service:     http://localhost:8005/docs"
echo "  RabbitMQ Admin:   http://localhost:15672 (user: codeconductor, pass: password)"
echo "  Grafana:          http://localhost:3000 (user: admin, pass: admin)"
echo "  Prometheus:       http://localhost:9090"
echo ""
echo "📝 Useful commands:"
echo "  View logs:        docker-compose logs -f"
echo "  Stop services:    docker-compose down"
echo "  Restart service:  docker-compose restart <service-name>"
echo "  Scale service:    docker-compose up -d --scale <service-name>=3"
echo ""
echo "🔍 Monitoring:"
echo "  Check all services: ./check_health.sh"
echo "  View service logs:  ./view_logs.sh"
echo "" 