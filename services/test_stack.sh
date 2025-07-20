#!/bin/bash

# CodeConductor Microservices Test Script
# Startar stacken och kör smoke tests

set -e

echo "🧪 CodeConductor Microservices Test Suite"
echo "=========================================="

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

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    docker-compose down
    echo "✅ Cleanup complete"
}

# Set trap to cleanup on script exit
trap cleanup EXIT

echo "🚀 Starting microservices stack..."
docker-compose up -d --build

echo "⏳ Waiting for services to be ready..."
sleep 15

echo "🔍 Running smoke tests..."
python smoke_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Smoke tests passed! Running full integration tests..."
    echo ""
    
    # Install test dependencies if needed
    pip install pytest pytest-asyncio httpx pika psycopg2-binary redis
    
    # Run integration tests
    python -m pytest tests/test_microservices.py -v -s
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 All tests passed! Microservices stack is working correctly."
        echo ""
        echo "📊 Service URLs:"
        echo "  Gateway API:      http://localhost:8000/docs"
        echo "  Agent Service:    http://localhost:8001/docs"
        echo "  Orchestrator:     http://localhost:8002/docs"
        echo "  Data Service:     http://localhost:8003/docs"
        echo "  Auth Service:     http://localhost:8005/docs"
        echo "  RabbitMQ Admin:   http://localhost:15672 (user: codeconductor, pass: password)"
        echo ""
        echo "🔍 To view logs: docker-compose logs -f"
        echo "🛑 To stop: docker-compose down"
        echo ""
        echo "Press Ctrl+C to stop the stack..."
        
        # Keep running until user stops
        while true; do
            sleep 1
        done
    else
        echo "❌ Integration tests failed"
        exit 1
    fi
else
    echo "❌ Smoke tests failed"
    exit 1
fi 