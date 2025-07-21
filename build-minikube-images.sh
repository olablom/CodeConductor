#!/bin/bash

# 🚀 Build Docker Images in Minikube
# This script builds images in Minikube's Docker daemon

set -e

echo "🚀 Building Docker images in Minikube..."

# Set up Minikube Docker environment for Windows
export DOCKER_TLS_VERIFY=1
export DOCKER_HOST=tcp://127.0.0.1:60222
export DOCKER_CERT_PATH=C:/Users/olabl/.minikube/certs
export MINIKUBE_ACTIVE_DOCKERD=minikube

echo "🔨 Building images in Minikube Docker daemon..."

cd services

# Build all services in Minikube's Docker daemon
echo "Building gateway-service..."
docker build -t codeconductor-gateway:latest -f gateway_service/Dockerfile .

echo "Building agent-service..."
docker build -t codeconductor-agent:latest -f agent_service/Dockerfile .

echo "Building orchestrator-service..."
docker build -t codeconductor-orchestrator:latest -f orchestrator_service/Dockerfile .

echo "Building auth-service..."
docker build -t codeconductor-auth:latest -f auth_service/Dockerfile .

echo "Building data-service..."
docker build -t codeconductor-data:latest -f data_service/Dockerfile .

cd ..

echo "✅ All images built in Minikube Docker daemon!"
echo "🎯 Now you can deploy to Kubernetes!" 