#!/bin/bash

# 🚀 CodeConductor Demo Deployment
# Quick deployment for local Kubernetes demo

set -e

NAMESPACE="codeconductor-demo"

echo "🚀 Deploying CodeConductor Demo to Kubernetes..."

# Create namespace
kubectl apply -f namespace-demo.yaml

# Deploy secrets
kubectl apply -f secrets-demo-final.yaml

# Deploy configmaps
kubectl apply -f configmaps.yaml

# Deploy services in order
echo "Deploying microservices..."
kubectl apply -f data-deployment.yaml
kubectl apply -f auth-deployment.yaml
kubectl apply -f agent-deployment.yaml
kubectl apply -f orchestrator-deployment.yaml
kubectl apply -f gateway-deployment.yaml

# Wait for deployments
echo "Waiting for deployments to be ready..."
kubectl rollout status deployment/data-service -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/auth-service -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/agent-service -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/orchestrator-service -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/gateway-service -n $NAMESPACE --timeout=300s

echo "🎉 Demo deployment completed!"
echo ""
echo "📊 Status:"
kubectl get pods -n $NAMESPACE
echo ""
echo "🌐 Access:"
echo "kubectl port-forward -n $NAMESPACE service/gateway-service 9000:80"
echo "Then visit: http://localhost:9000"
