#!/bin/bash

# 🚀 CodeConductor Demo Preparation Script
# Prepares Kubernetes manifests for local deployment

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Preparing CodeConductor for Kubernetes Demo${NC}"
echo "================================================"

# Function to build local Docker images
build_images() {
    echo -e "${BLUE}🔨 Building Docker images locally...${NC}"
    
    cd services
    
    # Build all services
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
    
    echo -e "${GREEN}✅ All images built successfully${NC}"
}

# Function to update deployment files for local images
update_deployments() {
    echo -e "${BLUE}🔄 Updating deployment files for local images...${NC}"
    
    # Update all deployment files to use local images
    find k8s -name "*-deployment.yaml" -type f -exec sed -i 's|your-username/codeconductor-|codeconductor-|g' {} \;
    find k8s -name "*-deployment.yaml" -type f -exec sed -i 's|:latest|:latest|g' {} \;
    
    echo -e "${GREEN}✅ Deployment files updated${NC}"
}

# Function to create demo namespace
create_demo_namespace() {
    echo -e "${BLUE}📦 Creating demo namespace...${NC}"
    
    cat > k8s/namespace-demo.yaml << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: codeconductor-demo
  labels:
    name: codeconductor-demo
    purpose: demo
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: codeconductor-demo-quota
  namespace: codeconductor-demo
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    pods: "20"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: codeconductor-demo-limits
  namespace: codeconductor-demo
spec:
  limits:
  - default:
      cpu: 500m
      memory: 512Mi
    defaultRequest:
      cpu: 100m
      memory: 128Mi
    type: Container
EOF
    
    echo -e "${GREEN}✅ Demo namespace created${NC}"
}

# Function to update secrets for demo
update_secrets() {
    echo -e "${BLUE}🔐 Updating secrets for demo...${NC}"
    
    # Copy demo secrets
    cp k8s/secrets-demo.yaml k8s/secrets-demo-final.yaml
    
    # Update namespace in secrets
    sed -i 's/namespace: codeconductor/namespace: codeconductor-demo/g' k8s/secrets-demo-final.yaml
    
    echo -e "${GREEN}✅ Demo secrets prepared${NC}"
}

# Function to create demo deployment script
create_demo_deploy_script() {
    echo -e "${BLUE}📝 Creating demo deployment script...${NC}"
    
    cat > k8s/deploy-demo.sh << 'EOF'
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
EOF
    
    chmod +x k8s/deploy-demo.sh
    echo -e "${GREEN}✅ Demo deployment script created${NC}"
}

# Main function
main() {
    build_images
    update_deployments
    create_demo_namespace
    update_secrets
    create_demo_deploy_script
    
    echo -e "\n${GREEN}🎉 Demo preparation completed!${NC}"
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Ensure Kubernetes is running"
    echo "2. Run: cd k8s && ./deploy-demo.sh"
    echo "3. Access your AI microservices at http://localhost:9000"
}

# Run main function
main "$@" 