#!/bin/bash

# 🚀 CodeConductor Kubernetes Deployment Script
# This script deploys all microservices to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="codeconductor"
DOCKER_USERNAME="${DOCKER_USERNAME:-your-username}"

echo -e "${BLUE}🚀 CodeConductor Kubernetes Deployment${NC}"
echo "=========================================="

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed or not in PATH${NC}"
        echo "Please install kubectl: https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi
    echo -e "${GREEN}✅ kubectl found${NC}"
}

# Function to check cluster connectivity
check_cluster() {
    echo -e "${BLUE}🔍 Checking cluster connectivity...${NC}"
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        echo "Please ensure your cluster is running and kubectl is configured"
        exit 1
    fi
    echo -e "${GREEN}✅ Connected to cluster${NC}"
}

# Function to update image names in manifests
update_images() {
    echo -e "${BLUE}🔄 Updating image names...${NC}"
    
    # Update all deployment files with correct Docker username
    find . -name "*-deployment.yaml" -type f -exec sed -i "s/your-username/${DOCKER_USERNAME}/g" {} \;
    
    echo -e "${GREEN}✅ Image names updated to use: ${DOCKER_USERNAME}${NC}"
}

# Function to create namespace and resources
deploy_namespace() {
    echo -e "${BLUE}📦 Creating namespace and resource quotas...${NC}"
    kubectl apply -f namespace.yaml
    echo -e "${GREEN}✅ Namespace created${NC}"
}

# Function to deploy secrets
deploy_secrets() {
    echo -e "${BLUE}🔐 Deploying secrets...${NC}"
    echo -e "${YELLOW}⚠️  Remember to update secrets.yaml with actual values!${NC}"
    kubectl apply -f secrets.yaml
    echo -e "${GREEN}✅ Secrets deployed${NC}"
}

# Function to deploy configmaps
deploy_configmaps() {
    echo -e "${BLUE}⚙️  Deploying configmaps...${NC}"
    kubectl apply -f configmaps.yaml
    echo -e "${GREEN}✅ ConfigMaps deployed${NC}"
}

# Function to deploy services
deploy_services() {
    echo -e "${BLUE}🔧 Deploying microservices...${NC}"
    
    # Deploy in order of dependencies
    kubectl apply -f data-deployment.yaml
    kubectl apply -f auth-deployment.yaml
    kubectl apply -f agent-deployment.yaml
    kubectl apply -f orchestrator-deployment.yaml
    kubectl apply -f gateway-deployment.yaml
    
    echo -e "${GREEN}✅ All services deployed${NC}"
}

# Function to wait for deployments
wait_for_deployments() {
    echo -e "${BLUE}⏳ Waiting for deployments to be ready...${NC}"
    
    deployments=("data-service" "auth-service" "agent-service" "orchestrator-service" "gateway-service")
    
    for deployment in "${deployments[@]}"; do
        echo -e "${BLUE}Waiting for ${deployment}...${NC}"
        kubectl rollout status deployment/${deployment} -n ${NAMESPACE} --timeout=300s
        echo -e "${GREEN}✅ ${deployment} is ready${NC}"
    done
}

# Function to show deployment status
show_status() {
    echo -e "${BLUE}📊 Deployment Status${NC}"
    echo "=================="
    
    echo -e "${BLUE}Pods:${NC}"
    kubectl get pods -n ${NAMESPACE}
    
    echo -e "\n${BLUE}Services:${NC}"
    kubectl get services -n ${NAMESPACE}
    
    echo -e "\n${BLUE}Ingress:${NC}"
    kubectl get ingress -n ${NAMESPACE}
    
    echo -e "\n${BLUE}Deployments:${NC}"
    kubectl get deployments -n ${NAMESPACE}
}

# Function to show access information
show_access_info() {
    echo -e "${BLUE}🌐 Access Information${NC}"
    echo "=================="
    
    # Get ingress host
    INGRESS_HOST=$(kubectl get ingress gateway-ingress -n ${NAMESPACE} -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "codeconductor.local")
    
    echo -e "${GREEN}Gateway URL:${NC} http://${INGRESS_HOST}"
    echo -e "${GREEN}Health Check:${NC} http://${INGRESS_HOST}/health"
    
    echo -e "\n${BLUE}Port Forward (if no ingress):${NC}"
    echo "kubectl port-forward -n ${NAMESPACE} service/gateway-service 9000:80"
    echo "Then access: http://localhost:9000"
    
    echo -e "\n${BLUE}Logs:${NC}"
    echo "kubectl logs -n ${NAMESPACE} -l app=codeconductor -f"
}

# Main deployment function
main() {
    check_kubectl
    check_cluster
    update_images
    deploy_namespace
    deploy_secrets
    deploy_configmaps
    deploy_services
    wait_for_deployments
    show_status
    show_access_info
    
    echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Update secrets.yaml with actual values"
    echo "2. Configure your domain in ingress"
    echo "3. Set up monitoring and observability"
    echo "4. Configure SSL/TLS certificates"
}

# Run main function
main "$@" 