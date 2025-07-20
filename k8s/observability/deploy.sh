#!/bin/bash

# 🚀 CodeConductor Observability Stack Deployment Script
# This script deploys Prometheus, Grafana, Jaeger, and Loki

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="observability"

echo -e "${BLUE}🚀 CodeConductor Observability Stack Deployment${NC}"
echo "=================================================="

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

# Function to create namespace and resources
deploy_namespace() {
    echo -e "${BLUE}📦 Creating observability namespace and resource quotas...${NC}"
    kubectl apply -f namespace.yaml
    echo -e "${GREEN}✅ Namespace created${NC}"
}

# Function to deploy Prometheus
deploy_prometheus() {
    echo -e "${BLUE}📊 Deploying Prometheus...${NC}"
    kubectl apply -f prometheus-deployment.yaml
    echo -e "${GREEN}✅ Prometheus deployed${NC}"
}

# Function to deploy Grafana
deploy_grafana() {
    echo -e "${BLUE}📈 Deploying Grafana...${NC}"
    kubectl apply -f grafana-deployment.yaml
    echo -e "${GREEN}✅ Grafana deployed${NC}"
}

# Function to deploy Loki
deploy_loki() {
    echo -e "${BLUE}📝 Deploying Loki and Promtail...${NC}"
    kubectl apply -f loki-deployment.yaml
    echo -e "${GREEN}✅ Loki and Promtail deployed${NC}"
}

# Function to deploy Jaeger
deploy_jaeger() {
    echo -e "${BLUE}🔍 Deploying Jaeger...${NC}"
    kubectl apply -f jaeger-deployment.yaml
    echo -e "${GREEN}✅ Jaeger deployed${NC}"
}

# Function to wait for deployments
wait_for_deployments() {
    echo -e "${BLUE}⏳ Waiting for deployments to be ready...${NC}"
    
    deployments=("prometheus" "grafana" "loki" "jaeger")
    
    for deployment in "${deployments[@]}"; do
        echo -e "${BLUE}Waiting for ${deployment}...${NC}"
        kubectl rollout status deployment/${deployment} -n ${NAMESPACE} --timeout=300s
        echo -e "${GREEN}✅ ${deployment} is ready${NC}"
    done
    
    # Wait for DaemonSet
    echo -e "${BLUE}Waiting for Promtail DaemonSet...${NC}"
    kubectl rollout status daemonset/promtail -n ${NAMESPACE} --timeout=300s
    echo -e "${GREEN}✅ Promtail is ready${NC}"
}

# Function to show deployment status
show_status() {
    echo -e "${BLUE}📊 Observability Stack Status${NC}"
    echo "================================"
    
    echo -e "${BLUE}Pods:${NC}"
    kubectl get pods -n ${NAMESPACE}
    
    echo -e "\n${BLUE}Services:${NC}"
    kubectl get services -n ${NAMESPACE}
    
    echo -e "\n${BLUE}Ingress:${NC}"
    kubectl get ingress -n ${NAMESPACE}
    
    echo -e "\n${BLUE}Deployments:${NC}"
    kubectl get deployments -n ${NAMESPACE}
    
    echo -e "\n${BLUE}DaemonSets:${NC}"
    kubectl get daemonsets -n ${NAMESPACE}
}

# Function to show access information
show_access_info() {
    echo -e "${BLUE}🌐 Access Information${NC}"
    echo "======================"
    
    # Get ingress hosts
    GRAFANA_HOST=$(kubectl get ingress grafana-ingress -n ${NAMESPACE} -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "grafana.codeconductor.local")
    JAEGER_HOST=$(kubectl get ingress jaeger-ingress -n ${NAMESPACE} -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "jaeger.codeconductor.local")
    
    echo -e "${GREEN}Grafana Dashboard:${NC} http://${GRAFANA_HOST}"
    echo -e "${GREEN}  Username:${NC} admin"
    echo -e "${GREEN}  Password:${NC} admin123"
    
    echo -e "\n${GREEN}Jaeger UI:${NC} http://${JAEGER_HOST}"
    
    echo -e "\n${BLUE}Port Forward (if no ingress):${NC}"
    echo "Grafana: kubectl port-forward -n ${NAMESPACE} service/grafana 3000:3000"
    echo "Prometheus: kubectl port-forward -n ${NAMESPACE} service/prometheus 9090:9090"
    echo "Jaeger: kubectl port-forward -n ${NAMESPACE} service/jaeger 16686:16686"
    echo "Loki: kubectl port-forward -n ${NAMESPACE} service/loki 3100:3100"
    
    echo -e "\n${BLUE}Logs:${NC}"
    echo "kubectl logs -n ${NAMESPACE} -l app=prometheus -f"
    echo "kubectl logs -n ${NAMESPACE} -l app=grafana -f"
    echo "kubectl logs -n ${NAMESPACE} -l app=loki -f"
    echo "kubectl logs -n ${NAMESPACE} -l app=jaeger -f"
}

# Function to configure CodeConductor services for observability
configure_codeconductor() {
    echo -e "${BLUE}🔧 Configuring CodeConductor services for observability...${NC}"
    
    # Add Prometheus annotations to CodeConductor services
    echo -e "${YELLOW}⚠️  Remember to add these annotations to your CodeConductor deployments:${NC}"
    echo ""
    echo "metadata:"
    echo "  annotations:"
    echo "    prometheus.io/scrape: \"true\""
    echo "    prometheus.io/port: \"9000\"  # or appropriate port"
    echo "    prometheus.io/path: \"/metrics\""
    echo ""
    echo "And ensure your services expose metrics on /metrics endpoint"
}

# Function to show next steps
show_next_steps() {
    echo -e "${BLUE}🎯 Next Steps${NC}"
    echo "============="
    echo "1. Configure your domain in ingress rules"
    echo "2. Set up SSL/TLS certificates"
    echo "3. Configure alerting rules in Prometheus"
    echo "4. Import additional Grafana dashboards"
    echo "5. Set up log retention policies in Loki"
    echo "6. Configure sampling strategies in Jaeger"
    echo "7. Add metrics endpoints to CodeConductor services"
    echo "8. Set up alerting (AlertManager, Slack, etc.)"
}

# Main deployment function
main() {
    check_kubectl
    check_cluster
    deploy_namespace
    deploy_prometheus
    deploy_grafana
    deploy_loki
    deploy_jaeger
    wait_for_deployments
    show_status
    show_access_info
    configure_codeconductor
    show_next_steps
    
    echo -e "\n${GREEN}🎉 Observability stack deployed successfully!${NC}"
    echo -e "${BLUE}Access Grafana at: http://${GRAFANA_HOST:-grafana.codeconductor.local}${NC}"
    echo -e "${BLUE}Access Jaeger at: http://${JAEGER_HOST:-jaeger.codeconductor.local}${NC}"
}

# Run main function
main "$@" 