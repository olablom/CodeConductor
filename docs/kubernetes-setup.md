# ☸️ Kubernetes Setup Guide

## 📋 Förutsättningar

1. **Kubernetes Cluster** - Lokalt (kind, minikube) eller moln (GKE, AKS, EKS)
2. **kubectl** - Kubernetes CLI tool
3. **Docker Hub Images** - Microservices pushade till Docker Hub
4. **Ingress Controller** - NGINX Ingress Controller eller liknande

## 🚀 Snabbstart

### **Steg 1: Förberedelser**

```bash
# Klona repository och navigera till k8s mappen
cd k8s

# Sätt din Docker Hub användarnamn
export DOCKER_USERNAME="your-dockerhub-username"

# Gör deployment scriptet körbart
chmod +x deploy.sh
```

### **Steg 2: Uppdatera Secrets**

Redigera `secrets.yaml` och ersätt placeholder-värdena:

```bash
# Generera base64-encoded secrets
echo -n "your-actual-secret-key" | base64
echo -n "your-jwt-secret" | base64
echo -n "your-dockerhub-token" | base64
```

### **Steg 3: Deploya**

```bash
# Kör deployment scriptet
./deploy.sh
```

## 🏗️ Detaljerad Setup

### **Lokalt Kluster (kind/minikube)**

#### **Kind (Kubernetes in Docker)**

```bash
# Installera kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Skapa kluster
kind create cluster --name codeconductor

# Konfigurera kubectl
kind export kubeconfig --name codeconductor
```

#### **Minikube**

```bash
# Installera minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Starta kluster
minikube start --cpus 4 --memory 8192 --disk-size 20g

# Aktivera ingress addon
minikube addons enable ingress
```

### **Molnet (GKE/AKS/EKS)**

#### **Google Kubernetes Engine (GKE)**

```bash
# Skapa kluster
gcloud container clusters create codeconductor \
  --zone=europe-west1-b \
  --num-nodes=3 \
  --machine-type=e2-standard-2 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10

# Konfigurera kubectl
gcloud container clusters get-credentials codeconductor --zone=europe-west1-b
```

#### **Azure Kubernetes Service (AKS)**

```bash
# Skapa kluster
az aks create \
  --resource-group codeconductor-rg \
  --name codeconductor-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Konfigurera kubectl
az aks get-credentials --resource-group codeconductor-rg --name codeconductor-cluster
```

#### **Amazon EKS**

```bash
# Skapa kluster med eksctl
eksctl create cluster \
  --name codeconductor \
  --region eu-west-1 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10
```

## 🔧 Konfiguration

### **Ingress Controller**

#### **NGINX Ingress Controller**

```bash
# Installera NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Vänta på att ingress controller är redo
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s
```

#### **Traefik Ingress Controller**

```bash
# Installera Traefik
helm repo add traefik https://traefik.github.io/charts
helm install traefik traefik/traefik
```

### **Storage Class**

Kontrollera tillgängliga storage classes:

```bash
kubectl get storageclass
```

Uppdatera `data-deployment.yaml` med rätt storage class för ditt kluster.

## 🔐 Security

### **Secrets Management**

För production, använd en secrets management lösning:

#### **HashiCorp Vault**

```bash
# Installera Vault
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault --set server.dev.enabled=true
```

#### **AWS Secrets Manager**

```bash
# Installera AWS Secrets Manager CSI Driver
helm repo add secrets-store-csi-driver https://kubernetes-sigs.github.io/secrets-store-csi-driver/charts
helm install csi-secrets-store secrets-store-csi-driver/secrets-store-csi-driver
```

### **Network Policies**

Skapa network policies för säkerhet:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: codeconductor-network-policy
  namespace: codeconductor
spec:
  podSelector:
    matchLabels:
      app: codeconductor
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 80
        - protocol: TCP
          port: 443
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: codeconductor
      ports:
        - protocol: TCP
          port: 8001
        - protocol: TCP
          port: 8002
        - protocol: TCP
          port: 8003
        - protocol: TCP
          port: 8004
```

## 📊 Monitoring & Observability

### **Grundläggande Monitoring**

```bash
# Installera Prometheus och Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack
```

### **Logging**

```bash
# Installera ELK Stack
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch
helm install kibana elastic/kibana
helm install filebeat elastic/filebeat
```

## 🚀 Deployment

### **Automatisk Deployment**

```bash
# Kör hela deployment
./deploy.sh
```

### **Manuell Deployment**

```bash
# Skapa namespace
kubectl apply -f namespace.yaml

# Deploya secrets (uppdatera först!)
kubectl apply -f secrets.yaml

# Deploya configmaps
kubectl apply -f configmaps.yaml

# Deploya services i ordning
kubectl apply -f data-deployment.yaml
kubectl apply -f auth-deployment.yaml
kubectl apply -f agent-deployment.yaml
kubectl apply -f orchestrator-deployment.yaml
kubectl apply -f gateway-deployment.yaml
```

## 🔍 Verifiering

### **Kontrollera Status**

```bash
# Kontrollera pods
kubectl get pods -n codeconductor

# Kontrollera services
kubectl get services -n codeconductor

# Kontrollera ingress
kubectl get ingress -n codeconductor

# Kontrollera deployments
kubectl get deployments -n codeconductor
```

### **Testa Tjänster**

```bash
# Port forward för gateway
kubectl port-forward -n codeconductor service/gateway-service 9000:80

# Testa health endpoint
curl http://localhost:9000/health

# Testa API endpoints
curl http://localhost:9000/api/v1/agent/health
curl http://localhost:9000/api/v1/orchestrator/health
curl http://localhost:9000/api/v1/auth/health
curl http://localhost:9000/api/v1/data/health
```

### **Loggar**

```bash
# Visa loggar för alla tjänster
kubectl logs -n codeconductor -l app=codeconductor -f

# Visa loggar för specifik tjänst
kubectl logs -n codeconductor deployment/gateway-service -f
```

## 🔧 Felsökning

### **Vanliga Problem**

#### **Image Pull Errors**

```bash
# Kontrollera image pull secrets
kubectl get secrets -n codeconductor

# Skapa Docker Hub secret
kubectl create secret docker-registry dockerhub-secret \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=your-username \
  --docker-password=your-token \
  --docker-email=your-email \
  -n codeconductor
```

#### **Pod Startup Issues**

```bash
# Beskriv pod för detaljer
kubectl describe pod -n codeconductor <pod-name>

# Kontrollera events
kubectl get events -n codeconductor --sort-by='.lastTimestamp'
```

#### **Ingress Issues**

```bash
# Kontrollera ingress controller
kubectl get pods -n ingress-nginx

# Kontrollera ingress status
kubectl describe ingress gateway-ingress -n codeconductor
```

## 📈 Skalning

### **Horizontal Pod Autoscaler**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gateway-hpa
  namespace: codeconductor
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gateway-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### **Vertical Pod Autoscaler**

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: gateway-vpa
  namespace: codeconductor
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gateway-service
  updatePolicy:
    updateMode: "Auto"
```

## 🎯 Nästa Steg

Efter Kubernetes deployment är klar:

1. **Observability Stack** - Prometheus, Grafana, Jaeger
2. **CI/CD Pipeline** - ArgoCD eller Flux för GitOps
3. **Security Scanning** - Falco, OPA Gatekeeper
4. **Backup & Recovery** - Velero för backup
5. **Production Hardening** - Network policies, RBAC, etc.

## 📚 Resurser

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [Prometheus Operator](https://prometheus-operator.dev/)
- [Helm Charts](https://helm.sh/docs/chart_template_guide/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
