# CI/CD Pipeline Documentation

## 🏗️ Projektstruktur

CodeConductor har två separata CI/CD-pipelines:

### 1. **Main CodeConductor CI** (`.github/workflows/main-ci.yml`)

- **Trigger**: Ändringar i root (exklusive `services/`)
- **Syfte**: Testa huvudapplikationen (monolitisk CodeConductor)
- **Aktivitet**: Linting, tester, Docker build

### 2. **Microservices CI/CD** (`.github/workflows/microservices-ci-cd.yml`)

- **Trigger**: Ändringar i `services/` mappen
- **Syfte**: Bygga, testa och deploya mikrotjänster-stacken
- **Aktivitet**: Matris-build, integrationstester, Docker push, deploy

## 🚀 Mikrotjänster CI/CD Pipeline

### **Jobs:**

#### **1. Build and Test (Matrix)**

```yaml
services:
  [
    gateway-service,
    agent-service,
    orchestrator-service,
    auth-service,
    data-service,
  ]
```

- Bygger varje tjänst parallellt
- Kör linting med Black
- Bygger Docker images
- Testar att images fungerar

#### **2. Integration Test**

- Startar hela stacken med Docker Compose
- Kör `test_full_stack.py` och `test_data_endpoints.py`
- Testar Gateway routing
- Cleanup efter test

#### **3. Deploy**

- Pushar Docker images till Docker Hub
- Förbereder för staging deployment
- Kräver secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`

#### **4. Security Scan**

- CodeQL säkerhetsscanning
- Analyserar Python-kod för säkerhetsproblem

#### **5. Notify**

- Meddelar om success/failure
- Kör alltid (även vid failure)

## 🔒 Secrets som behövs

### **Docker Hub**

```bash
DOCKERHUB_USERNAME=your_username
DOCKERHUB_TOKEN=your_access_token
```

### **Kubernetes (framtida)**

```bash
KUBE_CONFIG=base64_encoded_kubeconfig
KUBE_NAMESPACE=codeconductor
```

## 📊 Test Coverage

### **Unit Tests**

- Varje tjänst har egna tester i `services/{service}/tests/`
- Körs parallellt i matrix-build

### **Integration Tests**

- `test_full_stack.py`: Testar hela stacken
- `test_data_endpoints.py`: Testar Data Service endpoints
- Körs efter alla unit tests

### **End-to-End Tests**

- Gateway routing test
- Health check test
- Docker Compose integration

## 🎯 Deployment Strategy

### **Staging**

- Automatisk deploy på `main` branch
- Docker images pushas till registry
- Kubernetes deployment (framtida)

### **Production**

- Manuell deploy från staging
- Blue-green deployment (framtida)
- Rollback capability (framtida)

## 🔧 Lokal utveckling

### **Testa lokalt**

```bash
cd services
docker-compose up -d
python test_full_stack.py
```

### **Testa enskild tjänst**

```bash
cd services/data_service
docker build -t data-service .
docker run -p 9006:8003 data-service
```

## 📈 Monitoring

### **CI/CD Metrics**

- Build time per tjänst
- Test success rate
- Deployment frequency
- Lead time for changes

### **Runtime Metrics** (framtida)

- Prometheus metrics
- Grafana dashboards
- Alerting rules

## 🚨 Troubleshooting

### **Vanliga problem:**

1. **Docker build fails**

   - Kontrollera Dockerfile syntax
   - Verifiera dependencies i requirements.txt

2. **Integration tests fail**

   - Kontrollera att alla tjänster startar
   - Verifiera nätverkskonfiguration

3. **Deploy fails**
   - Kontrollera Docker Hub credentials
   - Verifiera image names och tags

### **Debug commands:**

```bash
# Kolla tjänstloggar
docker-compose logs [service-name]

# Testa enskild endpoint
curl http://localhost:9000/health

# Kolla container status
docker-compose ps
```

## 🔄 Workflow Triggers

### **Microservices Pipeline**

- Push till `main` med ändringar i `services/`
- Pull request till `main` med ändringar i `services/`

### **Main Pipeline**

- Push till `main` utan ändringar i `services/`
- Pull request till `main` utan ändringar i `services/`

## 📝 Best Practices

1. **Commit Messages**

   - Använd konventionell commit format
   - Länka till issues/PRs

2. **Branch Strategy**

   - `main`: Production-ready code
   - `feature/*`: Nya features
   - `hotfix/*`: Snabba fixes

3. **Testing**

   - Skriv tester för ny funktionalitet
   - Uppdatera integrationstester vid API-ändringar

4. **Security**
   - Uppdatera dependencies regelbundet
   - Granska säkerhetsscanningar
   - Använd secrets för känslig data
