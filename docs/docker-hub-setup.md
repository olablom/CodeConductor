# 🐳 Docker Hub Deployment Setup

## 📋 Förutsättningar

1. **Docker Hub Account** - Skapa ett konto på [hub.docker.com](https://hub.docker.com)
2. **GitHub Repository** - Detta repository med microservices
3. **GitHub Secrets** - Konfigurera secrets för Docker Hub autentisering

## 🔐 Steg 1: Skapa Docker Hub Access Token

1. **Logga in på Docker Hub**

   - Gå till [hub.docker.com](https://hub.docker.com)
   - Logga in med ditt konto

2. **Skapa Access Token**
   - Klicka på din profil (övre högra hörnet)
   - Välj "Account Settings"
   - Gå till "Security" → "New Access Token"
   - Ge token ett namn: `codeconductor-deploy`
   - Välj "Read & Write" permissions
   - Klicka "Generate"
   - **Kopiera token** (du ser den bara en gång!)

## 🔑 Steg 2: Konfigurera GitHub Secrets

1. **Gå till GitHub Repository**

   - Öppna [GitHub repository](https://github.com/olablom/CodeConductor)
   - Klicka "Settings" → "Secrets and variables" → "Actions"

2. **Lägg till secrets:**
   ```
   DOCKERHUB_USERNAME = ditt-dockerhub-användarnamn
   DOCKERHUB_TOKEN = din-access-token-från-steg-1
   ```

## 🚀 Steg 3: Testa Deployment

### **Automatisk Deployment**

- Pusha till `main` branch med ändringar i `services/`
- Workflow körs automatiskt
- Images pushas till Docker Hub

### **Manuell Deployment**

1. Gå till "Actions" → "Docker Hub Deployment"
2. Klicka "Run workflow"
3. Ange version (t.ex. "2.0.0")
4. Klicka "Run workflow"

## 📦 Steg 4: Verifiera Images

### **Lista images på Docker Hub**

```bash
# Sök efter dina images
docker search codeconductor-gateway
docker search codeconductor-agent
docker search codeconductor-orchestrator
docker search codeconductor-auth
docker search codeconductor-data
```

### **Testa images lokalt**

```bash
# Pull och kör gateway service
docker pull your-username/codeconductor-gateway:latest
docker run -p 9000:9000 your-username/codeconductor-gateway:latest

# Testa health endpoint
curl http://localhost:9000/health
```

## 🏷️ Image Tagging Strategy

### **Automatiska Tags**

- `latest` - Senaste commit på main branch
- `v2.0.0` - Semantic versioning (git tags)
- `v2.0` - Major.minor version
- `main` - Branch name

### **Exempel**

```bash
# Pull specifik version
docker pull your-username/codeconductor-gateway:v2.0.0

# Pull latest
docker pull your-username/codeconductor-gateway:latest
```

## 🔒 Security Features

### **Vulnerability Scanning**

- **Trivy** scannerar alla images automatiskt
- Resultat visas i GitHub Security tab
- CVEs rapporteras automatiskt

### **Best Practices**

- Multi-stage builds för mindre images
- Non-root user i containers
- Security updates automatiskt
- Image signing (kommer snart)

## 📊 Monitoring

### **Docker Hub Metrics**

- Pull statistics
- Image size tracking
- Security scan results
- Build history

### **GitHub Actions Metrics**

- Build success rate
- Deployment frequency
- Security scan results
- Performance metrics

## 🚀 Nästa Steg

Efter Docker Hub deployment är klar:

1. **Kubernetes Setup** - Skapa manifests för deployment
2. **Observability** - Prometheus, Grafana, Jaeger
3. **Production Deployment** - Deploya till cloud platform

## 🆘 Felsökning

### **Vanliga Problem**

**"Authentication failed"**

- Kontrollera DOCKERHUB_USERNAME och DOCKERHUB_TOKEN
- Verifiera att token har "Read & Write" permissions

**"Image not found"**

- Vänta på att build att slutföras
- Kontrollera att image namn är korrekt

**"Build failed"**

- Kontrollera Dockerfile syntax
- Verifiera att alla dependencies finns

### **Support**

- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Trivy Security Scanner](https://aquasecurity.github.io/trivy/)
