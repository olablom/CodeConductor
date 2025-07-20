# 📊 Observability Stack Setup Guide

## 🎯 Översikt

CodeConductor observability stack ger dig full insyn i dina microservices med:

- **📈 Prometheus** - Metrics collection och alerting
- **📊 Grafana** - Dashboards och visualisering
- **🔍 Jaeger** - Distributed tracing
- **📝 Loki** - Log aggregation med Promtail

## 🚀 Snabbstart

### **Steg 1: Förberedelser**

```bash
# Navigera till observability mappen
cd k8s/observability

# Gör deployment scriptet körbart
chmod +x deploy.sh
```

### **Steg 2: Deploya Stack**

```bash
# Kör deployment scriptet
./deploy.sh
```

### **Steg 3: Verifiera**

```bash
# Kontrollera att alla pods är redo
kubectl get pods -n observability

# Port forward för åtkomst
kubectl port-forward -n observability service/grafana 3000:3000
kubectl port-forward -n observability service/prometheus 9090:9090
kubectl port-forward -n observability service/jaeger 16686:16686
```

## 📊 Komponenter

### **Prometheus**

**Funktioner:**

- Metrics collection från Kubernetes och CodeConductor services
- Service discovery via Kubernetes API
- Alerting rules för CodeConductor-specifika metrics
- Persistent storage för historisk data

**Konfiguration:**

```yaml
# Scraping targets
- codeconductor-gateway (port 9000)
- codeconductor-agent (port 8001)
- codeconductor-orchestrator (port 8002)
- codeconductor-auth (port 8003)
- codeconductor-data (port 8004)
```

**Alerting Rules:**

- Service down alerts
- High error rate detection
- Response time thresholds
- Resource usage alerts

### **Grafana**

**Funktioner:**

- Pre-configured dashboards för CodeConductor
- Prometheus och Loki datasources
- Kubernetes cluster overview
- Customizable panels och alerts

**Pre-installerade Dashboards:**

1. **CodeConductor Overview**

   - Service health status
   - Request rates
   - Response times
   - Error rates

2. **Kubernetes Overview**
   - Pod status
   - CPU/Memory usage
   - Node metrics

**Access:**

- URL: `http://grafana.codeconductor.local`
- Username: `admin`
- Password: `admin123`

### **Jaeger**

**Funktioner:**

- Distributed tracing för microservices
- Sampling strategies för olika endpoints
- Integration med Prometheus för metrics
- Web UI för trace visualization

**Sampling Strategies:**

```yaml
- /api/v1/agent/generate: 100% (all traces)
- /api/v1/orchestrator/process: 100% (all traces)
- /api/v1/auth/login: 50% (half traces)
- /api/v1/data/query: 30% (30% traces)
```

**Access:**

- URL: `http://jaeger.codeconductor.local`

### **Loki**

**Funktioner:**

- Log aggregation från alla CodeConductor pods
- Promtail DaemonSet för log collection
- Integration med Grafana för log queries
- Efficient storage med compression

**Log Sources:**

- All CodeConductor microservices
- Kubernetes system logs
- Container logs

## 🔧 Konfiguration

### **CodeConductor Services Integration**

För att få full observability, lägg till dessa annotations till dina CodeConductor deployments:

```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9000" # eller rätt port
    prometheus.io/path: "/metrics"
```

### **Metrics Endpoints**

Se till att dina services exponerar metrics på `/metrics` endpoint:

```python
# Exempel för FastAPI service
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import FastAPI

app = FastAPI()

# Metrics
http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
http_request_duration_seconds = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### **Logging Configuration**

Konfigurera strukturerad logging för bästa integration med Loki:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'service': 'codeconductor-gateway',
            'component': 'gateway'
        }
        return json.dumps(log_entry)

# Konfigurera logger
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
```

## 📈 Dashboards

### **CodeConductor Overview Dashboard**

**Panels:**

1. **Service Health** - Status för alla microservices
2. **Request Rate** - Requests per sekund per service
3. **Response Time** - 95th percentile response times
4. **Error Rate** - 5xx errors per sekund
5. **Active Requests** - Concurrent requests
6. **Resource Usage** - CPU/Memory per pod

### **Kubernetes Cluster Dashboard**

**Panels:**

1. **Pod Status** - Running/Pending/Failed pods
2. **CPU Usage** - Per pod CPU consumption
3. **Memory Usage** - Per pod memory usage
4. **Network I/O** - Network traffic
5. **Disk I/O** - Storage operations

## 🔔 Alerting

### **Prometheus Alert Rules**

```yaml
# Service Down Alert
- alert: CodeConductorServiceDown
  expr: up{job=~"codeconductor-.*"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "CodeConductor service {{ $labels.job }} is down"

# High Error Rate
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
  for: 2m
  labels:
    severity: warning

# High Response Time
- alert: HighResponseTime
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
  for: 2m
  labels:
    severity: warning
```

### **AlertManager Integration**

För att få notifikationer, konfigurera AlertManager:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: observability
data:
  alertmanager.yml: |
    global:
      slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

    route:
      group_by: ['alertname']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'slack-notifications'

    receivers:
    - name: 'slack-notifications'
      slack_configs:
      - channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## 🔍 Tracing

### **Jaeger Integration**

För att få distributed tracing, lägg till OpenTelemetry till dina services:

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Konfigurera Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

# Sätt upp tracer provider
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

# Använd i din kod
@tracer.start_as_current_span("process_request")
def process_request(request_data):
    # Din kod här
    pass
```

## 📝 Logging

### **Loki Queries**

Exempel på användbara log queries i Grafana:

```logql
# Alla logs från CodeConductor services
{app="codeconductor"}

# Error logs
{app="codeconductor"} |= "ERROR"

# Logs från specifik service
{app="codeconductor", component="gateway"}

# Logs med specifik log level
{app="codeconductor"} | json | level="ERROR"

# Logs från senaste 1 timmen
{app="codeconductor"} [1h]
```

## 🚀 Production Deployment

### **Storage Configuration**

För production, använd persistent storage:

```yaml
# Prometheus storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-pvc
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 50Gi  # Öka för production
  storageClassName: fast-ssd  # Använd SSD för bättre performance

# Grafana storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-pvc
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 20Gi
  storageClassName: fast-ssd

# Loki storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: loki-pvc
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 100Gi  # Stora mängder log data
  storageClassName: fast-ssd
```

### **Resource Limits**

Justera resource limits för production:

```yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2
    memory: 4Gi
```

### **High Availability**

För HA, kör flera replicas:

```yaml
replicas: 3 # För Prometheus, Grafana
```

## 🔧 Felsökning

### **Vanliga Problem**

#### **Prometheus kan inte scrape metrics**

```bash
# Kontrollera service discovery
kubectl get endpoints -n codeconductor

# Kontrollera metrics endpoint
kubectl port-forward -n codeconductor service/gateway-service 9000:80
curl http://localhost:9000/metrics

# Kontrollera Prometheus targets
kubectl port-forward -n observability service/prometheus 9090:9090
# Öppna http://localhost:9090/targets
```

#### **Grafana kan inte ansluta till Prometheus**

```bash
# Kontrollera Prometheus service
kubectl get svc -n observability prometheus

# Testa anslutning från Grafana pod
kubectl exec -n observability deployment/grafana -- curl http://prometheus:9090/api/v1/status/config
```

#### **Loki tar inte emot logs**

```bash
# Kontrollera Promtail status
kubectl get pods -n observability -l app=promtail

# Kontrollera Promtail logs
kubectl logs -n observability -l app=promtail

# Testa Loki endpoint
kubectl port-forward -n observability service/loki 3100:3100
curl http://localhost:3100/ready
```

### **Performance Optimization**

#### **Prometheus**

```yaml
# Öka scrape interval för mindre viktiga metrics
scrape_interval: 30s # Istället för 15s

# Använd recording rules för komplexa queries
groups:
  - name: codeconductor.rules
    rules:
      - record: job:http_requests_total:rate5m
        expr: rate(http_requests_total[5m])
```

#### **Loki**

```yaml
# Konfigurera retention
limits_config:
  retention_period: 168h # 7 dagar
  max_query_length: 721h

# Använd chunk storage
schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: s3 # För production
```

## 📚 Resurser

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Loki Documentation](https://grafana.com/docs/loki/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Prometheus Client Python](https://github.com/prometheus/client_python)

## 🎯 Nästa Steg

Efter observability stack är igång:

1. **Alerting** - Konfigurera AlertManager med Slack/Email
2. **Custom Dashboards** - Skapa dashboards för specifika use cases
3. **SLO/SLI** - Definiera Service Level Objectives
4. **Capacity Planning** - Analysera trends för skalning
5. **Security** - Konfigurera RBAC och network policies
6. **Backup** - Backup av Prometheus och Grafana data
