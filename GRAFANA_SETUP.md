# 🚀 Grafana Setup Guide för CodeConductor

## 📋 Snabb setup

### 1. Öppna Grafana

Gå till: `http://localhost:3000`

- **Användarnamn:** `admin`
- **Lösenord:** `admin`

### 2. Lägg till Prometheus som datakälla

1. Gå till **Configuration** → **Data Sources**
2. Klicka **Add data source**
3. Välj **Prometheus**
4. **URL:** `http://host.docker.internal:9090`
5. Klicka **Save & Test**

### 3. Importera dashboard

1. Gå till **Dashboards** → **Import**
2. Kopiera innehållet från `grafana_dashboard.json`
3. Klicka **Load**
4. Välj Prometheus som datakälla
5. Klicka **Import**

## 🔧 Om du får nätverksfel

### Problem: "dial tcp: lookup prometheus"

**Lösning:** Använd `http://host.docker.internal:9090` istället för `http://prometheus:9090`

### Problem: "connection refused"

**Lösning:** Kontrollera att Prometheus körs:

```bash
curl http://localhost:9090/api/v1/query?query=up
```

## 📊 Vad du kommer att se

**Dashboard med:**

- **Model Health Status** - Antal friska modeller (6/6)
- **Total Requests** - Totalt antal förfrågningar
- **Success Rate** - Framgångsgrad i procent
- **Uptime** - Hur länge systemet kört
- **Model Response Times** - Graf över svarstider
- **System Health** - Tabell med alla modeller

## 🧪 Testa systemet

Kör test-scriptet för att generera aktivitet:

```bash
python test_monitoring.py
```

## 📈 Prometheus Queries

Här är några användbara Prometheus-queries:

### Model Health

```
sum(codeconductor_model_healthy)
```

### Success Rate

```
rate(codeconductor_successful_requests[5m]) / rate(codeconductor_total_requests[5m]) * 100
```

### Uptime

```
codeconductor_uptime_seconds / 3600
```

### Total Requests

```
codeconductor_total_requests
```

## 🎯 Nästa steg

1. **Konfigurera alerts** - Lägg till Slack/webhook notifikationer
2. **Skapa custom dashboards** - För specifika use cases
3. **Optimera queries** - För bättre prestanda
4. **Lägg till fler metrics** - För RLHF och Test-as-Reward
