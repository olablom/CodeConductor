# 🚀 Grafana Dashboard Update Guide

## 📊 Nya Paneler för Pipeline-Monitoring

Vi har lagt till nya paneler för att övervaka pipeline-prestanda och få bättre insyn i systemet.

### 🎯 **Nya Paneler:**

1. **Pipeline Response Time (s)** - Genomsnittlig svarstid för hela pipeline
2. **Pipeline Response Time by Task** - Svarstider per task-typ
3. **Request Rate (req/s)** - Antal requests per sekund
4. **Pipeline Performance Distribution** - Histogram över svarstider
5. **System Health Overview** - Tabell med modell-status

### 🔧 **Uppdatera Dashboard:**

#### Steg 1: Importera ny dashboard

1. Öppna Grafana: `http://localhost:3000`
2. Gå till **Dashboards** → **Import**
3. Kopiera innehållet från `grafana_dashboard.json`
4. Klicka **Load**
5. Välj Prometheus som datakälla
6. Klicka **Import**

#### Steg 2: Verifiera nya paneler

Du bör nu se följande paneler:

**Övre raden:**

- Model Health Status (6/6)
- Total Requests (58+)
- Success Rate (100%)
- Uptime (1.04h+)

**Mittre raden:**

- **Pipeline Response Time (s)** - Ny! 🆕
- **Pipeline Response Time by Task** - Ny! 🆕

**Nedre raden:**

- Model Response Times (uppdaterad)
- **Request Rate (req/s)** - Ny! 🆕
- **Pipeline Performance Distribution** - Ny! 🆕
- **System Health Overview** - Ny! 🆕

### 📈 **Vad du kommer att se:**

#### Pipeline Response Time (s)

- **Grön:** < 5 sekunder (optimal)
- **Gul:** 5-10 sekunder (varning)
- **Röd:** > 10 sekunder (kritisk)

#### Pipeline Response Time by Task

- Visar svarstider per task-typ
- Hjälper identifiera vilka tasks som är långsamma
- Trend-linjer över tid

#### Request Rate (req/s)

- Visar antal requests per sekund
- Hjälper identifiera belastning
- Trend-linjer över tid

#### Pipeline Performance Distribution

- Histogram över svarstider
- Visar fördelning av prestanda
- Hjälper identifiera outliers

### 🧪 **Testa nya paneler:**

Kör test-scriptet för att generera data:

```bash
python test_pipeline_metrics.py
```

Detta kommer att:

- Skicka 10 test-tasks med olika komplexitet
- Generera pipeline-metrics
- Uppdatera Grafana-paneler i realtid

### 🎯 **Mål för prestanda:**

- **Pipeline Response Time:** < 10 sekunder
- **Success Rate:** > 90%
- **Model Health:** 6/6 friska modeller
- **Request Rate:** Stabil under belastning

### 🔍 **Användbara Prometheus Queries:**

#### Pipeline Response Time

```promql
rate(codeconductor_response_time_seconds_sum[5m]) / rate(codeconductor_response_time_seconds_count[5m])
```

#### Request Rate

```promql
rate(codeconductor_total_requests[5m])
```

#### Model Health

```promql
sum(codeconductor_model_healthy)
```

#### Success Rate

```promql
rate(codeconductor_successful_requests[5m]) / rate(codeconductor_total_requests[5m]) * 100
```

### 🚀 **Nästa steg:**

1. **Övervaka prestanda** - Använd nya paneler för att identifiera flaskhalsar
2. **Optimera pipeline** - Justera baserat på metrics
3. **Sätt upp alerts** - Konfigurera notifikationer för kritiska värden
4. **Skapa custom dashboards** - För specifika use cases

### 📊 **Dashboard Layout:**

```
┌─────────────────┬─────────────────┐
│ Model Health    │ Total Requests  │
│ Status          │                 │
├─────────────────┼─────────────────┤
│ Success Rate    │ Uptime          │
│                 │                 │
├─────────────────┼─────────────────┤
│ Pipeline Resp.  │ Pipeline Resp.  │
│ Time (s)        │ Time by Task    │
├─────────────────┼─────────────────┤
│ Model Response  │ Request Rate    │
│ Times           │ (req/s)         │
├─────────────────┼─────────────────┤
│ Pipeline Perf.  │ System Health   │
│ Distribution    │ Overview        │
└─────────────────┴─────────────────┘
```

### 🎉 **Resultat:**

Du har nu en komplett monitoring-lösning som ger dig:

- **Realtids-insyn** i pipeline-prestanda
- **Detaljerad analys** per task-typ
- **Proaktiv övervakning** med alerts
- **Historisk data** för optimering

Allt redo för produktion! 🚀
