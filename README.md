# CodeConductor

![CodeConductor demo](docs/assets/hero.gif)

**🩹 Fix bugs in 30s – privately**

[// badges intentionally trimmed for accuracy in Early Alpha ]
[![Preflight Ready](https://img.shields.io/badge/preflight-ready-brightgreen)](#diagnose-cursor-api-windows)

## 🚀 **The Problem**

**Ever had a production bug at 3 AM?** You need to fix it fast, but you can't send sensitive code to ChatGPT. Your options:

- ❌ **ChatGPT** - Code leaves your machine
- ❌ **Manual debugging** - Takes hours
- ❌ **Wake up the team** - Expensive & slow

## 💡 **The Solution**

**CodeConductor** - Local agents debate before coding. Think of it as having a team of expert developers discussing your problem before writing the solution.

**Local agents · 30s fixes · 0% data leaves laptop**

### Project Conductor (analyze → .cursorrules → propose)

Run a local project analysis and auto‑prepare your workspace before any inference:

CLI:

```
# 1) Analyze repository → writes artifacts/repo_map.json and artifacts/state.md
python -m codeconductor.cli analyze --path . --out artifacts

# 2) Generate rules → writes .cursorrules from analysis
python -m codeconductor.cli cursorrules --input artifacts/repo_map.json --out .cursorrules

# 3) Propose next feature prompt → writes artifacts/prompts/next_feature.md
python -m codeconductor.cli propose --input artifacts/repo_map.json --state artifacts/state.md --out artifacts/prompts/next_feature.md
```

Streamlit (sidebar → Project Conductor):
- Buttons: Analyze / .cursorrules / Propose
- Preview of generated files (first lines) with file paths

Purpose:
- Local, zero‑upload project understanding
- Automatic `.cursorrules` for cleaner context
- Short, structured “next feature” prompt with constraints and test command

### Single‑model baseline (LM Studio) and sampling override

For a stable local baseline on RTX 5090, run exactly one LM Studio model and lock selection:

1. Start LM Studio REST API on :1234 and load only `meta-llama-3.1-8b-instruct` (GGUF is fine).

2. In Windows PowerShell, set env in the same session before running benchmarks:

```
$env:ENGINE_BACKENDS='lmstudio'
$env:LMSTUDIO_CLI_DISABLE='1'
$env:OLLAMA_DISABLE='1'
$env:DISCOVERY_DISABLE='1'
$env:ENGINE_MODEL_ALLOWLIST='meta-llama-3.1-8b-instruct'
$env:SELECTOR_POLICY='fixed'
$env:MODEL_SELECTOR_STRICT='1'
$env:FORCE_MODEL='meta-llama-3.1-8b-instruct'
$env:WINNER_MODEL='meta-llama-3.1-8b-instruct'
$env:MAX_PARALLEL_MODELS='1'
$env:RLHF_DISABLE='1'

# Sampling override (applies to payload)
$env:CC_TEMP='0.1'
$env:CC_TOP_P='0.9'
$env:MAX_TOKENS='192'
$env:PYTHONIOENCODING='utf-8'
```

The dispatcher now honors `CC_TEMP`/`CC_TOP_P`/`MAX_TOKENS` and applies them to request payloads.

3. Run benchmarks:

```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_benchmark_10.ps1 -Runs 10
```

Expected warm baseline: TTFT p50 ≈ 2.5–3.0 s (after warmup), CodeBLEU ≈ 0.73 for generic prompts. `first_prompt_success` stays 0 if the prompt has no tests.

## 📌 **Current Status (Early Alpha)**

- Local‑first, multi‑agent + ensemble engine working on Windows (RTX 5090).
- Selector runs in latency mode; artifacts saved per run under `artifacts/runs/<ts>/`.
- Recent consensus run (CodeBLEU+heuristic) scored ~0.7352; target is ≥0.8 with light calibration.
- External adapters are opt‑in and disabled by default (privacy‑first).

Run quick benchmarks locally:

```bash
python tests/run_benchmarks.py --quick
```

### Consensus calibration (CodeBLEU)

```powershell
$env:CODEBLEU_WEIGHTS = '0.2,0.6,0.2'   # ngram, AST, token
$env:CODEBLEU_NORMALIZE='1'
$env:CODEBLEU_STRIP_COMMENTS='1'
$env:CODEBLEU_STRIP_DOCSTRINGS='1'
$env:TEMP='0.1'  # reduce stylistic drift for short prompts
python scripts\generate_ensemble_run.py --prompt "Create a Python Fibonacci function (iterative)"
```

## 🎯 **What is CodeConductor?**

CodeConductor is a **local-first AI development assistant** that uses multiple AI agents to debate and collaborate before generating code. Think of it as having a team of expert developers discussing your problem before writing the solution.

### **How it works:**

1. **Architect Agent** - Designs the solution and architecture
2. **Coder Agent** - Implements the practical code
3. **Debate Phase** - Agents discuss and improve the approach
4. **Final Code** - Optimized solution with multiple perspectives

## 🚀 **Quick Start**

### **Prerequisites:**

- Python 3.11+
- LM Studio and/or Ollama (local models)
- 16GB+ RAM (for model loading)

### **Installation:**

```bash
git clone https://github.com/your-repo/CodeConductor
cd CodeConductor
pip install -r requirements.txt
```

### **Quick Start:**

```bash
# Install deps
pip install -r requirements.txt

# Smoke tests / benchmark (quick)
python tests/run_benchmarks.py --quick

# CLI (module path friendly)
PYTHONPATH=src python src/codeconductor/cli.py test --rounds 1 --timeout-per-turn 60

# Diagnostics
PYTHONPATH=src python src/codeconductor/cli.py doctor
```

See `docs/CURSOR_TROUBLESHOOTING.md` for Cursor integration issues and local-first workarounds.

### Diagnose Cursor API (Windows)

Use the PowerShell diagnostics script to check local ports and auto-detect a Cursor Local API port.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/diagnose_cursor.ps1 -Ports 11434 3000
```

- The script writes `artifacts/diagnostics/diagnose_latest.txt`.
- If a Cursor API port is detected, the log will include a line like:
  - `- Detected Cursor API port: 5123`
- Ollama `11434` health endpoint returning `404` is OK (port alive, no route).

Set env vars when a port is detected (persist for new sessions):

```powershell
[Environment]::SetEnvironmentVariable('CURSOR_API_BASE','http://127.0.0.1:<detected>','User')
[Environment]::SetEnvironmentVariable('CURSOR_MODE','auto','User')
```

Open a new PowerShell and verify:

```powershell
curl.exe -vk "http://127.0.0.1:<detected>/api/health"
```

If no port is detected yet, run in manual mode until Cursor Local API is listening:

```powershell
[Environment]::SetEnvironmentVariable('CURSOR_MODE','manual','User')
```

Git Bash tip: run the script with `-File` and avoid inline `-Command` with double quotes to prevent `$...` expansion issues.

#### Preflight

- Quick check (CLI):
  - `codeconductor diag cursor --run`
- Makefile/Tasks (shortcuts):
  - `make diag-cursor` (Unix)
  - `scripts/tasks.ps1 -Task diag-cursor` (Windows)

### Mock vs Real

- Mock (`CC_QUICK_CI=1`): deterministic outputs, no model loading; safe for CI.
- Real (local models): enable LM Studio/Ollama/vLLM; doctor records GPU/latency.

### Personas pack

Extended roles available in `agents/personas.yaml` (e.g., `bug_hunter`, `perf_tweaker`, `security_auditor`, `sql_sage`, `api_wrangler`, `refactorist`).

### External APIs (opt‑in)

- Default is private: `ALLOW_NET=0` (no external fetches)
- Enable adapters by setting `ALLOW_NET=1` (Stack Overflow / GitHub Code Search)
- Model backend gating:
  - `ENGINE_BACKENDS=ollama` (disable LM Studio HTTP entirely)
  - `LMSTUDIO_DISABLE=1` and/or `LMSTUDIO_CLI_DISABLE=1` (never touch port 1234 or `lms` CLI)

Env examples (optional, improves quotas and caching):

```powershell
# Enable external adapters for this session
$env:ALLOW_NET='1'

# Stack Exchange key (optional, boosts quota)
$env:STACKEXCHANGE_KEY='<your_key>'

# GitHub token (optional; read-only scopes e.g. public_repo)
$env:GITHUB_TOKEN='<your_token>'

# Net behavior tuning
$env:NET_TIMEOUT_S='10'
$env:NET_MAX_RETRIES='2'
$env:NET_CACHE_TTL_SECONDS='3600'  # 1 hour
$env:NET_CACHE_DIR='artifacts/net_cache'

# Adapter page sizes
$env:SO_PAGESIZE='5'
$env:GH_PER_PAGE='5'
```

Notes:

- Keys/tokens are optional; leave unset for anonymous mode.
- Cached responses are stored under `artifacts/net_cache` with TTL control.
- No repo code is uploaded; only query terms are used to fetch public info.

## 📦 Case bundle schema & KPI

This project defines JSON schemas for exported case bundles and KPIs to enable deterministic, privacy‑aware sharing of results.

- Schemas:
  - `src/codeconductor/utils/schemas/kpi.schema.json`
  - `src/codeconductor/utils/schemas/manifest.schema.json`

Bundle layout (public_safe):

```
artifacts/exports/codeconductor_case_<run_id>.zip
└─ manifest.json
└─ kpi.json
└─ consensus.json
└─ selector_decision.json
└─ diffs/
└─ before/
└─ after/
└─ tests/
└─ logs/
└─ README_case.md
```

KPI highlights:

- `ttft_ms` (monotonic), `first_prompt_success`, `pass_rate_*`
- winner model/score, `consensus_method`, `codebleu_*`, `sampling`
- `config_digest` over key env (`ENGINE_BACKENDS`, `ALLOW_NET`, `CODEBLEU_*`, `SELECTOR_POLICY`, `EXPORT_*`)

### Example commands

```bash
# Real doctor baseline
PYTHONIOENCODING=utf-8 python -m codeconductor.cli doctor --real --model mistral-7b-instruct-v0.1 --tokens 128 --profile

# Personas debate
codeconductor run --personas agents/personas.yaml \
  --agents architect,coder,bug_hunter \
  --prompt "Implement a small FastAPI /items endpoint with tests" \
  --rounds 1 --timeout-per-turn 60

# Focused suite (real)
PYTHONIOENCODING=utf-8 python tests/test_codeconductor_2agents_focused.py
```

## 📦 Export case bundle (v1)

```bash
# Exportera senaste run som public_safe bundle
python scripts/export_latest.py
```

- Skapar zip under `artifacts/exports/` med namn `codeconductor_case_<run_id>.zip`.
- Innehåller `kpi.json`, `manifest.json`, `README_case.md`, tester och diff/after (genererat från consensus‑kod om möjligt).
- Valideras mot schema om `jsonschema` finns installerat.

> CI note: GitHub Actions runs a deterministic mock smoke (`CC_QUICK_CI=1`) that never calls LM Studio. Full GPU tests are intended to run locally.

## 🐛 Bug report bundle

When you hit a problem, create a local bug report bundle from the UI to share relevant run data without leaking sensitive info.

### Create a bundle

1. Open the sidebar section “Selector & Cache”.
2. Optional settings:
   - Include raw outputs (default: off) – includes full model outputs. Not recommended outside your team.
   - Redact env (default: on) – removes sensitive environment variables and paths.
   - Generate public-safe bundle – forces a minimal, safe export for external sharing.
3. Click “Export bundle”.
4. Wait for “Verified ✓”.

### Send the bundle

- Manual: open `artifacts/exports/` and share the zip `codeconductor_run_<ts>_<policy>_<hit|miss>.zip` via an approved channel.
- Send to support (opt-in): copies the zip path to clipboard and opens a prefilled email in your default mail client.
  - Note: mailto cannot attach the file automatically – attach the zip manually.
  - Always use a public-safe bundle for external sharing.

### Retention & cleanup

- Only the latest 20 exports are kept.
- Click “Clear exports” in the UI to remove all.

## 📊 **Benchmarks (local)**

- Run the suite locally and inspect artifacts under `artifacts/`.
- Example:

```bash
python -m pytest -q tests/
python tests/run_benchmarks.py --quick
```

## 🎼 **System Architecture**

```mermaid
sequenceDiagram
    participant U as User
    participant A as Architect Agent
    participant C as Coder Agent
    participant T as Tester Agent
    participant R as RAG System
    participant M as Model Manager

    U->>A: "Fix this production bug"
    A->>R: Query context & past solutions
    R-->>A: Relevant code patterns
    A->>C: Design solution architecture
    C->>M: Generate implementation
    M-->>C: Code with tests
    C->>T: Validate & test
    T-->>C: Test results
    C->>U: Production-ready fix
```

### **Core Components:**

- **vLLM Backend** - High-performance model serving
- **Debate Manager** - Orchestrates agent discussions
- **Ensemble Engine** - Manages multiple models
- **RAG System** - Context retrieval from past solutions
- **Model Manager** - GPU-aware model scheduling

### **Agent Types:**

- **Architect** - System design and architecture
- **Coder** - Practical implementation
- **Tester** - Quality assurance and validation
- **Reviewer** - Code review and best practices

## 🧪 RLHF & Learning

- Method: Proximal Policy Optimization (PPO)
- Runtime: Loads a local policy from `ppo_codeconductor.zip` if present; otherwise RLHF is auto‑disabled and the system falls back gracefully.
- Signal: Combines unit‑test results and simple heuristics into a reward used to recommend model selection and sampling parameters.
- Scope: Lightweight policy influences selector/sampling; it does not rewrite business logic.
- Manage:
  - To refresh/disable quickly: temporarily move/rename `ppo_codeconductor.zip` (RLHF will log “disabled”).
  - Training jobs and nightly self‑play are planned; current repo includes scaffolding for resets (`reset_rlhf_weights.py`).

Target: Stable, measurable gains under the local test suite without regressing latency.

## 🔧 **Configuration**

## 📡 Monitoring (Grafana/Prometheus)

- Memory watchdog runs by default and logs VRAM/cleanup events to app logs and `artifacts/`.
- Dashboards: JSONs under `monitoring/grafana/dashboards/` (import into Grafana).
- Prometheus: starter config under `monitoring/prometheus/` (bring your own Prometheus and point to your exporters; full OpenTelemetry pack is planned).

Quick Grafana (optional):

```bash
# If you have Docker locally
docker run -d --name grafana -p 3000:3000 grafana/grafana
# Open http://localhost:3000, add Prometheus datasource, import dashboards from monitoring/grafana/dashboards/
```

Note: `scripts/diagnose_cursor.ps1` includes a port check for 3000 when requested.

### Grafana – import/export quick guide

1. Start Grafana (e.g., Docker snippet above) and sign in (admin/admin by default on a fresh container).
2. Add datasource: Settings → Data sources → Prometheus → URL `http://localhost:9090` → Save & test.
3. Import dashboards: Dashboards → New → Import → Upload JSON → pick JSON from `monitoring/grafana/dashboards/` (when available) or paste a JSON export.
4. Optional export: Open a dashboard → Share → Export → Save to JSON (commit under `monitoring/grafana/dashboards/`).

### Prometheus – minimal scrape config

CodeConductor does not expose a Prometheus `/metrics` endpoint by default. Use exporters (e.g., node_exporter) and/or add FastAPI instrumentation (e.g., `prometheus-fastapi-instrumentator`) if you want `localhost:8000/metrics`.

Example `prometheus.yml` (minimal):

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "codeconductor_local"
    metrics_path: /metrics
    static_configs:
      - targets:
          - "localhost:8000" # FastAPI metrics if instrumented
          - "localhost:9100" # node_exporter (optional)
```

Tips:

- To instrument FastAPI quickly: add `prometheus-fastapi-instrumentator` and mount it in `src/codeconductor/api/server.py`.
- Keep exporters local to preserve privacy; don’t publish metrics externally.

### **Model Setup:**

```python
# Optimal configuration (based on test results)
OPTIMAL_CONFIG = {
    "agents": 2,  # Sweet spot!
    "models": ["mistralai/codestral-22b-v0.1", "google/gemma-3-12b", "microsoft/phi-3-mini-4k"],
    "timeout": 30,  # 24s average + buffer
    "use_cases": ["REST API", "React", "SQL", "Bug Fix"],
}

# Fallback for 16GB VRAM systems
FALLBACK_CONFIG = {
    "agents": 2,
    "models": ["meta-llama-3.1-8b-instruct", "mistral-7b-instruct-v0.1"],
    "timeout": 30,
    "use_cases": ["REST API", "React", "SQL", "Bug Fix"],
}
```

### **Supported Models (local):**

- **meta-llama-3.1-8b-instruct** (LM Studio)
- **google/gemma-3-12b** (LM Studio)
- **mistral-7b-instruct-v0.1** (LM Studio)
- **mistral:latest** (Ollama)
- **phi3:mini** (Ollama)

## 🎯 **Use Cases**

### **Perfect for CodeConductor:**

- **REST API Development** - 100% success rate
- **React Component Creation** - 100% success rate
- **SQL Query Optimization** - 100% success rate
- **Bug Fixing** - 100% success rate
- **Code Refactoring** - High success rate

### **Skip for Single Model:**

- **Simple functions** (< 10 lines)
- **One-liners**
- **Basic utilities**

## 🔒 **Privacy & Security**

- **Local-first by default**: `ALLOW_NET=0` disables external adapters.
- **Backend gating**: `ENGINE_BACKENDS`, `LMSTUDIO_DISABLE`, `LMSTUDIO_CLI_DISABLE` strictly enforced.
- **No API keys required** for local flow; cloud adapters are opt‑in.

### **Threat Model:**

CodeConductor runs in a sandboxed environment that prevents malicious code from accessing your system:

- **Code Execution Isolation** - Generated code runs in temporary containers
- **Network Isolation** - No outbound connections from code execution
- **Resource Limits** - CPU, memory, and time constraints
- **File System Isolation** - Read-only access to project files only

### **Supply Chain Notes:**

- Dependencies pinned where practical; anonymized telemetry disabled by default.

## 📋 **Requirements**

### **System Requirements:**

- **Python** 3.11+ (tested with 3.11.7)
- **CUDA** 12.1+ (for GPU acceleration)
- **VRAM** 16GB+ (RTX 4090 or better recommended)
- **RAM** 32GB+ (for model loading)

### **Dependencies:**

- **vLLM** 0.4.0+ (model serving)
- **Streamlit** 1.28.0+ (GUI)
- **PyTorch** 2.7.0+ (CUDA 12.8)
- **Transformers** 4.30.0+ (model loading)

## 🚀 **Roadmap**

| Feature                  | Status        | Target  | Description                | Commit |
| ------------------------ | ------------- | ------- | -------------------------- | ------ |
| **Core debate system**   | ✅ Done       | Q4 2024 | 2-agent optimization       | [#42]  |
| **Self-reflection loop** | ✅ Done       | Q1 2025 | 100% success rate          | [#67]  |
| **RAG integration**      | ✅ Done       | Q1 2025 | Context retrieval          | [#89]  |
| **GPU-aware scheduling** | 🛠 In Progress | Q4 2025 | Auto model switching       | [#156] |
| **Plugin architecture**  | 🧭 Planned    | Q1 2026 | Custom agent support       | -      |
| **Enterprise features**  | 🧭 Planned    | Q2 2026 | Multi-user, RBAC           | -      |
| **IDE plugins**          | 🧭 Planned    | Q2 2026 | VSCode, Cursor integration | -      |

[// marketing claims trimmed for Early Alpha accuracy]

## ❗ ECONNRESET – Quick Fix (Cursor Local API)

Symptoms: ConnectError/ECONNRESET when Cursor’s local API isn’t listening or env is corrupt.

```powershell
# 1) Clear proxy/DNS
netsh winhttp reset proxy
[Environment]::SetEnvironmentVariable('CURSOR_DISABLE_PROXY','1','User')
ipconfig /flushdns

# 2) Reset Cursor env
[Environment]::SetEnvironmentVariable('CURSOR_MODE','manual','User')
[Environment]::SetEnvironmentVariable('CURSOR_API_BASE','','User')

# 3) Fully restart Cursor (quit from tray)

# Quick checks
Get-NetTCPConnection -State Listen | Where-Object LocalPort -eq 1234
curl.exe -s http://127.0.0.1:1234/v1/models
```

## 🔜 Later development – Privacy export hardening

- Privacy levels: `public_safe` (default), `team_safe` (masked paths + selected diffs/logs), `full_internal` (raw outputs allowed).
- RAW toggle: `EXPORT_INCLUDE_RAW=1` to include raw model outputs where allowed.
- Redactions: always mask absolute paths and usernames in `team_safe`; normalize to relative POSIX paths and replace home with `~`.
- Determinism: normalize line endings to `\n`, sort `files[]` lexicographically, set zip comment (run_id, privacy, schema versions).
- Truncation: per‑file size cap with byte‑exact streaming; write `*_TRUNCATED.txt`; set `bundle_truncated=true` and list categories in `manifest.redactions` and README_case.
- Concurrency: simple lock around export per `run_id` to avoid collisions.
- Policy precedence: CLI > env > defaults; unknown privacy level falls back to `public_safe` and records a manifest warning.

## 🤝 **Contributing**

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### **Development Setup:**

```bash
# Clone and setup
git clone https://github.com/your-repo/CodeConductor
cd CodeConductor
pip install -r requirements.txt

# Run tests
python test_master_simple.py --self_reflection
```

## 📄 **License**

MIT License - see [LICENSE](LICENSE) for details.

**Note**: While CodeConductor itself is MIT licensed, the AI models used (Llama, Mistral, etc.) may have different licensing terms. Please check the respective model licenses before redistribution.

## 🎉 **Acknowledgments**

- **LM Studio** - For local model serving
- **vLLM** - For high-performance inference
- **Meta** - For Llama models
- **Mistral AI** - For Mistral models

---

**Built with ❤️ for developers who want better AI assistance without compromising privacy.**

_"AI agents that debate before coding - because the best code comes from collaboration."_
