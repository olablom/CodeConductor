# 🎼 CodeConductor

> **Self-Learning Multi-Agent AI System for Intelligent Code Generation**

[![CI](https://github.com/olablom/CodeConductor/actions/workflows/ci.yml/badge.svg)](https://github.com/olablom/CodeConductor/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-277%2F336%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-61%25-yellow.svg)](tests/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://github.com/olablom/CodeConductor/pkgs/container/codeconductor)
[![PyPI](https://img.shields.io/badge/PyPI-v2.0.0-orange.svg)](https://pypi.org/project/codeconductor/)
[![RL](https://img.shields.io/badge/RL-Q--Learning-purple.svg)](agents/qlearning_agent.py)

**CodeConductor** är ett revolutionerande AI-system som kombinerar multi-agent diskussion, mänsklig godkännande och reinforcement learning för att generera högkvalitativ kod. Systemet lär sig kontinuerligt från feedback och optimerar sig själv över tid.

## 🚀 Live Demo

![CodeConductor Demo](docs/demo.gif)

_Se CodeConductor i aktion: Multi-agent diskussion → Mänsklig godkännande → Kodgenerering → RL-feedback_

## ✨ Nyckelfunktioner

- 🤖 **Multi-Agent System** - Architect, Review, CodeGen och Policy agents arbetar tillsammans
- 👤 **Human-in-the-Loop** - Mänsklig godkännande för kritiska beslut
- 🧠 **Reinforcement Learning** - Q-learning optimerar kodgenerering över tid
- 🛡️ **Safety First** - Policy agent säkerställer säker kod
- 📊 **Comprehensive Testing** - 35+ tester med 61% kodtäckning
- 🔄 **Self-Improving** - Systemet lär sig från varje iteration

## 🏗️ Systemarkitektur

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Multi-Agent   │    │   Human-in-the- │    │   CodeGen +     │
│   Discussion    │───▶│   Loop Approval │───▶│   Safety Check  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Consensus     │    │   Approval      │    │   Review +      │
│   Building      │    │   Interface     │    │   Policy Check  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CodeGenAgent  │    │   RewardAgent   │    │   QLearningAgent│
│   Generation    │───▶│   Calculation   │───▶│   Optimization  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Installation

#### **Option 1: Docker (Rekommenderat)**

```bash
# Dra ner och kör med Docker
docker pull ghcr.io/olablom/codeconductor:latest
docker run -it --rm ghcr.io/olablom/codeconductor:latest python pipeline.py --help
```

#### **Option 2: Pip Package**

```bash
# Installera från PyPI
pip install codeconductor

# Kör pipeline
codeconductor --prompt "Create a simple API" --iters 1 --offline
```

#### **Option 3: Local Development**

```bash
# Klona repository
git clone https://github.com/olablom/CodeConductor.git
cd CodeConductor

# Skapa virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# eller
.venv\Scripts\activate     # Windows

# Installera dependencies
pip install -r requirements.txt
```

### Kör din första pipeline

```bash
# Kör en enkel API-generering
python pipeline.py --prompt prompts/simple_api.md --iters 1 --offline

# Kör med online LLM (kräver LM Studio/Ollama)
python pipeline.py --prompt prompts/simple_api.md --iters 3 --online
```

### Exempel output

```
============================================================
PIPELINE COMPLETED
============================================================
Total iterations: 3
Successful iterations: 3
Results saved to: data/generated/pipeline_results_20250720_183003.json

Generated files:
  - data/generated/iter_1.py
  - data/generated/iter_2.py
  - data/generated/iter_3.py
============================================================
```

## 📁 Projektstruktur

```
CodeConductor/
├── agents/                 # AI-agenter
│   ├── base_agent.py      # Basklass för alla agenter
│   ├── architect_agent.py # Arkitektur-design
│   ├── review_agent.py    # Kodgranskning
│   ├── codegen_agent.py   # Kodgenerering
│   ├── policy_agent.py    # Säkerhetskontroll
│   ├── reward_agent.py    # Reward calculation
│   └── qlearning_agent.py # Q-learning optimization
├── cli/                   # Command-line interface
│   └── human_approval.py  # Mänsklig godkännande CLI
├── integrations/          # Externa tjänster
│   └── llm_client.py     # LLM integration
├── tests/                 # Test suite
│   ├── test_*.py         # Unit tester
│   └── conftest.py       # Test configuration
├── prompts/              # Prompt templates
├── data/                 # Generated code & metrics
│   ├── generated/        # Output files
│   ├── qtable.db        # Q-learning database
│   └── metrics.db       # Performance metrics
├── pipeline.py           # Main pipeline
└── requirements.txt      # Dependencies
```

## 🧠 AI-Agenter

### ArchitectAgent

- **Syfte**: Designar systemarkitektur och teknisk stack
- **Input**: Projektkrav och kontext
- **Output**: Arkitekturförslag med teknisk motivering

### ReviewAgent

- **Syfte**: Granskar kod för kvalitet och säkerhet
- **Input**: Genererad kod
- **Output**: Kvalitetsbedömning och förbättringsförslag

### CodeGenAgent

- **Syfte**: Genererar faktisk kod baserat på arkitektur
- **Input**: Arkitekturförslag och kontext
- **Output**: Komplett kodimplementation

### PolicyAgent

- **Syfte**: Säkerhetskontroll och policy-enforcement
- **Input**: Genererad kod
- **Output**: Säkerhetsbedömning och violations

### RewardAgent

- **Syfte**: Beräknar rewards för RL-system
- **Input**: Test results, quality metrics, human feedback
- **Output**: Comprehensive reward score

### QLearningAgent

- **Syfte**: Optimering genom reinforcement learning
- **Input**: States, actions, rewards
- **Output**: Optimal action selection

## 🔧 Konfiguration

### Environment Variables

```bash
# LLM Configuration
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=your-api-key

# Pipeline Configuration
PIPELINE_MAX_ITERATIONS=5
PIPELINE_REWARD_THRESHOLD=0.7
```

### Custom Prompts

Skapa egna prompts i `prompts/` mappen:

```markdown
# My Custom API

Create a REST API for managing books with the following features:

- CRUD operations for books
- Author management
- Category filtering
- Search functionality
```

## 🧪 Testing

```bash
# Kör alla tester
pytest

# Kör specifika tester
pytest tests/test_reward_agent.py -v
pytest tests/test_qlearning_agent.py -v

# Med coverage
pytest --cov=agents --cov-report=html
```

## 📊 Performance Metrics

- **Test Coverage**: 61%
- **Total Tests**: 35/35 passing
- **Average Reward**: 0.867 (good)
- **Q-table Entries**: 1+ (growing with usage)
- **Pipeline Success Rate**: 100%

## 🤝 Bidrag

Vi välkomnar bidrag! Se [CONTRIBUTING.md](CONTRIBUTING.md) för detaljer.

### Utvecklingsmiljö

```bash
# Setup development environment
git clone https://github.com/olablom/CodeConductor.git
cd CodeConductor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest -v

# Run linting
flake8 agents/ tests/
```

## 📚 Dokumentation

- [Getting Started Guide](GETTING_STARTED.md) - Detaljerad onboarding
- [Plugin Guide](PLUGIN_GUIDE.md) - Skapa egna agenter
- [API Documentation](docs/api.md) - Teknisk dokumentation
- [Architecture Guide](docs/architecture.md) - Systemdesign

## 🎯 Användningsfall

### 1. API Development

```bash
python pipeline.py --prompt prompts/rest_api.md --iters 3
```

### 2. Web Application

```bash
python pipeline.py --prompt prompts/web_app.md --iters 2
```

### 3. Data Processing

```bash
python pipeline.py --prompt prompts/data_pipeline.md --iters 1
```

## 🏆 Roadmap

- [x] **v1.0** - Multi-agent system
- [x] **v2.0** - Reinforcement learning
- [ ] **v3.0** - Plugin architecture
- [ ] **v4.0** - Cloud deployment
- [ ] **v5.0** - Enterprise features

## 📄 Licens

Detta projekt är licensierat under MIT License - se [LICENSE](LICENSE) filen för detaljer.

## 🙏 Acknowledgments

- **Multi-Agent Systems** - Inspiration från moderna AI-arkitekturer
- **Reinforcement Learning** - Q-learning implementation
- **Human-in-the-Loop** - Mänsklig centrerad AI-design
- **Open Source Community** - Alla som bidragit till detta projekt

---

**Byggt med ❤️ för framtidens AI-utveckling**

[![GitHub stars](https://img.shields.io/github/stars/olablom/CodeConductor?style=social)](https://github.com/olablom/CodeConductor)
[![GitHub forks](https://img.shields.io/github/forks/olablom/CodeConductor?style=social)](https://github.com/olablom/CodeConductor)
[![GitHub issues](https://img.shields.io/github/issues/olablom/CodeConductor)](https://github.com/olablom/CodeConductor/issues)
