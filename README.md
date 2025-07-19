# 🎼 CodeConductor v2.0

[![CI](https://github.com/olablom/CodeConductor/actions/workflows/ci.yml/badge.svg)](https://github.com/olablom/CodeConductor/actions/workflows/ci.yml)

> Multi-agent AI system with Reinforcement Learning for self-improving code generation

**Status**: Week -1/0 - Production Ready! 🚀

CodeConductor orchestrates intelligent LLM agents to improve code generation through multi-agent collaboration, reinforcement learning, distributed execution, and ML analytics.

## 🚀 Quick Start

### **Option 1: Docker (Recommended)**

```bash
# Clone and run with Docker
git clone https://github.com/olablom/CodeConductor.git
cd CodeConductor
docker-compose up --build

# Access the system:
# 📊 API: http://localhost:8000
# 🎨 GUI: http://localhost:8501
```

### **Option 2: Local Development**

```bash
# Clone and setup
git clone https://github.com/olablom/CodeConductor.git
cd CodeConductor
python -m venv .venv
source .venv/bin/activate  # På Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline
python pipeline.py --prompt prompts/hello_world.md --iters 10 --mock

# View results
streamlit run app.py
```

## 📊 Architecture

```
CodeConductor/
├── agents/                 # Multi-agent system
│   ├── base_agent.py      # Abstract base class
│   ├── orchestrator.py    # Agent coordination
│   └── orchestrator_distributed.py  # Distributed version
├── bandits/               # RL algorithms
│   └── q_learning.py      # Q-learning implementation
├── plugins/               # Plugin system
│   ├── base_simple.py     # Plugin base classes
│   ├── security_plugin.py # Security analysis
│   └── formatter_plugin.py # Code formatting
├── integrations/          # External integrations
│   ├── lm_studio.py       # LM Studio integration
│   ├── celery_app.py      # Celery configuration
│   └── github_webhook.py  # GitHub webhook
├── analytics/             # ML analytics
│   ├── ml_predictor.py    # Quality prediction
│   └── dashboard.py       # Analytics dashboard
├── config/                # Configuration
│   └── base.yaml          # Main configuration
├── data/                  # Data storage
│   └── metrics.db         # SQLite database
├── tests/                 # Test suite
├── pipeline.py            # Main pipeline
└── app.py                 # Streamlit dashboard
```

## 🧪 Complete Feature Set

### **Core Features**

- [x] **Multi-Agent Discussion System**
  - CodeGenAgent: Implementation strategy analysis
  - ArchitectAgent: Design pattern analysis
  - ReviewerAgent: Code quality & security analysis
  - PolicyAgent: Security and compliance checking
  - AgentOrchestrator: Consensus coordination
- [x] **Reinforcement Learning**
  - Q-learning optimization of strategies
  - Bandit algorithms for exploration/exploitation
  - Multi-factor reward calculation
  - Learning curves and convergence tracking
- [x] **Plugin Architecture**
  - Security plugin for code analysis
  - Formatter plugin for code quality
  - Extensible plugin system
  - Automatic plugin discovery and loading
- [x] **Distributed Execution**
  - Celery + Redis for parallel processing
  - Scalable agent execution
  - Graceful fallback to local execution
  - Real-time task monitoring
- [x] **ML Analytics**
  - Quality prediction using RandomForest
  - Trend analysis over time
  - Proactive warning system
  - Feature importance analysis
- [x] **GitHub Integration**
  - Automated PR analysis
  - Webhook-based triggers
  - Comment-based feedback
  - CI/CD integration

## 🔒 Week 2 Progress - PolicyAgent & PromptOptimizer

### **PolicyAgent Security System**

- [x] **PolicyAgent Security System**
  - Dangerous system call detection (`os.system`, `subprocess`)
  - File operation validation (secret files, write operations)
  - Network access blocking (`requests`, `urllib`)
  - License violation detection (GPL, AGPL headers)
  - Forbidden import blocking (`torch`, `tensorflow`, etc.)
  - Code size limits and syntax validation
- [x] **Pipeline Integration**
  - Automatic code validation after generation
  - Negative rewards for blocked code (-20.0)
  - Database tracking of violations
  - Real-time blocking statistics
- [x] **Dashboard Enhancements**
  - Policy violation analysis tab
  - Block reasons distribution charts
  - Model source tracking (mock vs LM Studio)
  - Security metrics overview

### **PromptOptimizerAgent Q-Learning System**

- [x] **Q-Learning Agent**
  - State vector: `(task_id, arm_prev, fail_bucket, complexity_bin, model_source)`
  - 6 prompt mutation actions: type hints, OOP, docstrings, simplify, examples, no change
  - ε-greedy exploration with configurable parameters
  - Q-table persistence and analysis
- [x] **Pipeline Integration**
  - Automatic prompt optimization after failures
  - Reward bonuses for green-on-first-retry (+10)
  - Iteration penalties (-1 per additional iteration)
  - Policy block penalties (-5)
  - Complexity bonuses (+2 for good complexity)
- [x] **Dashboard Enhancements**
  - PromptOptimizer analysis tab
  - Action distribution charts
  - Optimization timeline visualization
  - Q-table statistics and action usage

## 📈 Example Usage

```bash
# Test Multi-Agent System
python test_multi_agent.py

# Test Complete System with Human Approval
python test_complete_system.py

# Run with mock generator
python pipeline.py --prompt prompts/hello_world.md --iters 20 --mock

# Run with LM Studio (if available)
python pipeline.py --prompt prompts/calculator.md --iters 20 --online

# Run benchmark suite
python bench/run_suite.py --prompt_dir prompts --iters 50 --online

# Run tests
pytest tests/ -v

# View metrics
sqlite3 data/metrics.db "SELECT * FROM metrics ORDER BY iteration DESC LIMIT 10"

# View blocked code analysis
sqlite3 data/metrics.db "SELECT iteration, block_reasons FROM metrics WHERE blocked = 1"

# View prompt optimization data
sqlite3 data/metrics.db "SELECT iteration, optimizer_action, reward FROM metrics WHERE optimizer_action != 'no_change'"

## 🔮 Next Steps (Week 4-5)

- [x] **Multi-file project support** - Complex project generation ✅
- [x] **Docker deployment** - One-click deployment ✅
- [ ] **Cursor IDE integration** - Direct IDE plugin
- [ ] **SQLite persistence for RL history** - Advanced learning storage
- [ ] **Advanced prompt optimization** - Temperature/stop-token tweaks
- [ ] **Advanced security rules** - Custom policy configuration

## 🎯 Gabriel's Vision Status

✅ **Multi-agent collaboration** - Complete with 3 specialized agents
✅ **Reinforcement Learning** - Q-learning with convergence tracking
✅ **Human-in-the-Loop** - Approval system with decision logging
✅ **Local reasoning** - LM Studio integration with privacy-first approach

## 🚀 CI/CD

This project uses GitHub Actions for continuous integration:

- **Automated Testing**: Runs pytest on every push/PR
- **Pipeline Validation**: Tests the full CodeConductor pipeline
- **Quality Checks**: Ensures code quality and functionality

## 📝 License

MIT
```
