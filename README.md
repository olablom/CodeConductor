# 🎼 CodeConductor v2.0

[![CI](https://github.com/olablom/CodeConductor/actions/workflows/ci.yml/badge.svg)](https://github.com/olablom/CodeConductor/actions/workflows/ci.yml)

> Multi-agent AI system with Reinforcement Learning for self-improving code generation

**Status**: Week 3/10 - Gabriel's Vision Complete ✅

CodeConductor orchestrates intelligent LLM agents to improve code generation through multi-agent collaboration and reinforcement learning.

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
codeconductor/
├── agents/       # Base agent classes
├── bandits/      # LinUCB implementation
├── prompts/      # Input prompts
├── dashboard/    # Streamlit UI
└── pipeline.py   # Main orchestrator
```

## 🧪 Week 1-3 Progress

### **Week 1: Foundation**

- [x] LinUCB Bandit implementation
- [x] Mock Cursor CLI integration
- [x] Automated testing with pytest
- [x] Complexity metrics with radon
- [x] Real-time dashboard
- [x] SQLite metrics storage
- [x] LM Studio integration with fallback
- [x] Multi-prompt support

### **Week 2: RL & Security**

- [x] PolicyAgent Security System
- [x] PromptOptimizerAgent Q-Learning
- [x] Multi-factor reward calculation
- [x] Learning curves and convergence

### **Week 3: Gabriel's Vision - Multi-Agent + Human Control**

- [x] **Multi-Agent Discussion System**
  - CodeGenAgent: Implementation strategy analysis
  - ArchitectAgent: Design pattern analysis
  - ReviewerAgent: Code quality & security analysis
  - AgentOrchestrator: Consensus coordination
- [x] **Human-in-the-Loop Approval**
  - HumanGate: Approval interface
  - Decision logging and statistics
  - Edit and explain capabilities
- [x] **Complete Integration**
  - RL optimization of consensus
  - Real-time multi-agent coordination
  - Human approval workflow

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
