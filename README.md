# 🎼 CodeConductor

[![CI](https://github.com/olablom/CodeConductor/actions/workflows/ci.yml/badge.svg)](https://github.com/olablom/CodeConductor/actions/workflows/ci.yml)

> AI-driven code generation with Reinforcement Learning

CodeConductor uses Multi-Armed Bandits to learn optimal code generation strategies.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/olablom/CodeConductor.git
cd codeconductor
python -m venv .venv
source .venv/bin/activate  # På Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline
python pipeline.py --prompt prompts/hello_world.md --iters 10 --mock

# View results
streamlit run dashboard/app.py
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

## 🧪 Week 1 Progress

- [x] LinUCB Bandit implementation
- [x] Mock Cursor CLI integration
- [x] Automated testing with pytest
- [x] Complexity metrics with radon
- [x] Real-time dashboard
- [x] SQLite metrics storage
- [x] LM Studio integration with fallback
- [x] Multi-prompt support

## 🔒 Week 2 Progress - PolicyAgent

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

## 📈 Example Usage

```bash
# Run with mock generator
python pipeline.py --prompt prompts/hello_world.md --iters 20 --mock

# Run with LM Studio (if available)
python pipeline.py --prompt prompts/calculator.md --iters 20 --online

# Run tests
pytest tests/ -v

# View metrics
sqlite3 data/metrics.db "SELECT * FROM metrics ORDER BY iteration DESC LIMIT 10"

# View blocked code analysis
sqlite3 data/metrics.db "SELECT iteration, block_reasons FROM metrics WHERE blocked = 1"
```

## 🔮 Next Steps

- [ ] **PromptOptimizerAgent** - RL-based prompt optimization
- [ ] **Human feedback loop** - Manual code quality ratings
- [ ] **Multi-file project support** - Complex project generation
- [ ] **Advanced security rules** - Custom policy configuration

## 🚀 CI/CD

This project uses GitHub Actions for continuous integration:

- **Automated Testing**: Runs pytest on every push/PR
- **Pipeline Validation**: Tests the full CodeConductor pipeline
- **Quality Checks**: Ensures code quality and functionality

## 📝 License

MIT
