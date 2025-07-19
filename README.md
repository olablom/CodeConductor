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

## 📈 Example Usage

```python
# Run with custom prompt
python pipeline.py --prompt prompts/my_prompt.md --iters 20

# Run tests
pytest tests/ -v

# View metrics
sqlite3 data/metrics.db "SELECT * FROM metrics ORDER BY iteration DESC LIMIT 10"
```

## 🔮 Next Steps

- [ ] Real Cursor API integration
- [ ] Policy agent for security checks
- [ ] Human feedback loop
- [ ] Multi-file project support

## 🚀 CI/CD

This project uses GitHub Actions for continuous integration:

- **Automated Testing**: Runs pytest on every push/PR
- **Pipeline Validation**: Tests the full CodeConductor pipeline
- **Quality Checks**: Ensures code quality and functionality

## 📝 License

MIT
