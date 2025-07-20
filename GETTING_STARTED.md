# 🚀 Getting Started with CodeConductor

Välkommen till CodeConductor! Denna guide hjälper dig att komma igång med det självlärande AI-systemet för intelligent kodgenerering.

## 📋 Förkrav

### Systemkrav

- **Python**: 3.11 eller senare
- **RAM**: Minst 4GB (8GB rekommenderat)
- **Diskutrymme**: 2GB ledigt utrymme
- **OS**: Windows 10+, macOS 10.15+, eller Linux

### Valfria (för online LLM)

- **LM Studio**: För lokal LLM-körning
- **Ollama**: Alternativ för lokal LLM
- **Docker**: För containeriserad körning

## 🔧 Installation

### Steg 1: Klona Repository

```bash
git clone https://github.com/olablom/CodeConductor.git
cd CodeConductor
```

### Steg 2: Skapa Virtual Environment

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Steg 3: Installera Dependencies

```bash
pip install -r requirements.txt
```

### Steg 4: Verifiera Installation

```bash
python -c "import agents; print('✅ CodeConductor installerat framgångsrikt!')"
```

## 🎯 Din Första Pipeline

### Snabbstart (Offline Mode)

```bash
# Kör en enkel API-generering
python pipeline.py --prompt prompts/simple_api.md --iters 1 --offline
```

**Vad händer:**

1. 🤖 **Multi-agent diskussion** - Agenterna diskuterar arkitektur
2. 👤 **Mänsklig godkännande** - Du godkänner förslaget
3. 💻 **Kodgenerering** - CodeGenAgent skapar koden
4. 🛡️ **Säkerhetskontroll** - PolicyAgent kontrollerar säkerhet
5. 🧠 **RL-feedback** - Systemet lär sig från resultatet

### Exempel Output

```
============================================================
PIPELINE COMPLETED
============================================================
Total iterations: 1
Successful iterations: 1
Results saved to: data/generated/pipeline_results_20250720_183003.json

Generated files:
  - data/generated/iter_1.py
============================================================
```

## 🔍 Vad Händer Under Huven?

### Fas 1: Multi-Agent Diskussion

```
┌─────────────────┐
│ ArchitectAgent  │ → Designar systemarkitektur
├─────────────────┤
│ ReviewAgent     │ → Granskar kvalitet och säkerhet
├─────────────────┤
│ CodeGenAgent    │ → Genererar kodimplementation
└─────────────────┘
```

### Fas 2: Mänsklig Godkännande

```
┌─────────────────┐
│ Human Approval  │ → Du godkänner eller redigerar
│ Interface       │ → Interaktiv CLI
└─────────────────┘
```

### Fas 3: Kodgenerering & Säkerhet

```
┌─────────────────┐
│ CodeGenAgent    │ → Skapar faktisk kod
├─────────────────┤
│ PolicyAgent     │ → Säkerhetskontroll
├─────────────────┤
│ ReviewAgent     │ → Kvalitetsgranskning
└─────────────────┘
```

### Fas 4: Reinforcement Learning

```
┌─────────────────┐
│ RewardAgent     │ → Beräknar reward
├─────────────────┤
│ QLearningAgent  │ → Uppdaterar Q-table
└─────────────────┘
```

## 🎮 Interaktiv Demo

### Starta Live Demo

```bash
# Kör en komplett demo med 3 iterationer
python pipeline.py --prompt prompts/simple_api.md --iters 3 --offline
```

**Under körningen kommer du att se:**

1. **Agent-diskussion**:

   ```
   🤖 ArchitectAgent: "Jag rekommenderar en modulär arkitektur..."
   🤖 ReviewAgent: "Koden bör följa PEP8-standarder..."
   🤖 CodeGenAgent: "Jag kommer att implementera FastAPI..."
   ```

2. **Godkännande-gränssnitt**:

   ```
   ================================================================================
   🤖 AGENT CONSENSUS PROPOSAL
   ================================================================================

   🤔 What would you like to do?
      [A] Approve - Accept the proposal and continue
      [R] Reject - Reject the proposal and stop
      [E] Edit - Modify the proposal
      [H] Help - Show detailed help
      [Q] Quit - Exit without decision
   ```

3. **Resultat och feedback**:
   ```
   ✅ Proposal APPROVED!
   🧠 Reward calculated: 0.867 (good)
   📊 Q-table updated with new learning
   ```

## 📁 Utforska Resultaten

### Genererad Kod

```bash
# Visa genererad kod
cat data/generated/iter_1.py
```

### Pipeline Resultat

```bash
# Visa detaljerade resultat
cat data/generated/pipeline_results_*.json
```

### Q-Learning Data

```bash
# Visa Q-table statistik
python -c "
import sqlite3
conn = sqlite3.connect('data/qtable.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) as entries, AVG(q_value) as avg_q FROM q_table')
result = cursor.fetchone()
print(f'Q-table: {result[0]} entries, avg Q-value: {result[1]:.4f}')
conn.close()
"
```

## 🔧 Konfiguration

### Environment Variables

Skapa en `.env` fil i projektroten:

```bash
# LLM Configuration
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=your-api-key

# Pipeline Configuration
PIPELINE_MAX_ITERATIONS=5
PIPELINE_REWARD_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
```

### Online Mode (med LM Studio)

1. **Installera LM Studio**: https://lmstudio.ai/
2. **Ladda ner en modell** (t.ex. Llama 3.1 8B)
3. **Starta servern**:
   ```bash
   # I LM Studio, starta servern på localhost:1234
   ```
4. **Kör med online mode**:
   ```bash
   python pipeline.py --prompt prompts/simple_api.md --iters 2 --online
   ```

## 🧪 Testing

### Kör Tester

```bash
# Alla tester
pytest

# Specifika tester
pytest tests/test_reward_agent.py -v
pytest tests/test_qlearning_agent.py -v

# Med coverage
pytest --cov=agents --cov-report=html
```

### Verifiera Installation

```bash
# Testa alla komponenter
python -c "
from agents.architect_agent import ArchitectAgent
from agents.review_agent import ReviewAgent
from agents.codegen_agent import CodeGenAgent
from agents.policy_agent import PolicyAgent
from agents.reward_agent import RewardAgent
from agents.qlearning_agent import QLearningAgent
print('✅ Alla agenter importerade framgångsrikt!')
"
```

## 🎯 Vanliga Användningsfall

### 1. REST API Generering

```bash
python pipeline.py --prompt prompts/rest_api.md --iters 3
```

### 2. Web Application

```bash
python pipeline.py --prompt prompts/web_app.md --iters 2
```

### 3. Data Processing Pipeline

```bash
python pipeline.py --prompt prompts/data_pipeline.md --iters 1
```

### 4. Microservices

```bash
python pipeline.py --prompt prompts/microservices.md --iters 3
```

## 🛠️ Felsökning

### Vanliga Problem

**Problem**: `ModuleNotFoundError: No module named 'agents'`

```bash
# Lösning: Aktivera virtual environment
source .venv/bin/activate  # Linux/Mac
# eller
.venv\Scripts\activate     # Windows
```

**Problem**: `PermissionError` vid skapande av filer

```bash
# Lösning: Kontrollera skrivbehörigheter
chmod 755 data/
mkdir -p data/generated
```

**Problem**: LLM-anslutning misslyckas

```bash
# Lösning: Använd offline mode
python pipeline.py --prompt prompts/simple_api.md --offline
```

### Debug Mode

```bash
# Kör med debug logging
LOG_LEVEL=DEBUG python pipeline.py --prompt prompts/simple_api.md --offline
```

## 📚 Nästa Steg

### Lär Dig Mer

- [Plugin Guide](PLUGIN_GUIDE.md) - Skapa egna agenter
- [API Documentation](docs/api.md) - Teknisk dokumentation
- [Architecture Guide](docs/architecture.md) - Systemdesign

### Experimentera

- Skapa egna prompts i `prompts/` mappen
- Justera reward-parametrar i `agents/reward_agent.py`
- Utforska Q-learning statistik i `data/qtable.db`

### Bidra

- Rapportera bugs på GitHub Issues
- Skapa pull requests för förbättringar
- Dela dina egna agenter som plugins

## 🆘 Support

### Hjälp

```bash
# Visa pipeline hjälp
python pipeline.py --help

# Visa agent hjälp
python -c "from agents.architect_agent import ArchitectAgent; help(ArchitectAgent)"
```

### Community

- **GitHub Issues**: [Rapportera bugs](https://github.com/olablom/CodeConductor/issues)
- **Discussions**: [Community forum](https://github.com/olablom/CodeConductor/discussions)
- **Wiki**: [Dokumentation](https://github.com/olablom/CodeConductor/wiki)

---

**🎉 Grattis! Du är nu redo att använda CodeConductor!**

Börja med en enkel pipeline och utforska sedan systemets avancerade funktioner. Lycka till med din AI-driven kodgenerering! 🚀
