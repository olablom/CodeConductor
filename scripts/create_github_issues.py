#!/usr/bin/env python3
"""
GitHub Issues Generator for CodeConductor v2.0
Automatically creates issues from project roadmap
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any


class GitHubIssuesGenerator:
    def __init__(self):
        self.issues = []
        self.milestones = {}
        self.labels = {
            "backend": "Backend-komponenter",
            "agent": "Agent-relaterade features",
            "rl": "Reinforcement learning",
            "ui": "Användargränssnitt",
            "docs": "Dokumentation",
            "demo": "Demo och presentation",
            "testing": "Tester och QA",
            "infrastructure": "Infrastruktur och setup",
            "task": "Development task",
            "enhancement": "Feature enhancement",
        }

    def create_milestones(self):
        """Create milestone definitions"""
        start_date = datetime.now()

        self.milestones = {
            "Week 1": {
                "title": "Infrastructure & RL Foundation",
                "description": "Agent-bas, LLM-integration, Q-learning foundation",
                "due_on": (start_date + timedelta(days=7)).isoformat(),
                "state": "open",
            },
            "Week 2-3": {
                "title": "Multi-Agent Discussion",
                "description": "AgentOrchestrator, 3 agenter, diskussion",
                "due_on": (start_date + timedelta(days=21)).isoformat(),
                "state": "open",
            },
            "Week 3-4": {
                "title": "Human-in-the-Loop",
                "description": "CLI/Streamlit UI, godkännande-flöde",
                "due_on": (start_date + timedelta(days=28)).isoformat(),
                "state": "open",
            },
            "Week 4-6": {
                "title": "Implementation & Integration",
                "description": "Cursor/LM Studio, FastAPI fallback, testing",
                "due_on": (start_date + timedelta(days=42)).isoformat(),
                "state": "open",
            },
            "Week 6-9": {
                "title": "RL Feedback Loop",
                "description": "RewardAgent, PromptOptimizerAgent, metrics",
                "due_on": (start_date + timedelta(days=63)).isoformat(),
                "state": "open",
            },
            "Week 9-10": {
                "title": "Demo & Documentation",
                "description": "Live demo, dokumentation, workshop material",
                "due_on": (start_date + timedelta(days=70)).isoformat(),
                "state": "open",
            },
        }

    def create_phase_1_issues(self):
        """Create issues for Phase 1: Infrastructure & Grundarkitektur"""

        # 1.1 Agent-bas
        self.issues.append(
            {
                "title": "[TASK] Definiera BaseAgent abstract class",
                "body": """## 📋 Task: Definiera BaseAgent abstract class

### **Beskrivning**
Skapa en abstract base class för alla agenter med standardiserade metoder.

### **Fas**
- [x] Fas 1: Infrastruktur & Grundarkitektur

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 1 dag

### **Definition of Done**
- [ ] Abstract class med `analyze()`, `propose()`, `review()` metoder
- [ ] Type hints och docstrings
- [ ] Unit tester för base class
- [ ] Integration med befintlig agent-struktur

### **Teknisk detaljer**
- Skapa `agents/base_agent.py`
- Definiera abstract methods
- Implementera common functionality
- Uppdatera befintliga agenter att ärva från BaseAgent

### **Beroenden**
Inga

### **Tester**
- [x] Unit tester
- [ ] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "agent", "infrastructure"],
                "milestone": "Week 1",
            }
        )

        self.issues.append(
            {
                "title": "[TASK] Implementera kommunikationskanal (message bus)",
                "body": """## 📋 Task: Implementera kommunikationskanal

### **Beskrivning**
Skapa en message bus för agent-kommunikation.

### **Fas**
- [x] Fas 1: Infrastruktur & Grundarkitektur

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 1 dag

### **Definition of Done**
- [ ] Message bus implementation
- [ ] Agent message routing
- [ ] Error handling
- [ ] Performance monitoring

### **Teknisk detaljer**
- Implementera in-memory message bus
- Stöd för async kommunikation
- Message validation och routing
- Integration med agent-system

### **Beroenden**
- BaseAgent implementation

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "agent", "infrastructure"],
                "milestone": "Week 1",
            }
        )

        # 1.2 LLM-integration
        self.issues.append(
            {
                "title": "[TASK] Skapa LLMClient wrapper för Ollama/CodeLlama",
                "body": """## 📋 Task: LLMClient wrapper

### **Beskrivning**
Skapa en wrapper för LLM-integration med caching och retry-logic.

### **Fas**
- [x] Fas 1: Infrastruktur & Grundarkitektur

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 2 dagar

### **Definition of Done**
- [ ] LLMClient class implementation
- [ ] Ollama/CodeLlama integration
- [ ] Caching mechanism
- [ ] Retry logic och error handling
- [ ] Configuration management

### **Teknisk detaljer**
- Skapa `integrations/llm_client.py`
- Stöd för lokala modeller
- Response caching
- Exponential backoff för retries
- Config-driven model selection

### **Beroenden**
- BaseAgent implementation

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "infrastructure", "backend"],
                "milestone": "Week 1",
            }
        )

        # 1.3 YAML-konfiguration
        self.issues.append(
            {
                "title": "[TASK] Skapa config.yaml för agent-profiler och RL-hyperparametrar",
                "body": """## 📋 Task: YAML-konfiguration

### **Beskrivning**
Skapa konfigurationssystem för agent-profiler och RL-parametrar.

### **Fas**
- [x] Fas 1: Infrastruktur & Grundarkitektur

### **Prioritet**
- [x] Medium

### **Tidsuppskattning**
- [x] 1 dag

### **Definition of Done**
- [ ] config.yaml struktur
- [ ] Config loader och validator
- [ ] Agent profile definitions
- [ ] RL hyperparameter configuration
- [ ] Environment-specific configs

### **Teknisk detaljer**
- Skapa `config/` directory
- Implementera config loader med validation
- Stöd för development/production configs
- Integration med agent-system

### **Beroenden**
- LLMClient implementation

### **Tester**
- [x] Unit tester
- [ ] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "infrastructure"],
                "milestone": "Week 1",
            }
        )

    def create_phase_2_issues(self):
        """Create issues for Phase 2: RL-Infrastruktur"""

        self.issues.append(
            {
                "title": "[TASK] Implementera calculate_reward() funktion",
                "body": """## 📋 Task: Reward-funktion implementation

### **Beskrivning**
Implementera reward-funktion för RL-system enligt design.

### **Fas**
- [x] Fas 2: RL-Infrastruktur

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 2 dagar

### **Definition of Done**
- [ ] Test-pass rate reward
- [ ] Komplexitet bedömning
- [ ] Policy-block detection
- [ ] Reward normalisering
- [ ] Unit tester för reward-logic

### **Teknisk detaljer**
- Skapa `bandits/reward_calculator.py`
- Implementera olika reward-komponenter
- Weighted reward combination
- Normalisering och skalning
- Integration med Q-learning

### **Beroenden**
- BaseAgent implementation
- Config system

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "rl", "backend"],
                "milestone": "Week 2-3",
            }
        )

        self.issues.append(
            {
                "title": "[TASK] Implementera tabular Q-agent",
                "body": """## 📋 Task: Q-learning implementation

### **Beskrivning**
Skapa tabular Q-agent för prompt-optimering.

### **Fas**
- [x] Fas 2: RL-Infrastruktur

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 3 dagar

### **Definition of Done**
- [ ] Tabular Q-agent implementation
- [ ] State-action space definition
- [ ] Epsilon-greedy exploration
- [ ] Q-table persistence
- [ ] Learning rate och discount factor

### **Teknisk detaljer**
- Skapa `bandits/q_agent.py`
- Implementera Q-learning algorithm
- State representation för prompts
- Action space för prompt-modifieringar
- Q-table serialization

### **Beroenden**
- Reward function implementation

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "rl", "backend"],
                "milestone": "Week 2-3",
            }
        )

    def create_phase_3_issues(self):
        """Create issues for Phase 3: Multi-Agent Diskussion"""

        self.issues.append(
            {
                "title": "[TASK] Implementera AgentOrchestrator",
                "body": """## 📋 Task: AgentOrchestrator implementation

### **Beskrivning**
Skapa orchestrator för multi-agent diskussion och konsensus.

### **Fas**
- [x] Fas 3: Multi-Agent Diskussion

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 3 dagar

### **Definition of Done**
- [ ] Diskussionsturnering koordination
- [ ] Agent scheduling och timing
- [ ] Consensus building logic
- [ ] Conflict resolution
- [ ] Performance monitoring

### **Teknisk detaljer**
- Skapa `orchestrator/agent_orchestrator.py`
- Implementera discussion rounds
- Agent voting och consensus
- Timeout handling
- Integration med message bus

### **Beroenden**
- BaseAgent implementation
- Message bus
- Config system

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "agent", "backend"],
                "milestone": "Week 2-3",
            }
        )

        self.issues.append(
            {
                "title": "[TASK] Implementera CodeGenAgent, ArchitectAgent, ReviewAgent",
                "body": """## 📋 Task: Specialiserade agenter

### **Beskrivning**
Implementera de tre huvudagenterna med specifika ansvarsområden.

### **Fas**
- [x] Fas 3: Multi-Agent Diskussion

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 4 dagar

### **Definition of Done**
- [ ] CodeGenAgent - Kodgenerering fokus
- [ ] ArchitectAgent - Arkitektur och design
- [ ] ReviewAgent - Kodgranskning och kvalitet
- [ ] Agent-specifika prompts och logik
- [ ] Integration med orchestrator

### **Teknisk detaljer**
- Skapa `agents/codegen_agent.py`
- Skapa `agents/architect_agent.py`
- Skapa `agents/review_agent.py`
- Implementera agent-specifika prompts
- Integration med LLMClient

### **Beroenden**
- BaseAgent implementation
- AgentOrchestrator
- LLMClient

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "agent", "backend"],
                "milestone": "Week 2-3",
            }
        )

    def create_phase_4_issues(self):
        """Create issues for Phase 4: Human-in-the-Loop"""

        self.issues.append(
            {
                "title": "[TASK] Skapa CLI interface för agent-förslag",
                "body": """## 📋 Task: CLI interface

### **Beskrivning**
Skapa command-line interface för agent-förslag och godkännande.

### **Fas**
- [x] Fas 4: Human-in-the-Loop

### **Prioritet**
- [x] Medium

### **Tidsuppskattning**
- [x] 2 dagar

### **Definition of Done**
- [ ] Enkel meny för agent-förslag
- [ ] Approve/Reject funktionalitet
- [ ] Förhandsvisning av kod
- [ ] Keyboard shortcuts
- [ ] Error handling

### **Teknisk detaljer**
- Skapa `ui/cli_interface.py`
- Implementera interactive menu
- Kod preview med syntax highlighting
- Keyboard navigation
- Integration med agent-system

### **Beroenden**
- AgentOrchestrator
- Multi-agent system

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "ui", "backend"],
                "milestone": "Week 3-4",
            }
        )

        self.issues.append(
            {
                "title": "[TASK] Skapa Streamlit UI med dark mode",
                "body": """## 📋 Task: Streamlit UI

### **Beskrivning**
Skapa modern Streamlit interface med dark mode och live editing.

### **Fas**
- [x] Fas 4: Human-in-the-Loop

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 3 dagar

### **Definition of Done**
- [ ] Modern dark mode interface
- [ ] Live agent-diskussion visning
- [ ] Kod preview med syntax highlighting
- [ ] Feedback collection
- [ ] Real-time updates

### **Teknisk detaljer**
- Skapa `ui/streamlit_app.py`
- Implementera dark mode theme
- Live discussion visualization
- Code editor integration
- Real-time agent communication

### **Beroenden**
- AgentOrchestrator
- Multi-agent system

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "ui", "frontend"],
                "milestone": "Week 3-4",
            }
        )

    def create_phase_5_issues(self):
        """Create issues for Phase 5: Implementation"""

        self.issues.append(
            {
                "title": "[TASK] Integrera Cursor/LM Studio för kodgenerering",
                "body": """## 📋 Task: Cursor/LM Studio integration

### **Beskrivning**
Integrera Cursor och LM Studio för kodgenerering med fallback.

### **Fas**
- [x] Fas 5: Implementation

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 3 dagar

### **Definition of Done**
- [ ] API integration för kodgenerering
- [ ] Prompt engineering och optimization
- [ ] Response parsing och validation
- [ ] Error handling och fallback
- [ ] Performance monitoring

### **Teknisk detaljer**
- Skapa `integrations/cursor_client.py`
- Skapa `integrations/lm_studio_client.py`
- Implementera prompt templates
- Response validation och parsing
- Fallback mechanism

### **Beroenden**
- LLMClient base
- Agent system

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "backend", "integration"],
                "milestone": "Week 4-6",
            }
        )

        self.issues.append(
            {
                "title": "[TASK] Implementera FastAPI fallback generator",
                "body": """## 📋 Task: FastAPI fallback

### **Beskrivning**
Skapa fallback-generator för komplexa prompts när LLM misslyckas.

### **Fas**
- [x] Fas 5: Implementation

### **Prioritet**
- [x] Medium

### **Tidsuppskattning**
- [x] 2 dagar

### **Definition of Done**
- [ ] Integrera befintlig fallback-generator
- [ ] Komplexa prompt-hantering
- [ ] Code quality validation
- [ ] Performance optimization
- [ ] Integration med main pipeline

### **Teknisk detaljer**
- Skapa `generators/fastapi_fallback.py`
- Template-based code generation
- Code quality checks
- Integration med agent-system
- Performance monitoring

### **Beroenden**
- Cursor/LM Studio integration

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "backend", "generation"],
                "milestone": "Week 4-6",
            }
        )

    def create_phase_6_issues(self):
        """Create issues for Phase 6: RL-Feedback Loop"""

        self.issues.append(
            {
                "title": "[TASK] Implementera RewardAgent",
                "body": """## 📋 Task: RewardAgent implementation

### **Beskrivning**
Skapa RewardAgent för att analysera resultat och beräkna belöningar.

### **Fas**
- [x] Fas 6: RL-Feedback Loop

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 3 dagar

### **Definition of Done**
- [ ] Brainstorm belöningar baserat på resultat
- [ ] Test-resultat analys
- [ ] Komplexitet bedömning
- [ ] Policy-block detection
- [ ] Integration med Q-learning

### **Teknisk detaljer**
- Skapa `agents/reward_agent.py`
- Implementera result analysis
- Complexity assessment
- Policy violation detection
- Reward calculation logic

### **Beroenden**
- Q-learning implementation
- Test system

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "agent", "rl"],
                "milestone": "Week 6-9",
            }
        )

        self.issues.append(
            {
                "title": "[TASK] Implementera PromptOptimizerAgent",
                "body": """## 📋 Task: PromptOptimizerAgent

### **Beskrivning**
Skapa PromptOptimizerAgent som använder Q-learning för prompt-optimering.

### **Fas**
- [x] Fas 6: RL-Feedback Loop

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 4 dagar

### **Definition of Done**
- [ ] Q-agent integration
- [ ] Q-table uppdatering
- [ ] Prompt evolution
- [ ] Performance tracking
- [ ] Learning curve visualization

### **Teknisk detaljer**
- Skapa `agents/prompt_optimizer_agent.py`
- Q-learning integration
- Prompt modification strategies
- Performance tracking
- Learning visualization

### **Beroenden**
- RewardAgent
- Q-learning system

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "agent", "rl"],
                "milestone": "Week 6-9",
            }
        )

    def create_phase_7_issues(self):
        """Create issues for Phase 7: Live Demo & Dokumentation"""

        self.issues.append(
            {
                "title": "[TASK] Skapa live demo script",
                "body": """## 📋 Task: Live demo script

### **Beskrivning**
Skapa komplett demo script för att visa systemets funktionalitet.

### **Fas**
- [x] Fas 7: Live Demo & Dokumentation

### **Prioritet**
- [x] Hög

### **Tidsuppskattning**
- [x] 2 dagar

### **Definition of Done**
- [ ] Finslipa demo-script
- [ ] Microservices demonstration
- [ ] Live kodgenerering
- [ ] Performance showcase
- [ ] Error handling

### **Teknisk detaljer**
- Skapa `demo/live_demo.py`
- Interactive demonstration
- Real-time code generation
- Performance metrics
- Error recovery

### **Beroenden**
- Alla faser implementerade

### **Tester**
- [x] Unit tester
- [x] Integration tester
- [x] End-to-end tester""",
                "labels": ["task", "demo", "documentation"],
                "milestone": "Week 9-10",
            }
        )

        self.issues.append(
            {
                "title": "[TASK] Skapa workshop material",
                "body": """## 📋 Task: Workshop material

### **Beskrivning**
Skapa komplett workshop material för kursdeltagare.

### **Fas**
- [x] Fas 7: Live Demo & Dokumentation

### **Prioritet**
- [x] Medium

### **Tidsuppskattning**
- [x] 2 dagar

### **Definition of Done**
- [ ] Övningsuppgifter
- [ ] Quiz för deltagare
- [ ] Presentation slides
- [ ] Hands-on tutorials
- [ ] Evaluation forms

### **Teknisk detaljer**
- Skapa `workshop/` directory
- Interactive exercises
- Assessment materials
- Presentation templates
- Tutorial guides

### **Beroenden**
- Live demo script
- Dokumentation

### **Tester**
- [x] Unit tester
- [ ] Integration tester
- [ ] End-to-end tester""",
                "labels": ["task", "docs", "workshop"],
                "milestone": "Week 9-10",
            }
        )

    def generate_all_issues(self):
        """Generate all issues for the project"""
        self.create_milestones()
        self.create_phase_1_issues()
        self.create_phase_2_issues()
        self.create_phase_3_issues()
        self.create_phase_4_issues()
        self.create_phase_5_issues()
        self.create_phase_6_issues()
        self.create_phase_7_issues()

    def save_issues(self, filename: str = "github_issues.json"):
        """Save issues to JSON file"""
        output = {
            "milestones": self.milestones,
            "labels": self.labels,
            "issues": self.issues,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ Generated {len(self.issues)} issues")
        print(f"✅ Generated {len(self.milestones)} milestones")
        print(f"✅ Generated {len(self.labels)} labels")
        print(f"📁 Saved to {filename}")

    def print_summary(self):
        """Print summary of generated issues"""
        print("\n" + "=" * 60)
        print("🎯 GITHUB ISSUES SUMMARY")
        print("=" * 60)

        for milestone in self.milestones:
            milestone_issues = [
                i for i in self.issues if i.get("milestone") == milestone
            ]
            print(f"\n📅 {milestone}: {len(milestone_issues)} issues")
            for issue in milestone_issues:
                print(f"  - {issue['title']}")

        print(
            f"\n📊 Total: {len(self.issues)} issues across {len(self.milestones)} milestones"
        )
        print("=" * 60)


def main():
    """Main function to generate GitHub issues"""
    generator = GitHubIssuesGenerator()
    generator.generate_all_issues()
    generator.save_issues()
    generator.print_summary()

    print("\n🚀 Next steps:")
    print("1. Review generated issues in github_issues.json")
    print("2. Use GitHub CLI to create issues:")
    print("   gh issue create --title 'Title' --body-file body.md")
    print("3. Create project board and organize issues")
    print("4. Start with Phase 1 issues!")


if __name__ == "__main__":
    main()
