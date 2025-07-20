# 🔌 Plugin Guide - Skapa Egna Agenter

Denna guide visar hur du skapar egna agenter för CodeConductor. Plugin-systemet låter dig utöka systemets funktionalitet med specialiserade agenter för dina specifika behov.

## 🎯 Översikt

### Vad är en Agent?

En agent i CodeConductor är en specialiserad AI-komponent som:

- **Analyserar** input och kontext
- **Föreslår** lösningar eller åtgärder
- **Granskar** och validerar resultat
- **Lär sig** från feedback och rewards

### Agent-typer

- **ArchitectAgent** - Systemdesign och arkitektur
- **ReviewAgent** - Kodgranskning och kvalitet
- **CodeGenAgent** - Kodgenerering
- **PolicyAgent** - Säkerhetskontroll
- **RewardAgent** - Reward calculation
- **QLearningAgent** - Reinforcement learning

## 🏗️ Skapa Din Första Agent

### Steg 1: Grundstruktur

Skapa en ny fil `agents/my_custom_agent.py`:

```python
"""
My Custom Agent for CodeConductor

This agent specializes in [describe your agent's purpose].
"""

import logging
from typing import Dict, List, Any, Optional
from agents.base_agent import BaseAgent


class MyCustomAgent(BaseAgent):
    """
    Custom agent for [specific functionality].

    This agent focuses on:
    - [Feature 1]
    - [Feature 2]
    - [Feature 3]
    """

    def __init__(
        self, name: str = "my_custom_agent", config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the custom agent."""
        super().__init__(name, config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Custom configuration
        self.custom_config = config.get("custom", {}) if config else {}

        self.logger.info(f"MyCustomAgent '{name}' initialized")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze context for custom processing.

        Args:
            context: Context information

        Returns:
            Analysis results
        """
        self.logger.info(f"Analyzing context for {self.name}")

        # Your custom analysis logic here
        analysis_result = {
            "custom_insights": [],
            "recommendations": [],
            "confidence": 0.8
        }

        return analysis_result

    def propose(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose custom solution.

        Args:
            analysis: Analysis results
            context: Context information

        Returns:
            Custom proposal
        """
        self.logger.info(f"Proposing custom solution for {self.name}")

        # Your custom proposal logic here
        proposal = {
            "solution": "Custom solution",
            "approach": "Custom approach",
            "confidence": 0.8,
            "reasoning": "Custom reasoning"
        }

        return proposal

    def review(self, proposal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review custom proposal.

        Args:
            proposal: Proposal to review
            context: Context information

        Returns:
            Review results
        """
        self.logger.info(f"Reviewing custom proposal for {self.name}")

        # Your custom review logic here
        review_result = {
            "approved": True,
            "confidence": 0.9,
            "recommendations": ["Custom recommendation"]
        }

        return review_result
```

### Steg 2: Registrera Agenten

Lägg till din agent i `pipeline.py`:

```python
# I __init__ metoden
from agents.my_custom_agent import MyCustomAgent

# Lägg till i agents-listan
agents = [ArchitectAgent(), ReviewAgent(), CodeGenAgent(), MyCustomAgent()]
```

### Steg 3: Testa Din Agent

Skapa tester i `tests/test_my_custom_agent.py`:

```python
"""
Unit tests for MyCustomAgent
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.my_custom_agent import MyCustomAgent


class TestMyCustomAgent(unittest.TestCase):
    """Test cases for MyCustomAgent"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = MyCustomAgent("TestCustomAgent")
        self.sample_context = {
            "task_type": "custom",
            "requirements": "Custom requirements"
        }

    def test_init(self):
        """Test agent initialization"""
        agent = MyCustomAgent("TestAgent")
        self.assertEqual(agent.name, "TestAgent")

    def test_analyze(self):
        """Test analyze method"""
        result = self.agent.analyze(self.sample_context)
        self.assertIn("custom_insights", result)
        self.assertIn("recommendations", result)
        self.assertIn("confidence", result)

    def test_propose(self):
        """Test propose method"""
        analysis = {"custom_insights": []}
        result = self.agent.propose(analysis, self.sample_context)
        self.assertIn("solution", result)
        self.assertIn("approach", result)
        self.assertIn("confidence", result)

    def test_review(self):
        """Test review method"""
        proposal = {"solution": "test"}
        result = self.agent.review(proposal, self.sample_context)
        self.assertIn("approved", result)
        self.assertIn("confidence", result)
        self.assertIn("recommendations", result)


if __name__ == "__main__":
    unittest.main()
```

## 🎯 Exempel: Database Designer Agent

Här är ett komplett exempel på en Database Designer agent:

### `agents/database_designer_agent.py`

```python
"""
Database Designer Agent for CodeConductor

This agent specializes in database design and schema generation.
"""

import logging
from typing import Dict, List, Any, Optional
from agents.base_agent import BaseAgent


class DatabaseDesignerAgent(BaseAgent):
    """
    Database design agent for schema generation.

    This agent focuses on:
    - Database schema design
    - Table relationships
    - Index optimization
    - Migration scripts
    """

    def __init__(
        self, name: str = "database_designer_agent", config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the database designer agent."""
        super().__init__(name, config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Database configuration
        self.db_config = config.get("database", {}) if config else {}
        self.preferred_db = self.db_config.get("preferred_db", "postgresql")
        self.include_migrations = self.db_config.get("include_migrations", True)

        self.logger.info(f"DatabaseDesignerAgent '{name}' initialized with {self.preferred_db}")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze context for database design requirements.

        Args:
            context: Context information

        Returns:
            Database analysis results
        """
        self.logger.info(f"Analyzing database requirements for {self.name}")

        # Extract database requirements
        requirements = context.get("requirements", "")
        task_type = context.get("task_type", "unknown")

        # Analyze requirements for database needs
        db_entities = self._extract_entities(requirements)
        relationships = self._identify_relationships(db_entities)

        analysis_result = {
            "entities": db_entities,
            "relationships": relationships,
            "database_type": self.preferred_db,
            "complexity": self._assess_complexity(db_entities),
            "recommendations": self._generate_recommendations(db_entities)
        }

        return analysis_result

    def propose(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose database schema design.

        Args:
            analysis: Analysis results
            context: Context information

        Returns:
            Database schema proposal
        """
        self.logger.info(f"Proposing database schema for {self.name}")

        entities = analysis.get("entities", [])
        relationships = analysis.get("relationships", [])

        # Generate schema proposal
        schema = self._generate_schema(entities, relationships)
        migrations = self._generate_migrations(schema) if self.include_migrations else []

        proposal = {
            "schema": schema,
            "migrations": migrations,
            "database_type": self.preferred_db,
            "confidence": 0.9,
            "reasoning": f"Generated schema for {len(entities)} entities with {len(relationships)} relationships"
        }

        return proposal

    def review(self, proposal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review database schema proposal.

        Args:
            proposal: Proposal to review
            context: Context information

        Returns:
            Review results
        """
        self.logger.info(f"Reviewing database schema for {self.name}")

        schema = proposal.get("schema", {})

        # Review schema quality
        quality_score = self._assess_schema_quality(schema)
        issues = self._identify_schema_issues(schema)

        review_result = {
            "approved": quality_score > 0.7,
            "confidence": quality_score,
            "quality_score": quality_score,
            "issues": issues,
            "recommendations": self._generate_schema_recommendations(issues)
        }

        return review_result

    def _extract_entities(self, requirements: str) -> List[Dict[str, Any]]:
        """Extract database entities from requirements."""
        entities = []

        # Simple entity extraction (in practice, use NLP)
        if "user" in requirements.lower():
            entities.append({
                "name": "users",
                "fields": ["id", "username", "email", "created_at"],
                "primary_key": "id"
            })

        if "order" in requirements.lower():
            entities.append({
                "name": "orders",
                "fields": ["id", "user_id", "total", "status", "created_at"],
                "primary_key": "id",
                "foreign_keys": [{"field": "user_id", "references": "users.id"}]
            })

        return entities

    def _identify_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify relationships between entities."""
        relationships = []

        for entity in entities:
            for fk in entity.get("foreign_keys", []):
                relationships.append({
                    "from_table": entity["name"],
                    "from_field": fk["field"],
                    "to_table": fk["references"].split(".")[0],
                    "to_field": fk["references"].split(".")[1],
                    "type": "many_to_one"
                })

        return relationships

    def _assess_complexity(self, entities: List[Dict[str, Any]]) -> str:
        """Assess database complexity."""
        if len(entities) <= 3:
            return "simple"
        elif len(entities) <= 8:
            return "moderate"
        else:
            return "complex"

    def _generate_recommendations(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Generate database recommendations."""
        recommendations = []

        if len(entities) > 5:
            recommendations.append("Consider database normalization")

        if any(len(entity.get("fields", [])) > 10 for entity in entities):
            recommendations.append("Consider splitting large tables")

        return recommendations

    def _generate_schema(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate database schema."""
        schema = {
            "tables": {},
            "indexes": [],
            "constraints": []
        }

        for entity in entities:
            table_name = entity["name"]
            schema["tables"][table_name] = {
                "columns": self._generate_columns(entity),
                "primary_key": entity["primary_key"],
                "foreign_keys": entity.get("foreign_keys", [])
            }

        return schema

    def _generate_columns(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate column definitions."""
        columns = []

        for field in entity["fields"]:
            column = {
                "name": field,
                "type": self._infer_column_type(field),
                "nullable": field != entity["primary_key"],
                "default": self._get_default_value(field)
            }
            columns.append(column)

        return columns

    def _infer_column_type(self, field: str) -> str:
        """Infer column type from field name."""
        if field.endswith("_id"):
            return "INTEGER"
        elif field in ["created_at", "updated_at"]:
            return "TIMESTAMP"
        elif field in ["email"]:
            return "VARCHAR(255)"
        elif field in ["username"]:
            return "VARCHAR(100)"
        else:
            return "TEXT"

    def _get_default_value(self, field: str) -> str:
        """Get default value for field."""
        if field == "created_at":
            return "CURRENT_TIMESTAMP"
        elif field == "updated_at":
            return "CURRENT_TIMESTAMP"
        else:
            return "NULL"

    def _generate_migrations(self, schema: Dict[str, Any]) -> List[str]:
        """Generate migration scripts."""
        migrations = []

        for table_name, table_def in schema["tables"].items():
            migration = self._create_table_migration(table_name, table_def)
            migrations.append(migration)

        return migrations

    def _create_table_migration(self, table_name: str, table_def: Dict[str, Any]) -> str:
        """Create table migration script."""
        migration = f"CREATE TABLE {table_name} (\n"

        columns = []
        for col in table_def["columns"]:
            col_def = f"    {col['name']} {col['type']}"
            if not col["nullable"]:
                col_def += " NOT NULL"
            if col["default"] != "NULL":
                col_def += f" DEFAULT {col['default']}"
            columns.append(col_def)

        migration += ",\n".join(columns)
        migration += f"\n);"

        return migration

    def _assess_schema_quality(self, schema: Dict[str, Any]) -> float:
        """Assess schema quality score."""
        score = 1.0

        # Check for primary keys
        for table_name, table_def in schema["tables"].items():
            if not table_def.get("primary_key"):
                score -= 0.2

        # Check for proper foreign keys
        for table_name, table_def in schema["tables"].items():
            for fk in table_def.get("foreign_keys", []):
                if not fk.get("references"):
                    score -= 0.1

        return max(0.0, score)

    def _identify_schema_issues(self, schema: Dict[str, Any]) -> List[str]:
        """Identify schema issues."""
        issues = []

        for table_name, table_def in schema["tables"].items():
            if not table_def.get("primary_key"):
                issues.append(f"Table {table_name} missing primary key")

            if len(table_def["columns"]) > 15:
                issues.append(f"Table {table_name} has too many columns")

        return issues

    def _generate_schema_recommendations(self, issues: List[str]) -> List[str]:
        """Generate schema improvement recommendations."""
        recommendations = []

        for issue in issues:
            if "missing primary key" in issue:
                recommendations.append("Add primary key to all tables")
            elif "too many columns" in issue:
                recommendations.append("Consider normalizing large tables")

        return recommendations
```

### Testa Database Designer Agent

```python
# test_database_designer_agent.py
import unittest
from agents.database_designer_agent import DatabaseDesignerAgent


class TestDatabaseDesignerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DatabaseDesignerAgent()
        self.context = {
            "task_type": "api",
            "requirements": "Create a REST API for managing users and orders"
        }

    def test_analyze(self):
        result = self.agent.analyze(self.context)
        self.assertIn("entities", result)
        self.assertIn("relationships", result)
        self.assertIn("database_type", result)

    def test_propose(self):
        analysis = self.agent.analyze(self.context)
        proposal = self.agent.propose(analysis, self.context)
        self.assertIn("schema", proposal)
        self.assertIn("migrations", proposal)
        self.assertIn("confidence", proposal)

    def test_review(self):
        analysis = self.agent.analyze(self.context)
        proposal = self.agent.propose(analysis, self.context)
        review = self.agent.review(proposal, self.context)
        self.assertIn("approved", review)
        self.assertIn("quality_score", review)
        self.assertIn("issues", review)


if __name__ == "__main__":
    unittest.main()
```

## 🔧 Avancerade Plugin-funktioner

### Konfiguration via YAML

Skapa `config/my_agent.yaml`:

```yaml
my_custom_agent:
  name: "my_custom_agent"
  enabled: true
  config:
    custom:
      feature_1: true
      feature_2: false
      threshold: 0.8
    database:
      preferred_db: "postgresql"
      include_migrations: true
```

### Integration med Pipeline

Uppdatera `pipeline.py` för automatisk laddning:

```python
def load_agents_from_config(self):
    """Load agents from configuration."""
    import yaml

    with open("config/agents.yaml", "r") as f:
        config = yaml.safe_load(f)

    agents = []
    for agent_name, agent_config in config.items():
        if agent_config.get("enabled", True):
            agent_class = self._get_agent_class(agent_name)
            agent = agent_class(agent_config.get("config", {}))
            agents.append(agent)

    return agents

def _get_agent_class(self, agent_name: str):
    """Get agent class by name."""
    agent_map = {
        "architect_agent": ArchitectAgent,
        "review_agent": ReviewAgent,
        "codegen_agent": CodeGenAgent,
        "policy_agent": PolicyAgent,
        "reward_agent": RewardAgent,
        "qlearning_agent": QLearningAgent,
        "my_custom_agent": MyCustomAgent,
        "database_designer_agent": DatabaseDesignerAgent
    }

    return agent_map.get(agent_name)
```

### Custom Reward Integration

För att din agent ska påverka reward-beräkningen:

```python
class MyCustomAgent(BaseAgent):
    def calculate_custom_reward(self, result: Dict[str, Any]) -> float:
        """Calculate custom reward contribution."""
        # Your custom reward logic
        base_reward = 0.5

        if result.get("custom_metric") > 0.8:
            base_reward += 0.3

        return base_reward

    def propose(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        proposal = super().propose(analysis, context)

        # Add custom reward contribution
        proposal["custom_reward"] = self.calculate_custom_reward(proposal)

        return proposal
```

## 📊 Plugin Metrics och Monitoring

### Custom Metrics

```python
class MyCustomAgent(BaseAgent):
    def __init__(self, name: str = "my_custom_agent", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.metrics = {
            "proposals_generated": 0,
            "success_rate": 0.0,
            "average_confidence": 0.0
        }

    def propose(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        proposal = super().propose(analysis, context)

        # Update metrics
        self.metrics["proposals_generated"] += 1
        self.metrics["average_confidence"] = (
            (self.metrics["average_confidence"] * (self.metrics["proposals_generated"] - 1) +
             proposal.get("confidence", 0.0)) / self.metrics["proposals_generated"]
        )

        return proposal

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return self.metrics.copy()
```

### Dashboard Integration

```python
# I dashboard/app.py
def display_custom_agent_metrics():
    """Display custom agent metrics."""
    st.subheader("Custom Agent Metrics")

    for agent in pipeline.agents:
        if hasattr(agent, 'get_metrics'):
            metrics = agent.get_metrics()
            st.write(f"**{agent.name}**:")
            st.write(f"- Proposals: {metrics.get('proposals_generated', 0)}")
            st.write(f"- Success Rate: {metrics.get('success_rate', 0.0):.2%}")
            st.write(f"- Avg Confidence: {metrics.get('average_confidence', 0.0):.2f}")
```

## 🚀 Distribuera Din Plugin

### 1. Skapa Plugin Package

```
my-codeconductor-plugin/
├── setup.py
├── README.md
├── agents/
│   └── my_custom_agent.py
├── tests/
│   └── test_my_custom_agent.py
└── config/
    └── my_agent.yaml
```

### 2. Setup.py

```python
from setuptools import setup, find_packages

setup(
    name="codeconductor-my-plugin",
    version="1.0.0",
    description="My custom agent for CodeConductor",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "codeconductor>=2.0.0"
    ],
    entry_points={
        "codeconductor.agents": [
            "my_custom_agent = agents.my_custom_agent:MyCustomAgent"
        ]
    }
)
```

### 3. Installera Plugin

```bash
# Installera från lokal mapp
pip install -e ./my-codeconductor-plugin

# Eller från GitHub
pip install git+https://github.com/your-username/my-codeconductor-plugin.git
```

## 🎯 Best Practices

### 1. Följ Agent Interface

- Implementera alla krävda metoder (`analyze`, `propose`, `review`)
- Returnera konsistent datastruktur
- Hantera fel gracefully

### 2. Skriv Tester

- Testa alla publika metoder
- Mocka externa beroenden
- Testa edge cases

### 3. Dokumentation

- Skriv tydlig docstrings
- Inkludera exempel
- Beskriv konfigurationsalternativ

### 4. Performance

- Optimera för stora datasets
- Cacha resultat när möjligt
- Använd async/await för I/O-operationer

### 5. Logging

- Använd strukturerad logging
- Inkludera relevanta metrics
- Stöd debug-läge

## 🔍 Debugging Plugins

### Debug Mode

```bash
# Kör med debug logging
LOG_LEVEL=DEBUG python pipeline.py --prompt prompts/test.md --offline
```

### Agent-specifik Debug

```python
# I din agent
import logging
logging.getLogger(f"{__name__}.{self.__class__.__name__}").setLevel(logging.DEBUG)
```

### Metrics Debug

```python
# Visa agent metrics
python -c "
from agents.my_custom_agent import MyCustomAgent
agent = MyCustomAgent()
print(agent.get_metrics())
"
```

---

**🎉 Grattis! Du är nu redo att skapa egna agenter för CodeConductor!**

Börja med en enkel agent och bygg sedan vidare med mer avancerade funktioner. Lycka till med ditt plugin-utveckling! 🚀
