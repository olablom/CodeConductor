#!/usr/bin/env python3
import click
import numpy as np
import sqlite3
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple
from omegaconf import OmegaConf
from radon.complexity import cc_visit

from bandits.linucb import LinUCBBandit
from mocks.cursor_cli import run as cursor_run
from integrations.lm_studio import is_available, generate_code, ensure_models_ready
from agents.policy_agent import validate_code_safety, PolicyViolation
from agents.prompt_optimizer import prompt_optimizer, OptimizerState


# Initiera databas
def init_db(db_path: Path):
    """Skapa metrics-tabell om den inte finns"""
    conn = sqlite3.connect(db_path)

    # Kontrollera om tabellen finns
    cursor = conn.execute("PRAGMA table_info(metrics)")
    columns = [row[1] for row in cursor.fetchall()]

    if not columns:
        # Skapa ny tabell
        conn.execute("""
            CREATE TABLE metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER,
                timestamp DATETIME,
                reward REAL,
                pass_rate REAL,
                complexity REAL,
                arm_selected TEXT,
                features TEXT,
                model_source TEXT DEFAULT 'mock',
                blocked INTEGER DEFAULT 0,
                block_reasons TEXT DEFAULT '',
                optimizer_state TEXT DEFAULT '',
                optimizer_action TEXT DEFAULT ''
            )
        """)
    else:
        # Lägg till nya kolumner om de inte finns
        if "model_source" not in columns:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN model_source TEXT DEFAULT 'mock'"
            )
        if "blocked" not in columns:
            conn.execute("ALTER TABLE metrics ADD COLUMN blocked INTEGER DEFAULT 0")
        if "block_reasons" not in columns:
            conn.execute("ALTER TABLE metrics ADD COLUMN block_reasons TEXT DEFAULT ''")
        if "optimizer_state" not in columns:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN optimizer_state TEXT DEFAULT ''"
            )
        if "optimizer_action" not in columns:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN optimizer_action TEXT DEFAULT ''"
            )

    conn.commit()
    conn.close()


def extract_features(prompt_path: Path, iteration: int, history: list) -> np.ndarray:
    """Extrahera features för bandit"""
    prompt_content = prompt_path.read_text()

    # Enkla features
    prompt_length = len(prompt_content)
    code_blocks = prompt_content.count("```")
    prev_pass_rate = np.mean([h["passed"] for h in history[-5:]]) if history else 0.5

    # Normalisera
    features = np.array(
        [
            prompt_length / 1000.0,  # Normalisera till ~0-1
            code_blocks / 10.0,
            prev_pass_rate,
        ]
    )

    return features


def run_tests(code_path: Path, prompt_path: Path) -> Tuple[bool, float]:
    """Kör pytest på genererad kod baserat på prompt"""
    test_file = Path("tests/test_generated.py")

    # Bestäm vilken funktion vi testar baserat på prompt
    prompt_content = prompt_path.read_text().lower()

    if "fizzbuzz" in prompt_content:
        # Test för FizzBuzz
        test_content = f'''
import sys
sys.path.insert(0, "{code_path.parent}")
from {code_path.stem} import fizzbuzz

def test_fizzbuzz_exists():
    assert callable(fizzbuzz)

def test_fizzbuzz_works():
    assert fizzbuzz(3) == "Fizz"
    assert fizzbuzz(5) == "Buzz"
    assert fizzbuzz(15) == "FizzBuzz"
    assert fizzbuzz(7) == "7"
'''
    elif "add_numbers" in prompt_content or "calculator" in prompt_content:
        # Test för calculator
        test_content = f'''
import sys
sys.path.insert(0, "{code_path.parent}")
from {code_path.stem} import add_numbers

def test_add_numbers_exists():
    assert callable(add_numbers)

def test_add_numbers_works():
    result = add_numbers(5, 3)
    assert result == 8
'''
    else:
        # Default test för hello_world
        test_content = f'''
import sys
sys.path.insert(0, "{code_path.parent}")
from {code_path.stem} import hello_world

def test_hello_world_exists():
    assert callable(hello_world)

def test_hello_world_returns_string():
    result = hello_world()
    assert isinstance(result, str)
    assert len(result) > 0
'''

    test_file.write_text(test_content)

    # Kör pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-q"],
        capture_output=True,
        text=True,
    )

    passed = result.returncode == 0
    pass_rate = 1.0 if passed else 0.0

    # Cleanup
    test_file.unlink()

    return passed, pass_rate


def calculate_complexity(code_path: Path) -> float:
    """Beräkna kod-komplexitet med radon"""
    code = code_path.read_text()

    # Cyclomatic complexity
    cc_results = cc_visit(code)
    avg_complexity = (
        np.mean([block.complexity for block in cc_results]) if cc_results else 1.0
    )

    # Normalisera (lägre är bättre)
    complexity_score = 1.0 / (1.0 + avg_complexity)

    return complexity_score


def generate_multi_file_project(
    prompt_path: Path, strategy: str, project_dir: Path
) -> list:
    """Generate multi-file project using LM Studio"""
    try:
        # Load multi-file template
        template_path = Path("prompts/multi_file_template.md")
        if template_path.exists():
            template = template_path.read_text()
        else:
            template = "# Generate a complete Python project with multiple files"

        # Combine prompt with template
        full_prompt = f"{prompt_path.read_text()}\n\n{template}"

        # Generate project structure
        from integrations.lm_studio import generate_code

        project_code = generate_code(full_prompt, strategy)

        if not project_code:
            return []

        # Parse generated code and create files
        files_created = []
        current_file = None
        current_content = []

        for line in project_code.split("\n"):
            if line.startswith("# File:") or line.startswith("```"):
                # Save previous file
                if current_file and current_content:
                    file_path = project_dir / current_file
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text("\n".join(current_content))
                    files_created.append(file_path)

                # Start new file
                if line.startswith("# File:"):
                    current_file = line.split("File:")[1].strip()
                    current_content = []
                elif line.startswith("```"):
                    continue
            elif current_file and not line.startswith("```"):
                current_content.append(line)

        # Save last file
        if current_file and current_content:
            file_path = project_dir / current_file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("\n".join(current_content))
            files_created.append(file_path)

        return files_created

    except Exception as e:
        print(f"Error generating multi-file project: {e}")
        return []


def generate_mock_multi_file_project(
    prompt_path: Path, project_dir: Path, strategy: str
) -> list:
    """Generate mock multi-file project for testing"""
    prompt_content = prompt_path.read_text().lower()

    files_created = []

    # Generate main.py
    main_content = '''"""
Main application entry point
Generated by CodeConductor v2.0
"""

from utils import process_data
from config import get_config

def main():
    """Main application function"""
    config = get_config()
    print("🚀 Starting application...")
    
    # Process some data
    result = process_data("test data")
    print(f"✅ Result: {result}")
    
    return result

if __name__ == "__main__":
    main()
'''

    main_file = project_dir / "main.py"
    main_file.write_text(main_content, encoding="utf-8")
    files_created.append(main_file)

    # Generate utils.py
    utils_content = '''"""
Utility functions
Generated by CodeConductor v2.0
"""

def process_data(data: str) -> str:
    """Process input data and return result"""
    return f"Processed: {data.upper()}"

def validate_input(data: str) -> bool:
    """Validate input data"""
    return isinstance(data, str) and len(data) > 0
'''

    utils_file = project_dir / "utils.py"
    utils_file.write_text(utils_content, encoding="utf-8")
    files_created.append(utils_file)

    # Generate config.py
    config_content = '''"""
Configuration management
Generated by CodeConductor v2.0
"""

import os
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """Get application configuration"""
    return {
        "debug": os.getenv("DEBUG", "False").lower() == "true",
        "port": int(os.getenv("PORT", "8000")),
        "host": os.getenv("HOST", "localhost")
    }
'''

    config_file = project_dir / "config.py"
    config_file.write_text(config_content, encoding="utf-8")
    files_created.append(config_file)

    # Generate requirements.txt
    requirements_content = """# Generated by CodeConductor v2.0
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
pytest>=6.0.0
"""

    req_file = project_dir / "requirements.txt"
    req_file.write_text(requirements_content, encoding="utf-8")
    files_created.append(req_file)

    # Generate README.md
    readme_content = """# Generated Project

This project was generated by CodeConductor v2.0.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Test

```bash
pytest tests/
```
"""

    readme_file = project_dir / "README.md"
    readme_file.write_text(readme_content, encoding="utf-8")
    files_created.append(readme_file)

    # Generate tests directory
    tests_dir = project_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    # Generate __init__.py
    init_file = tests_dir / "__init__.py"
    init_file.write_text("# Tests package", encoding="utf-8")
    files_created.append(init_file)

    # Generate test_main.py
    test_main_content = '''"""
Tests for main module
Generated by CodeConductor v2.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main

def test_main_function_exists():
    """Test that main function exists"""
    assert callable(main)

def test_main_returns_result():
    """Test that main returns a result"""
    result = main()
    assert result is not None
'''

    test_main_file = tests_dir / "test_main.py"
    test_main_file.write_text(test_main_content, encoding="utf-8")
    files_created.append(test_main_file)

    # Generate test_utils.py
    test_utils_content = '''"""
Tests for utils module
Generated by CodeConductor v2.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import process_data, validate_input

def test_process_data():
    """Test data processing function"""
    result = process_data("hello")
    assert result == "Processed: HELLO"

def test_validate_input():
    """Test input validation"""
    assert validate_input("valid") == True
    assert validate_input("") == False
'''

    test_utils_file = tests_dir / "test_utils.py"
    test_utils_file.write_text(test_utils_content, encoding="utf-8")
    files_created.append(test_utils_file)

    # Generate .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""

    gitignore_file = project_dir / ".gitignore"
    gitignore_file.write_text(gitignore_content, encoding="utf-8")
    files_created.append(gitignore_file)

    return files_created


def calculate_reward(
    passed: bool, complexity: float, config: dict, code_content: str = ""
) -> float:
    """Beräkna total reward"""
    rewards = config.rewards

    base_reward = (
        passed * rewards.test_pass
        + complexity * rewards.complexity
        + 0.5 * rewards.lint_score  # Mock lint score
    )

    # Kreativitetsbonus för intressanta lösningar
    creativity_bonus = 0
    if passed and code_content:
        if (
            "emoji" in code_content.lower()
            or "🎉" in code_content
            or "🚀" in code_content
        ):
            creativity_bonus = rewards.creativity_bonus
        elif "import" in code_content and "time" in code_content:
            creativity_bonus = rewards.creativity_bonus * 0.5
        elif "random" in code_content:
            creativity_bonus = rewards.creativity_bonus * 0.3

    return base_reward + creativity_bonus


@click.command()
@click.option(
    "--prompt", type=click.Path(exists=True), required=True, help="Path to prompt file"
)
@click.option("--iters", default=10, help="Number of iterations")
@click.option("--mock", is_flag=True, help="Use mock Cursor CLI")
@click.option("--online", is_flag=True, help="Use local LM Studio instead of mock")
@click.option(
    "--multi-file", is_flag=True, help="Generate multi-file project structure"
)
@click.option(
    "--distributed", is_flag=True, help="Use distributed execution with Celery"
)
def main(prompt, iters, mock, online, multi_file, distributed):
    """Kör CodeConductor pipeline"""
    # Ladda config
    config = OmegaConf.load("config/base.yaml")

    # Initiera databas
    db_path = Path(config.database.path)
    db_path.parent.mkdir(exist_ok=True)
    init_db(db_path)

    # Initialize model manager if using online mode
    if online:
        try:
            from integrations.model_manager_simple import SimpleModelManager

            model_manager = SimpleModelManager()
            print("🔍 Checking model availability...")
            model_info = model_manager.get_model_info()
            print(f"📊 Model Status:")
            print(f"  Configured models: {len(model_info['configured_models'])}")
            print(
                f"  Available for installation: {len(model_info['available_for_installation'])}"
            )
            print(f"  Currently loaded: {len(model_info['currently_loaded'])}")
            print(
                f"  LM Studio CLI: {'✅' if model_info['lm_studio_cli_available'] else '❌'}"
            )

            # Ensure models are ready
            if model_info["lm_studio_cli_available"]:
                print("🔧 Ensuring models are ready...")
                results = model_manager.ensure_models()
                ready_count = sum(1 for ready in results.values() if ready)
                print(f"✅ {ready_count}/{len(results)} models ready")
            else:
                print("⚠️ LM Studio CLI not available, skipping model setup")

        except ImportError:
            print("⚠️ ModelManager not available, continuing without model management")
        except Exception as e:
            print(f"⚠️ Model setup failed: {e}, continuing...")

    # Initialize distributed orchestrator if requested
    orchestrator = None
    if distributed:
        try:
            from agents.orchestrator_distributed import DistributedAgentOrchestrator

            orchestrator = DistributedAgentOrchestrator(enable_plugins=True)
            print("🚀 Distributed execution enabled with Celery")

            # Check distributed stats
            stats = orchestrator.get_distributed_stats()
            if stats.get("distributed_enabled"):
                print(
                    f"✅ Celery broker available: {stats.get('broker_available', False)}"
                )
                print(
                    f"🔧 Worker concurrency: {stats.get('worker_concurrency', 'Unknown')}"
                )
            else:
                print(
                    "⚠️ Distributed mode not available, falling back to local execution"
                )
                orchestrator = None
        except Exception as e:
            print(f"⚠️ Failed to initialize distributed orchestrator: {e}")
            print("🔄 Falling back to local execution")
            orchestrator = None

    # Initiera bandit
    bandit = LinUCBBandit(d=config.bandit.feature_dim, alpha=config.bandit.alpha)

    # Arms (strategier)
    arms = ["conservative", "balanced", "exploratory"]

    # Historik
    history = []

    # PromptOptimizerAgent state tracking
    optimizer_state_prev = None
    optimizer_action_prev = None
    task_id = Path(prompt).stem

    # DEBUG: Verifiera prompt-inläsning
    print("DEBUG: Loading prompt from", prompt)
    try:
        with open(prompt, "r", encoding="utf-8") as f:
            prompt_text = f.read()
        print("DEBUG: Prompt text (first 200 chars):")
        print(prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text)
        print("DEBUG: Prompt length:", len(prompt_text), "characters")
    except Exception as e:
        print(f"DEBUG ERROR: Failed to read prompt file: {e}")

    click.echo(f"🚀 Starting CodeConductor pipeline with {iters} iterations...")

    for i in range(iters):
        click.echo(f"\n--- Iteration {i + 1}/{iters} ---")

        # 1. Extrahera features
        features = extract_features(Path(prompt), i, history)

        # 2. Välj arm
        arm = bandit.select_arm(arms, features)
        click.echo(f"Selected strategy: {arm}")

        # 3. PromptOptimizerAgent - optimera prompt om tidigare försök misslyckades
        original_prompt = Path(prompt).read_text()
        optimized_prompt = original_prompt
        optimizer_action = "no_change"

        if i > 0 and history:  # Inte första iterationen
            prev_result = history[-1]

            # Skapa optimizer state från föregående resultat
            optimizer_state_prev = prompt_optimizer.create_state(
                task_id=task_id,
                arm_prev=prev_result["arm"],
                passed=prev_result["passed"],
                blocked=prev_result["blocked"],
                complexity=prev_result.get("complexity", 0.5),
                model_source=prev_result.get("model_source", "mock"),
            )

            # Välj action och mutera prompt
            optimizer_action = prompt_optimizer.select_action(optimizer_state_prev)
            optimized_prompt = prompt_optimizer.mutate_prompt(
                original_prompt, optimizer_action
            )

            if optimizer_action != "no_change":
                click.echo(f"🤖 PromptOptimizer: {optimizer_action}")

                # Skapa temporär optimerad prompt-fil
                temp_prompt_path = Path(f"data/temp_prompt_{i}.md")
                temp_prompt_path.parent.mkdir(parents=True, exist_ok=True)
                temp_prompt_path.write_text(optimized_prompt)
                prompt_to_use = temp_prompt_path
            else:
                prompt_to_use = Path(prompt)
        else:
            prompt_to_use = Path(prompt)

        # 4. ML Quality Prediction
        ml_prediction = None
        try:
            from analytics.ml_predictor import create_predictor

            predictor = create_predictor()
            prediction_context = {
                "strategy": arm,
                "prev_pass_rate": np.mean([h["passed"] for h in history[-5:]])
                if history
                else 0.5,
                "complexity_avg": np.mean(
                    [h.get("complexity", 0.5) for h in history[-5:]]
                )
                if history
                else 0.5,
                "reward_avg": np.mean([h.get("reward", 30.0) for h in history[-5:]])
                if history
                else 30.0,
                "iteration_count": i + 1,
                "model_source": model_source,
            }

            ml_prediction = predictor.predict_quality(
                prompt_to_use.read_text(), prediction_context
            )

            if "error" not in ml_prediction:
                click.echo(
                    f"🤖 ML Prediction: Quality {ml_prediction['quality_score']:.2f}, Success {ml_prediction['success_probability']:.2f}"
                )

                # Show warnings
                for warning in ml_prediction.get("warnings", []):
                    click.echo(f"  {warning}")
            else:
                click.echo(f"⚠️ ML prediction failed: {ml_prediction['error']}")

        except Exception as e:
            click.echo(f"⚠️ ML prediction not available: {e}")

        # 5. Use distributed orchestrator if available
        if orchestrator:
            click.echo("🤖 Using distributed orchestrator for code generation...")
            try:
                discussion_result = orchestrator.facilitate_discussion(
                    prompt_to_use.read_text(),
                    {"strategy": arm, "iteration": i, "multi_file": multi_file},
                )

                # Extract consensus approach from discussion
                consensus = discussion_result.get("consensus", {})
                approach = consensus.get("synthesized_approach", "standard")
                click.echo(f"🎯 Consensus approach: {approach}")

                # Use the consensus approach for generation
                arm = approach if approach in arms else arm

            except Exception as e:
                click.echo(f"⚠️ Distributed orchestrator failed: {e}")
                click.echo("🔄 Falling back to standard generation...")

        # 5. Generera kod
        if multi_file:
            # Multi-file project generation
            project_dir = Path(f"data/generated/iter_{i}_project")
            project_dir.mkdir(parents=True, exist_ok=True)

            click.echo(f"📁 Generating multi-file project in: {project_dir}")

            # Bestäm kodkälla
            model_source = "mock"
            if online:
                click.echo(f"Online mode: checking LM Studio availability...")
                if is_available():
                    click.echo("Generating multi-file project via LM Studio...")
                    project_files = generate_multi_file_project(
                        prompt_to_use, arm, project_dir
                    )
                    if project_files:
                        model_source = "lm_studio"
                        click.echo(f"✅ LM Studio generated {len(project_files)} files")
                    else:
                        click.echo("LM Studio failed, falling back to mock...")
                        project_files = generate_mock_multi_file_project(
                            prompt_to_use, project_dir, arm
                        )
                else:
                    click.echo("LM Studio not available, using mock...")
                    project_files = generate_mock_multi_file_project(
                        prompt_to_use, project_dir, arm
                    )
            else:
                project_files = generate_mock_multi_file_project(
                    prompt_to_use, project_dir, arm
                )

            # Set output_path to main.py for compatibility
            output_path = project_dir / "main.py"
        else:
            # Single file generation (existing logic)
            output_path = Path(f"data/generated/iter_{i}.py")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Bestäm kodkälla
            model_source = "mock"
            if online:
                click.echo(f"Online mode: checking LM Studio availability...")
                if is_available():
                    click.echo("Generating code via LM Studio...")
                    code = generate_code(prompt_to_use, arm)
                    if code:
                        output_path.write_text(code)
                        model_source = "lm_studio"
                        click.echo(f"✅ LM Studio generated code: {len(code)} chars")
                    else:
                        click.echo("LM Studio failed, falling back to mock...")
                        cursor_run(prompt_to_use, output_path, strategy=arm)
                else:
                    click.echo("LM Studio not available, using mock...")
                    cursor_run(prompt_to_use, output_path, strategy=arm)
            else:
                cursor_run(prompt_to_use, output_path, strategy=arm)

        # 6. PolicyAgent säkerhetskontroll
        code_content = output_path.read_text()
        is_safe, violations = validate_code_safety(code_content, str(prompt_to_use))

        blocked = not is_safe
        block_reasons = ""

        if violations:
            click.echo(f"🚨 Code blocked by PolicyAgent!")
            for v in violations:
                click.echo(f"  - {v.reason.value}: {v.description}")
            block_reasons = "; ".join(
                [f"{v.reason.value}:{v.line_number}" for v in violations]
            )

            # Ge negativ reward för blockerad kod
            reward = -20.0
            passed = False
            pass_rate = 0.0
            complexity = 0.0
        else:
            click.echo(f"✅ Code passed PolicyAgent safety check")

            # 7. Kör tester
            if multi_file:
                # Test multi-file project
                from agents.test_agent import TestAgent

                test_agent = TestAgent()
                test_results = test_agent.run_tests("", "basic", project_dir)
                passed = test_results["tests_passed"] > 0
                pass_rate = test_results["tests_passed"] / max(
                    test_results["tests_run"], 1
                )
                click.echo(
                    f"Project tests: {test_results['tests_passed']}/{test_results['tests_run']} passed"
                )
            else:
                # Test single file
                passed, pass_rate = run_tests(output_path, prompt_to_use)
                click.echo(f"Tests passed: {passed}")

            # 8. Beräkna komplexitet
            complexity = calculate_complexity(output_path)
            click.echo(f"Complexity score: {complexity:.2f}")

            # 9. Beräkna reward med PromptOptimizer-bonus
            base_reward = calculate_reward(passed, complexity, config, code_content)
            reward = prompt_optimizer.calculate_reward(
                passed=passed,
                blocked=blocked,
                iterations=i + 1,
                complexity=complexity,
                base_reward=base_reward,
            )
            click.echo(f"Reward: {reward:.2f} (base: {base_reward:.2f})")

        # 10. Uppdatera bandit
        bandit.update(arm, features, reward)

        # 11. Uppdatera PromptOptimizerAgent
        if optimizer_state_prev and optimizer_action_prev:
            current_state = prompt_optimizer.create_state(
                task_id=task_id,
                arm_prev=arm,
                passed=passed,
                blocked=blocked,
                complexity=complexity,
                model_source=model_source,
            )
            prompt_optimizer.update(
                optimizer_state_prev, optimizer_action_prev, reward, current_state
            )

        # 11. Spara till databas
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO metrics (iteration, timestamp, reward, pass_rate, complexity, arm_selected, features, model_source, blocked, block_reasons, optimizer_state, optimizer_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                i,
                datetime.now(),
                reward,
                pass_rate,
                complexity,
                arm,
                str(features.tolist()),
                model_source,
                1 if blocked else 0,
                block_reasons,
                json.dumps(optimizer_state_prev.to_vector())
                if optimizer_state_prev
                else "",
                optimizer_action_prev or "",
            ),
        )
        conn.commit()
        conn.close()

        # 12. Uppdatera historik
        history.append(
            {
                "iteration": i,
                "passed": passed,
                "reward": reward,
                "arm": arm,
                "blocked": blocked,
                "complexity": complexity,
                "model_source": model_source,
            }
        )

        # 13. Spara aktuell state/action för nästa iteration
        optimizer_state_prev = prompt_optimizer.current_state
        optimizer_action_prev = optimizer_action

        # 14. Cleanup temporär prompt-fil
        if optimizer_action != "no_change":
            temp_prompt_path = Path(f"data/temp_prompt_{i}.md")
            if temp_prompt_path.exists():
                temp_prompt_path.unlink()

    click.echo("\n✅ Pipeline completed!")
    click.echo(f"📊 View results: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
