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
from integrations.lm_studio import is_available, generate_code
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
def main(prompt, iters, mock, online):
    """Kör CodeConductor pipeline"""
    # Ladda config
    config = OmegaConf.load("config/base.yaml")

    # Initiera databas
    db_path = Path(config.database.path)
    db_path.parent.mkdir(exist_ok=True)
    init_db(db_path)

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

        # 4. Generera kod
        output_path = Path(f"data/generated/iter_{i}.py")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Bestäm kodkälla
        model_source = "mock"
        if online and is_available():
            click.echo("Generating code via LM Studio...")
            code = generate_code(prompt_to_use, arm)
            if code:
                output_path.write_text(code)
                model_source = "lm_studio"
            else:
                click.echo("LM Studio failed, falling back to mock...")
                cursor_run(prompt_to_use, output_path, strategy=arm)
        else:
            cursor_run(prompt_to_use, output_path, strategy=arm)

        # 5. PolicyAgent säkerhetskontroll
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

            # 6. Kör tester
            passed, pass_rate = run_tests(output_path, prompt_to_use)
            click.echo(f"Tests passed: {passed}")

            # 7. Beräkna komplexitet
            complexity = calculate_complexity(output_path)
            click.echo(f"Complexity score: {complexity:.2f}")

            # 8. Beräkna reward med PromptOptimizer-bonus
            base_reward = calculate_reward(passed, complexity, config, code_content)
            reward = prompt_optimizer.calculate_reward(
                passed=passed,
                blocked=blocked,
                iterations=i + 1,
                complexity=complexity,
                base_reward=base_reward,
            )
            click.echo(f"Reward: {reward:.2f} (base: {base_reward:.2f})")

        # 9. Uppdatera bandit
        bandit.update(arm, features, reward)

        # 10. Uppdatera PromptOptimizerAgent
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
