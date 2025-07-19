#!/usr/bin/env python3
import click
import numpy as np
import sqlite3
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple
from omegaconf import OmegaConf
from radon.complexity import cc_visit

from bandits.linucb import LinUCBBandit
from mocks.cursor_cli import run as cursor_run


# Initiera databas
def init_db(db_path: Path):
    """Skapa metrics-tabell om den inte finns"""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER,
            timestamp DATETIME,
            reward REAL,
            pass_rate REAL,
            complexity REAL,
            arm_selected TEXT,
            features TEXT
        )
    """)
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
def main(prompt, iters, mock):
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

    click.echo(f"🚀 Starting CodeConductor pipeline with {iters} iterations...")

    for i in range(iters):
        click.echo(f"\n--- Iteration {i + 1}/{iters} ---")

        # 1. Extrahera features
        features = extract_features(Path(prompt), i, history)

        # 2. Välj arm
        arm = bandit.select_arm(arms, features)
        click.echo(f"Selected strategy: {arm}")

        # 3. Generera kod
        output_path = Path(f"data/generated/iter_{i}.py")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if mock:
            cursor_run(Path(prompt), output_path, strategy=arm)
        else:
            # TODO: Implementera riktig Cursor-integration
            cursor_run(Path(prompt), output_path, strategy=arm)

        # 4. Kör tester
        passed, pass_rate = run_tests(output_path, Path(prompt))
        click.echo(f"Tests passed: {passed}")

        # 5. Beräkna komplexitet
        complexity = calculate_complexity(output_path)
        click.echo(f"Complexity score: {complexity:.2f}")

        # 6. Beräkna reward
        code_content = output_path.read_text()
        reward = calculate_reward(passed, complexity, config, code_content)
        click.echo(f"Reward: {reward:.2f}")

        # 7. Uppdatera bandit
        bandit.update(arm, features, reward)

        # 8. Spara till databas
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO metrics (iteration, timestamp, reward, pass_rate, complexity, arm_selected, features)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                i,
                datetime.now(),
                reward,
                pass_rate,
                complexity,
                arm,
                str(features.tolist()),
            ),
        )
        conn.commit()
        conn.close()

        # 9. Uppdatera historik
        history.append({"iteration": i, "passed": passed, "reward": reward, "arm": arm})

    click.echo("\n✅ Pipeline completed!")
    click.echo(f"📊 View results: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
