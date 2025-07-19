from pathlib import Path
import random
import textwrap

TEMPLATES = [
    '''def hello_world():
    """Return a greeting"""
    return 'Hello, Bandit!'
''',
    '''def hello_world():
    """Return a greeting"""
    return 'Hello, RL Agent!'
''',
    '''def hello_world():
    """Return a greeting"""
    # This is a comment
    return "Hello, Learning System!"
''',
    """def hello_world():
    return 'Greetings, AI!'
""",
]


def run(prompt_path: Path, out_path: Path) -> bool:
    """Mock Cursor CLI - genererar slumpmässig kod"""
    # Läs prompt (för framtida användning)
    prompt_content = prompt_path.read_text()

    # Välj slumpmässig template
    code = random.choice(TEMPLATES)

    # Skriv till output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code)

    return True
