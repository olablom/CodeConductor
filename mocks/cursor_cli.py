from pathlib import Path
import random
import textwrap

TEMPLATES_BY_STRATEGY = {
    "conservative": [
        '''def hello_world():
    """Return a greeting"""
    return 'Hello, World!'
''',
        '''def hello_world():
    """Simple greeting"""
    return "Hello!"
''',
        """def hello_world():
    return 'Greetings!'
""",
    ],
    "balanced": [
        '''def hello_world():
    """Return greeting"""
    msg = "Hello, Bandit!"
    return msg
''',
        """def hello_world():
    # Greeting function
    return "Greetings!"
""",
        '''def hello_world():
    """Return a greeting"""
    greeting = "Hello, RL Agent!"
    return greeting
''',
    ],
    "exploratory": [
        """def hello_world():
    import random
    greetings = ["Hi!", "Hello!", "Hey!"]
    return random.choice(greetings)
""",
        """def hello_world():
    pass  # This will fail tests!
""",
        """def hello_world():
    return None  # This will also fail!
""",
        '''def hello_world():
    """Return a greeting"""
    # Complex but working
    import time
    return f"Hello at {time.strftime('%H:%M')}!"
''',
        '''def hello_world():
    """Return a greeting with emoji"""
    return "🎉 Hello, Explorer! 🚀"
''',
        """def hello_world():
    # TODO: implement this
    raise NotImplementedError("Coming soon!")
""",
        """def hello_world():
    return None  # Oops!
""",
        """def hello_world():
    return 42  # Wrong type!
""",
    ],
}


def run(prompt_path: Path, out_path: Path, strategy: str = None) -> bool:
    """Mock Cursor CLI - genererar kod baserat på strategi"""
    # Läs prompt (för framtida användning)
    prompt_content = prompt_path.read_text()

    # Välj strategi om inte given
    if strategy is None:
        strategy = random.choice(list(TEMPLATES_BY_STRATEGY.keys()))

    # Bestäm vilken typ av kod vi ska generera
    if "fizzbuzz" in prompt_content.lower():
        # FizzBuzz templates
        fizzbuzz_templates = {
            "conservative": [
                '''def fizzbuzz(n: int) -> str:
    """FizzBuzz implementation"""
    if n % 3 == 0 and n % 5 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    else:
        return str(n)
''',
                """def fizzbuzz(n: int) -> str:
    if n % 15 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    return str(n)
""",
            ],
            "balanced": [
                '''def fizzbuzz(n: int) -> str:
    """FizzBuzz with clear logic"""
    result = ""
    if n % 3 == 0:
        result += "Fizz"
    if n % 5 == 0:
        result += "Buzz"
    return result if result else str(n)
''',
                """def fizzbuzz(n: int) -> str:
    # FizzBuzz implementation
    if n % 3 == 0 and n % 5 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    return str(n)
""",
            ],
            "exploratory": [
                '''def fizzbuzz(n: int) -> str:
    """FizzBuzz with list comprehension"""
    return "FizzBuzz" if n % 15 == 0 else "Fizz" if n % 3 == 0 else "Buzz" if n % 5 == 0 else str(n)
''',
                """def fizzbuzz(n: int) -> str:
    # This will fail!
    return "Fizz"
""",
                """def fizzbuzz(n: int) -> str:
    raise NotImplementedError("Coming soon!")
""",
                '''def fizzbuzz(n: int) -> str:
    """FizzBuzz with lambda"""
    rules = [(15, "FizzBuzz"), (3, "Fizz"), (5, "Buzz")]
    for divisor, word in rules:
        if n % divisor == 0:
            return word
    return str(n)
''',
            ],
        }
        templates = fizzbuzz_templates.get(strategy, fizzbuzz_templates["balanced"])
    elif (
        "add_numbers" in prompt_content.lower()
        or "calculator" in prompt_content.lower()
    ):
        # Calculator templates
        calculator_templates = {
            "conservative": [
                '''def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
''',
                """def add_numbers(a: int, b: int) -> int:
    return a + b
""",
            ],
            "balanced": [
                '''def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    result = a + b
    return result
''',
                """def add_numbers(a: int, b: int) -> int:
    # Simple addition
    return a + b
""",
            ],
            "exploratory": [
                """def add_numbers(a: int, b: int) -> int:
    import math
    return math.fsum([a, b])
""",
                """def add_numbers(a: int, b: int) -> int:
    pass  # This will fail!
""",
                """def add_numbers(a: int, b: int) -> int:
    return a * b  # Wrong operation!
""",
                """def add_numbers(a: int, b: int) -> int:
    raise NotImplementedError("Coming soon!")
""",
                """def add_numbers(a: int, b: int) -> int:
    return "wrong type"  # Wrong return type!
""",
            ],
        }
        templates = calculator_templates.get(strategy, calculator_templates["balanced"])
    else:
        # Hello world templates
        templates = TEMPLATES_BY_STRATEGY.get(
            strategy, TEMPLATES_BY_STRATEGY["balanced"]
        )

    code = random.choice(templates)

    # Skriv till output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code)

    return True
