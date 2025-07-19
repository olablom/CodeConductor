import requests
import pathlib

_TEMP = {"conservative": 0.3, "balanced": 0.7, "exploratory": 1.2}
_BASE = "http://localhost:1234"


def is_available() -> bool:
    try:
        return requests.get(f"{_BASE}/v1/models", timeout=2).ok
    except requests.RequestException:
        return False


def generate_code(prompt_path: pathlib.Path, strategy: str) -> str | None:
    payload = {
        "prompt": (f"Write a Python function that:\n{prompt_path.read_text()}\n\ndef "),
        "max_tokens": 500,
        "temperature": _TEMP[strategy],
        "stop": ["```", "\n\n"],
    }
    try:
        r = requests.post(f"{_BASE}/v1/completions", json=payload, timeout=60)
        r.raise_for_status()
        response_data = r.json()
        if response_data.get("choices"):
            text = response_data["choices"][0]["text"].strip()
            # Kontrollera att texten ser ut som kod
            if text and ("def " in text or "return" in text):
                return text
        return None
    except requests.RequestException as e:
        print("[LM Studio]", e)
        return None
    except Exception as e:
        print("[LM Studio] Unexpected error:", e)
        return None
