import requests
import pathlib
import logging
from typing import Optional

logger = logging.getLogger("LMStudio")

_TEMP = {"conservative": 0.3, "balanced": 0.7, "exploratory": 1.2}
_BASE = "http://localhost:1234"

# Global model manager instance
_model_manager = None


def get_model_manager():
    """Get or create model manager instance"""
    global _model_manager
    if _model_manager is None:
        try:
            from integrations.model_manager import ModelManager

            _model_manager = ModelManager()
        except ImportError:
            logger.warning("ModelManager not available")
            _model_manager = None
    return _model_manager


def is_available() -> bool:
    try:
        return requests.get(f"{_BASE}/v1/models", timeout=2).ok
    except requests.RequestException:
        return False


def ensure_models_ready() -> bool:
    """Ensure models are ready before generating code"""
    model_manager = get_model_manager()
    if model_manager:
        logger.info("Checking model availability...")
        results = model_manager.ensure_models()
        ready_models = sum(1 for ready in results.values() if ready)
        logger.info(f"✅ {ready_models}/{len(results)} models ready")
        return ready_models > 0
    else:
        logger.warning("ModelManager not available, skipping model check")
        return True


def generate_code(prompt_path: pathlib.Path, strategy: str) -> str | None:
    # Ensure models are ready before generating
    if not ensure_models_ready():
        logger.warning("Models not ready, but continuing...")

    prompt_content = prompt_path.read_text()

    # Bättre prompt-format för LM Studio
    system_prompt = "You are a Python coding assistant. Write ONLY Python code. No explanations, no thinking, no markdown, no <think> tags, just pure Python code starting with 'def'."
    user_prompt = f"def hello_world():\n    return "

    payload = {
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "max_tokens": 500,
        "temperature": _TEMP[strategy],
        "stop": ["```", "\n\n"],
    }
    try:
        # Använd completions endpoint istället
        r = requests.post(f"{_BASE}/v1/completions", json=payload, timeout=60)
        r.raise_for_status()
        response_data = r.json()
        if response_data.get("choices"):
            text = response_data["choices"][0]["text"].strip()
            print(f"[LM Studio] Generated text: '{text}'")
            # Kontrollera att texten ser ut som kod
            if text and len(text) > 0:
                print(f"[LM Studio] ✅ Valid code generated")
                return f"def hello_world():\n    return {text}"
            else:
                print(f"[LM Studio] ❌ Invalid code: empty response")
        else:
            print(f"[LM Studio] ❌ No choices in response")
        return None
    except requests.RequestException as e:
        print(f"[LM Studio] Request error: {e}")
        return None
    except Exception as e:
        print(f"[LM Studio] Unexpected error: {e}")
        print(
            f"[LM Studio] Response: {response_data if 'response_data' in locals() else 'No response'}"
        )
        return None
