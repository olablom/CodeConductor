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

    # Läs prompt-filen och använd den
    prompt_content = prompt_path.read_text()
    print(f"[LM Studio DEBUG] Using prompt from file: {prompt_path}")
    print(
        f"[LM Studio DEBUG] Prompt content (first 100 chars): {prompt_content[:100]}..."
    )

    # Förbättra prompten för att få Python-kod
    enhanced_prompt = f"""Create Python code for this request:

{prompt_content}

def"""

    code_prompt = enhanced_prompt

    payload = {
        "prompt": code_prompt,
        "max_tokens": 500,  # Öka för längre svar
        "temperature": _TEMP[strategy],
        "stop": ["\n\n", "```"],  # Enklare stop-tokens
        "stream": False,
    }
    try:
        print(f"[LM Studio DEBUG] Sending request to {_BASE}/v1/completions")
        print(f"[LM Studio DEBUG] Payload: {payload}")

        # Använd completions endpoint istället
        r = requests.post(f"{_BASE}/v1/completions", json=payload, timeout=60)
        print(f"[LM Studio DEBUG] Response status: {r.status_code}")
        r.raise_for_status()
        response_data = r.json()
        print(f"[LM Studio DEBUG] Response data: {response_data}")
        if response_data.get("choices"):
            text = response_data["choices"][0]["text"].strip()
            print(f"[LM Studio] Generated text: '{text}'")

            if text:
                # Lägg till 'def' om det saknas
                if not text.strip().startswith("def "):
                    text = f"def {text.strip()}"

                print(f"[LM Studio] ✅ Valid code generated")
                return text

        print(f"[LM Studio] ❌ No valid response")
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
