#!/usr/bin/env python3
"""
Robust Ensemble Engine for CodeConductor PoC
With retries, strict JSON validation, and comprehensive error handling
"""

import json
import time
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_ENDPOINTS = ["http://localhost:11434/v1/completions",  # CodeLlama
                   "http://localhost:1234/v1/completions",   # Mistral
                   "http://localhost:2345/v1/completions"]   # Phi3

def call_model(url: str, prompt: str, timeout: float) -> str:
    # här byter du ut mot ditt HTTP‐client‐anrop, bara ett exempel:
    import requests
    return requests.post(url, json={"prompt": prompt}, timeout=timeout).text

def robust_ensemble(prompt: str) -> List[dict]:
    results = []
    for url in MODEL_ENDPOINTS:
        text = None
        for attempt in range(3):
            try:
                text = call_model(url, prompt, timeout=5)
                break
            except Exception as e:
                backoff = 2 ** attempt
                logger.warning(f"Timeout/fel mot {url}, attempt {attempt+1}, väntar {backoff}s…")
                time.sleep(backoff)
        if text is None:
            logger.error(f"Alla retries misslyckades mot {url}, hoppar över.")
            continue

        # strikt JSON‐validering
        try:
            spec = json.loads(text)
        except json.JSONDecodeError:
            logger.debug(f"Råmodell‐svar från {url}: {text!r}")
            logger.error("JSON parsing failed—fallback springer igång.")
            continue

        results.append({
            "model": url,
            "spec": spec,
            "confidence": 1.0  # placeholder
        })
    return results

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1]
    out = robust_ensemble(prompt)
    print(json.dumps(out, indent=2)) 