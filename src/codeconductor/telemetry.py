from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.enabled = os.getenv("CC_TELEMETRY", "0") == "1"
        self.path = Path(artifacts_dir) / "telemetry" / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def log(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        rec = {"ts": time.time(), "event": event, **payload}
        line = json.dumps(rec, ensure_ascii=False)
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


_LOGGER: JsonlLogger | None = None


def get_logger() -> JsonlLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = JsonlLogger()
    return _LOGGER
