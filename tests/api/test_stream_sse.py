import json
import sys
from pathlib import Path

# Ensure local package path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

try:
    # Prefer FastAPI wrapper when compatible
    from fastapi.testclient import TestClient  # type: ignore
except Exception:  # pragma: no cover
    from starlette.testclient import TestClient  # type: ignore
from codeconductor.api.server import app


def iter_sse_lines(resp):
    for chunk in resp.iter_lines():
        if not chunk:
            continue
        # Handle both bytes and string chunks
        if isinstance(chunk, bytes):
            if chunk.startswith(b"data: "):
                payload = chunk[len(b"data: ") :].decode("utf-8")
                data = json.loads(payload)
                yield data
        else:
            # Handle string chunks
            if chunk.startswith("data: "):
                payload = chunk[len("data: ") :]
                data = json.loads(payload)
                yield data


def test_stream_basic_order_and_done():
    try:
        # Try the standard way first
        client = TestClient(app)
        with client.stream("GET", "/stream", params={"prompt": "a b c"}) as resp:
            assert resp.status_code == 200
            tokens = []
            done_seen = False
            last_seq = 0
            for evt in iter_sse_lines(resp):
                assert evt["seq"] > last_seq
                last_seq = evt["seq"]
                if evt["done"]:
                    done_seen = True
                    break
                else:
                    tokens.append(evt["token"])
            assert done_seen
            assert tokens == ["a ", "b ", "c "]
    except TypeError as e:
        if "unexpected keyword argument 'app'" in str(e):
            # Skip this test if TestClient is not compatible
            import pytest

            pytest.skip("TestClient not compatible with current httpx version")
        else:
            raise
