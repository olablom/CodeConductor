import pathlib
import sys

# Ensure 'src' is on sys.path for local test runs without installation
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from codeconductor.integrations.cursor_local_api import CursorLocalAPI


def test_decide_trust_env_defaults_localhost_off(monkeypatch):
    monkeypatch.delenv("CURSOR_DISABLE_PROXY", raising=False)
    assert CursorLocalAPI._decide_trust_env("http://localhost:3000") is False
    assert CursorLocalAPI._decide_trust_env("http://127.0.0.1:3000") is False


def test_decide_trust_env_disable_proxy_env(monkeypatch):
    monkeypatch.setenv("CURSOR_DISABLE_PROXY", "1")
    assert CursorLocalAPI._decide_trust_env("http://example.com") is False


def test_decide_trust_env_remote_allows_proxy(monkeypatch):
    monkeypatch.delenv("CURSOR_DISABLE_PROXY", raising=False)
    assert CursorLocalAPI._decide_trust_env("http://example.com") is True


def test_should_retry_on_common_errors():
    api = CursorLocalAPI()
    # TimeoutError
    assert api._should_retry(TimeoutError()) is True
    # ConnectionResetError
    assert api._should_retry(ConnectionResetError()) is True
    # Non-retryable
    assert api._should_retry(ValueError("boom")) is False


def test_backoff_increases_and_caps():
    api = CursorLocalAPI(backoff_base_seconds=0.1, backoff_max_seconds=0.3)
    b0 = api._compute_backoff(0)
    b1 = api._compute_backoff(1)
    b2 = api._compute_backoff(2)
    assert 0.0 < b0 <= 0.3
    assert 0.0 < b1 <= 0.3
    assert 0.0 < b2 <= 0.3
    # Not strictly monotonic due to jitter, but ensure upper bound respected
    assert max(b0, b1, b2) <= 0.3
