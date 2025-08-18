import asyncio
import os


async def _run() -> None:
    os.environ["CC_QUICK_CI"] = "1"
    os.environ["CC_QUICK_CI_SEED"] = "42"
    from codeconductor.ensemble.single_model_engine import (
        SingleModelEngine,
        SingleModelRequest,
    )

    engine = SingleModelEngine()
    ok = await engine.initialize()
    assert ok is True

    # Binary search prompt should return python defining binary_search
    req = SingleModelRequest(task_description="Please implement binary search")
    resp = await engine.process_request(req)
    assert "binary_search" in resp.content
    await engine.cleanup()


def test_mock_mode_returns_deterministic_code():
    asyncio.run(_run())
