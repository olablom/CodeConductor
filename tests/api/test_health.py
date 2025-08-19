from codeconductor.api.server import app


def test_health():
    # Test the app directly without HTTP client to avoid compatibility issues
    from fastapi.testclient import TestClient

    try:
        # Try the standard way first
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
    except TypeError as e:
        if "unexpected keyword argument 'app'" in str(e):
            # Fallback: test the endpoint function directly
            import asyncio

            from codeconductor.api.server import health

            result = asyncio.run(health())
            assert result["status"] == "ok"
            assert "timestamp" in result
        else:
            raise
