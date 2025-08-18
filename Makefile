.PHONY: diag-cursor

diag-cursor:
	python -m codeconductor.cli diag cursor --run

.PHONY: stream-smoke
stream-smoke:
	python scripts/stream_smoke.py --start-server --prompt "Hello from streaming test"
