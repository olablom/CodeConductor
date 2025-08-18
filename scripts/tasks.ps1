Param(
  [ValidateSet('diag-cursor','stream-smoke')]
  [string]$Task = 'diag-cursor',
  [string]$Prompt = 'Hello from streaming test'
)

switch ($Task) {
  'diag-cursor' {
    python -m codeconductor.cli diag cursor --run
  }
  'stream-smoke' {
    python scripts/stream_smoke.py --start-server --prompt "$Prompt"
  }
}
