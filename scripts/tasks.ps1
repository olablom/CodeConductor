Param(
  [ValidateSet('diag-cursor')]
  [string]$Task = 'diag-cursor'
)

switch ($Task) {
  'diag-cursor' {
    python -m codeconductor.cli diag cursor --run
  }
}


