Param(
  [int[]]$Ports = @(11434, 3000),
  [string]$ArtifactsDir = 'artifacts/diagnostics'
)

$ErrorActionPreference = 'Continue'

New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $ArtifactsDir ("diagnose_{0}.txt" -f $stamp)
$latest = Join-Path $ArtifactsDir 'diagnose_latest.txt'

function Write-Log([string]$msg) {
  $msg | Tee-Object -FilePath $log -Append | Out-Null
}

Write-Log 'Diagnosing Cursor / Local API'
Write-Log ("=== {0} ===" -f (Get-Date))

Write-Log ''
Write-Log '[1/5] Proxy settings'
try {
  $proxy = [System.Net.WebRequest]::GetSystemWebProxy()
  $testUri = [Uri]'http://localhost'
  $proxyUri = $proxy.GetProxy($testUri)
  Write-Log (" System proxy for localhost -> {0}" -f $proxyUri)
} catch {
  Write-Log (" Could not read system proxy: {0}" -f $_.Exception.Message)
}

Write-Log ''
Write-Log '[2/5] DNS for cursor.sh'
try {
  $dns = Resolve-DnsName cursor.sh -ErrorAction Stop
  Write-Log (" DNS OK: {0}" -f $dns[0].IPAddress)
} catch {
  Write-Log (" DNS failed: {0}" -f $_.Exception.Message)
}

Write-Log ''
Write-Log '[3/5] Port checks'
foreach ($p in $Ports) {
  $tnc = Test-NetConnection -ComputerName 127.0.0.1 -Port $p
  Write-Log (" Port {0} -> TcpTestSucceeded={1}" -f $p, $tnc.TcpTestSucceeded)
  if ($tnc.TcpTestSucceeded) {
    try {
      if ($p -eq 11434) {
        try {
          $resp = Invoke-WebRequest -Uri 'http://127.0.0.1:11434/health' -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
          Write-Log ("  GET /health -> {0}" -f $resp.StatusCode)
        } catch {
          $sc = $null
          if ($_.Exception -and $_.Exception.Response -and $_.Exception.Response.StatusCode) {
            $sc = [int]$_.Exception.Response.StatusCode
          }
          if ($sc -eq 404) {
            Write-Log '  GET /health -> 404 (OK: service up, no health route)'
          } else {
            Write-Log ("  GET /health failed: {0}" -f $_.Exception.Message)
          }
        }
      } else {
        $ok = $false
        foreach ($path in @('/api/health', '/health')) {
          try {
            $u = "http://127.0.0.1:$p$path"
            $resp = Invoke-WebRequest -Uri $u -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
            Write-Log ("  GET {0} -> {1}" -f $path, $resp.StatusCode)
            $ok = $true
            break
          } catch {
            Write-Log ("  GET {0} failed: {1}" -f $path, $_.Exception.Message)
          }
        }
        if (-not $ok) { Write-Log '  No health endpoint responded' }
      }
    } catch {
      Write-Log ("  HTTP check error: {0}" -f $_.Exception.Message)
    }
  }
}

Write-Log ''
Write-Log '[4/5] LM Studio models (localhost:1234)'
try {
  $r = Invoke-WebRequest -Uri 'http://127.0.0.1:1234/v1/models' -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
  Write-Log ' LM Studio reachable'
} catch {
  Write-Log (" LM Studio not reachable: {0}" -f $_.Exception.Message)
}

# Auto-detect possible Cursor API port
$suggestedPort = $null
try {
  $procs = Get-Process | Where-Object { $_.ProcessName -match 'Cursor|electron|node|Helper' }
  $cand = @{}
  foreach ($p in $procs) {
    $conns = Get-NetTCPConnection -State Listen -OwningProcess $p.Id -ErrorAction SilentlyContinue
    foreach ($c in $conns) { $cand[$c.LocalPort] = $true }
  }
  foreach ($cp in ($cand.Keys | Sort-Object)) {
    if ($cp -eq 11434) { continue }
    foreach ($path in @('/api/health','/health')) {
      try {
        $u = "http://127.0.0.1:$cp$path"
        $resp = Invoke-WebRequest -Uri $u -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        $suggestedPort = $cp; break
      } catch { }
    }
    if ($suggestedPort) { break }
  }
} catch { }

Write-Log ''
Write-Log '[5/5] Suggested fixes'
Write-Log ' - Restart Cursor completely (tray -> Quit)'
Write-Log ' - Disable VPN/Proxy; clear caches: netsh winhttp reset proxy; ipconfig /flushdns'
Write-Log ' - Set local bypass: setx CURSOR_DISABLE_PROXY 1'
if ($suggestedPort) {
  Write-Log (" - Detected Cursor API port: {0}" -f $suggestedPort)
  Write-Log (" - To set: setx CURSOR_API_BASE http://127.0.0.1:{0}" -f $suggestedPort)
}
Write-Log ' - See docs/README_TROUBLESHOOTING_LINKS.md'

Copy-Item -Path $log -Destination $latest -Force -ErrorAction SilentlyContinue | Out-Null
Write-Host 'Diagnostics written to log file' -ForegroundColor Cyan


