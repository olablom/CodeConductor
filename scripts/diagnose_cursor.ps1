Param()

Write-Host "🔍 Diagnosing Cursor connection..." -ForegroundColor Cyan

Write-Host "[1/5] Proxy settings" -ForegroundColor Yellow
try {
    $proxy = [System.Net.WebRequest]::GetSystemWebProxy()
    $testUri = [Uri]"http://localhost"
    $proxyUri = $proxy.GetProxy($testUri)
    Write-Host " System proxy for localhost -> $proxyUri"
}
catch {
    Write-Host " Could not read system proxy: $_" -ForegroundColor Red
}

Write-Host "[2/5] DNS for cursor.sh" -ForegroundColor Yellow
try {
    $dns = Resolve-DnsName cursor.sh -ErrorAction Stop
    Write-Host " DNS OK: $($dns[0].IPAddress)" -ForegroundColor Green
}
catch {
    Write-Host " DNS failed: $_" -ForegroundColor Red
}

Write-Host "[3/5] Local Cursor API (localhost:3000)" -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -TimeoutSec 3 -ErrorAction Stop
    Write-Host " Local API status: $($r.StatusCode)" -ForegroundColor Green
}
catch {
    Write-Host " Local API unreachable: $_" -ForegroundColor Red
}

Write-Host "[4/5] LM Studio models (localhost:1234)" -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "http://localhost:1234/v1/models" -TimeoutSec 3 -ErrorAction Stop
    Write-Host " LM Studio reachable" -ForegroundColor Green
}
catch {
    Write-Host " LM Studio not reachable: $_" -ForegroundColor DarkYellow
}

Write-Host "[5/5] Suggested fixes" -ForegroundColor Yellow
Write-Host " - Restart Cursor completely (tray → Quit)"
Write-Host " - Disable VPN/Proxy; clear caches: netsh winhttp reset proxy; ipconfig /flushdns"
Write-Host " - Set local bypass: setx CURSOR_DISABLE_PROXY 1"
Write-Host " - See docs/CURSOR_TROUBLESHOOTING.md"


