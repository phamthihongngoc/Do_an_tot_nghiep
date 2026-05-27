$ErrorActionPreference = "Stop"

$demoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $demoDir
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python virtual environment not found: $python"
}

if (-not $env:DEMO_MODEL) {
    $env:DEMO_MODEL = "transformer"
}
if (-not $env:DEMO_TARGET_LEN) {
    $env:DEMO_TARGET_LEN = "201"
}
if (-not $env:DEMO_SAMPLE_RATE_HZ) {
    $env:DEMO_SAMPLE_RATE_HZ = "50"
}
if (-not $env:DEMO_WINDOW_SECONDS) {
    $env:DEMO_WINDOW_SECONDS = "4.0"
}
if (-not $env:DEMO_PREDICT_INTERVAL_SECONDS) {
    $env:DEMO_PREDICT_INTERVAL_SECONDS = "3.5"
}
if (-not $env:DEMO_ACTION_COOLDOWN_SECONDS) {
    $env:DEMO_ACTION_COOLDOWN_SECONDS = "3.5"
}
if (-not $env:DEMO_CONFIDENCE_THRESHOLD) {
    $env:DEMO_CONFIDENCE_THRESHOLD = "0.50"
}
if (-not $env:DEMO_MIN_GYRO_RMS_DPS) {
    $env:DEMO_MIN_GYRO_RMS_DPS = "6.0"
}
if (-not $env:DEMO_MIN_ACCEL_DYNAMIC_G) {
    $env:DEMO_MIN_ACCEL_DYNAMIC_G = "0.035"
}
if (-not $env:DEMO_CONTEXT_G2_MIN_CONFIDENCE) {
    $env:DEMO_CONTEXT_G2_MIN_CONFIDENCE = "0.08"
}
if (-not $env:DEMO_CONTEXT_G2_OVER_STOP_MIN_CONFIDENCE) {
    $env:DEMO_CONTEXT_G2_OVER_STOP_MIN_CONFIDENCE = "0.005"
}
if (-not $env:DEMO_GESTURE_CAPTURE_SECONDS) {
    $env:DEMO_GESTURE_CAPTURE_SECONDS = "3.5"
}
if (-not $env:DEMO_GESTURE_RECOVERY_SECONDS) {
    $env:DEMO_GESTURE_RECOVERY_SECONDS = "0.8"
}
if (-not $env:DEMO_STOP_CONFIRM_SECONDS) {
    $env:DEMO_STOP_CONFIRM_SECONDS = "6.0"
}
if (-not $env:DEMO_STOP_CONFIDENCE_THRESHOLD) {
    $env:DEMO_STOP_CONFIDENCE_THRESHOLD = "0.995"
}

Set-Location $root

$port = 8000
if ($env:DEMO_PORT) {
    $port = [int]$env:DEMO_PORT
}
while (Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue) {
    $port += 1
}

function Test-PortFree {
    param([int]$Port)
    $listener = $null
    try {
        $address = [System.Net.IPAddress]::Parse("127.0.0.1")
        $listener = [System.Net.Sockets.TcpListener]::new($address, $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener -ne $null) {
            $listener.Stop()
        }
    }
}

while (-not (Test-PortFree -Port $port)) {
    $port += 1
}

function Resolve-CocCocBrowser {
    if ($env:DEMO_BROWSER_PATH) {
        if (Test-Path -LiteralPath $env:DEMO_BROWSER_PATH) {
            return $env:DEMO_BROWSER_PATH
        }
        Write-Warning "DEMO_BROWSER_PATH khong ton tai: $env:DEMO_BROWSER_PATH"
    }

    $browserRoots = @(
        $env:LOCALAPPDATA,
        $env:ProgramFiles,
        ${env:ProgramFiles(x86)}
    )
    $candidates = foreach ($browserRoot in $browserRoots) {
        if ($browserRoot) {
            Join-Path $browserRoot "CocCoc\Browser\Application\browser.exe"
        }
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    return $null
}

function Start-BrowserWhenReady {
    param(
        [string]$BrowserPath,
        [string]$Url
    )

    Start-Job -ScriptBlock {
        param(
            [string]$BrowserPath,
            [string]$Url
        )

        $deadline = (Get-Date).AddSeconds(60)
        $ready = $false

        while ((Get-Date) -lt $deadline) {
            try {
                $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 1
                if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                    $ready = $true
                    break
                }
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }

        if ($ready) {
            Start-Process -FilePath $BrowserPath -ArgumentList $Url
        }
    } -ArgumentList $BrowserPath, $Url | Out-Null
}

$url = "http://127.0.0.1:$port/"
$browser = Resolve-CocCocBrowser

if ($browser) {
    Write-Host "Se mo Coc Coc: $url"
    Start-BrowserWhenReady -BrowserPath $browser -Url $url
} else {
    Write-Warning "Khong tim thay Coc Coc. Cai Coc Coc hoac dat DEMO_BROWSER_PATH toi browser.exe."
}

Write-Host "Smart Home demo: $url"
& $python -m uvicorn demo.backend.app:app --host 127.0.0.1 --port $port
