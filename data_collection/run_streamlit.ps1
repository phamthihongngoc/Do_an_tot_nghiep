$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Write-Host "Khong tim thay .venv. Dang tao moi truong ao..."
    python -m venv .venv
}

$python = ".\.venv\Scripts\python.exe"
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot ".tmp\matplotlib"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& $python -m streamlit --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Thieu thu vien. Dang cai requirements.txt..."
    & $python -m pip install -r requirements.txt
}

Write-Host "Dang chay giao dien Streamlit..."
Write-Host "Mo trinh duyet tai: http://localhost:8501"
& $python -m streamlit run streamlit_app.py --server.headless=false --server.port=8501 --browser.gatherUsageStats=false
