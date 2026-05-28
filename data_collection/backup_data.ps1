$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$backupDir = Join-Path $PSScriptRoot "data\backup"
$rawDir = Join-Path $PSScriptRoot "data\raw"
$manifestPath = Join-Path $PSScriptRoot "data\manifest.csv"

New-Item -ItemType Directory -Force -Path $backupDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupPath = Join-Path $backupDir "manual_backup_$timestamp.zip"

$items = @()
if (Test-Path -LiteralPath $rawDir) {
    $items += "data\raw"
}
if (Test-Path -LiteralPath $manifestPath) {
    $items += "data\manifest.csv"
}

if ($items.Count -eq 0) {
    Write-Host "Chua co data/raw hoac manifest.csv de backup."
    exit 0
}

Compress-Archive -Path $items -DestinationPath $backupPath -Force
Write-Host "Da tao backup: $backupPath"
