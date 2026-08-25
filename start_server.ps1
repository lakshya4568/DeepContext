<#
.SYNOPSIS
    One-Click Launcher for Deep Context Platform Web Studio on Windows.
#>

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $RepoRoot

Write-Host "Starting Deep Context Platform Web Studio..." -ForegroundColor Cyan
Write-Host "Opening http://localhost:8000 in your browser..." -ForegroundColor Yellow

Start-Process "http://localhost:8000"
uv run uvicorn deep_context.api.app:app --host 0.0.0.0 --port 8000 --reload
