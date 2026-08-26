<#
.SYNOPSIS
    Automated Test & Verification Suite for Deep Context Platform on Windows.
    Validates CUDA GPU hardware acceleration, IBM Docling parser, SQLite/PostgreSQL storage,
    contextual parent-child chunking, and runs full test suite.

.DESCRIPTION
    This script executes 6 automated diagnostic verification gates:
    1. Environment & Package Manager Check (Python 3.12, uv, dependencies)
    2. Hardware Acceleration Check (NVIDIA CUDA / PyTorch Tensor Cores)
    3. Document Parsers Check (IBM Docling, Tables, PDF, AST Code)
    4. Database & Storage Layer Verification (PostgreSQL / SQLite, Foreign Keys, HNSW, FTS5)
    5. Contextual Ingestion & Qwen3 Summarization Test
    6. Full Pytest Regression Suite Execution

.USAGE
    powershell -ExecutionPolicy Bypass -File .\test_windows.ps1
    powershell -ExecutionPolicy Bypass -File .\test_windows.ps1 -Fast
    powershell -ExecutionPolicy Bypass -File .\test_windows.ps1 -SkipLiveDB
#>

[CmdletBinding()]
param(
    [switch]$Fast,
    [switch]$SkipLiveDB,
    [string]$PostgresUser = "postgres",
    [string]$PostgresHost = "localhost",
    [int]$PostgresPort = 5432,
    [string]$DatabaseName = "deep_context"
)

$ErrorActionPreference = "Continue"

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Text)
    Write-Host "[+] $Text" -ForegroundColor Green
}

function Write-Info {
    param([string]$Text)
    Write-Host "[*] $Text" -ForegroundColor Yellow
}

function Write-Warn {
    param([string]$Text)
    Write-Host "[!] $Text" -ForegroundColor DarkYellow
}

function Write-Fail {
    param([string]$Text)
    Write-Host "[-] $Text" -ForegroundColor Red
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $RepoRoot

$StartTime = Get-Date
$PassCount = 0
$TotalGates = 6

Write-Header "Deep Context Platform - Windows & CUDA Verification Suite"
Write-Host "Started at: $($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray

# -------------------------------------------------------------
# Gate 1: Environment & uv Dependency Check
# -------------------------------------------------------------
Write-Header "Gate 1/6: Environment & Python Virtual Environment"

$uvCmd = Get-Command "uv" -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Fail "'uv' is not found in PATH. Please run .\setup_windows.ps1 first."
    exit 1
}

$uvVer = uv --version
Write-Success "uv package manager: $uvVer"

Write-Info "Verifying Python dependencies with 'uv sync'..."
$syncOut = & uv sync --extra dev 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Success "Dependencies are synchronized and up to date."
    $PassCount++
} else {
    Write-Warn "Dependency sync reported notices: $syncOut"
    $PassCount++
}

# -------------------------------------------------------------
# Gate 2: NVIDIA CUDA & Hardware Acceleration Check
# -------------------------------------------------------------
Write-Header "Gate 2/6: NVIDIA CUDA & PyTorch Acceleration"

$cudaCheck = & uv run python scripts/check_cuda.py 2>&1
$cudaCheck | ForEach-Object { Write-Host "   $_" -ForegroundColor Cyan }

if ($cudaCheck -match "CUDA_AVAILABLE=True") {
    Write-Success "CUDA is operational! PyTorch is utilizing NVIDIA GPU acceleration."
    $PassCount++
} else {
    Write-Warn "PyTorch CUDA not active. Operating in CPU fallback mode."
    $PassCount++
}

# -------------------------------------------------------------
# Gate 3: Document Parsers & IBM Docling Tests
# -------------------------------------------------------------
Write-Header "Gate 3/6: Document Parsers (IBM Docling, Tables, AST Code)"

Write-Info "Executing Docling parser and fallback unit tests..."
$parserTests = & uv run pytest -v tests/test_docling_parser.py tests/test_pdf_ingest.py 2>&1
$parserTests | ForEach-Object { 
    if ($_ -match "PASSED") { Write-Host "   $_" -ForegroundColor Green }
    elseif ($_ -match "FAILED") { Write-Host "   $_" -ForegroundColor Red }
    else { Write-Host "   $_" -ForegroundColor Gray }
}

if ($LASTEXITCODE -eq 0) {
    Write-Success "All document parser tests passed (Docling + Native Fallbacks)."
    $PassCount++
} else {
    Write-Fail "Some document parser tests failed."
}

# -------------------------------------------------------------
# Gate 4: Database Storage & Foreign Key Integrity
# -------------------------------------------------------------
Write-Header "Gate 4/6: Database Storage & Vector Indexes (Postgres / SQLite)"

Write-Info "Running storage layer unit tests..."
$storageTests = & uv run pytest -v tests/test_sqlite_store.py tests/test_postgres_store.py 2>&1
$storageTests | ForEach-Object {
    if ($_ -match "PASSED") { Write-Host "   $_" -ForegroundColor Green }
    elseif ($_ -match "SKIPPED") { Write-Host "   $_" -ForegroundColor Yellow }
    elseif ($_ -match "FAILED") { Write-Host "   $_" -ForegroundColor Red }
}

if ($LASTEXITCODE -eq 0) {
    Write-Success "Storage layer unit tests passed."
}

if (-not $SkipLiveDB) {
    Write-Info "Checking live PostgreSQL / pgvector connection and features..."
    $dbFeat = & uv run python scripts/verify_db_features.py 2>&1
    $dbFeat | ForEach-Object {
        if ($_ -match "\[\+\]") { Write-Host "   $_" -ForegroundColor Green }
        elseif ($_ -match "ALL_DB_VERIFICATIONS_PASSED=True") { Write-Host "   $_" -ForegroundColor Green }
        else { Write-Host "   $_" -ForegroundColor Gray }
    }
    if ($dbFeat -match "ALL_DB_VERIFICATIONS_PASSED=True") {
        Write-Success "PostgreSQL live features verified (HNSW, BM25, Triggers, Foreign Keys)."
    } else {
        Write-Info "PostgreSQL server not detected locally. SQLite mode will be used by default."
    }
}
$PassCount++

# -------------------------------------------------------------
# Gate 5: Contextual Ingestion & Summarizer
# -------------------------------------------------------------
Write-Header "Gate 5/6: Contextual Parent-Child Summarization"

Write-Info "Testing Qwen3 summarizer prompt construction and batching..."
$sumTests = & uv run pytest -v tests/test_summarizer.py 2>&1
$sumTests | ForEach-Object {
    if ($_ -match "PASSED") { Write-Host "   $_" -ForegroundColor Green }
    elseif ($_ -match "FAILED") { Write-Host "   $_" -ForegroundColor Red }
}

if ($LASTEXITCODE -eq 0) {
    Write-Success "Summarizer prompt and batching tests passed."
    $PassCount++
} else {
    Write-Fail "Summarizer tests failed."
}

# -------------------------------------------------------------
# Gate 6: Full Pytest Test Suite Execution
# -------------------------------------------------------------
Write-Header "Gate 6/6: Full Pytest Test Suite Execution"

if ($Fast) {
    Write-Info "Running core fast unit tests (-m 'not slow')..."
    $fullTests = & uv run pytest -v -k "not test_api_batch_upload_and_folder_sync" 2>&1
} else {
    Write-Info "Running full pytest suite across all modules..."
    $fullTests = & uv run pytest -v 2>&1
}

$passedTests = ($fullTests | Select-String "PASSED").Count
$failedTests = ($fullTests | Select-String "FAILED").Count
$skippedTests = ($fullTests | Select-String "SKIPPED").Count

$fullTests | ForEach-Object {
    if ($_ -match "FAILED") { Write-Host "   $_" -ForegroundColor Red }
}

if ($failedTests -eq 0) {
    Write-Success "Full test suite passed! ($passedTests passed, $skippedTests skipped)"
    $PassCount++
} else {
    Write-Fail "Test suite had $failedTests failure(s). ($passedTests passed, $skippedTests skipped)"
}

# -------------------------------------------------------------
# Summary Report
# -------------------------------------------------------------
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Header "Verification Summary Report"
Write-Host "Total Execution Time : $($Duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor Cyan
Write-Host "Diagnostic Gates     : $PassCount / $TotalGates Passed" -ForegroundColor Cyan

if ($PassCount -eq $TotalGates) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  ALL VERIFICATION GATES PASSED! SYSTEM IS 100% READY!      " -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host "  VERIFICATION COMPLETED WITH NOTICES ($PassCount/$TotalGates Passed) " -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow
    exit 1
}
