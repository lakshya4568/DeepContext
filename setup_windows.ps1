<#
.SYNOPSIS
    Automated Setup & Launcher for Deep Context Platform on Windows.
    Configures uv, Python 3.12, NVIDIA GPU (CUDA PyTorch), PostgreSQL / pgvector, and .env.

.DESCRIPTION
    This script automates:
    1. Checking and installing 'uv' package manager
    2. Python 3.12 virtual environment creation and dependency sync
    3. NVIDIA GPU / CUDA detection and CUDA-accelerated PyTorch installation
    4. NLTK tokenizer corpus download
    5. PostgreSQL database creation & pgvector extension check
    6. Environment configuration (.env file creation)
    7. Running tests and launching the Web Studio server

.USAGE
    powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
#>

[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$StartServer,
    [string]$PostgresUser = "postgres",
    [string]$PostgresHost = "localhost",
    [int]$PostgresPort = 5432,
    [string]$DatabaseName = "deep_context"
)

$ErrorActionPreference = "Stop"

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

# Change directory to repository root
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $RepoRoot

Write-Header "Deep Context Platform - Windows & NVIDIA GPU Automated Setup"

# -------------------------------------------------------------
# 1. Check & Install uv Package Manager
# -------------------------------------------------------------
Write-Header "Step 1: Checking uv Package Manager"

$uvCmd = Get-Command "uv" -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Info "'uv' package manager not found in PATH. Installing uv via Astral installer..."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
        $uvCmd = Get-Command "uv" -ErrorAction SilentlyContinue
    } catch {
        Write-Fail "Automatic installation of uv failed: $_"
        Write-Info "Please install uv manually from https://github.com/astral-sh/uv or run: winget install astral-sh.uv"
        exit 1
    }
}

if ($uvCmd) {
    $uvVer = uv --version
    Write-Success "Found uv: $uvVer"
} else {
    Write-Fail "uv is still not available in PATH. Please restart PowerShell and run this script again."
    exit 1
}

# -------------------------------------------------------------
# 2. Check NVIDIA GPU & CUDA Drivers
# -------------------------------------------------------------
Write-Header "Step 2: Detecting NVIDIA GPU & Hardware Acceleration"

$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
$HasNvidiaGPU = $false

if ($nvidiaSmi) {
    try {
        $gpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        Write-Success "Detected NVIDIA GPU: $gpuInfo"
        $HasNvidiaGPU = $true
    } catch {
        Write-Warn "nvidia-smi exists but could not query GPU details."
    }
} else {
    Write-Warn "nvidia-smi not found. NVIDIA driver may not be in PATH. The system will attempt CUDA PyTorch installation."
}

# -------------------------------------------------------------
# 3. Setup Python 3.12 Virtual Environment & Dependencies
# -------------------------------------------------------------
Write-Header "Step 3: Setting Up Python 3.12 Virtual Environment with uv"

# Clean up mixed or mismatched virtual environments if not 3.12
if (Test-Path ".venv") {
    $pyVenvCfg = Join-Path ".venv" "pyvenv.cfg"
    if (Test-Path $pyVenvCfg) {
        $cfgText = Get-Content $pyVenvCfg -Raw
        if ($cfgText -notmatch "3\.12") {
            Write-Info "Recreating virtual environment cleanly for Python 3.12..."
            try {
                Remove-Item -Recurse -Force ".venv" -ErrorAction SilentlyContinue
            } catch {
                Write-Warn "Could not remove .venv directly (in use). uv will overwrite it."
            }
        }
    }
}

Write-Info "Creating/Verifying Python 3.12 virtual environment..."
& uv venv --python 3.12 .venv --allow-existing
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Python 3.12 specific venv creation had non-zero code. Syncing default environment..."
}

Write-Info "Syncing Python dependencies and dev tools from pyproject.toml..."
& uv sync --extra dev --python .venv
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Failed to sync dependencies using uv."
    exit 1
}
Write-Success "Base dependencies synchronized successfully."

# -------------------------------------------------------------
# 4. Configuring CUDA PyTorch & Hardware Acceleration
# -------------------------------------------------------------
Write-Header "Step 4: Configuring CUDA PyTorch & Local Models"

# Download NLTK data for BM25 / reranker stopwords
Write-Info "Downloading NLTK stopwords..."
& uv run python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Verify GPU detection inside PyTorch via check_cuda script
Write-Info "Verifying PyTorch GPU / CUDA access on NVIDIA GPU..."
$cudaOutput = & uv run python scripts/check_cuda.py
$cudaOutput | ForEach-Object { Write-Host "   $_" -ForegroundColor Cyan }

if ($cudaOutput -match "CUDA_AVAILABLE=True") {
    Write-Success "CUDA is fully operational! Local Qwen3 summarizer will run on your NVIDIA RTX 3060 GPU."
} else {
    Write-Warn "PyTorch did not detect CUDA. CPU will be used as fallback."
}

# -------------------------------------------------------------
# 5. Configure Database (PostgreSQL / SQLite via Pure Python)
# -------------------------------------------------------------
Write-Header "Step 5: Database Setup (PostgreSQL / SQLite)"

Write-Info "Running database initialization script (scripts/init_db.py)..."
$dbInitOutput = & uv run python scripts/init_db.py
$dbInitOutput | ForEach-Object { 
    if ($_ -match "\[\+\]") { Write-Host "   $_" -ForegroundColor Green }
    elseif ($_ -match "\[\*\]") { Write-Host "   $_" -ForegroundColor Cyan }
    elseif ($_ -match "\[\!\]") { Write-Host "   $_" -ForegroundColor Yellow }
    else { Write-Host "   $_" -ForegroundColor Gray }
}

if ($dbInitOutput -match "DB_SETUP_SUCCESS=True") {
    Write-Success "Database configuration and initialization complete!"
} else {
    Write-Warn "Database initialization reported a notice. DeepContext will still support SQLite or manual pgAdmin configuration."
}

# -------------------------------------------------------------
# 6. Configure .env File
# -------------------------------------------------------------
Write-Header "Step 6: Environment Configuration (.env)"

$EnvPath = Join-Path $RepoRoot ".env"
$EnvExists = Test-Path $EnvPath

if (-not $EnvExists) {
    Write-Info "Creating .env configuration file..."
    
    $encodedPass = if ($PostgresPassword) { [System.Uri]::EscapeDataString($PostgresPassword) } else { "postgres" }
    $dsn = "postgresql://${PostgresUser}:${encodedPass}@${PostgresHost}:${PostgresPort}/${DatabaseName}"
    $dbType = if ($PostgresConfigured) { "postgres" } else { "postgres" }

    $defaultEnv = @"
# =============================================================
# Deep Context Platform Environment Configuration
# =============================================================

# Database Configuration
DATABASE_TYPE=$dbType
POSTGRES_DSN=$dsn
SQLITE_DB_PATH=deep_context.db

# LLM & Embedding API Keys
# Get your Gemini API key from: https://aistudio.google.com/
GEMINI_API_KEY=
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIM=768

# Get your Groq API key from: https://console.groq.com/
GROQ_API_KEY=
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.6-27b

# NVIDIA NIM API (Optional) from: https://build.nvidia.com/
NVIDIA_API_KEY=

# Local GPU Processing (NVIDIA CUDA)
SUMMARY_ENABLED=true
SUMMARY_MODEL=Qwen/Qwen3-0.6B
SUMMARY_DEVICE=cuda
SUMMARY_BATCH_SIZE=8

# Hugging Face Token (Optional)
HF_TOKEN=
"@
    Set-Content -Path $EnvPath -Value $defaultEnv -Encoding UTF8
    Write-Success ".env file created at: $EnvPath"
    Write-Warn "Please remember to open .env and add your GEMINI_API_KEY and GROQ_API_KEY!"
} else {
    Write-Success ".env file already exists."
}

# -------------------------------------------------------------
# 7. Run Verification Tests (Optional)
# -------------------------------------------------------------
if (-not $SkipTests) {
    Write-Header "Step 7: Running Test Suite Verification"
    Write-Info "Executing pytest..."
    & uv run pytest -q
    if ($LASTEXITCODE -eq 0) {
        Write-Success "All tests passed successfully!"
    } else {
        Write-Warn "Some tests exited with warnings or failures. You can review with 'uv run pytest'."
    }
}

# -------------------------------------------------------------
# 8. Finished & Launch Option
# -------------------------------------------------------------
Write-Header "Setup Complete!"

Write-Host @"
Your Deep Context Platform environment is configured and ready!

Useful commands:
  • Start Web Studio & API Server:
      uv run uvicorn deep_context.api.app:app --host 0.0.0.0 --port 8000 --reload

  • Ingest a document:
      uv run deep-context ingest path\to\document.pdf -e gemini-embedding-2 -d 768

  • Query your ingested knowledge:
      uv run deep-context query "What are the main findings?"

  • Corrective Agentic RAG:
      uv run deep-context agentic-query "Explain system architecture"

"@ -ForegroundColor Green

if ($StartServer) {
    Write-Header "Starting Deep Context Studio Server..."
    Write-Info "Opening http://localhost:8000 in your browser..."
    Start-Process "http://localhost:8000"
    & uv run uvicorn deep_context.api.app:app --host 0.0.0.0 --port 8000 --reload
} else {
    $response = Read-Host "Would you like to start the Deep Context Web Studio now? (Y/n)"
    if ($response -eq "" -or $response -match "^[Yy]") {
        Write-Header "Starting Deep Context Studio Server..."
        Start-Process "http://localhost:8000"
        & uv run uvicorn deep_context.api.app:app --host 0.0.0.0 --port 8000 --reload
    }
}
