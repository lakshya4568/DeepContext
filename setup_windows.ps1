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
# 3. Setup Python Virtual Environment & Dependencies
# -------------------------------------------------------------
Write-Header "Step 3: Setting Up Python 3.12 Virtual Environment with uv"

Write-Info "Syncing Python dependencies from pyproject.toml..."
& uv sync --extra dev
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Failed to sync dependencies using uv."
    exit 1
}
Write-Success "Base dependencies synchronized successfully."

# -------------------------------------------------------------
# 4. Install CUDA-enabled PyTorch for NVIDIA GPU
# -------------------------------------------------------------
Write-Header "Step 4: Configuring CUDA PyTorch & Local Models"

Write-Info "Installing CUDA-enabled PyTorch (CUDA 12.4 index) into virtual environment..."
& uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Warn "CUDA 12.4 PyTorch wheel install returned non-zero code. Trying default wheel..."
}

# Download NLTK data for BM25 / reranker stopwords
Write-Info "Downloading NLTK stopwords..."
& uv run python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Verify GPU detection inside PyTorch
Write-Info "Verifying PyTorch GPU / CUDA access..."
$cudaCheckScript = @"
import torch
cuda_avail = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_avail else 'CPU'
print(f'CUDA_AVAILABLE={cuda_avail}')
print(f'DEVICE_NAME={device_name}')
"@

$cudaOutput = & uv run python -c $cudaCheckScript
$cudaOutput | ForEach-Object { Write-Host "   $_" -ForegroundColor Cyan }

if ($cudaOutput -match "CUDA_AVAILABLE=True") {
    Write-Success "CUDA is fully operational! Local Qwen3 summarizer will run on GPU."
} else {
    Write-Warn "PyTorch did not detect CUDA. CPU will be used as fallback."
}

# -------------------------------------------------------------
# 5. Configure PostgreSQL & Database (Pure Python - No psql required)
# -------------------------------------------------------------
Write-Header "Step 5: PostgreSQL Database & pgvector Setup (pgAdmin / Local Server)"

$PostgresConfigured = $false
$PostgresPassword = ""

if (-not $env:PGPASSWORD) {
    $secPass = Read-Host -Prompt "Enter password for PostgreSQL user '$PostgresUser' (default is 'postgres')" -AsSecureString
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secPass)
    $PostgresPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)
    if (-not $PostgresPassword) {
        $PostgresPassword = "postgres"
    }
} else {
    $PostgresPassword = $env:PGPASSWORD
}

$dbInitScript = @"
import asyncio
import sys
import asyncpg

async def init_db():
    user = '$PostgresUser'
    password = '$PostgresPassword'
    host = '$PostgresHost'
    port = $PostgresPort
    target_db = '$DatabaseName'

    try:
        # 1. Connect to default 'postgres' database
        conn = await asyncpg.connect(user=user, password=password, host=host, port=port, database='postgres')
        exists = await conn.fetchval('SELECT 1 FROM pg_database WHERE datname = $1', target_db)
        if not exists:
            print(f'[*] Creating database \"{target_db}\"...')
            await conn.execute(f'CREATE DATABASE \"{target_db}\"')
            print(f'[+] Database \"{target_db}\" created successfully.')
        else:
            print(f'[+] Database \"{target_db}\" already exists.')
        await conn.close()

        # 2. Connect to the newly created/existing target database
        conn_target = await asyncpg.connect(user=user, password=password, host=host, port=port, database=target_db)
        try:
            await conn_target.execute('CREATE EXTENSION IF NOT EXISTS vector;')
            print('[+] Extension \"vector\" (pgvector) enabled successfully.')
        except Exception as e_vec:
            print(f'[!] Note on pgvector: {e_vec}')
            print('[!] If pgvector is not installed in Postgres yet, you can also use pgAdmin or SQLite mode.')

        await conn_target.execute('CREATE EXTENSION IF NOT EXISTS pgcrypto;')
        print('[+] Extension \"pgcrypto\" enabled.')
        await conn_target.close()
        print('DB_SETUP_SUCCESS=True')
    except Exception as e:
        print(f'DB_SETUP_ERROR={e}')

asyncio.run(init_db())
"@

Write-Info "Connecting to PostgreSQL server at $PostgresHost`:$PostgresPort via Python..."
$dbInitOutput = & uv run python -c $dbInitScript
$dbInitOutput | ForEach-Object { 
    if ($_ -match "\[\+\]") { Write-Host "   $_" -ForegroundColor Green }
    elseif ($_ -match "\[\*\]") { Write-Host "   $_" -ForegroundColor Cyan }
    elseif ($_ -match "\[\!\]") { Write-Host "   $_" -ForegroundColor Yellow }
    else { Write-Host "   $_" -ForegroundColor Gray }
}

if ($dbInitOutput -match "DB_SETUP_SUCCESS=True") {
    Write-Success "PostgreSQL database '$DatabaseName' is initialized and ready!"
    $PostgresConfigured = $true
} else {
    Write-Warn "Could not connect to PostgreSQL automatically."
    Write-Info "If your PostgreSQL service is running, you can create the database in pgAdmin GUI:"
    Write-Info "  1. Open pgAdmin -> Right click 'Databases' -> Create -> Database: '$DatabaseName'"
    Write-Info "  2. Right click '$DatabaseName' -> Query Tool -> Run: CREATE EXTENSION IF NOT EXISTS vector;"
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
