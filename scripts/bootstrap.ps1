$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repositoryRoot

try {
    & python scripts/verify_environment.py
    if ($LASTEXITCODE -ne 0) {
        throw "Environment validation failed. Resolve the diagnostics above and rerun setup."
    }

    & uv sync --group dev
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE."
    }

    & npm install
    if ($LASTEXITCODE -ne 0) {
        throw "npm install failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

