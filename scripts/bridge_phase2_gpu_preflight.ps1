<#
.SYNOPSIS
    GPU-host preflight for OpenAlpha Bridge-2K Phase 2 (Windows hosts).

.DESCRIPTION
    Verifies the host BEFORE any provider access or asset download. On a
    CPU-only machine this returns a typed NO_ACCELERATOR or
    MISSING_OPTIONAL_DEPENDENCY blocker and a non-zero exit code.

.EXAMPLE
    scripts\bridge_phase2_gpu_preflight.ps1 -RunDir D:\openalpha\runs\phase2 -CacheDir D:\openalpha\cache\phase2
#>
[CmdletBinding()]
param(
    [string]$RunDir = $(if ($env:OPENALPHA_RUN_DIR) { $env:OPENALPHA_RUN_DIR } else { "$env:LOCALAPPDATA\openalpha\runs\phase2" }),
    [string]$CacheDir = $(if ($env:OPENALPHA_CACHE_DIR) { $env:OPENALPHA_CACHE_DIR } else { "$env:LOCALAPPDATA\openalpha\cache\phase2" })
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Output "== OpenAlpha Bridge-2K Phase 2 GPU preflight =="
Write-Output "repository : $RepoRoot"
Write-Output "run dir    : $RunDir"
Write-Output "cache dir  : $CacheDir"

if ($CacheDir.StartsWith($RepoRoot, [StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "BLOCKER: CACHE_INSIDE_GIT - cache directory must live outside the worktree"
    exit 2
}

foreach ($dir in @($RunDir, $CacheDir)) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
}

Write-Output "-- git worktree state --"
git -C $RepoRoot rev-parse HEAD

Write-Output "-- forbidden tracked files --"
$forbidden = git -C $RepoRoot ls-files | Where-Object { $_ -match '\.(pt|pth|bin|safetensors|ckpt|onnx|npy|npz|parquet|csv|pkl|env)$' }
if ($forbidden) {
    $forbidden
    Write-Error "BLOCKER: FORBIDDEN_TRACKED_ARTIFACT"
    exit 2
}
Write-Output "none"

Write-Output "-- structured preflight --"
Push-Location $RepoRoot
try {
    uv run python -m openalpha_bridge.phase2 preflight `
        --experiment research/bridge-v0 `
        --run-dir $RunDir `
        --cache-dir $CacheDir `
        --repository-root $RepoRoot `
        --json
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
