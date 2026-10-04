param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    & $Python -m scripts.prepare_shared_cache
    if ($LASTEXITCODE -ne 0) { throw "Preparing shared resources failed." }
    & $Python scripts/install_dependencies.py --wheelhouse data/wheels/common --requirements requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Installing local dependencies failed." }
    & $Python -m scripts.prepare_shared_cache
    if ($LASTEXITCODE -ne 0) { throw "Exporting installed resources failed." }
} finally {
    Pop-Location
}
