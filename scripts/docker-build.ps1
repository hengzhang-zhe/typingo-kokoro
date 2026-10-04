param(
    [switch]$Gpu,
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    & $Python -m scripts.prepare_shared_cache
    if ($LASTEXITCODE -ne 0) { throw "Preparing shared resources failed." }
    $composeArguments = @("compose", "-f", "docker-compose.yml")
    if ($Gpu) { $composeArguments += @("-f", "docker-compose.gpu.yml") }
    & docker @composeArguments build
    if ($LASTEXITCODE -ne 0) { throw "Docker build failed." }
    # Export portable dependencies and models from Docker back to the shared folder.
    # This short-lived container uses no network and starts no server.
    $wheelMount = "type=bind,source=$projectRoot\data\wheels\common,target=/shared"
    $helperMount = "type=bind,source=$projectRoot\scripts\prepare_shared_cache.py,target=/prepare.py,readonly"
    & docker run --rm --network none --mount $wheelMount --mount $helperMount --entrypoint python typing-kokoro:latest /prepare.py --destination /shared --pip-cache /nonexistent
    if ($LASTEXITCODE -ne 0) { throw "Exporting container language resources failed." }
} finally {
    Pop-Location
}
