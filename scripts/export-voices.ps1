param(
    [string]$BaseUrl = "http://localhost:9000",
    [string]$Output = ".\\output\\voice-catalog.zip"
)

$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $Output
if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
if ($Output.EndsWith(".zip", [StringComparison]::OrdinalIgnoreCase)) {
    Invoke-WebRequest -Uri "$BaseUrl/v1/audio/voices/download" -OutFile $Output
} else {
    Invoke-WebRequest -Uri "$BaseUrl/v1/audio/voices" -OutFile $Output
}
Write-Host "Voice catalog exported: $Output"
