param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile,
    [string]$BaseUrl = "http://localhost:9000",
    [string]$OutputRoot = ".\output",
    [switch]$DownloadAudio
)

$ErrorActionPreference = "Stop"
if (!(Test-Path -LiteralPath $InputFile -PathType Leaf)) { throw "Input file not found: $InputFile" }
Add-Type -AssemblyName System.Net.Http
$client = New-Object System.Net.Http.HttpClient
$client.Timeout = [TimeSpan]::FromMinutes(15)
$source = [System.IO.File]::OpenRead((Resolve-Path -LiteralPath $InputFile).Path)
if ($source.Length -gt 32MB) { $source.Dispose(); $client.Dispose(); throw "Task file exceeds 32MB; reduce the exported content count" }
$chunkBytes = 4MB
$uploadId = [Guid]::NewGuid().ToString()
$totalChunks = [int][Math]::Ceiling($source.Length / $chunkBytes)
$totalSize = $source.Length
try {
    for ($index = 0; $index -lt $totalChunks; $index++) {
        $length = [int][Math]::Min($chunkBytes, $totalSize - $source.Position)
        $buffer = New-Object byte[] $length
        $filled = 0
        while ($filled -lt $length) {
            $read = $source.Read($buffer, $filled, $length - $filled)
            if (!$read) { throw "Unexpected end of task file" }
            $filled += $read
        }
        for ($attempt = 0; ; $attempt++) {
            $multipart = New-Object System.Net.Http.MultipartFormDataContent
            try {
                $content = New-Object System.Net.Http.ByteArrayContent(,$buffer)
                $multipart.Add($content, "file", "task.part")
                foreach ($field in @{uploadId=$uploadId; index=$index; totalChunks=$totalChunks; totalSize=$totalSize}.GetEnumerator()) {
                    $multipart.Add((New-Object System.Net.Http.StringContent([string]$field.Value)), [string]$field.Key)
                }
                $response = $client.PostAsync("$BaseUrl/v1/batches/import/chunks", $multipart).GetAwaiter().GetResult()
                $payload = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
                $success = $response.IsSuccessStatusCode
                $response.Dispose()
                if ($success) { break }
                if ($attempt -ge 2) { throw "Task chunk upload failed: $payload" }
            } finally { $multipart.Dispose() }
        }
    }
    $body = @{uploadId=$uploadId;totalChunks=$totalChunks;totalSize=$totalSize} | ConvertTo-Json
    $job = Invoke-RestMethod -Uri "$BaseUrl/v1/batches/import/chunks/complete" -Method Post -ContentType "application/json" -Body $body
} finally {
    $source.Dispose()
    $client.Dispose()
    try { Invoke-RestMethod -Uri "$BaseUrl/v1/batches/import/chunks/$uploadId" -Method Delete | Out-Null } catch { }
}
if ($job.audioCount -eq 0) { Write-Host "No missing audio in this task."; return }
$job = Invoke-RestMethod -Uri "$BaseUrl/v1/batches/$($job.id)/generate" -Method Post
while ($job.status -eq "running") {
    $done = $job.completed + $job.failed
    Write-Progress -Activity "Generating Kokoro batch" -Status "$done / $($job.audioCount)" -PercentComplete ([int](100 * $done / $job.audioCount))
    Start-Sleep -Seconds 1
    $job = Invoke-RestMethod -Uri "$BaseUrl/v1/batches/$($job.id)"
}
Write-Progress -Activity "Generating Kokoro batch" -Completed
if ($job.status -eq "failed") { throw "Generation failed: $($job.error)" }
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$parts = [Math]::Max(1, $job.downloadParts)
for ($part = 1; $part -le $parts; $part++) {
    $result = Join-Path $OutputRoot "typingo-audio-batch-$($job.id)-part-$part.zip"
    Invoke-WebRequest -Uri "$BaseUrl/v1/batches/$($job.id)/download?part=$part" -OutFile $result
    if ($DownloadAudio) {
        $audio = Join-Path $OutputRoot "typingo-audio-only-$($job.id)-part-$part.zip"
        Invoke-WebRequest -Uri "$BaseUrl/v1/batches/$($job.id)/audio/download?part=$part" -OutFile $audio
    }
    Write-Host "Result ZIP: $result"
}
Write-Host "Output mode: $($job.outputMode); generated: $($job.completed); failed: $($job.failed)"
if ($job.failed -gt 0) { Write-Warning "Import successful results, then re-export missing tasks from Typingo." }
