# presto - GGUF smoke test for Windows CI (PowerShell 5.1 compatible)
param(
    [string]$ModelUrl = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories260K.gguf",
    [string]$OutPath = "models/stories260K.gguf",
    [int]$MaxTokens = 16,
    [string]$Prompt = "Once upon a time"
)
$ErrorActionPreference = "Stop"

Write-Host "::group::[smoke-gguf] download model"
Write-Host "[smoke-gguf] url: $ModelUrl"
New-Item -ItemType Directory -Force -Path (Split-Path $OutPath) | Out-Null
$downloaded = $false
for ($i = 1; $i -le 3; $i++) {
    try {
        Invoke-WebRequest -Uri $ModelUrl -OutFile $OutPath -UseBasicParsing -ErrorAction Stop
        $downloaded = $true
        break
    } catch {
        Write-Host "[smoke-gguf] attempt ${i}/3 failed: $_"
        Start-Sleep -Seconds 3
    }
}
if (-not $downloaded) {
    Write-Host "::endgroup::"
    Write-Host "[smoke-gguf] FAILED: download after retries"
    exit 1
}
$size = (Get-Item $OutPath).Length
if ($size -lt 1000000) {
    Write-Host "::endgroup::"
    Write-Host "[smoke-gguf] FAILED: model suspiciously small ($size bytes)"
    exit 1
}
Write-Host "[smoke-gguf] model size: $size bytes"
Write-Host "::endgroup::"

Write-Host "::group::[smoke-gguf] locate binary"
$exe = Get-ChildItem -Path build -Recurse -Filter presto.exe -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $exe) {
    Write-Host "::endgroup::"
    Write-Host "[smoke-gguf] FAILED: presto.exe not found under build/"
    exit 1
}
Write-Host "[smoke-gguf] using $($exe.FullName)"
& $exe.FullName version | Tee-Object -Variable verOut
Write-Host "$verOut"
Write-Host "::endgroup::"

Write-Host "::group::[smoke-gguf] generate $MaxTokens tokens"
$env:PRESTO_SMOKE = "1"
$output = & $exe.FullName run $OutPath --prompt $Prompt --max-tokens $MaxTokens 2>&1 |
    Tee-Object -FilePath smoke-gguf.log
$exitCode = $LASTEXITCODE
foreach ($line in $output) { Write-Host $line }
Write-Host "::endgroup::"

$marker = ($output | Select-String -SimpleMatch "ok=true" | Where-Object { $_.Line -like "*presto-smoke*" })
if (-not $marker) {
    Write-Host "[smoke-gguf] FAILED: success marker missing"
    Get-Content smoke-gguf.log -Tail 50
    exit 1
}
Write-Host "[smoke-gguf] SUCCESS: $($marker.Line)"
exit $exitCode
