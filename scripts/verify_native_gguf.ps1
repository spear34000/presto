param(
  [string]$Presto = ".\build-native-core\presto.exe",
  [string]$Models = ".\models"
)

$ErrorActionPreference = "Stop"
$prestoPath = (Resolve-Path -LiteralPath $Presto).Path
$modelsPath = (Resolve-Path -LiteralPath $Models).Path
$files = @(Get-ChildItem -LiteralPath $modelsPath -Filter "*.gguf" -File | Sort-Object Name)
if ($files.Count -eq 0) { throw "No GGUF files found in $modelsPath" }

$failures = 0
foreach ($file in $files) {
  $output = & $prestoPath info $file.FullName 2>&1
  if ($LASTEXITCODE -ne 0) {
    $failures++
    Write-Error "FAIL $($file.Name): $($output -join ' ')" -ErrorAction Continue
  } else {
    $summary = ($output | Select-String -Pattern "^summary|tensor_count|tensor_types" |
      ForEach-Object { $_.Line.Trim() }) -join " | "
    Write-Host "OK $($file.Name): $summary"
  }
}
if ($failures -ne 0) { throw "$failures of $($files.Count) GGUF files failed native inspection" }
Write-Host "Native GGUF inspection passed: $($files.Count) file(s)"
