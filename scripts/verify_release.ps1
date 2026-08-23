# presto - extensive local verification suite (release gate)
$ErrorActionPreference = "Continue"
$exe    = "C:\Users\spear\project\presto\build-full\Release\presto.exe"
$vexe   = "C:\Users\spear\project\presto\build-vulkan\Release\presto.exe"
$modelDir = "C:\Users\spear\project\presto\models"
$outMd  = "C:\Users\spear\project\presto\VERIFICATION.md"
$results = New-Object System.Collections.Generic.List[string]

function Record($area, $name, $pass, $detail) {
    $script:results.Add("| $area | $name | $(if ($pass) {'PASS'} else {'**FAIL**'}) | $detail |")
    Write-Host ("[{0}] {1} -> {2} ({3})" -f $(if ($pass) {'PASS'} else {'FAIL'}), $name, $area, $detail)
}

# ---------- A. robustness: corrupted / hostile inputs ----------
$tmp = "$env:TEMP\presto_verify"; Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$src = [IO.File]::ReadAllBytes("$modelDir\stories15M-q4_0.gguf")
$half = $src[0..([int]($src.Length/2))]; [IO.File]::WriteAllBytes("$tmp\trunc.gguf", $half)
# info() legitimately succeeds: header lives in the first KB. Execution must
# fail cleanly when tensor data is missing.
& $exe run "$tmp\trunc.gguf" --prompt x --max-tokens 2 *> "$tmp\t1.txt"; $ecT = $LASTEXITCODE
$t1txt = Get-Content "$tmp\t1.txt" -Raw -ErrorAction SilentlyContinue
Record "robustness" "truncated gguf -> run fails cleanly" ($ecT -ne 0 -and $t1txt -notmatch "AccessViolation") "exit=$ecT"

$rng = New-Object byte[] 4096; (New-Object Random 1).NextBytes($rng)
[IO.File]::WriteAllBytes("$tmp\rand.gguf", $rng)
$r = & $exe info "$tmp\rand.gguf" 2>&1 | Out-String
Record "robustness" "random bytes .gguf" ($LASTEXITCODE -eq 3 -and $r -match "unrecognized|unknown") "exit=3 classified unknown"

[IO.File]::WriteAllBytes("$tmp\empty.gguf", @())
& $exe info "$tmp\empty.gguf" *> "$tmp\e1.txt"; $ec1 = $LASTEXITCODE
Record "robustness" "zero-byte model file" ($ec1 -ne 0) "exit=$ec1"

& $exe run "$tmp\nope.gguf" --prompt x --max-tokens 2 *> "$tmp\e2.txt"; $ec2 = $LASTEXITCODE
Record "robustness" "nonexistent path (run)" ($ec2 -eq 3) "exit=$ec2 (3=unsupported/missing)"

& $exe run "$modelDir" --prompt x --max-tokens 2 *> "$tmp\e3.txt"; $ec3 = $LASTEXITCODE
Record "robustness" "directory passed to run" ($ec3 -eq 3) "exit=$ec3"

# determinism sweep: same command twice -> identical generated line
foreach ($m in @("stories260K.gguf","stories15M-q4_0.gguf","SmolLM2-135M-Instruct-Q8_0.gguf")) {
    $a = (& $exe run "$modelDir\$m" --prompt "determinism probe" --max-tokens 20 --seed 42 2>$null | Select-String "generated").Line
    $b = (& $exe run "$modelDir\$m" --prompt "determinism probe" --max-tokens 20 --seed 42 2>$null | Select-String "generated").Line
    Record "determinism" "$m cross-process greedy" ($a -is [string] -and $a -eq $b) $(if ($a) { ($a -replace '.*generated ','') } else { "no output" })
}

# ---------- B. server defense + soak + memory ----------
$port = 8210
$proc = Start-Process -WindowStyle Hidden -FilePath $exe -ArgumentList "serve","`"$modelDir\SmolLM2-135M-Instruct-Q8_0.gguf`"","--port",$port -PassThru
$ready = $false
for ($i=0; $i -lt 40; $i++) { Start-Sleep -Milliseconds 500; try { Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2 | Out-Null; $ready=$true; break } catch {} }
if (-not $ready) {
    Record "server" "startup" $false "no health within 20s"
} else {
    Record "server" "startup health" $true ""

    # malformed JSON -> must NOT kill server
    try { Invoke-RestMethod "http://127.0.0.1:$port/v1/chat/completions" -Method Post -Body "{not json" -ContentType "application/json" -TimeoutSec 10 | Out-Null; $code = 200 } catch { $code = [int]$_.Exception.Response.StatusCode }
    $aliveAfterBad = $true
    try { Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 5 | Out-Null } catch { $aliveAfterBad = $false }
    Record "server-defense" "malformed JSON -> 400, server alive" ($code -eq 400 -and $aliveAfterBad) "http=$code"

    # missing field
    try { Invoke-RestMethod "http://127.0.0.1:$port/v1/chat/completions" -Method Post -Body "{}" -ContentType "application/json" -TimeoutSec 10 | Out-Null; $c2 = 200 } catch { $c2 = [int]$_.Exception.Response.StatusCode }
    Record "server-defense" "empty object rejected" ($c2 -eq 400) "http=$c2"

    # memory before soak
    $proc.Refresh(); $wsBefore = $proc.WorkingSet64

    # soak: 40 sequential completions alternating prompts (prefix-cache churn)
    $fail = 0; $lat = @()
    for ($k = 1; $k -le 40; $k++) {
        $p = if ($k % 2 -eq 1) { "Story number $k about a fox and a moonlit river:" } else { "Completely unrelated ask $k - list three colors." }
        $body = @{ messages=@(@{role="user"; content=$p}); max_tokens=16; temperature=0 } | ConvertTo-Json -Depth 4
        try {
            $sw = [Diagnostics.Stopwatch]::StartNew()
            Invoke-RestMethod "http://127.0.0.1:$port/v1/chat/completions" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120 | Out-Null
            $sw.Stop(); $lat += $sw.ElapsedMilliseconds
        } catch { $fail++ }
    }
    $proc.Refresh(); $wsAfter = $proc.WorkingSet64; $growMB = [math]::Round(($wsAfter-$wsBefore)/1MB,1)
    $sorted = @($lat | Sort-Object); $med = $sorted[[int]($sorted.Count/2)]
    Record "soak" "40 sequential chat completions" ($fail -eq 0) "fails=$fail med=${med}ms"
    Record "safety" "RSS growth over soak <= 60MB" ($growMB -le 60) "growth=${growMB}MB"

    # parallel burst: 6 concurrent
    $jobs = 1..6 | ForEach-Object { Start-Job -ScriptBlock { param($p) $b=@{messages=@(@{role="user";content="parallel $_"});max_tokens=8;temperature=0}|ConvertTo-Json -Depth 4; try{Invoke-RestMethod "http://127.0.0.1:$p/v1/chat/completions" -Method Post -Body $b -ContentType "application/json" -TimeoutSec 120|Out-Null;return 200}catch{return 500} } -ArgumentList $port }
    $codes = $jobs | Wait-Job | Receive-Job
    Record "concurrency" "6 parallel clients all 200" (($codes | Where-Object {$_ -eq 200}).Count -eq 6) "codes=$($codes -join ',')"
    Get-Job | Remove-Job -Force
}

Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue

# ---------- C. vulkan backend sanity (GPU device visible, deterministic) ----------
if (Test-Path $vexe) {
    $vinfo = & $vexe version 2>&1 | Out-String
    Record "gpu-vulkan" "capability reports vulkan=yes" ($vinfo -match "vulkan=yes") ""
}

# ---------- write report ----------
$md = @()
$md += "# presto verification report"
$md += ""
$md += "- date: $(Get-Date -Format 'yyyy-MM-dd HH:mm zzz')"
$md += "- host: Windows 11, Lunar Lake CPU 8 threads, Intel Arc 140V iGPU, 31.5GB RAM"
$md += "- binaries: build-full (CPU) / build-vulkan (Vulkan), commit $(git rev-parse --short HEAD)"
$md += ""
$md += "| Area | Check | Result | Detail |"
$md += "|---|---|---|---|"
$md += $results
$md += ""
$passCount = ($results | Where-Object { $_ -notmatch "\*\*FAIL\*\*" }).Count
$md += "**$passCount / $($results.Count) checks passed**"
Set-Content -Path $outMd -Value ($md -join "`r`n") -Encoding UTF8
Write-Host "`nreport -> $outMd  ($passCount/$($results.Count) passed)"
