<#
.SYNOPSIS
    Empirically determines the fastest presto execution route (cpu / vulkan / sycl)
    available on this machine by running a short benchmark against each built binary.

.DESCRIPTION
    For every route whose binary exists, runs:
        presto bench "<model>" --steps 32 --warmup 1 --runs 2 --temp 0
    parses the '[presto-bench] ... med_tps=<num>' summary line, compares the
    median tok/s across routes and prints one recommendation line:

        BEST_ROUTE=<cpu|vulkan|sycl> tok/s=<x>

    preceded by exactly one detail line per route:
        <route>: <tps> tok/s      (measured)
        <route>: not built        (binary missing)
        <route>: failed (<why>)   (crashed / timed out / no bench output)

    Route binaries checked (relative to the repo root):
        cpu     -> build-full\Release\presto.exe
        vulkan  -> build-vulkan\Release\presto.exe
        sycl    -> build-sycl\Release\presto.exe
                   (fallback: build-sycl\presto.exe - the Ninja single-config
                    layout produced by scripts\build_sycl.bat)

    The SYCL binary silently crashes without the Intel oneAPI environment, so
    its benchmark is wrapped in: cmd /c "...oneapi-vars.bat" && "...presto.exe" ...
    with stdout/stderr redirected to separate temp files.

    Every route runs under a wall-clock timeout guard: if the process has not
    exited after -TimeoutSec seconds (default 600) the whole process tree is
    killed and the route is reported as failed.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\route_probe.ps1 models\stories15M-q4_0.gguf

.NOTES
    Exit codes: 0 = at least one route measured, 3 = no route measured.
    PowerShell 5.1 compatible (no ternary operators).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $Model,

    # Wall-clock timeout per route; the process tree is killed when exceeded.
    [int] $TimeoutSec = 600
)

Set-StrictMode -Version 2.0

# '[presto-bench] backend=... med_tps=...' machine-readable summary is only
# emitted when PRESTO_SMOKE=1 (see src/cli.cpp); children inherit this.
$env:PRESTO_SMOKE = '1'

$BenchArgsTemplate = 'bench "{0}" --steps 32 --warmup 1 --runs 2 --temp 0'
$SyclVarsBat       = 'C:\Program Files (x86)\Intel\oneAPI\2026.0\oneapi-vars.bat'
$MedTpsRegex       = '\[presto-bench\][^\r\n]*?med_tps=([0-9]+(?:\.[0-9]+)?)'

function Format-Tps {
    param([double] $Value)
    return $Value.ToString('0.##', [System.Globalization.CultureInfo]::InvariantCulture)
}

function Invoke-BenchRun {
    # Runs one benchmark process with hard stdout/stderr separation and a
    # wall-clock timeout guard. Returns a PSCustomObject:
    #   Ok=$true / Tps=<double>            on success
    #   Ok=$false / Reason=<string>        on any failure
    param(
        [string] $Route,
        [string] $FilePath,
        [string] $CmdArguments,
        [string] $StdOutFile,
        [string] $StdErrFile,
        [int]    $RunTimeoutSec
    )

    Remove-Item -LiteralPath @($StdOutFile, $StdErrFile) -Force -ErrorAction SilentlyContinue

    $proc = $null
    try {
        $proc = Start-Process -FilePath $FilePath `
            -ArgumentList $CmdArguments `
            -NoNewWindow -PassThru `
            -RedirectStandardOutput $StdOutFile `
            -RedirectStandardError  $StdErrFile
    }
    catch {
        return [pscustomobject] @{ Ok = $false; Tps = 0.0;
            Reason = 'failed to start: {0}' -f $_.Exception.Message }
    }

    # Cache the handle now so .ExitCode is readable after exit.
    $null = $proc.Handle

    # Timeout guard: poll instead of a blind WaitForExit so we can kill.
    # NOTE: stay on one clock - mixing UtcNow with local Get-Date breaks the
    # comparison on non-UTC timezones (DateTime -lt compares raw ticks).
    $deadline = (Get-Date).AddSeconds($RunTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { break }
        Start-Sleep -Milliseconds 500
    }

    if (-not $proc.HasExited) {
        # Kill the whole tree (matters for the cmd /c SYCL wrapper).
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try { $null = (& taskkill /F /T /PID $proc.Id) 2>&1 }
        finally { $ErrorActionPreference = $prevEap }
        try { $proc.WaitForExit(5000) | Out-Null } catch { }

        return [pscustomobject] @{ Ok = $false; Tps = 0.0;
            Reason = 'timed out after {0}s (killed)' -f $RunTimeoutSec }
    }

    $exitCode = $proc.ExitCode

    # Capture stdout through Out-String so CR/LF quirks normalize away.
    $text = ''
    if (Test-Path -LiteralPath $StdOutFile) {
        $text = Get-Content -LiteralPath $StdOutFile | Out-String
    }

    # Keep the LAST '[presto-bench]' summary line (the final stdout line on success).
    $lastValue = $null
    foreach ($m in [regex]::Matches($text, $MedTpsRegex)) {
        $lastValue = $m.Groups[1].Value
    }

    if ($null -eq $lastValue) {
        return [pscustomobject] @{ Ok = $false; Tps = 0.0;
            Reason = 'no [presto-bench] med_tps output (exit={0})' -f $exitCode }
    }

    $tps = [double]::Parse($lastValue, [System.Globalization.CultureInfo]::InvariantCulture)
    return [pscustomobject] @{ Ok = $true; Tps = $tps; Reason = '' }
}

# --- main --------------------------------------------------------------------

if ($PSScriptRoot) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
}
else {
    $repoRoot = (Get-Location).Path
}

$modelQuoted = $Model
$modelResolved = Resolve-Path -LiteralPath $Model -ErrorAction SilentlyContinue
if ($null -eq $modelResolved) {
    Write-Warning ("model path not found: {0} (routes will likely fail)" -f $Model)
}
else {
    $modelQuoted = $modelResolved.Path
}

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) `
    ('presto-route-probe-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempDir -Force -ErrorAction Stop | Out-Null

# cpu/vulkan use multi-config generators -> Release\ layout.
# sycl uses Ninja via scripts\build_sycl.bat -> flat layout; accept both.
$syclCandidates = @(
    (Join-Path $repoRoot 'build-sycl\Release\presto.exe'),
    (Join-Path $repoRoot 'build-sycl\presto.exe')
)
$syclExe = $null
foreach ($candidate in $syclCandidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $syclExe = $candidate
        break
    }
}

$routeDefs = @(
    @{ Name = 'cpu';    Exe = Join-Path $repoRoot 'build-full\Release\presto.exe' },
    @{ Name = 'vulkan'; Exe = Join-Path $repoRoot 'build-vulkan\Release\presto.exe' },
    @{ Name = 'sycl';   Exe = $syclExe }
)

$results = @()
$routeIndex = 0
foreach ($def in $routeDefs) {
    $routeIndex++
    $route = $def.Name
    Write-Progress -Activity 'probing presto execution routes' `
        -Status ('route {0} ({1}/{2})' -f $route, $routeIndex, $routeDefs.Count) `
        -PercentComplete (($routeIndex - 1) * 100 / $routeDefs.Count)

    if ($null -eq $def.Exe -or -not (Test-Path -LiteralPath $def.Exe -PathType Leaf)) {
        $results += [pscustomobject] @{ Route = $route; State = 'notbuilt'; Tps = 0.0; Detail = '' }
        continue
    }

    if ($route -eq 'sycl') {
        if (-not (Test-Path -LiteralPath $SyclVarsBat -PathType Leaf)) {
            $results += [pscustomobject] @{ Route = $route; State = 'failed'; Tps = 0.0;
                Detail = 'oneAPI env not found: {0}' -f $SyclVarsBat }
            continue
        }
        # SYCL builds silently crash without the oneAPI environment: run under
        # cmd /c after importing oneapi-vars.bat. The whole chain gets an extra
        # outer pair of quotes to survive cmd's quote-stripping rules.
        $inner = '"{0}" && "{1}" {2}' -f $SyclVarsBat, $def.Exe, ($BenchArgsTemplate -f $modelQuoted)
        $filePath = $env:ComSpec
        if ([string]::IsNullOrEmpty($filePath)) { $filePath = 'cmd.exe' }
        $cmdArgs = '/c "{0}"' -f $inner
    }
    else {
        # cpu/vulkan: launch the exe directly (no shell wrapper).
        $filePath = $def.Exe
        $cmdArgs = $BenchArgsTemplate -f $modelQuoted
    }

    $stdOut = Join-Path $tempDir ($route + '.stdout.txt')
    $stdErr = Join-Path $tempDir ($route + '.stderr.txt')

    $run = Invoke-BenchRun -Route $route -FilePath $filePath -CmdArguments $cmdArgs `
        -StdOutFile $stdOut -StdErrFile $stdErr -RunTimeoutSec $TimeoutSec

    if ($run.Ok) {
        $results += [pscustomobject] @{ Route = $route; State = 'ok'; Tps = $run.Tps; Detail = '' }
    }
    else {
        $results += [pscustomobject] @{ Route = $route; State = 'failed'; Tps = 0.0; Detail = $run.Reason }
    }
}

Write-Progress -Activity 'probing presto execution routes' -Completed

# --- report ------------------------------------------------------------------

foreach ($r in $results) {
    if ($r.State -eq 'ok') {
        Write-Host ('{0}: {1} tok/s' -f $r.Route, (Format-Tps $r.Tps))
    }
    elseif ($r.State -eq 'notbuilt') {
        Write-Host ('{0}: not built' -f $r.Route)
    }
    else {
        Write-Host ('{0}: failed ({1})' -f $r.Route, $r.Detail)
    }
}

$measured = @($results | Where-Object { $_.State -eq 'ok' })

Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue

if ($measured.Count -eq 0) {
    exit 3
}

# First max wins -> ties resolve in cpu > vulkan > sycl order.
$winner = $measured[0]
foreach ($cand in $measured) {
    if ($cand.Tps -gt $winner.Tps) { $winner = $cand }
}

Write-Host ('BEST_ROUTE={0} tok/s={1}' -f $winner.Route, (Format-Tps $winner.Tps))
exit 0
