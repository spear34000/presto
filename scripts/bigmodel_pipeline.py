#!/usr/bin/env python3
"""bigmodel_pipeline.py - end-to-end big-model evaluation loop for presto.

Stages
------
1. Download a GGUF model from a direct HuggingFace URL using ``curl.exe``
   with ``--retry 5 -C -`` (resumable). File size is polled every 30 s and
   progress is printed as a percentage of the Content-Length reported by a
   HEAD request (``curl.exe -sIL``).
2. Verify the final byte size against the reported Content-Length.
3. Optionally requantize the model (``--quant Q2_K`` style argument) with
   ``build-vulkan\\bin\\Release\\llama-quantize.exe``; the benchmark then
   targets the requantized artifact.
4. Benchmark every available presto build:

   ======  ==============================================================
   route   binary
   ======  ==============================================================
   full    ``build-full\\Release\\presto.exe``
   vulkan  ``build-vulkan\\Release\\presto.exe``
   sycl    ``build-sycl\\{Release\\,}presto.exe`` launched under
           ``cmd /c "...oneapi-vars.bat" && ...`` so the oneAPI environment
           is imported first
   ==============================================================

   Each route runs ``bench <model> --steps 64 --warmup 2 --runs 3 --temp 0``
   with a hard 1800 s timeout that kills the whole process tree. Missing
   binaries are skipped with ``<route>: not built``.
5. Print a summary table (route -> med_tps) and write it to
   ``scripts/last_bigmodel_report.txt``.

Exit codes
----------
0  success (at least one route benchmarked successfully)
1  pipeline stage failure (download, size verification, or requantization)
2  usage error (bad arguments, curl.exe not found)
4  no route succeeded (nothing built, or every benchmark failed)

Resume safety: partial downloads are never deleted; rerunning the script
continues the transfer where it stopped.

Standard library only. Target platform: Windows, Python 3.9+.

Usage::

    python scripts/bigmodel_pipeline.py --url <hf-gguf-url> --out models\\x.gguf [--quant Q2_K]
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths and constants (anchored to the repo root, independent of cwd)
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "scripts" / "last_bigmodel_report.txt"

QUANTIZE_EXE = REPO_ROOT / "build-vulkan" / "bin" / "Release" / "llama-quantize.exe"
ONEAPI_VARS_BAT = Path(r"C:\Program Files (x86)\Intel\oneAPI\2026.0\oneapi-vars.bat")

DOWNLOAD_POLL_SECONDS = 30
HEAD_TIMEOUT_SECONDS = 120
ROUTE_TIMEOUT_SECONDS = 1800

BENCH_STEPS = 64
BENCH_WARMUP = 2
BENCH_RUNS = 3
BENCH_TEMP = 0

EXIT_OK = 0
EXIT_STAGE_FAILED = 1
EXIT_USAGE = 2
EXIT_NO_ROUTE = 4

_CONTENT_LENGTH_RE = re.compile(
    r"^[ \t]*content-length:[ \t]*(\d+)[ \t]*\r?$", re.IGNORECASE | re.MULTILINE
)
_BENCH_RESULT_RE = re.compile(
    r"\[presto-bench\][^\r\n]*?"
    r"med_tps=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    r"[ \t]+sigma=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
)


@dataclass
class RouteResult:
    """Outcome of one benchmark route."""

    name: str
    status: str  # "ok" | "not built" | "failed" | "timeout"
    med_tps: float | None = None
    sigma: float | None = None
    note: str = ""
    elapsed_s: float = 0.0


# --------------------------------------------------------------------------
# Stage 1+2: download and verify
# --------------------------------------------------------------------------


def head_content_length(url: str) -> int | None:
    """Return the Content-Length of *url* via ``curl.exe -sIL`` (HEAD).

    Redirect chains produce several header blocks; the last non-zero value
    wins. Returns ``None`` when the header is unavailable.
    """
    try:
        proc = subprocess.run(
            ["curl.exe", "-sIL", url],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=HEAD_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[download] HEAD request failed: {exc}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(
            f"[download] HEAD request exited {proc.returncode}: "
            f"{(proc.stderr or '').strip()}",
            file=sys.stderr,
        )
        return None
    sizes = [int(m) for m in _CONTENT_LENGTH_RE.findall(proc.stdout or "") if int(m) > 0]
    return sizes[-1] if sizes else None


def _print_download_progress(dest: Path, expected_size: int | None) -> None:
    current = dest.stat().st_size if dest.exists() else 0
    if expected_size:
        pct = min(100.0, 100.0 * current / expected_size)
        print(f"[download] {current:,} / {expected_size:,} bytes ({pct:5.1f}%)", flush=True)
    else:
        print(f"[download] {current:,} bytes", flush=True)


def download_model(url: str, dest: Path, expected_size: int | None) -> bool:
    """Download *url* to *dest* with curl.exe, resuming any partial file.

    Never deletes an existing partial file; ``curl -C -`` continues where a
    previous run stopped. Returns True when *dest* holds the full payload.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        print(f"[download] {dest.name} already complete ({expected_size:,} bytes); skipping")
        return True

    cmd = [
        "curl.exe",
        "--retry", "5",
        "--continue-at", "-",   # resume-safe: never restart from zero
        "--location",           # follow HF redirect to the CDN
        "--fail",
        "--silent",             # we print our own progress below
        "--show-error",
        "--output", str(dest),
        url,
    ]
    print(f"[download] GET {url}")
    print(f"[download] -> {dest}")
    try:
        proc = subprocess.Popen(cmd)
    except OSError as exc:
        print(f"[download] cannot launch curl.exe: {exc}", file=sys.stderr)
        return False

    try:
        while True:
            try:
                returncode = proc.wait(timeout=DOWNLOAD_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                _print_download_progress(dest, expected_size)
    except KeyboardInterrupt:
        proc.kill()
        print("\n[download] interrupted; partial file kept (rerun to resume)", file=sys.stderr)
        raise

    # Already-complete reruns can surface as a 416 range error; accept them.
    actual = dest.stat().st_size if dest.exists() else 0
    if returncode != 0:
        if expected_size is not None and actual == expected_size:
            print(f"[download] curl exited {returncode} but file is complete ({actual:,} bytes)")
            return True
        print(
            f"\n[download] curl.exe exited with code {returncode}; "
            f"partial file kept at {dest} ({actual:,} bytes, rerun to resume)",
            file=sys.stderr,
        )
        return False

    if expected_size is not None and actual != expected_size:
        print(
            f"[download] size mismatch: got {actual:,} bytes, expected {expected_size:,}; "
            f"file kept for resume",
            file=sys.stderr,
        )
        return False

    _print_download_progress(dest, expected_size)
    return True


# --------------------------------------------------------------------------
# Process-tree helpers
# --------------------------------------------------------------------------


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill *proc* and all of its children (Windows process tree)."""
    if proc.poll() is not None:
        return
    try:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run_captured(argv: Sequence[str], timeout_s: int) -> tuple[int | None, str, bool]:
    """Run *argv* capturing merged stdout+stderr.

    Returns ``(returncode, output, timed_out)``; ``returncode`` is ``None``
    when the timeout hit, in which case the whole process tree is killed.
    """
    try:
        proc = subprocess.Popen(
            list(argv),
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return -1, f"failed to launch {argv[0]}: {exc}", False
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        return proc.returncode, out or "", False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, _ = proc.communicate(timeout=15)
        except (subprocess.TimeoutExpired, ValueError):
            out = ""
        return None, out or "", True


def _output_tail(output: str, lines: int = 15) -> str:
    stripped = (output or "").strip()
    if not stripped:
        return "(no output)"
    return "\n".join(stripped.splitlines()[-lines:])


# --------------------------------------------------------------------------
# Stage 3: optional requantization
# --------------------------------------------------------------------------


def requantized_path(model: Path, quant_type: str) -> Path:
    """models/x.gguf + Q2_K -> models/x.Q2_K.gguf"""
    return model.with_name(f"{model.stem}.{quant_type}.gguf")


def requantize(model: Path, quant_type: str) -> Path | None:
    """Requantize *model* with llama-quantize.exe; returns the new path."""
    out_path = requantized_path(model, quant_type)
    if not QUANTIZE_EXE.exists():
        print(f"error: llama-quantize not built: {QUANTIZE_EXE}", file=sys.stderr)
        return None

    cmd = [str(QUANTIZE_EXE), str(model), str(out_path), quant_type]
    print(f"[quant] {' '.join(cmd)}")
    started = time.monotonic()
    returncode, output, timed_out = run_captured(cmd, ROUTE_TIMEOUT_SECONDS)
    elapsed = time.monotonic() - started

    if timed_out:
        print(f"[quant] timed out after {elapsed:.0f}s (limit {ROUTE_TIMEOUT_SECONDS}s)", file=sys.stderr)
        return None
    if returncode != 0:
        print(f"[quant] failed (exit {returncode}); output tail:\n{_output_tail(output)}", file=sys.stderr)
        return None
    if not out_path.exists():
        print(f"[quant] reported success but {out_path} does not exist", file=sys.stderr)
        return None

    print(f"[quant] wrote {out_path} ({out_path.stat().st_size:,} bytes) in {elapsed:.0f}s")
    return out_path


# --------------------------------------------------------------------------
# Stage 4: benchmarks across presto builds
# --------------------------------------------------------------------------


def discover_routes() -> list[tuple[str, Path | None, Path | None]]:
    """Return (route_name, presto_exe_or_None, env_bat_or_None) triples."""
    sycl_candidates = [
        REPO_ROOT / "build-sycl" / "Release" / "presto.exe",
        REPO_ROOT / "build-sycl" / "presto.exe",  # Ninja single-config layout
    ]
    sycl_exe = next((p for p in sycl_candidates if p.exists()), None)
    return [
        ("full", REPO_ROOT / "build-full" / "Release" / "presto.exe", None),
        ("vulkan", REPO_ROOT / "build-vulkan" / "Release" / "presto.exe", None),
        ("sycl", sycl_exe, ONEAPI_VARS_BAT),
    ]


def bench_argv(exe: Path, env_bat: Path | None, model: Path) -> list[str]:
    """Build the argv for one route's ``bench`` invocation."""
    args = [
        "bench",
        str(model),
        "--steps", str(BENCH_STEPS),
        "--warmup", str(BENCH_WARMUP),
        "--runs", str(BENCH_RUNS),
        "--temp", str(BENCH_TEMP),
    ]
    if env_bat is None:
        return [str(exe)] + args
    # SYCL builds crash without the oneAPI environment: import it via cmd /c.
    # The whole chain keeps its own quotes; cmd /s strips only the outer pair.
    inner = f'"{env_bat}" && ' + subprocess.list2cmdline([str(exe)] + args)
    return ["cmd", "/d", "/s", "/c", inner]


def run_route(name: str, exe: Path | None, env_bat: Path | None, model: Path) -> RouteResult:
    label = f"[bench:{name}]"
    if exe is None or not exe.exists():
        print(f"{label} {name}: not built")
        return RouteResult(name=name, status="not built", note="not built")

    if env_bat is not None and not env_bat.exists():
        print(f"{label} {name}: not built (oneAPI env missing: {env_bat})")
        return RouteResult(name=name, status="not built", note=f"missing {env_bat}")

    argv = bench_argv(exe, env_bat, model)
    print(f"{label} running: {subprocess.list2cmdline(argv)}")
    started = time.monotonic()
    returncode, output, timed_out = run_captured(argv, ROUTE_TIMEOUT_SECONDS)
    elapsed = time.monotonic() - started

    match = _BENCH_RESULT_RE.search(output or "")
    if timed_out:
        print(f"{label} TIMEOUT after {elapsed:.0f}s (limit {ROUTE_TIMEOUT_SECONDS}s); tree killed")
        return RouteResult(
            name=name, status="timeout",
            note=f"killed after {ROUTE_TIMEOUT_SECONDS}s", elapsed_s=elapsed,
        )
    if match is None:
        print(f"{label} FAILED (exit {returncode}); no [presto-bench] line. Output tail:")
        print(_output_tail(output))
        return RouteResult(
            name=name, status="failed",
            note=f"exit {returncode}, no bench result", elapsed_s=elapsed,
        )

    med_tps = float(match.group(1))
    sigma = float(match.group(2))
    print(f"{label} med_tps={med_tps:.2f} sigma={sigma:.2f} ({elapsed:.0f}s)")
    return RouteResult(name=name, status="ok", med_tps=med_tps, sigma=sigma, elapsed_s=elapsed)


# --------------------------------------------------------------------------
# Stage 5: summary report
# --------------------------------------------------------------------------


def render_report(results: Sequence[RouteResult], model: Path, quant_type: str | None) -> str:
    lines = [
        "presto big-model pipeline report",
        f"generated : {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"model     : {model}",
    ]
    if quant_type:
        lines.append(f"quant     : {quant_type}")
    lines.append("")
    header = f"{'route':<10}{'med_tps':>14}{'sigma':>14}  {'status':<10}note"
    lines.append(header)
    lines.append("-" * max(len(header), 60))
    for r in results:
        if r.status == "ok" and r.med_tps is not None and r.sigma is not None:
            note = f"{r.elapsed_s:.0f}s"
            lines.append(f"{r.name:<10}{r.med_tps:>14.2f}{r.sigma:>14.2f}  {'ok':<10}{note}")
        else:
            note = r.note or r.status
            lines.append(f"{r.name:<10}{'-':>14}{'-':>14}  {r.status:<10}{note}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def resolve_out_path(raw: str) -> Path:
    """Resolve --out relative to the repo root so cwd never matters."""
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return Path(os.path.normpath(str(p)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bigmodel_pipeline.py",
        description=(
            "Download a GGUF model, optionally requantize it, benchmark every "
            "available presto build, and write scripts/last_bigmodel_report.txt."
        ),
        epilog=(
            "example: python scripts/bigmodel_pipeline.py "
            "--url https://huggingface.co/<org>/<repo>/resolve/main/model.gguf "
            '--out models\\model.gguf --quant Q2_K'
        ),
    )
    parser.add_argument("--url", required=True, help="direct HuggingFace GGUF download URL")
    parser.add_argument("--out", required=True, help="target GGUF path, e.g. models\\x.gguf")
    parser.add_argument(
        "--quant",
        default=None,
        metavar="TYPE",
        help="optional requantization type for llama-quantize.exe, e.g. Q2_K",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if shutil.which("curl.exe") is None:
        print("error: curl.exe not found on PATH", file=sys.stderr)
        return EXIT_USAGE

    quant_type = args.quant.strip().upper() if args.quant else None
    if quant_type and not re.fullmatch(r"[A-Z0-9_]{2,}", quant_type):
        print(f"error: invalid --quant value {args.quant!r} (expected e.g. Q2_K)", file=sys.stderr)
        return EXIT_USAGE

    model_path = resolve_out_path(args.out)

    # ---- stages 1+2: download and verify ---------------------------------
    expected_size = head_content_length(args.url)
    if expected_size is None:
        print("[download] warning: Content-Length unavailable; size verification disabled")
    if not download_model(args.url, model_path, expected_size):
        return EXIT_STAGE_FAILED
    actual_size = model_path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        print(
            f"[download] final size mismatch: {model_path} has {actual_size:,} bytes, "
            f"expected {expected_size:,}",
            file=sys.stderr,
        )
        return EXIT_STAGE_FAILED
    print(f"[download] verified: {model_path} ({actual_size:,} bytes)")

    # ---- stage 3: optional requantization ----------------------------------
    if quant_type:
        quantized = requantize(model_path, quant_type)
        if quantized is None:
            return EXIT_STAGE_FAILED
        model_path = quantized

    # ---- stage 4: benchmarks ------------------------------------------------
    results = [
        run_route(name, exe, env_bat, model_path)
        for name, exe, env_bat in discover_routes()
    ]

    # ---- stage 5: summary ----------------------------------------------------
    report = render_report(results, model_path, quant_type)
    print("\n" + report, end="")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[report] written to {REPORT_PATH}")

    if not any(r.status == "ok" for r in results):
        print("error: no route succeeded", file=sys.stderr)
        return EXIT_NO_ROUTE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
