import json, os, re, subprocess, sys, time

EXE = r"build-full\Release\presto.exe"
VEXE = r"build-vulkan\Release\presto.exe"
MODELS = [
    ("stories15M-q4_0",              r"models\stories15M-q4_0.gguf",              "cpu"),
    ("SmolLM2-135M-Instruct Q8_0",   r"models\SmolLM2-135M-Instruct-Q8_0.gguf",   "cpu"),
    ("Qwen2.5-0.5B-Instruct Q8_0",   r"models\qwen2.5-0.5b-instruct-Q8_0.gguf",   "cpu"),
    ("phi-4 Q4_K_M",                 r"models\phi-4-Q4_K_M.gguf",                 "cpu"),
    ("Qwen3-8B Q4_K_M (imatrix)",    r"C:\Users\spear\.lmstudio\models\DavidAU\Qwen3-8B-Hivemind-Instruct-Heretic-Abliterated-Uncensored-NEO-Imatrix-GGUF\Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat.gguf", "cpu"),
    ("Dolphin3.0-Llama3.1-8B Q4_K_S",r"C:\Users\spear\.lmstudio\models\dphn\Dolphin3.0-Llama3.1-8B-GGUF\Dolphin3.0-Llama3.1-8B-Q4_K_S.gguf", "cpu"),
    ("Bonsai-27B Q1_0",              r"C:\Users\spear\.lmstudio\models\lmstudio-community\Bonsai-27B-GGUF\Bonsai-27B-Q1_0.gguf", "cpu"),
    ("gemma-4-E4B-it Q4_K_M",        r"C:\Users\spear\.lmstudio\models\lmstudio-community\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q4_K_M.gguf", "cpu"),
    ("gemma-4-26B-A4B-it Q3_K_M",    r"models\gemma4-26b-a4b-it-Q3_K_M.gguf",     "gpu"),
    ("gpt-oss-20b Q4_K_M",           r"models\gpt-oss-20b-Q4_K_M.gguf",           "gpu"),
    ("Qwen3.5-9B Q4_K_M",            r"models\qwen35-9b-instruct-Q4_K_M.gguf",    "cpu"),
]

STEPS = 48
RUNS = 3

def run_bench(exe, model):
    env = dict(os.environ, PRESTO_CTX="4096")
    p = subprocess.run([exe, "bench", model, "--steps", str(STEPS),
                        "--warmup", "2", "--runs", str(RUNS)],
                       capture_output=True, text=True, timeout=1800, env=env)
    m = re.search(r"decode\s+: min/med/max = ([\d.]+) / ([\d.]+) / ([\d.]+) tok/s\s+sigma=([\d.]+)", p.stdout)
    if not m:
        return None
    return dict(min=float(m.group(1)), med=float(m.group(2)),
                max=float(m.group(3)), sigma=float(m.group(4)))

def reproducible(exe, model):
    args = [exe, "run", model, "--prompt", "consistency probe", "--max-tokens", "16", "--seed", "7"]
    try:
        a = subprocess.run(args, capture_output=True, text=True, timeout=600)
        b = subprocess.run(args, capture_output=True, text=True, timeout=600)
        ma = re.search(r"generated \d+ token\(s\): ([\d,]+)", a.stdout)
        mb = re.search(r"generated \d+ token\(s\): ([\d,]+)", b.stdout)
        return bool(ma and mb and ma.group(1) == mb.group(1))
    except Exception:
        return False

rows = []
for name, path, dev in MODELS:
    exe = VEXE if dev == "gpu" else EXE
    if not os.path.exists(path):
        print(f"[skip] {name}: file missing", flush=True)
        continue
    r = run_bench(exe, path)
    rep = reproducible(exe, path)
    if r:
        rows.append((name, dev, r["med"], r["sigma"], rep))
        print(f"[ok] {name}: {r['med']:.1f} tok/s repro={rep}", flush=True)
    else:
        rows.append((name, dev, None, None, rep))
        print(f"[fail] {name}: bench produced no numbers", flush=True)

print("\n=== MATRIX ===")
for name, dev, med, sig, rep in rows:
    if med:
        print(f"| {name} | {dev} | {med:.1f} | ±{sig:.1f}% | {'✅' if rep else '❌'} |")
