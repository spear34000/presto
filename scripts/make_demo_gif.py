"""Render a demo GIF from real presto CLI captures (no fabricated text)."""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 780, 440
BG = (12, 12, 13)
FG = (232, 232, 235)
DIM = (120, 120, 128)
ACCENT = (255, 157, 0)
GREEN = (126, 231, 135)
TITLE = (255, 213, 0)

font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 17)
font_b = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 17)
font_s = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 14)

def load_lines(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    return [l for l in txt.replace("\r", "").split("\n") if l.strip()]

gpu = load_lines(os.path.join(os.environ.get("TEMP", "."), "demo_gpu.txt"))
cpu = load_lines(os.path.join(os.environ.get("TEMP", "."), "demo_cpu.txt"))
bench = load_lines(os.path.join(os.environ.get("TEMP", "."), "demo_bench.txt"))

def scene(header, body_lines, accent_last=2):
    """yield progressive states: list of visible lines"""
    yield [header]
    shown = [header]
    for line in body_lines:
        limit = len(line)
        for cut in range(0, limit, 3):
            shown[-1] = line[:cut]
            yield list(shown)
        shown[-1] = line
        shown.append("")
        yield list(shown)
    # hold
    for _ in range(10):
        yield list(shown)

frames = []

def render(lines, tag):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 34], fill=(24, 24, 27))
    d.text((14, 8), "presto", font=font_b, fill=TITLE)
    d.text((90, 10), "v0.2.0 - unified LLM runtime", font=font_s, fill=DIM)
    d.text((W - 150, 10), tag, font=font_s, fill=ACCENT)
    y = 52
    for i, line in enumerate(lines[-16:]):
        color = FG
        if line.startswith("===") or line.startswith("backend"):
            color = TITLE
        elif line.startswith("decode") or line.startswith("tempo"):
            color = GREEN
        elif line.startswith("---"):
            color = ACCENT
        d.text((18, y), line[:92], font=font, fill=color)
        y += 26
    d.rectangle([0, H - 26, W, H], fill=(24, 24, 27))
    d.text((14, H - 22), "github.com/spear34000/presto - one binary, every format, any silicon",
           font=font_s, fill=DIM)
    return im

S1_H = "> presto run gpt-oss-20b-Q4_K_M.gguf --prompt \"Explain what presto means in music.\""
S1 = scene(S1_H, [l for l in gpu if l.strip()][:4])
S2_H = "> presto run qwen2.5-0.5b-instruct-Q8_0.gguf --prompt \"List three colors...\""
S2 = scene(S2_H, [l for l in cpu if l.strip()][:3])
S3_H = "> presto bench gpt-oss-20b-Q4_K_M.gguf --steps 64 --runs 3"
S3 = scene(S3_H, [l for l in bench if l.strip()][:7], accent_last=3)

for gen, tag in ((S1, "[Vulkan · Arc GPU]"), (S2, "[CPU x8]"), (S3, "[Tempo Report]")):
    for state in gen:
        frames.append(render(state, tag))

frames[0].save(r"assets\demo.gif", save_all=True, append_images=frames[1:],
               duration=90, loop=0, optimize=True)
import os
print("frames:", len(frames), "size:", round(os.path.getsize(r"assets\demo.gif")/1e6, 2), "MB")
