"""Studio grid: inputs on the left, then one row per pose with every model (name + scores)."""
import json, pathlib, sys
from PIL import Image, ImageDraw, ImageFont
bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20, index=1)
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
ORDER = ["nano_banana_pro", "gpt_2_5_flare", "seedream_5_pro", "qwen", "klein"]
for run in sys.argv[1:]:
    run = pathlib.Path(run); m = json.loads((run / "manifest.json").read_text())
    by = {r["variant"]: r for r in m["results"] if "file" in r}
    W, H = 360, 420; cols = 1 + len(ORDER)
    s = Image.new("RGB", (cols * W, 2 * H), "white"); d = ImageDraw.Draw(s)
    def tile(f, x, y, a, b, colour):
        im = Image.open(f).convert("RGB"); im.thumbnail((W - 8, H - 64))
        s.paste(im, (x + (W - im.width) // 2, y + 58)); d.text((x + 5, y + 6), a, fill=colour, font=bold); d.text((x + 5, y + 32), b, fill="black", font=font)
    tile(m["face"], 0, 0, "INPUT customer", f"{m['cid']}", "red"); tile(m["product"], 0, H, "INPUT product", f"PID {m['pid']}", "red")
    for row, pose in enumerate(("front", "three_quarter")):
        for c, k in enumerate(ORDER):
            r = by.get(f"{pose}_{k}")
            if not r: continue
            q = r.get("qc") or {}
            tile(run / r["file"], (c + 1) * W, row * H, r["model"], f"{pose.replace('_', ' ')} · id {q.get('id_arc')} / sf {q.get('id_sf')}", "#0b57d0")
    s.save(run / "grid.jpg", quality=86); print(run / "grid.jpg")
