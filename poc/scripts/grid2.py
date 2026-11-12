"""Contact sheet for a round-2 run: inputs, then every variant with its QC scores."""
import json, pathlib, sys
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 19)
bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22, index=1)
MODEL = {"seedream_4_5": "Seedream 4.5", "seedream_5_pro": "Seedream 5 Pro", "gpt_2_5_flare": "GPT Image 2.5 Flare",
         "qwen": "Qwen Image Edit 2511 (open)", "klein": "FLUX.2 Klein 9B (open)", "faceswap": "Face swap (open)"}
def model_of(r):
    if r.get("model"): return r["model"]
    for k, v in MODEL.items():
        if k in r["variant"]: return v
    return "Nano Banana Pro"
for run in sys.argv[1:]:
    run = pathlib.Path(run); m = json.loads((run / "manifest.json").read_text())
    tiles = [("INPUT scan", m["face"]), ("INPUT head crop (identity ref)", m["head"]), ("INPUT catalog shot", m["scene"]), ("INPUT product", m["product"])]
    for r in m["results"]:
        if "file" not in r: continue
        q = r.get("qc") or {}
        lab = f"{r['variant']}  id {q.get('id_arc', '-')}" + (f"  leak {q['leak']}" if "leak" in q else "")
        tiles.append((model_of(r) + "|" + lab, str(run / r["file"])))
    W, H, cols = 400, 466, 5; rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * H), "white"); d = ImageDraw.Draw(sheet)
    for i, (label, f) in enumerate(tiles):
        im = Image.open(f).convert("RGB"); im.thumbnail((W - 8, H - 66)); x, y = (i % cols) * W, (i // cols) * H
        sheet.paste(im, (x + (W - im.width) // 2, y + 60))
        if "|" in label:
            mname, rest = label.split("|", 1); d.text((x + 5, y + 6), mname, fill="#0b57d0", font=bold); d.text((x + 5, y + 34), rest, fill="black", font=font)
        else: d.text((x + 5, y + 20), label, fill="red", font=bold)
    sheet.save(run / "grid.jpg", quality=86); print(run / "grid.jpg")
