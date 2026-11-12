"""Contact sheet: inputs + every model output with time and est cost."""
import json, pathlib, sys
from PIL import Image, ImageDraw, ImageFont
run = pathlib.Path(sys.argv[1]); m = json.loads((run / "manifest.json").read_text())
try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
except Exception: font = ImageFont.load_default()
tiles = [("INPUT customer", m["face"]), ("INPUT catalog scene", m["scene"]), ("INPUT product", m["product"])]
tiles += [(f"{r['model']}  {r['seconds']}s  ${r['est_cost']}", str(run / r["file"])) for r in m["results"] if r["ok"]]
W, H, cols = 420, 460, 5
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * W, rows * H), "white"); d = ImageDraw.Draw(sheet)
for i, (label, f) in enumerate(tiles):
    im = Image.open(f).convert("RGB"); im.thumbnail((W - 10, H - 45))
    x, y = (i % cols) * W, (i // cols) * H
    sheet.paste(im, (x + (W - im.width) // 2, y + 38))
    d.text((x + 6, y + 8), label, fill="red" if label.startswith("INPUT") else "black", font=font)
sheet.save(run / "grid.jpg", quality=88); print(run / "grid.jpg")
