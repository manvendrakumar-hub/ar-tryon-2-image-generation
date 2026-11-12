"""Download a PID's catalog gallery from the public Lenskart product API into poc/pids/<pid>/.

Files are saved in gallery order as g00, g01, ... with their original extension, which is
how the other scripts refer to them (for example poc/pids/208956/g04.png is a studio
worn-by-model shot). Gallery order can change when the catalog team edits a PID, so
check the contact sheet (_sheet_<pid>.jpg) before reusing an index.

python3 poc/scripts/fetch_pids.py                 # the six POC PIDs
python3 poc/scripts/fetch_pids.py 208956 244418   # specific PIDs
"""
import json
import pathlib
import sys
import urllib.request

import truststore

truststore.inject_into_ssl()

from PIL import Image, ImageDraw  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1] / "pids"
API = "https://api-gateway.juno.lenskart.com/v2/products/product/{}"
POC_PIDS = ["208956", "201388", "244418", "242765", "135212", "238524"]


def get(url):
    req = urllib.request.Request(url, headers={"x-api-client": "desktop", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def fetch(pid):
    out = ROOT / pid
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(get(API.format(pid)))["result"]
    (out / "product.json").write_text(json.dumps(data, indent=2))
    files = []
    for i, url in enumerate(data.get("imageUrls", [])):
        f = out / f"g{i:02d}.{url.rsplit('.', 1)[-1].lower()}"
        if not f.exists():
            f.write_bytes(get(url))
        files.append(f)
    sheet(pid, files)
    return len(files)


def sheet(pid, files):
    tiles = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im.thumbnail((260, 260))
        c = Image.new("RGB", (260, 280), "white")
        c.paste(im, ((260 - im.width) // 2, 20))
        ImageDraw.Draw(c).text((4, 2), f.name, fill="red")
        tiles.append(c)
    cols = 6
    s = Image.new("RGB", (cols * 260, max(1, (len(tiles) + cols - 1) // cols) * 280), "white")
    for i, t in enumerate(tiles):
        s.paste(t, ((i % cols) * 260, (i // cols) * 280))
    s.save(ROOT / f"_sheet_{pid}.jpg", quality=80)


if __name__ == "__main__":
    for pid in sys.argv[1:] or POC_PIDS:
        print(pid, fetch(pid), "images")
