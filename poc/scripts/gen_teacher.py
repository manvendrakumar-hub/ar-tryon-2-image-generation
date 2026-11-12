"""Teacher data for distillation: frontier model (GPT Image 2.5 Flare, identity-first
swap prompt) on customers and PIDs that are NOT in the test set. Each output is
QC-scored; only passing pairs go into the LoRA training zip."""
import json, pathlib, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import falx, qc
from round2 import swap_prompt, call

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "poc/out/teacher"; OUT.mkdir(parents=True, exist_ok=True)
PID = {
    "208956": ("translucent matte grey full-rim square Lenskart Hustlr eyeglasses", ", lenses stay clear"),
    "201388": ("gunmetal metal double-bridge John Jacobs eyeglasses", ", lenses stay clear"),
    "242765": ("translucent grey square Lenskart Hustlr sunglasses with dark grey lenses", ", lens colour and tint unchanged"),
    "238524": ("black Meller SHIPO aviator sunglasses with orange lenses", ", lens colour and tint unchanged"),
}
M = ["208956/g04.png", "201388/g02.png", "242765/g02.png", "238524/g20.jpg", "208956/g05.png", "201388/g04.png",
     "242765/g04.png", "238524/g10.jpg", "201388/g03.png", "242765/g05.png", "208956/g02.png"]
F = ["208956/g06.png", "242765/g06.png", "238524/g15.jpg", "208956/g07.png", "242765/g03.png", "238524/g16.jpg",
     "242765/g07.png", "208956/g03.png", "238524/g21.jpg"]
CUST = {"c00": M, "c01": M, "c02": M, "c03": M, "c04": M, "c06": F, "c07": F, "c08": F, "c10": F, "c11": F}
HAIR = "facial hair or lack of it"

def jobs():
    for i, (cid, pool) in enumerate(CUST.items()):
        k = list(CUST).index(cid) % 5
        for j in range(3):
            yield cid, pool[(k * 3 + j) % len(pool)]

def one(job):
    cid, scene = job; pid = scene.split("/")[0]; name = f"{cid}_{pid}_{pathlib.Path(scene).stem}"
    dest = OUT / f"{name}.png"; eyewear, lens = PID[pid]
    prompt = swap_prompt(eyewear, HAIR, lens=lens)
    rec = dict(name=name, cid=cid, pid=pid, scene=str(ROOT / "poc/pids" / scene), head=str(ROOT / f"poc/faces/{cid}_head.jpg"),
               face=str(ROOT / f"poc/faces/{cid}.jpg"), prompt=prompt, file=str(dest))
    try:
        if not dest.exists():
            urls = [falx.upload(rec["head"]), falx.upload(rec["scene"])]
            call("openai/gpt-image-2.5/flare/edit", dict(prompt=prompt, image_urls=urls, image_size={"width": 1024, "height": 1024},
                                                         quality="high", output_format="png"), dest)
        c, s, o = qc.feats(rec["face"]), qc.feats(rec["scene"]), qc.feats(dest)
        rec["qc"] = None if o is None or c is None or s is None else dict(
            id_arc=round(float(o["arc"] @ c["arc"]), 3), id_sf=round(float(o["sf"] @ c["sf"]), 3),
            leak=round(float(o["arc"] @ s["arc"]), 3), scale=round(float(o["box"][3] / s["box"][3]), 3))
    except Exception as e:
        rec["error"] = str(e)[:200]
    return rec

if __name__ == "__main__":
    with ThreadPoolExecutor(10) as ex: recs = list(ex.map(one, jobs()))
    keep = []
    for r in recs:
        q = r.get("qc"); ok = bool(q) and q["id_arc"] >= 0.45 and q["leak"] <= 0.2 and 0.8 <= q["scale"] <= 1.25
        r["keep"] = ok; keep.append(r) if ok else None
        print(f"{r['name']:22} {'KEEP' if ok else 'drop'} {q or r.get('error')}")
    (OUT / "teacher.json").write_text(json.dumps(recs, indent=2))
    z = OUT / "lora_train.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in keep:
            for src, suffix in ((r["head"], "start"), (r["scene"], "start2"), (r["file"], "end")):
                tmp = OUT / f"_{r['name']}_{suffix}.jpg"; Image.open(src).convert("RGB").save(tmp, quality=95)
                zf.write(tmp, f"{r['name']}_{suffix}.jpg"); tmp.unlink()
            zf.writestr(f"{r['name']}.txt", r["prompt"])
    print(f"kept {len(keep)}/{len(recs)}  zip {z}  {z.stat().st_size/1e6:.1f} MB  est ${0.20*len(recs):.2f}")
