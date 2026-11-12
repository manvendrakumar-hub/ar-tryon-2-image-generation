"""Locked-pose studio shot: no catalog photo as input. The customer (head crop) and the
product photo go in; the shot (white background, close-up, fixed pose, lighting) is
written in the prompt, identical for every customer and PID.

python3 poc/scripts/studio.py   (runs the round-2 test pairs x 2 poses x 5 models)
"""
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import falx  # noqa: E402
import qc  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SQ = {"width": 1024, "height": 1024}

POSES = {
    "front": "facing the camera straight on, head level, eyes looking into the lens",
    "three_quarter": "head and shoulders turned about 30 degrees to the camera's left, eyes looking into the lens",
}

PAIRS = [  # same customers and PIDs as rounds 2 and 3
    ("c05", "244418", "poc/pids/244418/g00.webp", "black full-rim square Meller NURU eyeglasses", "beard and moustache", False),
    ("c09", "244418", "poc/pids/244418/g00.webp", "black full-rim square Meller NURU eyeglasses", "facial proportions", False),
    ("c05", "135212", "poc/pids/135212/g06.png", "silver metal double-bridge John Jacobs sunglasses with dark grey lenses", "beard and moustache", True),
    ("c09", "135212", "poc/pids/135212/g06.png", "silver metal double-bridge John Jacobs sunglasses with dark grey lenses", "facial proportions", True),
]


def prompt(eyewear, facial_hair, pose, sunglasses, n="Image"):
    lens = ", with the same dark grey lens tint and opacity" if sunglasses else ", with clear lenses"
    return f"""{n} 1 is the identity source: a photo of a real customer. {n} 2 is the product: {eyewear}, photographed on white.

Create a studio catalog portrait of the person from {n} 1 wearing exactly the eyewear from {n} 2.

Person: keep the facial features exactly the same as {n} 1: face shape, eyes, eyebrows, nose, lips, skin tone, {facial_hair}, hairline and hairstyle. Someone who knows this person must recognise them immediately. Ignore {n} 1's background, clothing, lighting and any glasses worn there.

Eyewear: the same frame shape, proportions, colour and finish, rim thickness, bridge and temples as {n} 2{lens}. It sits naturally on the nose bridge and over the ears, at a realistic size for this face.

Shot: close-up head-and-shoulders portrait, {POSES[pose]}, relaxed neutral expression, wearing a plain dark grey crew-neck T-shirt. Seamless pure white studio background. Soft, even, diffused front lighting with gentle fill and no harsh shadows. 85 mm lens at eye level, sharp focus on the eyes and the frame.

Render it as an unretouched photograph of a real person, with natural skin texture."""


MODELS = {
    "nano_banana_pro": ("Nano Banana Pro", "fal-ai/nano-banana-pro/edit", "Image",
                        lambda p, u: dict(prompt=p, image_urls=u, aspect_ratio="1:1", resolution="2K", output_format="png"), 0.15),
    "gpt_2_5_flare": ("GPT Image 2.5 Flare", "openai/gpt-image-2.5/flare/edit", "Image",
                      lambda p, u: dict(prompt=p, image_urls=u, image_size=SQ, quality="high", output_format="png"), 0.20),
    "seedream_5_pro": ("Seedream 5 Pro", "bytedance/seedream/v5/pro/edit", "Figure",
                       lambda p, u: dict(prompt=p, image_urls=u, image_size=SQ, output_format="png"), 0.072),
    "qwen": ("Qwen Image Edit 2511 (open)", "fal-ai/qwen-image-edit-2511", "Image",
             lambda p, u: dict(prompt=p, image_urls=u, image_size=SQ, output_format="png"), 0.03),
    "klein": ("FLUX.2 Klein 9B (open)", "fal-ai/flux-2/klein/9b/edit", "Image",
              lambda p, u: dict(prompt=p, image_urls=u, image_size=SQ, num_inference_steps=8, output_format="png"), 0.044),
}


def white_bg(path):
    """Share of border pixels that are near-white (1.0 = clean white background)."""
    g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    b = max(4, g.shape[0] // 20)
    border = np.concatenate([g[:b].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
    return round(float((border > 230).mean()), 3)


def run_pair_raw(pair):
    cid, pid, product, eyewear, hair, sun = pair
    out = ROOT / f"poc/out/r4_studio_{cid}_{pid}"
    out.mkdir(parents=True, exist_ok=True)
    head, face = ROOT / f"poc/faces/{cid}_head.jpg", ROOT / f"poc/faces/{cid}.jpg"
    prod = ROOT / product
    if prod.suffix == ".webp":
        png = out / "product.png"
        Image.open(prod).convert("RGB").save(png)
        prod = png
    urls = [falx.upload(head), falx.upload(prod)]
    results = []

    def job(key, pose):
        label, ep, n, args, cost = MODELS[key]
        name = f"{pose}_{key}"
        t0 = time.time()
        try:
            res = falx.run(ep, args(prompt(eyewear, hair, pose, sun, n), urls))
            falx.download(res["images"][0]["url"], out / f"{name}.png")
            return dict(variant=name, model=label, pose=pose, file=f"{name}.png", est_cost=cost, seconds=round(time.time() - t0, 1))
        except Exception as e:
            return dict(variant=name, model=label, pose=pose, error=str(e)[:200], est_cost=0)

    with ThreadPoolExecutor(10) as ex:
        results = list(ex.map(lambda kp: job(*kp), [(k, p) for p in POSES for k in MODELS]))
    return out, results


def finish(pair, out, results):
    """Score and write the manifest. Not thread-safe (OpenCV face nets): call sequentially."""
    cid, pid, product, eyewear, hair, sun = pair
    head, face = ROOT / f"poc/faces/{cid}_head.jpg", ROOT / f"poc/faces/{cid}.jpg"
    prod = out / "product.png" if (out / "product.png").exists() else ROOT / product
    cust = qc.feats(face)
    for r in results:
        if "file" not in r:
            continue
        o = qc.feats(out / r["file"])
        r["qc"] = None if o is None else dict(
            id_arc=round(float(o["arc"] @ cust["arc"]), 3), id_sf=round(float(o["sf"] @ cust["sf"]), 3),
            yaw=round(float(o["pose"][0]), 3), white_bg=white_bg(out / r["file"]))
    m = dict(cid=cid, pid=pid, face=str(face), head=str(head), product=str(prod),
             prompt_front=prompt(eyewear, hair, "front", sun), prompt_three_quarter=prompt(eyewear, hair, "three_quarter", sun),
             results=results, total_est_cost=round(sum(r["est_cost"] for r in results), 3))
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    return m


def rescore(pair):
    """Rebuild results from the saved images (no API calls) and score them."""
    cid, pid = pair[0], pair[1]
    out = ROOT / f"poc/out/r4_studio_{cid}_{pid}"
    mf = out / "manifest.json"
    old = {r["variant"]: r for r in json.loads(mf.read_text())["results"]} if mf.exists() else {}
    results = []
    for pose in POSES:
        for key, (label, _, _, _, cost) in MODELS.items():
            name = f"{pose}_{key}"
            if (out / f"{name}.png").exists():
                r = dict(variant=name, model=label, pose=pose, file=f"{name}.png", est_cost=cost,
                         seconds=old.get(name, {}).get("seconds", "-"))
                results.append(r)
    return finish(pair, out, results)


def report(m):
    print(f"\n== {m['cid']} x {m['pid']}  est ${m['total_est_cost']}")
    for r in m["results"]:
        print(f"  {r['variant']:28} {r.get('seconds', '-'):>6}s {r.get('qc') or r.get('error')}")


if __name__ == "__main__":
    if "--score" in sys.argv:
        for p in PAIRS:
            report(rescore(p))
    else:
        with ThreadPoolExecutor(4) as ex:
            gen = list(ex.map(lambda p: (p, run_pair_raw(p)), PAIRS))
        for p, (out, results) in gen:
            report(finish(p, out, results))
