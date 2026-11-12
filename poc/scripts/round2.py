"""Round 2: prompt-pipeline variants for one (customer, catalog shot) pair.

Variants (all Nano Banana Pro unless the name says otherwise):
  A_r1_baseline   round-1 prompt and image order (scene, raw scan, product), 1K
  B_swap          identity first (head crop), scene second, role-based prompt, 2K
  C_swap_product  B + product photo as a third, detail-only reference
  D_two_step      chat-style: (1) put the glasses on the customer's own photo,
                  (2) restage that person into the catalog shot
  E_refine        B, then one targeted follow-up edit of the face ("edit, don't re-roll")
  B_* other models: the B prompt on Seedream 4.5 / Seedream 5 Pro / GPT Image 2.5 Flare

python3 poc/scripts/round2.py --cid c05 --pid 244418 --scene poc/pids/244418/g03.webp \
   --product poc/pids/244418/g00.webp --eyewear "black square Meller NURU eyeglasses" \
   --hair "beard" --out poc/out/r2_c05_244418 [--sunglasses]
"""

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import falx  # noqa: E402
import prompt as P1  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
NBP = "fal-ai/nano-banana-pro/edit"


def swap_prompt(eyewear, facial_hair, n="Image", product=False, lens=""):
    extra = f"\n{n} 3 is a product photo of the same eyewear, for detail reference only." if product else ""
    return f"""{n} 1 is the identity source: a photo of a real customer. {n} 2 is the target photo: a catalog shot of a model wearing {eyewear}.{extra}

Replace the model in {n} 2 with the person from {n} 1. Keep the person's facial features exactly the same as {n} 1: face shape, eyes, eyebrows, nose, lips, skin tone, {facial_hair}, hairline and hairstyle. Someone who knows this person must recognise them immediately.

Keep everything else in {n} 2 exactly the same: the eyewear (same frame shape, colour, thickness, size and position on the face{lens}), the clothing, pose, head angle, gaze direction, expression, framing, background, camera angle and lighting. Match skin texture, facial shadows, white balance and grain so the person belongs in {n} 2.

Render it as an unretouched photograph of a real person, with natural skin texture."""


def tryon_prompt(eyewear, lens=""):
    return f"""Put the {eyewear} from Image 2 on the person in Image 1. The eyewear sits naturally on the nose bridge and over the ears, at a realistic size for the face, with the same frame shape, colour and thickness as Image 2{lens}.

Keep the person's face, facial features, skin tone, facial hair, hair and expression exactly the same as Image 1. Keep the pose, clothing, background and lighting of Image 1 exactly the same. Change only one thing: the person is now wearing the eyewear."""


def restage_prompt(eyewear, facial_hair, lens=""):
    return f"""Image 1 is the identity source: a real customer, already wearing {eyewear}. Image 2 is a close reference of the same customer's face. Image 3 is the target photo: a catalog shot of a model wearing the same eyewear.

Replace the model in Image 3 with the customer from Images 1 and 2. Keep the customer's facial features exactly the same: face shape, eyes, eyebrows, nose, lips, skin tone, {facial_hair}, hairline and hairstyle.

Keep everything else in Image 3 exactly the same: the eyewear (same frame shape, colour, thickness, size and position{lens}), the clothing, pose, head angle, gaze direction, expression, framing, background, camera angle and lighting. Match skin texture, facial shadows, white balance and grain so the person belongs in Image 3.

Render it as an unretouched photograph of a real person, with natural skin texture."""


REFINE = """Image 1 is the identity source. Image 2 is the target photo.

The person in Image 2 should be the person from Image 1, but the face has drifted. Change only the face and facial features of the person in Image 2 so they match Image 1 exactly: face shape, eyes, eyebrows, nose, lips, skin tone and facial hair.

Keep the eyewear, hair outline, clothing, pose, head angle, background, camera angle, lighting and composition of Image 2 exactly the same."""


def call(endpoint, args, dest):
    t0 = time.time()
    res = falx.run(endpoint, args)
    img = (res.get("images") or [res.get("image")])[0]
    falx.download(img["url"], dest)
    return img["url"], round(time.time() - t0, 1)


def nbp(prompt, urls, res="2K"):
    return dict(prompt=prompt, image_urls=urls, aspect_ratio="1:1", resolution=res, output_format="png")


def main():
    a = argparse.ArgumentParser()
    for k in ("cid", "pid", "scene", "product", "eyewear", "hair", "out"):
        a.add_argument("--" + k, required=True)
    a.add_argument("--sunglasses", action="store_true")
    a.add_argument("--frame-color", default="")
    x = a.parse_args()
    out = pathlib.Path(x.out)
    out.mkdir(parents=True, exist_ok=True)
    lens = ", lenses stay dark and opaque" if x.sunglasses else ", lenses stay clear"

    def as_png(p):  # fal models do not all accept webp
        p = pathlib.Path(p)
        if p.suffix.lower() != ".webp":
            return p
        q = out / (p.stem + "_in.png")
        Image.open(p).convert("RGB").save(q)
        return q

    face, head = ROOT / f"poc/faces/{x.cid}.jpg", ROOT / f"poc/faces/{x.cid}_head.jpg"
    scene, product = as_png(x.scene), as_png(x.product)
    u_face, u_head, u_scene, u_prod = (falx.upload(p) for p in (face, head, scene, product))
    swap = swap_prompt(x.eyewear, x.hair, lens=lens)
    swap_fig = swap_prompt(x.eyewear, x.hair, n="Figure", lens=lens)
    sq1, sq2 = {"width": 1024, "height": 1024}, {"width": 2048, "height": 2048}

    def A():
        p = P1.build(x.eyewear, x.frame_color or "the same", x.sunglasses, "dark")
        _, s = call(NBP, nbp(p, [u_scene, u_face, u_prod], "1K"), out / "A_r1_baseline.png")
        return dict(variant="A_r1_baseline", seconds=s, est_cost=0.15, calls=1)

    def B_then_E():
        url, s = call(NBP, nbp(swap, [u_head, u_scene]), out / "B_swap.png")
        _, s2 = call(NBP, nbp(REFINE, [u_head, url]), out / "E_refine.png")
        return [dict(variant="B_swap", seconds=s, est_cost=0.15, calls=1),
                dict(variant="E_refine", seconds=round(s + s2, 1), est_cost=0.30, calls=2)]

    def C():
        p = swap_prompt(x.eyewear, x.hair, product=True, lens=lens)
        _, s = call(NBP, nbp(p, [u_head, u_scene, u_prod]), out / "C_swap_product.png")
        return dict(variant="C_swap_product", seconds=s, est_cost=0.15, calls=1)

    def D():
        a1 = dict(prompt=tryon_prompt(x.eyewear, lens), image_urls=[u_head, u_prod],
                  aspect_ratio="auto", resolution="2K", output_format="png")
        url, s = call(NBP, a1, out / "D_step1_tryon.png")
        _, s2 = call(NBP, nbp(restage_prompt(x.eyewear, x.hair, lens), [url, u_head, u_scene]), out / "D_two_step.png")
        return [dict(variant="D_step1_tryon", seconds=s, est_cost=0.15, calls=1, intermediate=True),
                dict(variant="D_two_step", seconds=round(s + s2, 1), est_cost=0.30, calls=2)]

    def other(name, endpoint, args, cost):
        def f():
            _, s = call(endpoint, args, out / f"{name}.png")
            return dict(variant=name, seconds=s, est_cost=cost, calls=1)
        return f

    jobs = [A, B_then_E, C, D,
            other("B_seedream_4_5", "fal-ai/bytedance/seedream/v4.5/edit",
                  dict(prompt=swap_fig, image_urls=[u_head, u_scene], image_size=sq2), 0.04),
            other("B_seedream_5_pro", "bytedance/seedream/v5/pro/edit",
                  dict(prompt=swap_fig, image_urls=[u_head, u_scene], image_size=sq1, output_format="png"), 0.072),
            other("B_gpt_2_5_flare", "openai/gpt-image-2.5/flare/edit",
                  dict(prompt=swap, image_urls=[u_head, u_scene], image_size=sq1, quality="high", output_format="png"), 0.20)]

    def safe(job):
        try:
            r = job()
            return r if isinstance(r, list) else [r]
        except Exception as e:
            return [dict(variant=job.__name__, error=str(e)[:300], est_cost=0)]

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        results = [r for rs in ex.map(safe, jobs) for r in rs]
    for r in results:
        if "error" not in r:
            r["file"] = r["variant"] + ".png"
    m = dict(cid=x.cid, pid=x.pid, sunglasses=x.sunglasses, face=str(face), head=str(head), scene=str(scene),
             product=str(product), prompts=dict(swap=swap, tryon=tryon_prompt(x.eyewear, lens),
             restage=restage_prompt(x.eyewear, x.hair, lens), refine=REFINE), results=results,
             # E and D already include their first call, so B and D step 1 are not added twice
             total_est_cost=round(sum(r["est_cost"] for r in results if r["variant"] not in ("B_swap", "D_step1_tryon")), 3))
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    for r in results:
        print(f"{x.cid}x{x.pid} {r['variant']:18} {r.get('seconds', '-'):>6}s {r.get('error', 'OK')[:140]}")
    print("est $", m["total_est_cost"])


if __name__ == "__main__":
    main()
