"""Run one (customer, PID scene) pair across fal edit models and save a manifest.

python3 poc/scripts/run_tryon.py --face F.jpg --scene S.png --product P.jpg \
    --pid 208956 --pid-desc "..." --frame-color "..." [--sunglasses --lens-color grey] \
    --out poc/out/run1 [--models a,b]
"""

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import falx  # noqa: E402
import prompt as P  # noqa: E402
from models import MODELS  # noqa: E402


def one(name, cfg, text, urls, out):
    t0 = time.time()
    try:
        res = falx.run(cfg["endpoint"], cfg["args"](text, urls))
        img = (res.get("images") or [res.get("image")])[0]
        path = falx.download(img["url"], out / f"{name}.png")
        return dict(model=name, ok=True, file=path.name, seconds=round(time.time() - t0, 1),
                    est_cost=cfg["est_cost"], endpoint=cfg["endpoint"], license=cfg["license"])
    except Exception as e:  # keep going; one model failing must not kill the run
        return dict(model=name, ok=False, error=str(e)[:300], seconds=round(time.time() - t0, 1),
                    est_cost=0, endpoint=cfg["endpoint"], license=cfg["license"])


def main():
    a = argparse.ArgumentParser()
    for k in ("face", "scene", "product", "pid", "pid_desc", "frame_color", "out"):
        a.add_argument("--" + k.replace("_", "-"), required=True)
    a.add_argument("--sunglasses", action="store_true")
    a.add_argument("--lens-color", default="")
    a.add_argument("--models", default=",".join(MODELS))
    args = a.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = P.build(args.pid_desc, args.frame_color, args.sunglasses, args.lens_color)
    (out / "prompt.txt").write_text(text)
    urls = [falx.upload(args.scene), falx.upload(args.face), falx.upload(args.product)]

    names = args.models.split(",")
    with ThreadPoolExecutor(max_workers=len(names)) as ex:
        results = list(ex.map(lambda n: one(n, MODELS[n], text, urls, out), names))

    manifest = dict(pid=args.pid, face=args.face, scene=args.scene, product=args.product,
                    results=results, total_est_cost=round(sum(r["est_cost"] for r in results), 4))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for r in results:
        print(f"{r['model']:24} {'OK ' if r['ok'] else 'ERR'} {r['seconds']:6}s ${r['est_cost']:<7} {r.get('error', '')[:120]}")
    print("total est $", manifest["total_est_cost"])


if __name__ == "__main__":
    main()
