"""Round 3, part 1: open-source models with no training. For each round-2 test pair,
generate 4 candidates per model with the identity-first swap prompt; QC later picks the best."""
import json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import falx
from round2 import swap_prompt

ROOT = pathlib.Path(__file__).resolve().parents[2]
SQ = {"width": 1024, "height": 1024}
MODELS = {
    "qwen": ("Qwen Image Edit 2511 (open)", "fal-ai/qwen-image-edit-2511", lambda p, u: dict(prompt=p, image_urls=u, image_size=SQ, num_images=4, output_format="png"), 0.03),
    "klein": ("FLUX.2 Klein 9B (open)", "fal-ai/flux-2/klein/9b/edit", lambda p, u: dict(prompt=p, image_urls=u, image_size=SQ, num_images=4, num_inference_steps=8, output_format="png"), 0.033),
}

def run_pair(src, lora=None, tag=None):
    src = pathlib.Path(src); m2 = json.loads((src / "manifest.json").read_text())
    out = ROOT / "poc/out" / src.name.replace("r2_", "r3_"); out.mkdir(exist_ok=True)
    mf = out / "manifest.json"
    m = json.loads(mf.read_text()) if mf.exists() else dict(cid=m2["cid"], pid=m2["pid"], face=m2["face"], head=m2["head"], scene=m2["scene"], product=m2["product"], results=[])
    prompt = m2["prompts"]["swap"]; urls = [falx.upload(m2["head"]), falx.upload(m2["scene"])]
    todo = {tag: ("Qwen Image Edit 2511 + our LoRA (open)", "fal-ai/qwen-image-edit-2511/lora",
                  lambda p, u: dict(prompt=p, image_urls=u, image_size=SQ, num_images=4, output_format="png", loras=[dict(path=lora, scale=1.0)]), 0.035)} if lora else MODELS
    for key, (label, ep, args, cost) in todo.items():
        try:
            res = falx.run(ep, args(prompt, urls))
            for i, im in enumerate(res["images"]):
                name = f"{key}_s{i+1}"; falx.download(im["url"], out / f"{name}.png")
                m["results"] = [r for r in m["results"] if r["variant"] != name]
                m["results"].append(dict(variant=name, model=label, file=f"{name}.png", est_cost=cost, seconds=0, group=key))
        except Exception as e:
            print(src.name, key, "ERR", str(e)[:200])
    mf.write_text(json.dumps(m, indent=2)); return out.name

if __name__ == "__main__":
    lora = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("http") else None
    tag = sys.argv[2] if lora else None
    srcs = [ROOT / "poc/out" / d for d in ("r2_c05_244418", "r2_c09_244418", "r2_c05_135212", "r2_c09_135212")]
    with ThreadPoolExecutor(4) as ex: print(list(ex.map(lambda s: run_pair(s, lora, tag), srcs)))
