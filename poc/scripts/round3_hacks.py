"""Round 3, part 2: no-training hacks. Per pair: pick the best open-model candidate by QC,
then run the local face-swap pass on (a) the raw catalog shot and (b) that best candidate, plus grain."""
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import faceswap as FS

def score(r):  # identity minus leak, zero if framing is off
    q = r.get("qc") or {}
    return -1 if not q or not (0.8 <= q.get("scale", 0) <= 1.25) else q["id_arc"] - max(0, q["leak"])

for run in sys.argv[1:]:
    run = pathlib.Path(run); m = json.loads((run / "manifest.json").read_text())
    m["results"] = [r for r in m["results"] if not r["variant"].startswith(("H_", "L_"))]
    def add(name, model, f, note=""):
        m["results"].append(dict(variant=name, model=model, file=f, est_cost=0, seconds=0, note=note))
    t0 = time.time()
    if FS.swap(m["face"], m["scene"], run / "H_faceswap_on_catalog.png"):
        add("H_faceswap_on_catalog", "Face swap only (open, local)", "H_faceswap_on_catalog.png")
        FS.grain(run / "H_faceswap_on_catalog.png", run / "H_faceswap_on_catalog_grain.png")
        add("H_faceswap_catalog+grain", "Face swap only + grain", "H_faceswap_on_catalog_grain.png")
    for grp, label in (("klein", "Klein 9B best-of-4"), ("qwen", "Qwen best-of-4"), ("lora", "Qwen + our LoRA best-of-4")):
        cands = [r for r in m["results"] if r.get("group") == grp]
        if not cands: continue
        best = max(cands, key=score)
        add(f"L_{grp}_best", f"{label} (QC pick: {best['variant']})", best["file"])
        if FS.swap(m["face"], run / best["file"], run / f"L_{grp}_best_faceswap.png"):
            add(f"L_{grp}_best+faceswap", f"{label} + face swap", f"L_{grp}_best_faceswap.png")
    m["faceswap_seconds"] = round(time.time() - t0, 1)
    (run / "manifest.json").write_text(json.dumps(m, indent=2)); print(run.name, m["faceswap_seconds"], "s")
