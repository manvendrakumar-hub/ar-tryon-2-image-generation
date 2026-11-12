"""Variant F (own-head restyle): start from the customer's own photo already wearing
the eyewear (D_step1_tryon) and restyle everything around the head to the catalog
shot. The head pose stays the customer's, so the model never has to invent the face
from a new angle."""
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import falx
from concurrent.futures import ThreadPoolExecutor

PROMPT = """Image 1 is a real customer wearing eyewear. Image 2 is a catalog photo that defines the look we want.

Restyle Image 1 into the catalog look of Image 2: use Image 2's background, clothing, framing, crop, lighting and colour grade, so the result reads as a photo from the same shoot as Image 2.

Keep the customer's head from Image 1 exactly the same: face, facial features, skin tone, facial hair, hair, expression, head angle and the eyewear they are wearing (same frame shape, colour, size and position). The person in the result is the customer from Image 1, not the model from Image 2.

Render it as an unretouched photograph of a real person, with natural skin texture."""

def one(run):
    run = pathlib.Path(run); m = json.loads((run / "manifest.json").read_text())
    t0 = time.time()
    urls = [falx.upload(run / "D_step1_tryon.png"), falx.upload(m["scene"])]
    res = falx.run("fal-ai/nano-banana-pro/edit", dict(prompt=PROMPT, image_urls=urls, aspect_ratio="1:1", resolution="2K", output_format="png"))
    falx.download(res["images"][0]["url"], run / "F_own_head_restyle.png")
    m["results"] = [r for r in m["results"] if r["variant"] != "F_own_head_restyle"]
    m["results"].append(dict(variant="F_own_head_restyle", seconds=round(time.time() - t0, 1), est_cost=0.30, calls=2, file="F_own_head_restyle.png"))
    m["prompts"]["own_head_restyle"] = PROMPT
    m["total_est_cost"] = 1.362  # 1.212 for the first eight variants + 0.15 for this call
    (run / "manifest.json").write_text(json.dumps(m, indent=2)); return run.name

with ThreadPoolExecutor(4) as ex: print(list(ex.map(one, sys.argv[1:])))
