"""Train a Qwen Image Edit 2511 LoRA on the QC-filtered teacher pairs (fal trainer).
Result (LoRA URL) is written to poc/out/teacher/lora_qwen.json."""
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import falx
OUT = pathlib.Path(__file__).resolve().parents[1] / "out/teacher"
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
t0 = time.time()
url = falx.upload(OUT / "lora_train.zip")
res = falx.run("fal-ai/qwen-image-edit-2511-trainer", dict(image_data_url=url, steps=steps, learning_rate=1e-4))
res["minutes"] = round((time.time() - t0) / 60, 1); res["steps"] = steps
(OUT / "lora_qwen.json").write_text(json.dumps(res, indent=2))
print("done", res["minutes"], "min", res["diffusers_lora_file"]["url"])
