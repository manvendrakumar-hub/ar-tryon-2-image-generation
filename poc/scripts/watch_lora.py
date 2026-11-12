"""Wait for the fal LoRA training job to finish, then download the result. Costs nothing:
it only reads fal's list of completed requests (once a minute) and fetches the output."""
import json, os, pathlib, sys, time, urllib.parse, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import falx
import fal_client

EP = "fal-ai/qwen-image-edit-2511-trainer"
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "models/lora"; DEST.mkdir(parents=True, exist_ok=True)
falx.ensure_key()

def finished():
    q = urllib.parse.urlencode({"endpoint_id": EP, "limit": 5})
    req = urllib.request.Request("https://api.fal.ai/v1/models/requests/by-endpoint?" + q, headers={"Authorization": "Key " + os.environ["FAL_KEY"]})
    return json.load(urllib.request.urlopen(req, timeout=30))["items"]

deadline = time.time() + 4 * 3600
while time.time() < deadline:
    try:
        items = finished()
    except Exception as e:
        items = []; print("poll error", str(e)[:100], flush=True)
    if items:
        it = items[0]; print("finished:", it["request_id"], "status", it["status_code"], "duration", round(it["duration"] / 60, 1), "min", flush=True)
        if it["status_code"] != 200:
            print("TRAINING FAILED on fal; nothing to download"); sys.exit(2)
        res = fal_client.result(EP, it["request_id"])
        res["request_id"] = it["request_id"]; res["minutes"] = round(it["duration"] / 60, 1); res["steps"] = 1000
        f = falx.download(res["diffusers_lora_file"]["url"], DEST / "qwen_tryon_v0.safetensors")
        falx.download(res["config_file"]["url"], DEST / "qwen_tryon_v0_config.json")
        (ROOT / "out/teacher/lora_qwen.json").write_text(json.dumps(res, indent=2))
        print("downloaded", f, round(f.stat().st_size / 1e6, 1), "MB"); sys.exit(0)
    time.sleep(60)
print("gave up after 4 hours: job never showed as finished"); sys.exit(3)
