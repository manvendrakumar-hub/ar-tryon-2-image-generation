# AR Try-On 2.0: personalised product images

Turn a customer's face scan into their own product photos: the exact frame (PID) on their own face, from the front and from the side, in place of the worn-by-model shots on a product page.

**Showcase (results, failures, recommendation, costs):** https://ar-tryon-2-showcase.vercel.app

This repo holds the proof of concept: the scripts that generated and scored every test image, the planning documents, and the showcase site.

## Where it landed

Four test rounds (11 to 12 November 2026) across 11 image models on fal.ai:

| Round | What changed | Best identity score* |
|---|---|---|
| 1 | One customer swapped into a catalog shot, 11 models | 0.61 (GPT Image 2.5 Flare) |
| 2 | Research-based prompts, head crop, two-step edits | 0.60 (GPT Image 2.5 Flare) |
| 3 | Open-source models, shortcuts, a 20-pair LoRA on Qwen | open models could not swap reliably |
| 4 | **Pose written in the prompt, white studio, front + side** | **0.74 (GPT Image 2.5 Flare)** |

\* ArcFace cosine similarity between the customer's scan and the generated image. Two different people score about 0; the same photo edited scores about 0.8. Small samples, not yet calibrated against human ratings.

**Recommended setup**
1. Generate with **GPT Image 2.5 Flare** (about $0.20 per image, token-billed estimate), with **Nano Banana Pro** ($0.15 per image) as backup. Two images per frame: front and side.
2. **Existing customers** get a stored profile (best scan, face signature, face attributes). A returning customer is recognised and their images are made ahead of time, so they show instantly.
3. **New customers** are generated live, about 45 seconds per frame, with a "your images are being generated" message.

GPT Image 2.5 and Nano Banana Pro do not accept custom LoRA weights. The customer data can train LoRAs for an in-house open model later.

The full write-up is in [docs/POC-PLAN.md](docs/POC-PLAN.md). The quality-check design is in [docs/QC-PLAN.md](docs/QC-PLAN.md).

## Repo layout

```
docs/
  POC-PLAN.md          plan, decisions, results of every round, caching design, cost model
  QC-PLAN.md           automatic quality checks: what exists, what is planned, calibration
poc/scripts/
  falx.py              fal.ai client (upload, run, download); reads FAL_KEY
  prep.py              face detect + head crop + upscale of the customer scan (identity reference)
  prompt.py            round-1 prompt template
  models.py            fal endpoint registry with unit prices (round 1)
  run_tryon.py         round 1: one customer x one catalog shot across 11 models
  round2.py            round 2: prompt and pipeline variants (A to E) on several models
  variant_f.py         round 2: "keep own head, restyle around it" variant
  round3.py            round 3: open models (Qwen, Klein) best-of-4; also runs a LoRA by URL
  round3_hacks.py      round 3: QC best-of-N pick + local face-swap pass
  faceswap.py          local InsightFace face swap + film grain (research licence, test only)
  gen_teacher.py       teacher images from GPT Image 2.5, QC-filtered, zipped for LoRA training
  train_lora.py        Qwen Image Edit 2511 LoRA training on fal
  watch_lora.py        waits for a fal training job and downloads the LoRA
  studio.py            round 4 and the recommended pipeline: locked studio pose, front + side
  identity.py          ArcFace embedding and similarity (YuNet detect + 5-point align)
  qc.py                QC scores: identity (ArcFace + SFace), leak, pose, framing, scene-keep
  grid.py, grid2.py, grid_studio.py   contact sheets with model names and scores
  fetch_pids.py        download PID catalog galleries from the Lenskart product API
  get_models.sh        download the local face models
showcase-site/         the Vercel showcase (static HTML + images)
requirements.txt
```

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo 'FAL_KEY=<your fal.ai key>' > .env   # git ignores .env
bash poc/scripts/get_models.sh    # face models for QC (add WITH_FACESWAP=1 for the round-3 swapper)
python3 poc/scripts/fetch_pids.py # catalog images for the six test PIDs
```

On a network that inspects TLS (corporate proxy), Python's own certificate bundle fails. The scripts call `truststore.inject_into_ssl()` at import time so they use the operating system's trust store instead.

## Running the recommended pipeline

1. Put a customer scan at `poc/faces/<id>.jpg` and make the identity reference:
   ```bash
   python3 poc/scripts/prep.py poc/faces/c05.jpg poc/faces/c05_head.jpg
   ```
2. Edit `PAIRS` in `poc/scripts/studio.py` (customer id, PID, product photo, eyewear description, facial hair, sunglasses or not) and the `MODELS` you want, then:
   ```bash
   python3 poc/scripts/studio.py           # generate, then score
   python3 poc/scripts/studio.py --score   # re-score saved images without new API calls
   python3 poc/scripts/grid_studio.py poc/out/r4_studio_c05_244418
   ```
   Outputs land in `poc/out/r4_studio_<customer>_<pid>/` with a `manifest.json` (scores, time, unit cost) and `grid.jpg`.

Each script's docstring has its own usage line. Every paid call goes through fal, so check the unit prices in `models.py` / `studio.py` before a large run.

## Unit prices (fal.ai, 2026-11-11)

| Model | Per image | Notes |
|---|---|---|
| GPT Image 2.5 Flare (high, 1024²) | ~$0.20 | token-billed estimate |
| Nano Banana Pro (1K/2K) | $0.15 | 4K is double |
| Seedream 5 Pro | ~$0.07 | |
| Qwen Image Edit 2511 | $0.03 per MP | open weights, Apache-2.0 |
| FLUX.2 Klein 9B | ~$0.04 | non-commercial weights |
| Qwen Image Edit 2511 LoRA training | $4 per 1,000 steps | |

## Not in this repo, on purpose

- **Customer data** (`data/`): the profiling and scan CSVs contain names, mobile numbers and face-photo links.
- **Customer photos and generated test images** (`poc/faces/`, `poc/out/`). The showcase site carries the 31 images shown on the live page.
- **Model weights** (`poc/models/`): large, and the InsightFace ArcFace and inswapper weights are licensed for non-commercial research only. `get_models.sh` re-downloads them. Production needs a commercially licensed face recogniser; SFace (Apache-2.0) is included and tracks ArcFace closely (correlation 0.85 in our tests).
- **The trained LoRA** (295 MB, 20 training pairs): an early experiment, not useful for production.
- **API keys**: put `FAL_KEY` in `.env`, which git ignores.

## Showcase site

`showcase-site/` is a static page deployed on Vercel. It is not indexed by search engines, but anyone with the link can open it, and it shows real customer scans: share it internally only. To redeploy:

```bash
cd showcase-site && vercel deploy --prod
```
