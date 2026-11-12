# AR Try-On 2.0: Personalised PID Image Generation, POC and Integration Plan

Status: v0.2 (2026-11-11). Run 1 (1 customer × 1 PID × 11 models) is done and cost ≈ $1.18.

## 0. Decisions (2026-11-11)

- **Data.** Customer data may be used for the POC and for training. The data lives in `data/`, as two CSVs: profiling + scans.
- **Target images.** Only the **worn-by-model shots** on each PID page get replaced.
- **POC scope.**
  - Run 1: 1 customer × 1 PID across models (done).
  - Then **10 customers × 6 PIDs**: 3 eyeglasses + 3 sunglasses, covering Hustlr, John Jacobs and Meller.
- **Compute.** Cloud GPUs for production; 5090s are available in-house.
- **Frame measurements.** Face-scan mm data will be requested later. For now, use the general process.
- **Latency UX.** A cache hit is instant. A cache miss shows "generating…".

### POC PIDs (all have male + female worn-by-model shots unless noted)

| Brand | Eyeglasses | Sunglasses |
|---|---|---|
| Hustlr | 208956 Grey Stone square (Air) | 242765 Hustlr Z Graphite square |
| John Jacobs | 201388 Supreme Steel gunmetal double-bridge (**male shots only**) | 135212 JJ Tints silver mirror-grey aviator |
| Meller | 244418 NURU black square | 238524 SHIPO black/orange aviator |

## 0b. Run 1 results: customer c04 (male, 30–34, beard) × PID 208956, studio shot

Identity is measured with ArcFace cosine similarity against the customer's scan. For calibration:
- The catalog model against the customer scores −0.01.
- Other customers against c04 score between −0.09 and 0.16.

"Leak" is the score against the original catalog model; lower is better.

| Model | Identity | Leak | Kept catalog pose | Time | $/img |
|---|---|---|---|---|---|
| GPT Image 2.5 Flare | 0.611 | 0.133 | no (turned frontal) | 40 s | ~0.20 |
| Seedream 4.5 | 0.599 | 0.293 | yes | 46 s | 0.04 |
| GPT Image 2.5 Sunburst | 0.573 | 0.134 | no | 59 s | ~0.20 |
| FLUX.2 Pro | 0.571 | 0.100 | no (pulled in the customer's own T-shirt and earbud) | 28 s | 0.075 |
| Nano Banana Pro | 0.486 | 0.187 | yes (best scene fidelity) | 33 s | 0.15 |
| Seedream 5 Pro | 0.448 | 0.273 | yes | 52 s | 0.077 |
| Nano Banana 2 | 0.424 | 0.295 | yes | 28 s | 0.08 |
| GPT Image 2 high | 0.360 | 0.443 | no | 118 s | ~0.20 |
| FLUX.2 Klein 9B | 0.257 | 0.329 | yes; frame drifted lighter | 9 s | 0.044 |
| Qwen Image Edit 2511 | 0.199 | 0.606 | yes; checkerboard background artifact | 48 s | 0.03 |
| FLUX Kontext Max multi | −0.005 | 0.911 | did not swap the face at all | 21 s | 0.08 |

Outputs, grid and manifest are in `poc/out/r1_c04_208956/`. One sample cannot settle anything; these are directional only.

## 0c. Scan-match caching

**The goal:** reuse earlier generations when the person in today's scan is the same person *and still looks the same*.

On every new scan:
1. **Embed the scan** with ArcFace (`poc/scripts/identity.py`). This is the same scorer used for QA.
2. **Identity check.** Compare the embedding to the stored reference embedding for that customer ID. If there is no ID, as with walk-ins, do a nearest-neighbour lookup over store embeddings.
   - Cosine ≥ T_same (to calibrate; expected ~0.5–0.6 for scan-vs-scan of the same person) means it is the same person.
   - Below that, treat it as a new identity.
3. **Look-freshness check.** This addresses the "the customer sees an old version of himself" risk. Run the existing attribute labellers on the new scan (`facialHair`, skin tone, glasses on or off, plus hairstyle and hair length via a VLM) and diff them against the stored attributes. Also apply a scan-age TTL (e.g. 90 days).
4. **Decide.**
   - Same person and unchanged look: **serve cached images instantly**, labelled "from your scan on <date> · refresh".
   - Same person but changed look, or TTL expired: regenerate. Show "generating…", serve each new image as it lands, then swap the reference.
   - New identity: full generation.
5. **What gets stored per identity version:** the clean reference face crop, the embedding, the attribute vector, and a "hero" generated image. The hero image becomes an extra reference that keeps the face consistent across PIDs and makes the next PID cheaper and more stable.
6. **Cache key:** `identity_version × pid × shot_id × model_version`.

**Calibration still needed.** The web-scan and store-scan CSVs share no customer IDs. A real "same person, two scans" threshold therefore needs a small staff set: 10 people × 2 scans on different days, one with a changed look (shaved, glasses, new haircut).

## 0f. Round 4: locked studio pose and the LoRA test (≈ $4.50)

**Locked studio pose.** No catalog photo goes in. The inputs are the customer's head crop and the product photo. The shot is written in the prompt and is the same for every customer and PID: white seamless background, close-up head and shoulders, fixed pose (front or three-quarter), soft even light, plain dark grey T-shirt. The prompt is in `poc/scripts/studio.py`; the grids are in `poc/out/r4_studio_*/grid.jpg`.

Averages over 2 customers × 2 PIDs × 2 poses (8 images per model):

| Model | Identity (ArcFace, worst) | SFace | Time | $/img | Licence |
|---|---|---|---|---|---|
| GPT Image 2.5 Flare | 0.736 (0.68) | 0.736 | 45 s | ~0.20 | API |
| **Qwen Image Edit 2511** | **0.734 (0.67)** | 0.684 | 22 s | **0.03** | **Apache-2.0, self-hostable** |
| Seedream 5 Pro | 0.712 (0.64) | 0.700 | 43 s | 0.072 | API |
| Nano Banana Pro | 0.594 (0.53) | 0.639 | 34 s | 0.15 | API |
| FLUX.2 Klein 9B | 0.509 (0.45) | 0.528 | 11 s | 0.044 | non-commercial |

Per model: GPT Image 2.5 rose from 0.60 to 0.74, Seedream 5 Pro from 0.51 to 0.71, and Nano Banana Pro from 0.50 (single call) to 0.59, the same as its two-step result at half the cost. Moving the customer into someone else's pose and lighting was the hard part; describing a neutral pose removes it.

**The open-source model now matches the frontier.** Base Qwen, with no training, scores the same as GPT Image 2.5 at about a seventh of the price and half the time on fal, and it can run on our own 5090s.

Things seen by eye that the scores don't capture:
- GPT Image 2.5 and Seedream keep small identity details (c09's bindi). Qwen, Nano Banana Pro and Klein drop them.
- Frames are right in shape and colour on all five; Seedream's three-quarter frames come out a little thicker. Not yet checked by an automatic frame check.
- The three-quarter pose comes out as a mild turn (about 20–30 degrees), not a strong profile.

**LoRA test (Qwen + our 20-pair LoRA, catalog-pose swap task).** On the half-body street shot it worked partly: it swapped the face where base Qwen didn't (identity up to 0.52, against −0.03 for base). On the tight close-up it failed: it blended the two faces and pulled in the customer's store background. With 20 pairs and 1,000 steps that is expected. The studio route makes this LoRA unnecessary for now, because base Qwen already does the studio task.

Round-4 grids carry the model name on every tile.

## 0d. Round 2 results (2 new customers × 2 new PIDs, 10 pipeline variants, ≈ $5.45)

Pairs: c05 (male, beard) and c09 (female) × Meller NURU 244418 (eyeglasses, tight 3/4 close-up) and John Jacobs 135212 (sunglasses, half-body street shot). Grids: `poc/out/r2_*/grid.jpg`.

Average over the 4 pairs. Identity is ArcFace cosine against the scan; the worst of the four is in brackets.

| Variant | Model | Identity | Leak | Calls | Time | $ |
|---|---|---|---|---|---|---|
| Try-on on the customer's own photo (step 1 only) | Nano Banana Pro | **0.756** (0.68) | n/a | 1 | 46 s | 0.15 |
| B: identity-first swap prompt | GPT Image 2.5 Flare | 0.599 (0.53) | 0.04 | 1 | 34 s | ~0.20 |
| D: two-step (try-on, then restage into catalog shot) | Nano Banana Pro | 0.592 (0.53) | 0.07 | 2 | 90 s | 0.30 |
| A: round-1 prompt | Nano Banana Pro | 0.546 (0.43) | 0.05 | 1 | 42 s | 0.15 |
| B: identity-first swap prompt | Seedream 5 Pro | 0.513 (0.45) | 0.23 | 1 | 47 s | 0.07 |
| C: B + product photo | Nano Banana Pro | 0.512 (0.40) | 0.03 | 1 | 57 s | 0.15 |
| B: identity-first swap prompt | Nano Banana Pro | 0.497 (0.38) | 0.06 | 1 | 60 s | 0.15 |
| F: keep own head, restyle around it | Nano Banana Pro | 0.481 (0.41) | 0.07 | 2 | 47 s | 0.30 |
| E: B + one corrective follow-up edit | Nano Banana Pro | 0.480 (0.38) | 0.05 | 2 | 97 s | 0.30 |
| B: identity-first swap prompt | Seedream 4.5 | 0.334 (0.20) | 0.43 | 1 | 43 s | 0.04 |

What this says (4 pairs only, so gaps under ~0.05 are noise):
- **The "Gemini chat" quality is the step-1 edit.** Adding glasses to the customer's own photo keeps the face pixels, so identity is 0.68–0.81. Every route that moves the customer into the catalog pose scores 0.15–0.25 lower, on every model.
- **The research-based prompt alone did not beat the round-1 prompt on Nano Banana Pro** (0.497 vs 0.546). It did fix GPT Image 2.5: it now keeps the catalog pose and scored highest of the single-call swaps.
- **Two-step is the best Nano Banana Pro route** (0.592), at twice the cost and time.
- **Seedream 4.5's round-1 result did not hold.** It left the catalog model's face in 3 of 4 images.
- **Half-body shots are easier than tight close-ups.** The small-face street shots look convincing on most models; the 3/4 close-up is the hard case.
- **The lever that is left is the input, not the prompt.** The scan is one frontal 480-px photo and the catalog shots are 3/4 views. A scan taken at 2–3 angles would give the model a reference at the right pose.

### Prompt pipeline rules adopted (from Google's, OpenAI's and fal's own guides)
1. Identity photo goes in slot 1, cropped to the head and upscaled (`poc/scripts/prep.py`).
2. Each image gets one named role; the smallest reference set that works (2 images for a swap).
3. Plain sentences: what transfers, then "keep everything else in Image 2 exactly the same" with the list.
4. No long negative lists; one short realism line.
5. Per-model wording: "Image N" for Gemini and GPT, "Figure N" for Seedream.
6. 2K output on Nano Banana Pro (same price as 1K).

## 0e. Cache POC with power customers

The scan-match cache (§0c) will be tested in store on **power customers**: the most frequent buyers. Data to request per customer:
- every historical face scan with its timestamp (at least 2 scans, ideally months apart),
- customer ID and the attribute labels already produced per scan,
- PIDs bought or tried, to pick which images to pre-generate.

This one data set does three jobs: it calibrates the same-person threshold, it tests the look-freshness check on real changes over time, and it calibrates the QC identity threshold (see `docs/QC-PLAN.md` §4).

## 1. Goal

The customer takes a face scan (on their phone, or on an iPad in a Lenskart store). On every PID page, the model-shot catalog images are then replaced with images of **that customer** wearing **that exact frame**.

There are two non-negotiables:
1. **Identity lock.** The person in the image must be recognisably the customer. Face shape, skin tone, facial hair and features must not drift.
2. **PID lock.** The frame must be exactly the SKU: shape, colour, rim type, bridge, temples, lens tint and logo. It must also be **the correct size relative to the face**.

Secondary goals are cost per image, latency, and integration with the existing Lenskart stack.

## 2. What the data tells us

`face_scan_links_web_app_last7d.csv`:
- 36,819 rows, all unique customers, one scan each, from 2026-10-28 to 2026-11-04. That is **about 5,300 scans/day**, with a peak of 5,808.
- There are two sources: `static5.lenskart.com/juno-fus/.../frameSizeImage_Personalisation` (80%) and `static.lenskart.io/frame-size` (20%). These are the frame-size scans that already exist.
- Images are **480×640 JPEG** (a few are 405×720).
- In a sample I checked by eye: there is motion blur, low-angle selfies, people lying in bed, mixed or dim indoor light, half-closed eyes, and bystanders in frame.

**Implication:** the input is low-resolution and uncontrolled. Identity preservation will be hardest on exactly these inputs. The pipeline therefore needs a **scan quality gate** up front, and the in-store iPad capture should be guided (good light, frontal, eyes open).

`Google Sheet (customer profiling)`: **not accessible.** It returns 401, so it is restricted to the org domain. It is needed to build a stratified test set that covers age, gender, skin tone, face shape, facial hair and existing spectacles.

## 2b. Prior work we reuse (from `AR Try On`, `ComfyUI`, `Reflection Images Project`)

- **fal wrapper.** `AR Try On/scripts/fal_setup.py` handles upload, run, download and a free content-policy probe. It includes the `truststore.inject_into_ssl()` fix needed on the corporate proxy. `FAL_KEY` comes from `.env`.
- **Proven prompt structure.** Name the reference roles (PLATE / CHARACTER / PID). Use the "NATURAL RENDER LOCK" realism block, a colour lock ("do not add tortoiseshell mottling"), and an eye-colour lock. Sample N=3 per image and pick the best.
- **Known frontier failure modes:**
  - The temple logo comes out random.
  - Plain frames pick up invented tortoiseshell.
  - Eye colour drifts.
  - Nano Banana garbles printed logo text ("VINCENT CHASE").
  - Seedance rejects photoreal faces.
- **Protected generation** (`Reflection Images Project/src/protected_generation.py`): pasting the original product pixels back in via a mask gave 0/255 drift. **This is the core of Route C.**
- **Catalogue fetch.** `Browser Automation/examples/higgsfield-reflection/download_from_feed.py` reads the Merchant feed (3,205 eyewear rows) and prefers `_Image_PLA`. It also finds 49 rows with `_model.png` worn-by-model shots, which are exactly the images we would replace.
- **BiRefNet** (`fal-ai/birefnet/v2`) cuts out clear and lilac frames cleanly; luma keying fails on them.
- **Local ComfyUI 0.42.0** has blueprints for Qwen-Image-Edit 2511, FLUX.2 Klein 4B, Flux.1 Fill, Mediapipe face detection, SAM3 and BiRefNet. **None of the image-edit weights are downloaded.** The only weights on disk are MiniMax H3 (video) and SeedVR2. The `lk_catalogue` node source lives on the 4090 box, not on this Mac.

## 3. Two corrections to the framing

1. **We are not fine-tuning an LLM here.** We fine-tune an **image-edit model** (a diffusion transformer) with a LoRA. An LLM or VLM is only used as a QA judge and for prompt building.
2. **Customer scans alone cannot train the model.** An edit LoRA needs *pairs*: (face without glasses + PID product image) → (same face wearing that PID). Those pairs do not exist. The standard approach is **distillation**:
   - Generate pairs with the frontier models.
   - Keep only the ones that pass QA.
   - Train the open-source LoRA on them.

   That way the frontier spend becomes a one-time training-data cost instead of a per-customer cost.
   *Per-customer LoRAs are not viable* at 5k scans/day. The model has to be **zero-shot**: one general "LK eyewear try-on" LoRA, with the face and the PID passed as reference images at inference.

## 4. Architecture options (what we will benchmark)

| Route | How | Identity | PID accuracy | Cost/img | Notes |
|---|---|---|---|---|---|
| **A. Frontier full edit** | Scan + PID product shot(s) + prompt → GPT Image 2/2.5, Nano Banana Pro, Seedream 5 Pro (via fal) | Good, not perfect (re-renders whole face) | Good on shape and colour; weak on logos and thin metal | $0.04–0.20+ | Best quality ceiling, highest cost, third-party data exposure |
| **B. Open-source full edit** | Same inputs → Qwen-Image-Edit-2511 / FLUX.2 / FLUX.1 Kontext in ComfyUI on 4090/5090 | Medium (rises with LoRA) | Medium (rises with LoRA) | ~$0.002–0.005 | Needs distillation LoRA to be competitive |
| **C. Composite → refine (recommended to test hardest)** | Detect face landmarks → place the **real PID product cutout** at the correct scale (from face-scan measurements + frame width in mm) → **masked inpaint** only the eyewear region for shadows, nose-pad contact, lens reflections and relighting | **Excellent** (face pixels outside the mask are untouched) | **Excellent** (the real product pixels are used) | ~$0.001–0.003 local | Front view only, unless 3D frame assets exist for angled shots |

Hypothesis: **C wins on front views for identity and PID accuracy at the lowest cost. A/B are needed for angled and lifestyle shots.** The production answer is probably C for front, and B+LoRA for angles, with A as the fallback when QA fails.

## 5. Phase 1: POC (in this repo)

### 1a. Inputs
- **Faces:** 30 scans from the CSV, chosen to be diverse (and stratified by the profiling sheet once we have access). Add 5–10 consented in-house faces for clean comparison.
- **PIDs:** 10 PIDs covering full-rim acetate, thin metal, rimless, aviator, cat-eye, round, tinted sunglasses, transparent frames and a logo-heavy frame. For each: front, 45° and side product shots, plus frame dimensions in mm.

### 1b. Scan quality gate
Use InsightFace/RetinaFace to check: single face, frontal pose (yaw/pitch < ~20°), eyes open, blur (Laplacian variance), face pixel height, and whether the customer is already wearing glasses. This yields pass, fix or reject. It also lets us measure what % of the real scans are usable.

### 1c. Model shoot-out
- **Smoke test:** 6 faces × 3 PIDs × 6 endpoints, about 108 images, **≈ $10–15**.
  - Endpoints: GPT Image 2 edit, GPT Image 2.5 edit, Nano Banana Pro edit, Nano Banana 2 edit, Seedream 5 Pro edit, Seedream 4.5 edit, plus Qwen-Image-Edit / FLUX Kontext on fal as an open-source proxy.
- **Full benchmark:** 30 faces × 10 PIDs × the top 3 models plus Route C, about 900 paid images, **≈ $80–120**.
- **Prompt system:** one structured template (PID attributes pulled from the catalog: shape, colour, material, rim, lens) plus fixed realism rules. It is tuned once and applied to every face. No per-face prompting.

### 1d. Automatic scoring for every output
- **Identity:** ArcFace cosine similarity between the scan and the output. The threshold is calibrated on the in-house faces.
- **PID fidelity:** crop the frame region, then DINOv2/CLIP similarity against the product shot, plus a VLM checklist judge (colour, shape, rim, bridge, temple, logo, tint, scale).
- **Artifacts:** double glasses, changed hair or skin tone, warped lenses, and missing temple arms.
- **Human review grid:** an HTML page with side-by-side images and 1–5 ratings, to calibrate the automatic scores.

**Deliverable:** a scorecard per model and route (quality, cost per image, latency), plus a recommended route.

## 6. Phase 2: Open-source on 4090/5090 (ComfyUI)

- Author the workflows in the local ComfyUI. The Mac is fine for building them, but too slow and MPS-limited for benchmarking. Then run them on the 4090/5090 boxes.
- **Candidate models (license check is critical for Lenskart commercial use):**
  - **Qwen-Image-Edit-2511.** Apache-2.0 and multi-image reference, so it is commercially safe. **Primary.**
  - **FLUX.2 [klein] 4B.** Apache-2.0, very fast. A secondary candidate.
  - **FLUX.1 Kontext [dev] / FLUX.2 [dev] / klein 9B.** Non-commercial licences, so they need a BFL commercial licence for production. Benchmark only.
- **Speed-ups:** Lightning/Turbo 4–8-step LoRAs; FP8 on 4090; NVFP4 (Nunchaku) on 5090.
- **Estimated throughput** (to be verified by benchmark):

  | GPU | Qwen-Edit 8-step, ~1MP | Images/hour/GPU |
  |---|---|---|
  | RTX 5090 32GB | ~5–8 s | ~450–700 |
  | RTX 4090 24GB | ~9–15 s | ~250–400 |

## 7. Phase 3: LoRA fine-tune (distillation) and quantisation

1. **Dataset.** Generate 2,000–3,000 pairs from the winning frontier model using **consented** faces (or synthetic faces + consented). Use the QA filter and expect about 30% rejects. Cost **≈ $200–450**.
2. **Training.** Train a LoRA on Qwen-Image-Edit-2511 with ai-toolkit or musubi-tuner on the **5090** (needs 32GB headroom). Estimated **6–12 GPU-hours** per run, with 2–3 runs.
3. **Re-benchmark.** Re-run the Phase 1 scorecard. The target is ≥ 90% of the frontier score at < 5% of the cost.
4. **Quantise.** FP8 for 4090, NVFP4 for 5090. Re-check identity scores after quantising, because quantisation can cost fidelity.

## 8. Phase 4: Production integration (for Engineering/DevOps)

```
Phone / Store iPad ──scan──▶ Face-scan service (exists) ──▶ scan_id + measurements
                                                   │
PDP page (PID) ──GET /tryon/{scan_id}/{pid}/{angle}▼
              ┌────────────── Try-On API (FastAPI) ──────────────┐
              │ cache hit? → CDN URL                              │
              │ miss → enqueue job (Redis/SQS)                    │
              └───────────────┬───────────────────────────────────┘
                              ▼
          GPU workers: ComfyUI headless (API-format workflow JSON, models baked into image)
                              ▼
          QA gate (ArcFace + PID check) ── fail ──▶ fallback: frontier API (fal)
                              ▼ pass
          S3 + CDN  ◀── cache key = hash(scan) + pid + angle + model_version
```

- **What it ships as:** a Docker image per worker (ComfyUI + custom nodes + pinned model weights). The workflow is checked in as `workflow_api.json`, so it is versioned like code. The API is a small Python service. DevOps sees a normal stateless service plus a GPU worker pool that **autoscales on queue depth**.
- **Hosting options:** the owned 4090/5090 boxes (cheapest at steady load), or RunPod/Modal serverless for bursts (4090 ≈ $1.10/hr, 5090 ≈ $1.58/hr serverless; pods ≈ $0.34–0.99/hr).
- **Generation strategy.** Pre-generating the whole catalogue per customer is not feasible. Instead:
  - When a scan lands, pre-generate the **top-N recommended or best-seller PIDs** (front shot).
  - Then generate **lazily on PDP view**, and cache.
- **UX for latency:** show the existing AR/2D overlay instantly, then swap in the generated photo when it is ready (~5–15 s).
- **Model upgrades** are a `model_version` bump. The cache key invalidates automatically.

## 9. Cost model (estimates, to be replaced by measured numbers)

Volume assumption: 5,300 customers/day × ~16 images (8 PIDs × 2 shots) ≈ **85k images/day**. Use per-100k as the unit:

| Route | $/image | Per 100k images/day | Per month |
|---|---|---|---|
| GPT Image 2 edit, high, portrait | ~$0.20+ | ~$20,000 | ~$600k |
| Nano Banana Pro edit | $0.15 | $15,000 | ~$450k |
| Seedream 5 Pro edit (<1536px) | ~$0.0675 | $6,750 | ~$200k |
| Seedream 4.5 edit | $0.04 | $4,000 | ~$120k |
| **Self-hosted Qwen-Edit + LoRA (5090)** | ~$0.002–0.005 | ~$200–500 | **~$6–15k** (cloud) / power only (owned) |

Self-hosted capacity at 100k/day: ≈ 150–220 GPU-hours/day, which is **~8–10 × 5090 continuous, or ~20–25 for store-hour peaks**. A 4090 needs roughly 1.8× the count.

POC frontier spend, all phases: **≈ $300–600 total.**

## 10. Risks

- **Privacy and consent (DPDP Act 2023).** Customer faces go to third parties (fal → OpenAI/Google/ByteDance) and get used for training. Legal/consent sign-off is needed. Until then, the POC uses consented in-house faces plus a small internal sample.
- **Scan quality.** A share of scans will be unusable. This needs measuring, and the capture UX needs fixing.
- **Frame scale.** Generative models do not know millimetres. Correct scale requires face-scan measurements plus the PID frame width. Route C handles this explicitly.
- **Licences.** FLUX dev-family weights are non-commercial. Default to Apache-2.0 models.
- **Angled shots** need either 3D frame assets (does Lenskart's 3D try-on have GLBs per PID?) or the generative route.
