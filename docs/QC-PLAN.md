# QC Plan: automatic quality control for generated try-on images

Status: v0.1 (2026-11-11). QC v0.2 is running in `poc/scripts/qc.py`; the rest is planned.

## 1. What QC has to decide

For every generated image, in under ~100 ms, with no human in the loop:

| Decision | Meaning |
|---|---|
| **Pass** | Show it to the customer and cache it. |
| **Retry** | Regenerate once (same model, new attempt). |
| **Fallback** | Send to the stronger, more expensive model. |
| **Reject input** | The scan itself is unusable; ask for a new photo. |

Two errors matter. Showing a customer a face that is not theirs is the expensive one. Rejecting good images only costs money.

## 2. The six checks

| # | Check | Question | How | Status |
|---|---|---|---|---|
| G0 | Scan gate | Is the customer photo usable? | Face found, one face, face size, blur, exposure, pose, eyes open, already wearing glasses | planned (labels already exist, see §5a) |
| G1 | Identity | Is it the customer? | Face-embedding cosine: output vs scan | **built** |
| G2 | Leak | Is the catalog model's face still there? | Face-embedding cosine: output vs catalog model | **built** |
| G3 | Product | Is it exactly this PID? | Segment the eyewear, compare to the product photo (shape, colour, lens tint, logo) | planned |
| G4 | Scene | Same pose, framing and background as the catalog shot? | Landmark pose difference, face-size ratio, pixel difference outside the head | **built** |
| G5 | Artifacts | Does it look like a real photo? | Double glasses, warped lenses, extra people, waxy skin: VLM checklist, later a small classifier | planned |

## 3. What round 2 taught us about the current QC

Measured on 40 scored images (4 pairs × 10 variants):

- **G2 leak works.** Seedream 4.5 left the catalog model's face in 3 of 4 images. Leak scored 0.38–0.58 on those and ≤ 0.14 on every correct swap.
- **G4 framing works.** One "refine" output returned the customer's store photo instead of the catalog shot. Face-size ratio was 0.48 (normal: 0.88–1.08), so it was caught. One other output re-composed the scene (ratio 1.61) and was caught the same way.
- **A commercially usable recogniser is viable.** ArcFace r50 (`buffalo_l`) is research-licence only. SFace from the OpenCV zoo is Apache-2.0. The two correlate at 0.85 across the 40 images, and they rank the variants in almost the same order.
- **Detection needed a fix.** YuNet missed faces that fill the frame (catalog close-ups). Shrinking and padding the image before detection fixed it.
- **We do not yet know the pass threshold.** See §4.

### Round 3 addition: a face-match score alone can be fooled

A local open-source face swapper (inswapper) was run on the raw catalog shots. It scored 0.60–0.72 on ArcFace, higher than the frontier models. The images are wrong:
- the catalog model's hair, skin tone and head shape are still there, so it does not look like the customer;
- on close-ups the swapper redraws through the lenses and damages the frame.

The swapper is driven by an ArcFace embedding, so it optimises the very number G1 measures. What this changes in the plan:
- G1 must use a recogniser that the generator was not built on. SFace scored the same images lower (0.52–0.64), but not low enough to reject them, so a second signal is needed.
- Add **G1b, head match**: skin tone of the face and neck region, and hair colour and outline, compared with the scan. These are what the face swap left unchanged.
- G3 (product check) is mandatory before any pass decision, not a later nice-to-have: only it catches the damaged frame.

## 4. The main gap: calibration

Identity scores so far:

| Comparison | ArcFace cosine |
|---|---|
| Different people | −0.09 to 0.16 |
| Customer's own photo with glasses added (same photo, edited) | 0.68–0.81 |
| Best swaps into the catalog shot | 0.53–0.70 |
| A **real** second photo of the same person, different pose, wearing glasses | **unknown** |

The last row is the number that sets the threshold. A real second photo will not score 1.0: pose, lighting and glasses all lower it. If real pairs score ~0.55, then today's best swaps are already at the ceiling. If they score ~0.75, there is a real gap.

**What is needed:** the power-customer set (customers with several scans over time), or 10–20 staff photographed twice, once wearing glasses and once at the catalog angle. Either gives the same-person distribution, and from it:
- T_pass (identity high enough to show),
- T_same (same person, for the cache check),
- separate thresholds for eyeglasses and sunglasses, and for close-up and half-body shots.

## 5. Building our own, piece by piece

### 5a. Scan gate (G0): train it on labels we already own
`data/customer_data_merged_with_product.csv` has 73,741 scans already labelled `image_quality` (GOOD/POOR) and `quality_issues` (occluded, blurry, face_small, backlit_uneven, too_dark, overexposed), plus an audit column.
- Train a small classifier on frozen image embeddings (DINOv2-small or SigLIP) for the six issue labels.
- Runs in a few ms. Output: usable / fixable (crop, relight) / reject with a reason the store staff can act on.
- This is the cheapest part to own and likely the biggest quality lever, since a weak identity photo is the main cause of drift.

### 5b. Identity and leak (G1, G2): optimise what exists
- One detection pass per image, embeddings batched on GPU: target < 15 ms per image.
- Store one embedding per customer per scan. With power customers, average several scans into one reference; that is more stable than a single photo.
- Occlusion-aware scoring for sunglasses: compare against a reference that is also wearing the glasses (the step-1 try-on image), so both sides are occluded the same way.
- Recogniser choice for production: SFace (Apache-2.0) now; evaluate a paid InsightFace licence if its accuracy is needed. Training a recogniser from scratch is not realistic: it needs millions of identities.

### 5c. Product fidelity (G3): the part nobody sells
1. Segment the eyewear in the output with a text-prompted segmenter ("eyeglasses").
2. Compare against the catalog shot's own eyewear region (same pose, so it is a like-for-like comparison) and against the product photo:
   - shape: mask overlap after landmark alignment,
   - colour: mean colour difference of the frame pixels (ΔE),
   - lens: tint and opacity,
   - appearance: DINOv2 embedding similarity of the eyewear crop.
3. A VLM checklist for what metrics miss: logo text, hinge and bridge type, added patterns.
4. Later: distil the VLM answers plus human labels into a small classifier so the VLM is out of the hot path.

### 5d. Golden set and human labels
- 300 generated images rated by 2–3 people: identity (1–5), product (1–5), realism (1–5), ship / don't ship.
- Every automatic score is fitted against these ratings. Without this, thresholds are guesses.
- Reuse the same set as a regression test whenever the model, prompt or LoRA changes.

### 5e. Decision logic
```
G0 fail                      → reject input, ask for a new scan
G2 leak high or G4 scale off → retry once (the model did the wrong job)
G1 below T_pass              → fallback model
G3 or G5 fail                → retry once, then fallback
all pass                     → show + cache, store scores with the image
```
Every score is logged with the image, so thresholds can be re-fitted later without regenerating.

## 6. On using /reverse-engineer-anything

That skill takes apart shipped apps and binaries. It does not help build the checks above, which are models and thresholds we train and fit ourselves.

It could help in one place: if nobody has the source for how the existing face-scan app produces `image_quality` and the facial measurements, we could inspect the shipped app to learn it. Asking the team that owns it is faster and cleaner, so I have not run it.

## 7. Order of work

| Step | Output | Needs |
|---|---|---|
| 1 | Product check (G3) v1: segmentation + colour/shape + DINO | nothing new |
| 2 | Scan gate (G0) classifier from existing labels | nothing new |
| 3 | Calibration of identity thresholds | power-customer or staff photo pairs |
| 4 | Golden set of 300 rated images | 2–3 raters, ~3 hours |
| 5 | Decision logic + speed pass (GPU batch) | steps 1–4 |
