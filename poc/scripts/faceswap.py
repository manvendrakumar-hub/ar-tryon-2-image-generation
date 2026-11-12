"""Local open-source face-swap pass (InsightFace inswapper_128, CPU, ~1 s, $0).

Used as an identity 'snap' on top of an image that already has the right scene and
eyewear. The swapper only works at 128 px, so on close-ups the swapped face is soft;
we put the fine skin texture of the original back (frequency separation).

Caveats: inswapper is driven by the ArcFace embedding, so the ArcFace score is
inflated for these outputs (SFace is the independent check). The weights are
research/non-commercial; production needs a licensed or self-trained swapper.

python3 poc/scripts/faceswap.py <customer_scan> <target_image> <out.png>
"""
import pathlib
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import identity as I  # noqa: E402
import qc  # noqa: E402

_P = str(I.M / "inswapper_128.onnx")
_EMAP = numpy_helper.to_array(onnx.load(_P).graph.initializer[-1])
_opt = ort.SessionOptions(); _opt.log_severity_level = 3
_SW = ort.InferenceSession(_P, _opt, providers=["CPUExecutionProvider"])


def _landmarks_fullres(path):
    """5 landmarks in original-image pixels (qc.load detects on a shrunk, padded copy)."""
    full = cv2.imread(str(path))
    small, pad, p, f = qc.load(path)
    if f is None:
        return full, None
    s = small.shape[0] / full.shape[0]
    return full, (f[4:14].reshape(5, 2) - p) / s


def swap(customer, target, out, texture=True):
    src = qc.feats(customer)
    full, pts = _landmarks_fullres(target)
    if src is None or pts is None:
        return False
    latent = src["arc"].reshape(1, -1) @ _EMAP
    latent = (latent / np.linalg.norm(latent)).astype(np.float32)
    A, _ = cv2.estimateAffinePartial2D(pts.astype(np.float32), I.ARC_TEMPLATE * (128 / 112), method=cv2.LMEDS)
    crop = cv2.warpAffine(full, A, (128, 128), borderValue=0)
    blob = cv2.dnn.blobFromImage(crop, 1 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
    pred = _SW.run(None, {"target": blob, "source": latent})[0][0]
    fake = cv2.cvtColor(np.clip(pred.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    h, w = full.shape[:2]
    inv = cv2.invertAffineTransform(A)
    scale = np.sqrt(abs(np.linalg.det(inv[:, :2])))  # output pixels per swapper pixel
    back = cv2.warpAffine(fake, inv, (w, h), flags=cv2.INTER_CUBIC, borderValue=0)
    if texture and scale > 1.5:
        # swapped face carries identity (low/mid frequencies); original carries pores and sharpness
        k = int(scale * 2) | 1
        back = np.clip(back.astype(np.float32) + (full.astype(np.float32) - cv2.GaussianBlur(full, (k, k), 0).astype(np.float32)), 0, 255).astype(np.uint8)
    mask = np.zeros((128, 128), np.float32); mask[10:118, 10:118] = 1
    mask = cv2.warpAffine(mask, inv, (w, h))
    e = max(3, int(12 * scale)) | 1
    mask = cv2.GaussianBlur(cv2.erode(mask, np.ones((e, e), np.uint8)), (2 * e + 1, 2 * e + 1), 0)[..., None]
    cv2.imwrite(str(out), (mask * back + (1 - mask) * full).astype(np.uint8))
    return True


def grain(src, out, amount=5.0, seed=7):
    """Luminance film grain: hides the over-clean 'AI' surface. Cosmetic only."""
    img = cv2.imread(str(src)).astype(np.float32)
    n = np.random.default_rng(seed).normal(0, amount, img.shape[:2]).astype(np.float32)
    n = cv2.GaussianBlur(n, (3, 3), 0) * 1.6
    cv2.imwrite(str(out), np.clip(img + n[..., None], 0, 255).astype(np.uint8))


if __name__ == "__main__":
    print(swap(sys.argv[1], sys.argv[2], sys.argv[3]))
