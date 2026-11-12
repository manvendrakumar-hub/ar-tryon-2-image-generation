"""ArcFace identity similarity (YuNet detect + 5-point align + w600k_r50).

Same building block serves two jobs:
  1. QA: does the generated image still look like the customer?
  2. Cache match: is today's scan the same person as the stored scan?

Note: insightface buffalo_l weights are research/non-commercial. Fine for the POC;
production needs a commercially licensed recogniser.
"""

import pathlib
import sys

import cv2
import numpy as np
import onnxruntime as ort

M = pathlib.Path(__file__).resolve().parents[1] / "models"
ARC_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
_sess = ort.InferenceSession(str(M / "buffalo_l/w600k_r50.onnx"), providers=["CPUExecutionProvider"])


def _largest_face(img):
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(str(M / "yunet.onnx"), "", (w, h), 0.6)
    _, faces = det.detect(img)
    if faces is None:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def embed(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    # YuNet misses faces that fill the frame (catalog close-ups): shrink and pad.
    s = 640 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    p = max(img.shape[:2]) // 4
    img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    f = _largest_face(img)
    if f is None:
        return None
    pts = f[4:14].reshape(5, 2).astype(np.float32)
    M_, _ = cv2.estimateAffinePartial2D(pts, ARC_TEMPLATE, method=cv2.LMEDS)
    crop = cv2.warpAffine(img, M_, (112, 112), borderValue=0)
    blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    v = _sess.run(None, {_sess.get_inputs()[0].name: blob})[0][0]
    return v / np.linalg.norm(v)


def similarity(a, b):
    ea, eb = embed(a), embed(b)
    if ea is None or eb is None:
        return None
    return float(np.dot(ea, eb))


if __name__ == "__main__":
    print(similarity(sys.argv[1], sys.argv[2]))
