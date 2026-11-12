"""QC v0.2 for one run folder: identity (ArcFace + SFace), leak, pose, framing, scene-keep.

  id_arc / id_sf   cosine vs the customer's scan (ArcFace r50 = research licence,
                   SFace = Apache-2.0, the commercially usable candidate)
  leak             ArcFace cosine vs the catalog model (should be near 0)
  pose_err         difference in yaw/pitch/roll proxies vs the catalog shot (0 = same pose)
  scale            face size relative to the catalog shot (1.0 = same framing)
  scene_keep       1 - mean abs pixel difference outside the head region (1.0 = untouched)
"""
import json, math, pathlib, sys
import cv2, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import identity as I

SF = cv2.FaceRecognizerSF.create(str(I.M / "sface.onnx"), "")

def load(path):
    img = cv2.imread(str(path)); s = 640 / max(img.shape[:2])
    if s < 1: img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    p = max(img.shape[:2]) // 4
    pad = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img, pad, p, I._largest_face(pad)

def feats(path):
    img, pad, p, f = load(path)
    if f is None: return None
    pts = f[4:14].reshape(5, 2).astype(np.float32)
    A, _ = cv2.estimateAffinePartial2D(pts, I.ARC_TEMPLATE, method=cv2.LMEDS)
    blob = cv2.dnn.blobFromImage(cv2.warpAffine(pad, A, (112, 112)), 1/127.5, (112, 112), (127.5,)*3, swapRB=True)
    arc = I._sess.run(None, {I._sess.get_inputs()[0].name: blob})[0][0]; arc /= np.linalg.norm(arc)
    sf = SF.feature(SF.alignCrop(pad, f.reshape(1, -1)))[0]; sf = sf / np.linalg.norm(sf)
    le, re, nose = pts[0], pts[1], pts[2]; d = np.linalg.norm(re - le); mid = (le + re) / 2
    pose = np.array([(nose[0]-mid[0])/d, (nose[1]-mid[1])/d, math.atan2(re[1]-le[1], re[0]-le[0])])
    h, w = img.shape[:2]
    box = np.array([f[0]-p, f[1]-p, f[2], f[3]]) / np.array([w, h, w, h])
    return dict(arc=arc, sf=sf, pose=pose, box=box, img=img)

def scene_keep(a, b):
    g = lambda im: cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (256, 256)).astype(np.float32)
    A, B = g(a["img"]), g(b["img"]); mask = np.ones((256, 256), bool)
    for f in (a, b):
        x, y, w, h = f["box"] * 256
        mask[max(0, int(y-0.9*h)):int(y+1.5*h), max(0, int(x-0.6*w)):int(x+1.6*w)] = False
    return float(1 - np.abs(A - B)[mask].mean() / 255) if mask.any() else None

def main(run):
    run = pathlib.Path(run); m = json.loads((run / "manifest.json").read_text())
    cust, scene = feats(m["face"]), feats(m["scene"])
    for r in m["results"]:
        if "file" not in r: continue
        o = feats(run / r["file"])
        if o is None: r["qc"] = None; continue
        q = dict(id_arc=float(o["arc"] @ cust["arc"]), id_sf=float(o["sf"] @ cust["sf"]))
        if not r.get("intermediate"):
            q.update(leak=float(o["arc"] @ scene["arc"]), pose_err=float(np.abs(o["pose"] - scene["pose"]).sum()),
                     scale=float(o["box"][3] / scene["box"][3]), scene_keep=scene_keep(o, scene))
        r["qc"] = {k: (None if v is None else round(v, 3)) for k, v in q.items()}
    (run / "manifest.json").write_text(json.dumps(m, indent=2))
    return m

if __name__ == "__main__":
    for run in sys.argv[1:]:
        m = main(run)
        print(f"\n== {m['cid']} x {m['pid']}")
        for r in m["results"]:
            print(f"  {r['variant']:18}", r.get("qc"))
