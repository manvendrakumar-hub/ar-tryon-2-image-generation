"""Identity reference prep: crop the head (face + hair + a little shoulder) and
upscale so the face is large in the reference. Research finding: a small or
soft face in the identity slot is the main cause of 'close but not the same
person'. Lanczos only: restoration upscalers invent facial detail."""

import pathlib
import sys

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import identity as I  # noqa: E402


def head_crop(src, dst, size=1024, margin=1.0):
    img = cv2.imread(str(src))
    f = I._largest_face(img)
    if f is None:
        raise RuntimeError(f"no face in {src}")
    x, y, w, h = f[:4]
    cx, cy = x + w / 2, y + h / 2 - 0.1 * h  # shift up a little to keep hair
    half = max(w, h) * (0.5 + margin)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half))
    crop = img[y0:y1, x0:x1]
    s = size / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dict(face_px=int(max(w, h)), crop_px=crop.shape[:2])


if __name__ == "__main__":
    print(head_crop(sys.argv[1], sys.argv[2]))
