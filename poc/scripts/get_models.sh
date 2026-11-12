#!/usr/bin/env bash
# Download the local QC / face models into poc/models/ (about 1 GB in total).
#
# Licences, read before any production use:
#   yunet.onnx, sface.onnx (OpenCV zoo)        Apache-2.0 / MIT, fine for commercial use
#   buffalo_l (InsightFace ArcFace)           non-commercial research only
#   inswapper_128.onnx (InsightFace swapper)  non-commercial research only, used only in the round-3 test
set -euo pipefail
cd "$(dirname "$0")/../models" 2>/dev/null || { mkdir -p "$(dirname "$0")/../models"; cd "$(dirname "$0")/../models"; }

fetch() { [ -s "$2" ] && echo "have $2" || { echo "get  $2"; curl -fL --retry 3 -o "$2" "$1"; }; }

fetch https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx yunet.onnx
fetch https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx sface.onnx
fetch https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip buffalo_l.zip
[ -d buffalo_l ] || unzip -q buffalo_l.zip -d buffalo_l

if [ "${WITH_FACESWAP:-0}" = "1" ]; then
  fetch https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx inswapper_128.onnx
fi
echo "done: $(pwd)"
