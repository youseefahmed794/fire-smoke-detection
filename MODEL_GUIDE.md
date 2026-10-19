# Model evaluation guide — recommendations unverified

The verified local demo uses `models_weights/firedetect-11s.pt` on CPU, with
`yolov8n.pt` as an auxiliary screen-object detector. Technical operation on one
sample and synthetic inputs does not establish detection accuracy or prove
real/spoof authenticity. `models_weights/firedetect-11x.pt` is an existing local
alternative; it has not been benchmarked against the default by this guide.

## Historical source references — not validated recommendations

The following landing-page references were already present in this guide.
They are retained for provenance review only. They were not visited or checked
for availability, model identity, license, training data, or redistribution
permission in this work. No claim of "best", "latest", "highest precision", or
improved speed/accuracy is supported by a comparative evaluation here.

- Previously listed YOLOv11 specialized-model reference: [existing Roboflow reference](https://universe.roboflow.com/ds-nslpt/fire-smoke-detection-yolov11).
- Previously listed YOLOv11s FireDetect reference: [existing Hugging Face reference](https://huggingface.co/leeyunjai/yolo11-firedetect).
- Previously listed YOLOv8x fire/smoke reference: [existing GitHub reference](https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection).

These pointers do not verify the origin of the local weight files. No new
sources, download locations, licenses, or benchmark results are proposed.
**Unresolved code licensing/attribution and model/media provenance are public
release blockers.** Do not infer permission to download, use, or redistribute a
model from a link or its architecture name. Weights and media are not in Git.

## Evaluating an already reviewed local model

Keep the original `.env` off-limits. The isolated launcher takes an explicit
`--model` argument and bypasses dotenv; no configuration edit or credentials
are needed. Only use an existing, reviewed local file with the expected Fire /
Smoke class semantics and compatible outputs. Missing files must cause an error,
not a download. Keep the auxiliary local `yolov8n.pt` available as documented
in [README.md](README.md).

This is a **template**, not a command verified for an arbitrary replacement.
Replace the uppercase filename placeholder with your reviewed local file.
The launcher remains CPU-only, headless, and bounded:

```bash
.venv/bin/python -I -B local_video_demo.py \
  --model "models_weights/REPLACE_WITH_REVIEWED_MODEL.pt" \
  --max-frames 10 --timeout 90
```

Model compatibility, class names, confidence calibration, throughput, and
failure cases require separate evaluation. Keep thresholds and heuristic
settings fixed when comparing models, and use a reviewed labeled dataset to
measure precision/recall and false rejections. This guide does not recommend
lowering thresholds or tuning them to the sample clip. New runtime behavior
requires a separately reviewed change, not edits to the original `.env`.

## Interpretation limits

- Fire/Smoke confidence is the model's score, not a probability of authentic fire.
- `Fire candidate - low temporal change` and `Fire candidate - screen overlap`
  remain rejected candidates; neither heuristic proves spoofing. Passing them
  does not prove authenticity either.
- HSV color and associated-ion labels are unvalidated image-color estimates and
  static lookups, not chemical measurements or identification of accelerants.
- Frame counts, a black-frame prediction, or a short successful run do not prove
  accuracy, safety, or improved performance. The user-reported fire rejection
  remains unresolved as an authenticity question.
- The sample advertises 1,883 frames but previously decoded 1,880; partial output
  must not be reported as complete.
- A fresh virtual environment on the same Ubuntu/Python host and existing working
  tree passed all 31 credential-free isolated tests. A clean checkout and real-model
  loading/inference in that virtual environment were not verified. See
  [README.md](README.md) for the exact verification scope and results.

Use only the isolated demo instructions for the verified path. Camera/API/
notification services and Docker/Compose are legacy and untested; Docker
exclusions do not make their execution safe.
