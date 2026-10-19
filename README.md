# Isolated local CPU fire/smoke video demo

## Verified scope

This launcher runs headlessly on the CPU and prints a JSON summary. The default
is still a bounded 10-frame check with no saved video. Explicit `--full-video`
processes through end of file, and `--save-video` saves the detector's existing
annotated frames to a new, uniquely named local MP4. It does not start cameras or
API servers, create detection databases/logs, send notifications, or upload images.
These runs check technical operation, not detection accuracy.

The existing `.env` is never read or used by this launcher or its focused tests.
Before application imports, the launcher starts a fresh subprocess with an
explicit non-secret environment, bypasses both dotenv loading paths, blocks
dotenv/service imports and `.env` reads, and installs a Linux seccomp filter
that denies networking for Python, native libraries, and subprocesses. It fails
before application imports if these protections cannot be installed. Ultralytics
sync and automatic installation are disabled, and library settings/caches stay
in a temporary directory outside the repository. Model-loading security defaults
are preserved. This isolation applies specifically to `local_video_demo.py`;
the other entry points retain their existing configuration behavior.

### Dependency snapshots and Ubuntu / Python 3.12 setup

The observed environment on 2026-10-19 is **Ubuntu 24.04.5 LTS, Linux x86_64,
Python 3.12.3** in the project `.venv`, with system site packages disabled.
The snapshots below were generated from installed distribution metadata, without
importing third-party packages or application modules. Every version pin is an
observed installed version, not a newly selected compatibility recommendation.

| File | Scope |
| --- | --- |
| `requirements.txt` | Existing legacy application declarations, including API, notification/cloud, and other features. Preserved unchanged; not a frozen or verified demo environment. |
| `requirements-demo-cpu.txt` | Five direct non-PyTorch dependencies for the isolated detector/demo and tests: Ultralytics, cvzone, OpenCV, NumPy, requests. |
| `requirements-demo-cpu-torch.txt` | Separate CPU prerequisites: `torch==2.14.1+cpu`, `torchvision==0.29.1+cpu`. Install using the official PyTorch CPU index. |
| `requirements-demo-cpu-resolved.txt` | All 45 installed distributions in this project environment, including transitive dependencies, pip/setuptools, and optional packages already present. Use as exact-version constraints/reference, not a standalone PyPI install recipe. |

The direct files describe the demo subset; the resolved snapshot records the
entire existing environment. In particular, `cloudinary`, `GPUtil`, and
`python-dotenv` are present in that snapshot but are not needed by the isolated
path: dotenv loading is bypassed and cloud/notification/monitoring services are
disabled. `psutil`, `requests`, and `nvidia-ml-py` are required by the installed
Ultralytics metadata, even though this demo uses CPU inference and disables
external services. The detector unconditionally imports cvzone, OpenCV, NumPy,
and Ultralytics. Tests use Python's built-in `unittest`, not pytest, and mock
requests/uploads rather than exercising external accounts.

Linux and **`libseccomp.so.2`** are required for the launcher to fail closed on
networking before application imports. On Ubuntu 24.04, arrange for Python 3.12,
its `python3.12-venv` support, and the `libseccomp2` runtime to be available first.
OpenCV also needs working native video/FFmpeg runtime libraries. Native library
versions and OS packages are not captured by a Python dependency snapshot.
No CUDA toolkit, GPU driver, camera, or credentials are needed for this CPU demo.

**Fresh-virtual-environment verification passed for the credential-free tests
on the same Ubuntu / Python host and the existing working tree. A clean checkout
has NOT been verified, and real-model inference was NOT rerun in that environment.**
The user reported successfully running both documented dependency installation
steps in a separate temporary environment. The package metadata, `pip check`, and
isolated tests were then independently checked as recorded below. No installation
or network access was performed during that independent verification.

The commands below are the setup procedure for a future environment; they were
not rerun as installation commands during verification. Never overwrite an
existing `.venv`; inspect or reuse it instead. Run from the repository root only
after the OS prerequisites are available and dependency downloading is authorized:

```bash
# Only when .venv is absent in the future clean checkout.
python3.12 -m venv .venv

# FIRST: obtain the exact CPU builds from the official PyTorch CPU wheel index.
.venv/bin/python -I -B -m pip --isolated install \
  --index-url https://download.pytorch.org/whl/cpu \
  -r requirements-demo-cpu-torch.txt -c requirements-demo-cpu-resolved.txt

# THEN: install the demo's non-PyTorch dependencies with observed-version constraints.
.venv/bin/python -I -B -m pip --isolated install \
  --index-url https://pypi.org/simple \
  -r requirements-demo-cpu.txt -c requirements-demo-cpu-resolved.txt

# Local metadata consistency check; no package installation.
.venv/bin/python -I -B -m pip --isolated --disable-pip-version-check --no-cache-dir check

# Isolated tests: model loading and external services are mocked.
.venv/bin/python -I -B local_video_demo.py --run-tests --timeout 90
```

The CPU wheels are sourced separately from the **official PyTorch CPU index**
(`https://download.pytorch.org/whl/cpu`), not ordinary PyPI. Official installation
entry point: https://pytorch.org/get-started/locally/. Neither reference site was
accessed during the independent verification. The user reported successful
installation using the documented indexes. Installed metadata confirms the
`+cpu` versions; download origins and future artifact availability were not
independently checked. The first
step installs them before the PyPI step; the resolved constraints preserve those
exact CPU versions. If any matching artifact is unavailable, stop and investigate
rather than silently choosing different versions or GPU wheels.

The proposed procedure installs the demo subset plus its constrained transitive
dependencies. It does not install every optional package or force pip's version
to match the complete observed environment. That snapshot also includes pip
24.0 and setuptools 78.1.0; setuptools is a dependency of the installed torch.
No package hashes, download archives, or native OS-library lock are provided.
A snapshot and a passing `pip check` do not prove that a clean installation can
reproduce the environment, video decoding, model loading, or previous results.

Offline documentation verification (2026-10-19): the complete isolated demo
suite passed **31 tests in 0.708 seconds**, exit 0, with all model loading mocked.
The metadata consistency command above printed `No broken requirements found.`
and exited 0. All 5 direct pins, 2 CPU pins, and 45 full-environment pins matched
installed metadata; the full snapshot covered every installed distribution in
the project environment. `libseccomp.so.2` loaded successfully, and the tests
verified dotenv bypass and service/network guards. No dependency installation,
new environment, original `.env` access, model loading, inference, or network
access was performed in this milestone. These checks apply to the existing
environment only; the later fresh-virtual-environment check below has a
separate, limited scope.

### Fresh virtual environment: independently verified test scope (2026-10-19)

The user reported that both documented dependency installation steps completed
in the separate temporary virtual environment, `pip check` found no broken
requirements, and 31 isolated tests passed in 0.894 seconds. The independent
verification used that same new interpreter without installation or network
access; the existing project `.venv`, dependencies, and source were preserved.

- Host: Ubuntu 24.04.5 LTS, Linux x86_64, Python 3.12.3; pip 24.0.
- Exact direct versions: torch 2.14.1+cpu, torchvision 0.29.1+cpu,
  Ultralytics 8.4.182, cvzone 2.0.0, opencv-python 5.0.0.93, NumPy 2.5.2,
  and requests 2.34.2.
- All 42 installed distributions match the resolved snapshot; no extra packages
  or version drift were found. The snapshot's optional cloudinary, GPUtil, and
  python-dotenv packages are absent. The credential-free suite passed without them.
- `pip check`: `No broken requirements found.`, exit 0.
- Complete isolated suite: **31 tests passed in 0.807 seconds**, `OK`, exit 0.
- Isolation confirmed a clean subprocess environment, dotenv loading disabled,
  dotenv reads blocked, kernel networking blocked, and services disabled before
  project imports. Model loading and external services were mocked; synthetic
  temporary fixtures were used without real weights or private media.

For an already prepared environment, set `verified_python` to its interpreter
and run these commands from the repository root. The uppercase path below is an
explicit placeholder; replace it before running. These are check commands,
not installation commands:

```bash
verified_python="REPLACE_WITH_FRESH_ENV_INTERPRETER"
"$verified_python" -I -B -m pip --isolated --disable-pip-version-check --no-cache-dir check
"$verified_python" -I -B local_video_demo.py --run-tests --timeout 90
```

This verifies a fresh virtual environment **on the same host and existing working
tree**. It does not verify a clean checkout, a separately provisioned Ubuntu
machine, native-library reproducibility, real-model loading/inference, or video
accuracy in that environment. Docker/Compose, cameras, API services, notifications,
and uploads remain untested. Prior real-video results below came from the
original environment and are not attributed to this new environment. Licensing,
attribution, model/dataset provenance, and redistribution questions remain open.

### Required local assets and isolated execution

Provide these existing local files before inference:

- `test_fire_smoke.mp4` (default sample video)
- `models_weights/firedetect-11s.pt` (default primary model)
- `yolov8n.pt` (auxiliary TV/laptop/cell-phone detector)

`models_weights/firedetect-11x.pt` is an optional alternative, selected with
`--model`. **Model weights and media are not included in Git**; these paths are
ignored and no weight/media files are in the current Git index. Their
**redistribution rights and download sources have not yet been verified**.
A checkout and dependency install alone therefore cannot reproduce inference;
obtain reviewed, legally usable local assets separately. No model downloads are
attempted by the isolated launcher, and missing local assets cause an error.
Do not assume the demo's isolation applies to other entry points.

The original `.env` remains off-limits and is bypassed even if it exists. No new
`.env` is required. The clean subprocess environment contains explicit non-secret
demo settings; dotenv imports/reads, notification/cloud/camera services and
workers, uploads, Python networking, and native network syscalls remain blocked.
Generated settings stay in a temporary directory outside the repository, and
saved media/reports stay local under the verified Git-ignored `demo-output/`.

Known limitations remain: low temporal change can reject a Fire candidate
without proving spoofing, and passing the heuristic does not verify authenticity.
The supplied sample advertises **1,883 frames but decoded only 1,880** in the
previous technical evaluation. A full run is marked partial with exit 1 when
that discrepancy occurs; it is not silently treated as complete. No accuracy or
resolution of the reported false rejection is claimed. These are previously
verified observations; no models or full-video inference were run in this
documentation milestone.

No assets or packages are downloaded automatically. No credentials are needed.
The following execution commands were verified in the existing environment in
previous milestones; they do not constitute a fresh-environment reproduction.
Run these commands from the repository root:

```bash
# Focused tests: mocked model loading, HTTP services, uploads, and video capture.
.venv/bin/python -I -B local_video_demo.py --run-tests --timeout 90

# Bounded smoke run: process the first 10 decoded frames on CPU.
.venv/bin/python -I -B local_video_demo.py --max-frames 10 --timeout 90

# Full local sample, annotated output, CPU, with a 15-minute overall limit.
.venv/bin/python -I -B local_video_demo.py --full-video --save-video --timeout 900
```

The commands print an `ISOLATION` record before application imports. A successful
demo then prints `DEMO_RESULT` with processed/written frame counts, selected-label
frame counts, CPU throughput, elapsed time, completion status, and the output path.
Exit codes are `0` for success, `1` for a runtime/isolation failure,
`2` for invalid CLI arguments, and `124` for the subprocess time limit. A shorter
video stops at end of file; zero decoded frames is an error. Inference errors
fail the demo rather than silently appearing as "no detection."

To select other existing local assets, pass `--video`, `--model`, and
`--screen-model`. For example, use `--model models_weights/firedetect-11x.pt` to
select the larger fire/smoke model. URLs, device indices, missing files, and
unsupported file types are rejected. `--max-frames` must be positive and
`--timeout` must be between 1 and 900 seconds (default: 60 seconds). `--full-video`
and `--max-frames` are mutually exclusive. The timeout includes subprocess startup
and reserves up to five seconds for cleanup; interrupted runs are explicitly
partial. A forced kill can leave an unreadable partial video.

Both MP4 output and the JSON run/progress report live in the Git-ignored
`demo-output/` directory. The launcher checks the ignore rules before creating
media, refuses to overwrite files, and grants the child write access only to its
two explicit output files in addition to its temporary directory. The writer
uses source FPS and the detector's processed dimensions, releases on failure or
completion, and saves no audio. The saved video is decoded in full to verify its
first/last frames and frame count before a successful result is reported.

Annotations retain the detector's actual box confidences. The existing bottom
`Conf`/`IOU` values are configured thresholds, not per-frame confidence or accuracy.
No thresholds, detection logic, or scores are changed or invented for output.
`label_frame_counts` groups the selected per-frame assessment into `Fire`,
`Smoke`, `Fire candidate - low temporal change`,
`Fire candidate - screen overlap`, and `No Detection`. It is not an exhaustive
count of all boxes in the frame. These counts are **not precision, recall, or accuracy**;
there are no ground-truth annotations in this technical check.

To independently verify an existing saved output without loading any models:

```bash
.venv/bin/python -I -B local_video_demo.py \
  --verify-output "demo-output/REPLACE_WITH_YOUR_OUTPUT.mp4" \
  --expected-frames REPLACE_WITH_FRAMES_WRITTEN --timeout 90
```

This is a command template: replace both `REPLACE_WITH_...` placeholders with
the output path and integer `frames_written` from your own run. Previous local
outputs are not included in a checkout. The expected count must be the recorded
`frames_written`, not an assumed full-source count for a partial run.
Missing assets and output-verification failures produce errors.

### Uncertain heuristic assessments (2026-10-19)

The isolated demo separates three kinds of information:

- **Model detection:** Fire/Smoke class and the model's actual confidence.
- **Heuristic assessment:** `No Flicker` displays as
  `Fire candidate - low temporal change`; `Screen` displays as
  `Fire candidate - screen overlap`. Both remain rejected candidates.
- **Authenticity:** unverified for every candidate, including accepted Fire.
  Low temporal change does not prove a spoof; screen overlap does not prove
  that the detected fire is fake. Passing either heuristic does not prove real fire.

Some accepted Fire annotations also contain HSV color and associated-ion labels
from a static lookup table. These are **unvalidated color-based estimates and
lookups, not chemical measurements**, accelerant identification, or verified
fire-type classification. Motion and screen-overlap rules are likewise
unvalidated estimates, not proof of real/spoof authenticity. Box colors and
model confidences do not establish either chemistry or authenticity.

Annotations, diagnostic panels, and current summary counts use these uncertain
labels. Structured assessments record `model_class`, `rejected`,
`rejection_reason`, `reason_code`, `alert_eligible`, and `authenticity` separately
from display text. The historical candidate `eligible` field means confidence
eligibility only; it does not override rejection. `alert_eligible` describes
candidate gates only, before any legacy cooldown/dispatch rules. All notification
services remain disabled in this isolated launcher regardless of these fields.

The detector's legacy returned status (and diagnostic `legacy_selected_status`)
retains its old wording for compatibility with existing notification checks.
Only the isolated paths opt in to the new presentation. Rejected candidates do
not become alert-eligible. Candidate selection, thresholds, motion history,
models, CPU settings, and Fire/Smoke acceptance are unchanged by this milestone.
Earlier videos and historical results below retain their original labels.

```bash
# Process sequentially from frame zero through 16 seconds; save only [14, 16).
.venv/bin/python -I -B local_video_demo.py --diagnose-start 14 --diagnose-end 16 --diagnostic-tag current --save-video --timeout 300

# Complete demo suite, including label, eligibility, selection, and isolation regressions.
.venv/bin/python -I -B local_video_demo.py --run-tests --timeout 90
```

Verified result: all **31 tests passed** in 0.685 seconds (exit 0). The diagnostic
command above exited 0: 480 frames were processed sequentially in 73.494 seconds,
and 60 interval frames (14.014–15.983 source seconds) were saved. The MP4 opened
and fully decoded all 60 frames, including first and last; metadata also reports
60. That recording and its same-stem JSON report are ignored, local-only
artifacts, not files supplied with this repository. Rerun the interval command
above to create a uniquely named output for your own reviewed local assets.

The interval selected Fire on 36 frames and `Fire candidate - low temporal
change` on 24 frames. At 15.015 seconds, the Fire model confidence was 0.805992,
motion was 12.9994, and the displayed status was `Fire candidate - low temporal
change`, with `rejected=true`, `reason_code=low_temporal_change`, and
`alert_eligible=false`. At 14.014 and 15.983 seconds, Fire passed the existing
checks. No screen-overlap rejection occurred in this interval; that label and
its rejection/eligibility behavior were verified with mocked regression cases.
The model hashes, settings, all candidate measurements, selected indices, and
legacy selected statuses matched the same 60 frames of the preceding diagnostic
exactly. These frame counts and confidences do not measure authenticity or accuracy.

This is a presentation correction, not evidence of improved accuracy or a
resolution of the user-reported false rejection. No real/spoof ground truth
has been validated. Shared ROI history, warm-up, low-motion rejections, and
screen-overlap heuristics retain their existing limitations.

### Verified local sample result (2026-10-19)

At the annotated-video milestone, the test command passed all 23 tests (exit 0).
The later investigation expanded the suite to 28 tests; the uncertainty-label
milestone expanded it to 31. The full-video command used
`firedetect-11s.pt` on CPU and wrote 1,880 annotated frames in 302.760 seconds,
averaging 6.210 processed frames/second including model initialization and writing.
It **exited 1 and is marked partial**, because the source reports 1,883 frames but
OpenCV reached decoder EOF after 1,880. It did not reach the 900-second timeout.
No source frames were deliberately skipped.

The selected per-frame status counts were Fire: 628; Smoke: 0; Fake Fire: 540;
No Detection: 712. These are the detector's existing labels, including its
anti-spoofing heuristics, and are not ground-truth assessments or accuracy metrics.

The historical recording and its same-stem JSON report are local-only and
ignored; they are not supplied in a checkout. Independent verification of that
recording exited 0: its metadata count and
fully decoded count were both 1,880, and its first and last frames decoded.
All frames actually written are present; this does not resolve the source's
three-frame discrepancy or establish completeness against its advertised count.

This independent source-only diagnostic, which loads no models, also decoded
1,880 frames and exited 1 against the advertised count of 1,883:

```bash
.venv/bin/python -I -B local_video_demo.py --verify-output test_fire_smoke.mp4 --expected-frames 1883 --timeout 90
```

The cause of the source metadata/decoder discrepancy is unconfirmed. The source
video has not been repaired or modified. A reviewed replacement sample or a
separate investigation would be needed to assert full advertised-frame coverage.

### Fake Fire investigation and focused correction (2026-10-19)

A user reported visible real fire being labeled `Fake Fire` around seconds
15–31. This is a user observation, not validated ground truth. The investigation
used `firedetect-11s.pt` on CPU with identical model-file hashes and settings in
both runs. Every frame from the beginning through 34 seconds was processed, so
the temporal state was warmed up without seeking. Only `[12, 34)` was saved:
659 frames, source timestamps 12.012–33.967 seconds, after 1,019 processed frames
including warm-up. Timestamps are frame indices divided by source FPS.

The diagnostic mode preserves the existing isolation, checks output ignores,
saves unique MP4/JSON files, and adds a decision panel beside the original frame.
The JSON records every interval candidate's actual confidence, box, rejection
reason, measured motion, history, and selected status. Console output summarizes
transitions and representative times instead of printing every frame.

```bash
# Current corrected detector; bounded to ten minutes including warm-up/cleanup.
.venv/bin/python -I -B local_video_demo.py --diagnose-start 12 --diagnose-end 34 --diagnostic-tag after --save-video --timeout 600

# Complete suite, including regression and isolation tests.
.venv/bin/python -I -B local_video_demo.py --run-tests --timeout 90
```

The original rules sort primary candidates by descending confidence. A fire box
is rejected as `Screen` when more than half its area overlaps an auxiliary TV,
laptop, or cell-phone box (auxiliary confidence threshold 0.3). Otherwise its
grayscale ROI is resized to 100x100 and compared with the last ten fire ROIs.
The mean absolute pixel difference must be **greater than 15**; otherwise the
reason is `No Flicker`. The first two ROIs are accepted as warm-up. History is
shared across fire candidates and persists through frames without fire; it is
not object tracking or a fixed-duration temporal window.

Two code defects were reproduced with synthetic regression cases and corrected:

1. A highest-confidence rejected fire candidate previously prevented a lower
   confidence candidate that passed the existing checks from becoming the frame
   status. Selection now uses the highest-confidence passing candidate, with the
   highest-confidence rejected candidate retained only when none pass. Rejected
   boxes keep their original rejection labels.
2. Drawing each candidate before measuring later overlapping ROIs contaminated
   those measurements with colored box overlays. Measurements now use an
   unannotated frame copy. This defect was reproduced in a synthetic overlapping
   case; no contaminated fire ROIs occurred in the measured sample interval.

No confidence, overlap, flicker, or other threshold was changed. The screen and
low-motion safeguards remain active, including rejection of stationary ROIs.
The regressions produced three failures before correction (Fire/Smoke suppression
and ROI contamination); all **28 tests passed after correction**, exit 0.

| Selected frame status, seconds 12–34 | Before | After |
| --- | ---: | ---: |
| Fire | 357 | 357 |
| Smoke | 0 | 105 |
| Fake Fire | 275 | 170 |
| No Detection | 27 | 27 |

All candidate records—including confidences, measured motion, history, and
rejection reasons—were identical between runs. Exactly 105 overall statuses
changed from `Fake Fire` to `Smoke` at source times 29.196–33.967 seconds, because
eligible Smoke candidates were no longer suppressed. **All 275 individual Fire
candidate `No Flicker` rejections remained unchanged.** The corrected selection
does not establish those fire candidates as authentic or resolve the reported
fire-authenticity concern.

Representative evidence (confidence is the model's class confidence):

| Source time | Candidates / measured motion | Before → after selected status |
| --- | --- | --- |
| 15.015 s | Fire 0.806; motion 12.999, below 15 | Fake Fire → Fake Fire |
| 17.985 s | Fire 0.733; motion 27.304 | Fire → Fire |
| 20.988 s | Fire 0.732; motion 22.251 | Fire → Fire |
| 26.994 s | Fire 0.780; motion 17.965 | Fire → Fire |
| 29.997 s | Fire 0.883, motion 5.028; Smoke 0.526 | Fake Fire → Smoke |
| 30.998 s | Fire 0.851; motion 8.321 | Fake Fire → Fake Fire |

All observed rejections were `No Flicker`, with no screen-overlap rejections.
Baseline selected statuses changed 53 times; corrected statuses changed 72 times,
including Smoke/Fake Fire switches. This remains an unstable heuristic, not a
validated authenticity classifier. Both runs completed (exit 0) in 160.605 and
162.455 seconds respectively; both saved videos opened and decoded all 659
frames, including their first and last frames.

The before/after videos, JSON traces, and comparison report were local-only
artifacts under ignored `demo-output/`; none are included in this repository.
The interval command above generates a new trace of the current detector.
Recreating the historical comparison would additionally require the corresponding
historical code and reviewed local assets; current code alone does not regenerate
the old behavior.

Remaining options require a separately reviewed behavior change: validate on
labeled real-fire and spoof clips, evaluate per-object/time-based temporal
history and ROI alignment, or review the alert policy for uncertain evidence.
The presentation milestone above does not change that policy. No thresholds were tuned to this
clip. Candidate counts and this before/after comparison are not precision,
recall, accuracy, or proof that visible fire is real or fake.

## Legacy application reference — unverified and outside the isolated demo

Camera, API, notifications, uploads, long-running monitoring, and Docker/Compose
operation have **not been validated by the isolated demo tests**. This section
describes intended legacy interfaces, not verified setup instructions or a
recommendation to start them. They can load configuration, create databases/logs,
start workers, connect to cameras, and contact external services. They do not
inherit the isolated launcher's guards.

| Legacy interface | Intended behavior; unverified |
| --- | --- |
| `main.py --mode video --source "path/to/reviewed-local-video.mp4"` | Legacy video processing; can initialize external-service infrastructure. The path is a placeholder. |
| `main.py --mode camera` | Camera processing using legacy configuration; not tested here. |
| `main.py --logs` / `main.py` | Database history / interactive CLI; not tested here. |
| `server.py` and `src/api/app.py` | API, video feed, and status routes backed by camera-system infrastructure; startup and service behavior are unverified. |
| `src/services/notification_service.py` | Intended Telegram/WhatsApp delivery and cloud uploads; delivery, security, and account configuration are unverified. |
| `src/performance_monitor.py` | Intended CPU/RAM/GPU reporting; logging metrics does not establish stability or prevent overheating/memory leaks. |

The original `.env` must remain untouched and must not be reused. The isolated
demo needs no credentials. Future legacy-service work requires a separate
configuration and security review, fresh independently supplied credentials,
explicit authorization for external connections, and mocked-service tests first.
The sanitized `.env.example` documents variable names only; it is not a working
account configuration. Do not copy old credentials into a new setup.
`requirements.txt` remains the legacy dependency declaration; the CPU-demo
snapshots above describe the verified local environment instead.

### Docker and Compose — legacy and untested

`Dockerfile` and `docker-compose.yml` remain **legacy and untested**. They use a
Python 3.10 base and legacy requirements, rather than the documented Ubuntu /
Python 3.12 demo environment and separate CPU-wheel procedure. The default
command starts `server.py`; Compose includes an `.env` mount and camera/service
infrastructure. Do not use these as an isolated-demo launcher.

`.dockerignore` excludes environment/credential files, virtual environments,
databases, logs, weights, media, generated outputs, and backups from the build
context, while retaining the sanitized root `.env.example`. These exclusions
reduce accidental packaging; **they do not make Docker execution safe**, disable
service behavior, validate mounts, or prove that an image builds or runs. No
Docker build, Compose startup, or service execution was performed for this
cleanup. Docker/Compose need a separate review before use or distribution.

## Release blockers: licensing, attribution, and asset provenance

No project license has been assigned by this work. Code ownership, required
attribution, third-party dependency obligations, and permission to publish must
be reviewed before public release. No compliance conclusion is implied.
Installed dependency metadata includes an AGPL-3.0 notice for Ultralytics;
that observation does not determine the license of this repository.

Model weights and media remain excluded from Git. Their source correspondence,
licenses, and redistribution rights are unverified; do not bundle or redistribute
them on the strength of this README. The historical references in
[MODEL_GUIDE.md](MODEL_GUIDE.md) are unverified pointers, not endorsements or
proof of permission. Clean-checkout and native-host reproduction, real-model
inference in the fresh environment, and Docker/Compose testing remain incomplete. Technical smoke checks are not a detection-accuracy,
fire-authenticity, chemistry, or safety certification.
