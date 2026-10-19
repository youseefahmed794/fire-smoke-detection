"""Decision tracing for a local interval, with sequential temporal warm-up."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from src.local_video_demo import detection_label, local_asset, reserve_file, verify_video
from src.detection_status import display_result


def summarize(records):
    labels = Counter()
    reasons = Counter()
    transitions = []
    previous = None
    contaminated = 0
    suppressed = 0
    confidences = []
    scores = []
    for record in records:
        status = detection_label(record["selected_status"], record.get("selected_result"))
        labels[status] += 1
        selected = record["selected_candidate"]
        candidate = record["candidates"][selected] if selected is not None else None
        reason = candidate["rejection_reason"] if candidate else None
        if reason:
            reasons[reason] += 1
        state = (status, reason)
        if state != previous:
            transitions.append({"source_seconds": record["source_seconds"],
                                "status": status, "reason": reason})
            previous = state
        suppressed += bool(record["fake_suppressed_eligible_candidate"])
        contaminated += any(c["roi_contains_prior_annotations"] for c in record["candidates"])
        if reason == "No Flicker":
            confidences.append(candidate["confidence"])
            if candidate["motion_score"] is not None:
                scores.append(candidate["motion_score"])
    representatives = []
    if records:
        targets = [records[0]["source_seconds"], 15, 18, 21, 24, 27, 30, 31,
                   records[-1]["source_seconds"]]
        seen = set()
        for target in targets:
            record = min(records, key=lambda r: abs(r["source_seconds"] - target))
            if record["frame_index"] in seen:
                continue
            seen.add(record["frame_index"])
            representatives.append({"source_seconds": record["source_seconds"],
                                    "selected_status": record["selected_status"],
                                    "assessment": display_result(record.get("selected_result"), record["selected_status"]),
                                    "candidates": [{k: c[k] for k in (
                                        "class", "confidence", "rejection_reason", "motion_score",
                                        "history_size_before", "roi_contains_prior_annotations")}
                                                   for c in record["candidates"]]})
    return {"label_frame_counts": dict(labels), "selected_rejection_counts": dict(reasons),
            "fake_suppressed_eligible_frames": suppressed,
            "prior_annotation_roi_frames": contaminated, "transition_count": len(transitions),
            "transition_examples": transitions[:12], "representatives": representatives,
            "no_flicker_confidence_range": [min(confidences), max(confidences)] if confidences else None,
            "no_flicker_motion_range": [min(scores), max(scores)] if scores else None}


def diagnostic_frame(frame, record):
    """Keep original annotations and add a panel containing measured reasons."""
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    canvas = np.zeros((max(height, 480), width + 640, 3), dtype=frame.dtype)
    canvas[:height, :width] = frame
    lines = [f"Source {record['source_seconds']:.3f}s | frame {record['frame_index']}",
             "Selected: " + display_result(record.get("selected_result"), record["selected_status"])["display_status"],
             "Authenticity: unverified (real/spoof not established)"]
    for index, candidate in enumerate(record["candidates"][:5]):
        lines.append(f"C{index} {candidate['class']} confidence={candidate['confidence']:.4f}")
        if "assessment" in candidate:
            lines.append("  " + candidate["assessment"]["display_label"])
        lines.append("  reason=" + str(candidate["rejection_reason"] or "passes safeguards"))
        score = candidate["motion_score"]
        text = "not measured / warm-up" if score is None else f"{score:.3f} (must exceed {candidate['flicker_threshold']:g})"
        lines.append("  motion=" + text)
        lines.append(f"  history={candidate['history_size_before']}; painted ROI={candidate['roi_contains_prior_annotations']}")
    if len(record["candidates"]) > 5:
        lines.append("Additional candidates are recorded in the JSON report.")
    if not record["candidates"]:
        lines.append("No primary candidates.")
    for index, text in enumerate(lines):
        cv2.putText(canvas, text[:92], (width + 10, 22 + 20 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def diagnose_video(video, model, screen_model, start, end, output_path, report_path, *,
                   deadline=None, should_stop=None, capture_factory=None,
                   detector_factory=None, writer_factory=None):
    """Always process from frame zero; only write records/video for [start, end)."""
    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
        raise ValueError("Diagnostic interval must satisfy 0 <= start < end")
    video = local_asset(video, {".mp4", ".avi", ".mov", ".mkv"})
    model = local_asset(model, {".pt"})
    screen_model = local_asset(screen_model, {".pt"})
    import cv2
    if capture_factory is None:
        capture_factory = lambda path: cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if detector_factory is None:
        from src.services.fire_detection_service import Detector
        detector_factory = Detector
    output_path, report_path = Path(output_path), Path(report_path)
    reserve_file(output_path)
    reserve_file(report_path)
    capture = writer = detector = None
    records = []
    processed = 0
    completed = False
    started = time.monotonic()
    data = {"completed": False, "interval_seconds": [start, end], "frames": records,
            "output_path": str(output_path), "report_path": str(report_path),
            "device": "cpu", "processed_from_frame_zero": True,
            "model_path": str(model), "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "screen_model_sha256": hashlib.sha256(screen_model.read_bytes()).hexdigest()}
    try:
        capture = capture_factory(str(video))
        if not capture.isOpened():
            raise RuntimeError("Diagnostic source could not be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Source FPS is unavailable")
        data["source_fps"] = fps
        detector = detector_factory(model, screen_model_path=screen_model, device="cpu",
                                    raise_errors=True, capture_decisions=True, uncertain_labels=True)
        data["settings"] = {"min_confidence": detector.min_confidence,
                            "smoke_confidence": detector.smoke_confidence,
                            "iou_threshold": detector.iou_threshold,
                            "flicker_threshold": detector.flicker_threshold,
                            "flicker_buffer_size": detector.flicker_buffer.maxlen,
                            "target_height": detector.target_height,
                            "screen_confidence": 0.3}
        while processed / fps < end:
            if (deadline is not None and time.monotonic() >= deadline) or (should_stop and should_stop()):
                raise TimeoutError("Diagnostic interval did not finish within the time limit")
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Source ended before the requested diagnostic interval")
            annotated, status = detector.process_frame(frame)
            timestamp = processed / fps
            if timestamp >= start:
                record = detector.last_decision
                if record is None:
                    raise RuntimeError("Detector did not provide decision diagnostics")
                record["source_seconds"] = round(timestamp, 6)
                record["frame_index"] = processed
                records.append(record)
                panel = diagnostic_frame(annotated, record)
                if writer is None:
                    if writer_factory is None:
                        writer_factory = lambda path, rate, size: cv2.VideoWriter(
                            str(path), cv2.VideoWriter_fourcc(*"mp4v"), rate, size)
                    writer = writer_factory(output_path, fps, (panel.shape[1], panel.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Diagnostic video writer could not be opened")
                writer.write(panel)
            processed += 1
            if processed % 100 == 0:
                data.update(frames_processed_including_warmup=processed, frames_written=len(records))
                report_path.write_text(json.dumps(data))
                print("DIAGNOSTIC_PROGRESS " + json.dumps({"frames_processed": processed,
                      "source_seconds": round(timestamp, 3), "interval_frames": len(records),
                      "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)
        completed = True
    except BaseException as error:
        data["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        try:
            if writer is not None:
                writer.release()
        finally:
            if capture is not None:
                capture.release()
            data.update(completed=completed, frames_processed_including_warmup=processed,
                        frames_written=len(records), elapsed_seconds=round(time.monotonic() - started, 3),
                        summary=summarize(records))
            report_path.write_text(json.dumps(data, indent=2) + "\n")
    if not records:
        raise RuntimeError("No diagnostic interval frames were recorded")
    try:
        data["output_verification"] = verify_video(output_path, len(records))
    except Exception as error:
        data.update(completed=False, error=type(error).__name__ + ": " + str(error))
        report_path.write_text(json.dumps(data, indent=2) + "\n")
        raise
    report_path.write_text(json.dumps(data, indent=2) + "\n")
    return {key: value for key, value in data.items() if key != "frames"}
