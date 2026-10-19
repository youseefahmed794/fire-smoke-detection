"""Bounded local-file inference; imported only by the isolated launcher."""

from collections import Counter
import json
import math
import os
from pathlib import Path
import time

from src.detection_status import UNCERTAIN_LABELS, display_result


def local_asset(value, suffixes):
    """Reject URLs, devices, missing assets, and unexpected file types."""
    if "://" in str(value) or str(value).isdigit():
        raise ValueError("Only an existing local file is accepted")
    path = Path(value).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in suffixes:
        raise ValueError("Asset has an unsupported file type")
    return path


def detection_label(detection, result=None):
    """Summarize structured assessments, with support for legacy report inputs."""
    return display_result(result, detection)["display_label"]


def reserve_file(path):
    """Reserve a new file atomically, including protection against symlinks."""
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    os.close(descriptor)


def verify_video(path, expected_frames):
    """Decode the whole saved file, including its first and last frames."""
    import cv2
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            raise RuntimeError("Saved video could not be opened")
        metadata_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        decoded = 0
        first = last = False
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError("Saved video contains an undecodable frame")
            decoded += 1
            first = True
            last = decoded == expected_frames
        result = {"opened": True, "first_frame_decoded": first,
                  "last_frame_decoded": last, "decoded_frames": decoded,
                  "metadata_frames": metadata_frames, "expected_frames": expected_frames}
        if not first or not last or decoded != expected_frames or metadata_frames != expected_frames:
            raise RuntimeError("Saved video frame-count verification failed: " + json.dumps(result))
        return result
    finally:
        capture.release()


def run_video(video, model, screen_model, max_frames=10, *, output_path=None,
              report_path=None, deadline=None, should_stop=None,
              capture_factory=None, detector_factory=None, writer_factory=None):
    """Process local frames; None explicitly requests full-video processing."""
    if max_frames is not None and (
            not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1):
        raise ValueError("max_frames must be a positive integer or None for a full video")
    video = local_asset(video, {".mp4", ".avi", ".mov", ".mkv"})
    model = local_asset(model, {".pt"})
    screen_model = local_asset(screen_model, {".pt"})
    # Validate every asset before importing the detector or constructing YOLO.
    import cv2
    if capture_factory is None:
        capture_factory = lambda path: cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if detector_factory is None:
        from src.services.fire_detection_service import Detector
        detector_factory = Detector

    output_path = Path(output_path) if output_path is not None else None
    report_path = Path(report_path) if report_path is not None else None
    if output_path is not None:
        reserve_file(output_path)
    if report_path is not None:
        reserve_file(report_path)
    capture = None
    writer = None
    detector = None
    count = 0
    written = 0
    detections = Counter()
    labels = Counter({label: 0 for label in ("Fire", "Smoke", *UNCERTAIN_LABELS.values(), "No Detection")})
    started = time.monotonic()
    stop_reason = "frame_limit"
    source_frames = 0
    fps = 0.0
    error_text = None
    output_size = None

    def snapshot(completed=False):
        elapsed = time.monotonic() - started
        return {
            "device": "cpu", "frames_processed": count, "frame_limit": max_frames,
            "full_video": max_frames is None, "completed": completed,
            "source_complete": stop_reason == "end_of_video" and (
                source_frames == 0 or count == source_frames),
            "stop_reason": stop_reason, "source_frame_count": source_frames,
            "source_fps": fps, "detections": dict(detections),
            "label_frame_counts": dict(labels), "elapsed_seconds": round(elapsed, 3),
            "throughput_fps": round(count / elapsed, 3) if elapsed else 0.0,
            "output_path": str(output_path) if output_path else None,
            "report_path": str(report_path) if report_path else None,
            "frames_written": written, "notifications": False, "uploads": False,
            "camera": False, "frames_saved": written > 0, "error": error_text,
        }

    def report(completed=False):
        result = snapshot(completed)
        if report_path is not None:
            report_path.write_text(json.dumps(result, indent=2) + "\n")
        return result

    try:
        capture = capture_factory(str(video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the local video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_frames = int(round(source_count)) if math.isfinite(source_count) and source_count > 0 else 0
        if output_path is not None and (not math.isfinite(fps) or fps <= 0):
            raise RuntimeError("Source FPS is unavailable; refusing to invent output timing")
        detector = detector_factory(
            model, screen_model_path=screen_model, device="cpu", raise_errors=True,
            uncertain_labels=True)
        while max_frames is None or count < max_frames:
            if (deadline is not None and time.monotonic() >= deadline) or (
                    should_stop is not None and should_stop()):
                stop_reason = "timeout"
                break
            ok, frame = capture.read()
            if not ok:
                stop_reason = "end_of_video"
                break
            if frame is None or frame.size == 0:
                raise RuntimeError("Video decoder returned an empty frame")
            processed, detection = detector.process_frame(frame)
            if processed is None:
                raise RuntimeError("Detector returned no processed frame")
            count += 1
            result = display_result(getattr(detector, "last_result", None), detection)
            if detection:
                detections[result["display_status"]] += 1
            labels[result["display_label"]] += 1
            if output_path is not None:
                height, width = processed.shape[:2]
                if writer is None:
                    output_size = (width, height)
                    if writer_factory is None:
                        writer_factory = lambda path, rate, size: cv2.VideoWriter(
                            str(path), cv2.VideoWriter_fourcc(*"mp4v"), rate, size)
                    writer = writer_factory(output_path, fps, output_size)
                    if not writer.isOpened():
                        raise RuntimeError("Annotated video writer could not be opened")
                if (width, height) != output_size:
                    raise RuntimeError("Processed frame dimensions changed during recording")
                # Preserve the detector's existing boxes, labels and real confidence values.
                writer.write(processed)
                written += 1
            report()
            if count % 100 == 0:
                print("DEMO_PROGRESS " + json.dumps(snapshot(), sort_keys=True), flush=True)
        if count == 0 and stop_reason != "timeout":
            raise RuntimeError("No video frames could be decoded")
        if max_frames is None and stop_reason == "end_of_video" and source_frames and count != source_frames:
            raise RuntimeError("Video ended before its advertised frame count; output is partial")
    except BaseException as error:
        stop_reason = "error"
        error_text = type(error).__name__ + ": " + str(error)
        raise
    finally:
        try:
            if writer is not None:
                writer.release()
        finally:
            try:
                if capture is not None:
                    capture.release()
            finally:
                detector = None
                report()
    return report(completed=stop_reason != "timeout")
