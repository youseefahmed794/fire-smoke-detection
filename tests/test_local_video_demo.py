"""Focused tests, run exclusively through the isolated demo launcher."""

import ctypes
import errno
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import cv2
import numpy as np

import local_video_demo as launcher
from src.local_video_demo import local_asset, run_video, verify_video


class LocalVideoDemoTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.directory.cleanup)
        self.directory = Path(self.directory.name)
        self.video = self.directory / "fixture.mp4"
        self.model = self.directory / "fixture.pt"
        self.screen = self.directory / "screen.pt"
        for path in (self.video, self.model, self.screen):
            path.write_bytes(b"nonfunctional local test fixture")
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.capture = mock.Mock()
        self.capture.isOpened.return_value = True
        self.capture.read.return_value = (True, self.frame)
        self.capture.get.side_effect = lambda field: {
            cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_COUNT: 0.0}.get(field, 0.0)
        self.detector = mock.Mock()
        self.detector.process_frame.side_effect = lambda frame: (frame, None)
        self.factory = mock.Mock(return_value=self.detector)
        self.capture_factory = mock.Mock(return_value=self.capture)
        self.http = self.enterContext(mock.patch(
            "requests.sessions.Session.request",
            side_effect=AssertionError("HTTP services must never run")))
        self.upload = mock.Mock(side_effect=AssertionError("Uploads must never run"))
        cloud = types.ModuleType("cloudinary")
        uploader = types.ModuleType("cloudinary.uploader")
        uploader.upload = self.upload
        cloud.uploader = uploader
        self.enterContext(mock.patch.dict(sys.modules, {
            "cloudinary": cloud, "cloudinary.uploader": uploader}))
        self.addCleanup(self.http.assert_not_called)
        self.addCleanup(self.upload.assert_not_called)

    def run_fixture(self, **kwargs):
        return run_video(self.video, self.model, self.screen,
                         capture_factory=self.capture_factory,
                         detector_factory=self.factory, **kwargs)

    def test_exact_frame_limit_and_cpu_with_services_mocked(self):
        result = self.run_fixture(max_frames=10)
        self.assertEqual(result["frames_processed"], 10)
        self.assertEqual(self.capture.read.call_count, 10)
        self.factory.assert_called_once_with(
            self.model, screen_model_path=self.screen, device="cpu", raise_errors=True,
            uncertain_labels=True)
        self.capture.release.assert_called_once()
        for flag in ("notifications", "uploads", "camera", "frames_saved"):
            self.assertFalse(result[flag])

    def test_end_of_file_stops_without_reading_extra_frames(self):
        self.capture.read.side_effect = [(True, self.frame), (False, None)]
        self.assertEqual(self.run_fixture(max_frames=10)["frames_processed"], 1)
        self.capture.release.assert_called_once()

    def test_decoder_failure_is_an_error_and_releases_capture(self):
        self.capture.read.return_value = (False, None)
        with self.assertRaisesRegex(RuntimeError, "No video frames"):
            self.run_fixture()
        self.capture.release.assert_called_once()

    def test_inference_failure_propagates_and_releases_capture(self):
        self.detector.process_frame.side_effect = RuntimeError("synthetic inference failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            self.run_fixture()
        self.capture.release.assert_called_once()

    def test_missing_weights_stop_before_detector_or_capture_creation(self):
        self.model.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_fixture()
        self.factory.assert_not_called()
        self.capture_factory.assert_not_called()

    def test_urls_devices_and_invalid_frame_limits_are_rejected(self):
        for value in ("rtsp://invalid.example/stream", "https://invalid.example/video.mp4", "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                local_asset(value, {".mp4"})
        for limit in (0, -1, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.run_fixture(max_frames=limit)
        self.factory.assert_not_called()

    def test_environment_is_an_explicit_allowlist(self):
        clean = launcher.clean_environment(self.directory)
        for name in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "TEXTMEBOT_API_KEY",
                     "CLOUDINARY_URL", "CLOUDINARY_API_SECRET", "USERNAME", "PASSWORD",
                     "IP", "RTSP_URL", "HOME", "PYTHONPATH"):
            self.assertNotIn(name, clean)
        self.assertEqual(clean["FIRE_DEMO_ISOLATED"], "1")
        self.assertEqual(clean["PYTHON_DOTENV_DISABLED"], "1")
        self.assertEqual(clean["YOLO_OFFLINE"], "true")
        self.assertEqual(clean["YOLO_AUTOINSTALL"], "false")

    def test_configuration_skips_all_dotenv_calls_and_file_probes(self):
        fake = types.ModuleType("dotenv")
        fake.load_dotenv = mock.Mock(side_effect=AssertionError("dotenv is forbidden"))
        fake.find_dotenv = mock.Mock(side_effect=AssertionError("dotenv discovery is forbidden"))
        fake.dotenv_values = mock.Mock(side_effect=AssertionError("dotenv reads are forbidden"))
        spec = importlib.util.spec_from_file_location("isolated_config_test", launcher.ROOT / "src/config.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"dotenv": fake}), mock.patch.object(
                Path, "exists", side_effect=AssertionError("Configuration must not probe files")):
            spec.loader.exec_module(module)
        fake.load_dotenv.assert_not_called()
        fake.find_dotenv.assert_not_called()
        fake.dotenv_values.assert_not_called()
        self.assertIsNone(module.Config.TELEGRAM_TOKEN)
        self.assertIsNone(module.Config.CLOUDINARY_API_SECRET)
        self.assertIsNone(module.Config.USERNAME)

    def test_dotenv_read_guard_uses_only_a_nonexistent_temporary_sentinel(self):
        with self.assertRaisesRegex(PermissionError, "blocks .env"):
            (self.directory / ".env").read_bytes()

    def test_service_imports_and_python_network_events_are_blocked(self):
        for name in ("src.services.notification_service", "src.services.camera_system",
                     "src.background_workers", "src.api.app", "dotenv"):
            with self.subTest(module=name), self.assertRaises(ImportError):
                importlib.import_module(name)
        with self.assertRaisesRegex(PermissionError, "Networking disabled"):
            sys.audit("socket.connect", None, None)

    def test_native_network_syscalls_are_blocked_without_any_connection_attempt(self):
        libc = ctypes.CDLL(None, use_errno=True)
        ctypes.set_errno(0)
        self.assertEqual(libc.connect(-1, None, 0), -1)
        self.assertEqual(ctypes.get_errno(), errno.EPERM)
        ctypes.set_errno(0)
        self.assertEqual(libc.sendto(-1, None, 0, 0, None, 0), -1)
        self.assertEqual(ctypes.get_errno(), errno.EPERM)

    def test_detector_passes_cpu_to_both_models_and_surfaces_errors(self):
        from src.services import fire_detection_service as service
        primary, screen = mock.Mock(), mock.Mock()
        primary.model.names = {}
        primary.return_value = []
        screen.return_value = []
        with mock.patch.object(service, "YOLO", side_effect=[primary, screen]) as load:
            detector = service.Detector(self.model, screen_model_path=self.screen,
                                        device="cpu", raise_errors=True)
        self.assertEqual(load.call_args_list, [mock.call(str(self.model)), mock.call(str(self.screen))])
        detector.process_frame(self.frame)
        self.assertEqual(primary.call_args.kwargs["device"], "cpu")
        self.assertEqual(screen.call_args.kwargs["device"], "cpu")
        primary.side_effect = RuntimeError("mock model failure")
        with self.assertLogs(detector.logger, level="ERROR"), self.assertRaisesRegex(
                RuntimeError, "mock model failure"):
            detector.process_frame(self.frame)

    def test_real_video_output_is_created_written_and_decoded(self):
        path = self.directory / "annotated.mp4"
        result = self.run_fixture(max_frames=3, output_path=path)
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 0)
        self.assertEqual(result["frames_written"], 3)
        self.assertTrue(result["frames_saved"])
        self.assertTrue(result["completed"])
        verified = verify_video(path, 3)
        self.assertTrue(verified["first_frame_decoded"])
        self.assertTrue(verified["last_frame_decoded"])
        self.assertEqual(verified["decoded_frames"], 3)
        with self.assertRaisesRegex(RuntimeError, "frame-count verification failed"):
            verify_video(path, 4)

    def test_writer_receives_existing_annotated_frames_and_source_fps(self):
        annotated = np.ones_like(self.frame)
        self.detector.process_frame.side_effect = lambda frame: (annotated, "Smoke (Rank 1)")
        writer = mock.Mock()
        factory = mock.Mock(return_value=writer)
        output = self.directory / "annotated.mp4"
        result = self.run_fixture(max_frames=2, output_path=output, writer_factory=factory)
        factory.assert_called_once_with(output, 30.0, (64, 48))
        self.assertEqual(writer.write.call_count, 2)
        self.assertIs(writer.write.call_args.args[0], annotated)
        writer.release.assert_called_once()
        self.assertEqual(result["label_frame_counts"]["Smoke"], 2)

    def test_writer_open_failure_releases_writer_and_capture(self):
        writer = mock.Mock()
        writer.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "writer could not be opened"):
            self.run_fixture(output_path=self.directory / "failed.mp4",
                             writer_factory=mock.Mock(return_value=writer))
        writer.write.assert_not_called()
        writer.release.assert_called_once()
        self.capture.release.assert_called_once()

    def test_writer_failure_records_partial_status_and_releases_resources(self):
        writer = mock.Mock()
        writer.write.side_effect = OSError("synthetic writer failure")
        report = self.directory / "partial.json"
        with self.assertRaisesRegex(OSError, "synthetic writer failure"):
            self.run_fixture(output_path=self.directory / "partial.mp4", report_path=report,
                             writer_factory=mock.Mock(return_value=writer))
        writer.release.assert_called_once()
        self.capture.release.assert_called_once()
        import json
        result = json.loads(report.read_text())
        self.assertFalse(result["completed"])
        self.assertEqual(result["stop_reason"], "error")
        self.assertEqual(result["frames_written"], 0)

    def test_later_inference_failure_still_finalizes_existing_writer(self):
        self.detector.process_frame.side_effect = [
            (self.frame, None), RuntimeError("later inference failure")]
        writer = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "later inference failure"):
            self.run_fixture(output_path=self.directory / "partial.mp4",
                             writer_factory=mock.Mock(return_value=writer))
        writer.write.assert_called_once()
        writer.release.assert_called_once()
        self.capture.release.assert_called_once()

    def test_existing_output_is_never_overwritten(self):
        output = self.directory / "existing.mp4"
        output.write_bytes(b"keep this existing file")
        with self.assertRaises(FileExistsError):
            self.run_fixture(output_path=output)
        self.assertEqual(output.read_bytes(), b"keep this existing file")
        self.capture_factory.assert_not_called()
        self.factory.assert_not_called()

    def test_explicit_full_mode_reaches_eof_and_counts_selected_labels(self):
        self.capture.read.side_effect = [(True, self.frame)] * 4 + [(False, None)]
        self.capture.get.side_effect = lambda field: {
            cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_COUNT: 4.0}.get(field, 0.0)
        self.detector.process_frame.side_effect = [
            (self.frame, "Fire (Rank 1) [Color: Yellow]"),
            (self.frame, "Smoke (Rank 2)"),
            (self.frame, "Fake Fire (No Flicker)"), (self.frame, None)]
        result = self.run_fixture(max_frames=None)
        self.assertTrue(result["full_video"])
        self.assertTrue(result["source_complete"])
        self.assertEqual(result["frames_processed"], 4)
        self.assertEqual(result["label_frame_counts"], {
            "Fire": 1, "Smoke": 1, "Fire candidate - low temporal change": 1,
            "Fire candidate - screen overlap": 0, "No Detection": 1})

    def test_early_decoder_end_does_not_claim_a_complete_full_video(self):
        self.capture.read.side_effect = [(True, self.frame), (False, None)]
        self.capture.get.side_effect = lambda field: {
            cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_COUNT: 5.0}.get(field, 0.0)
        with self.assertRaisesRegex(RuntimeError, "output is partial"):
            self.run_fixture(max_frames=None)
        self.capture.release.assert_called_once()

    def test_timeout_is_partial_and_releases_capture(self):
        result = self.run_fixture(max_frames=None, deadline=0)
        self.assertFalse(result["completed"])
        self.assertFalse(result["source_complete"])
        self.assertEqual(result["stop_reason"], "timeout")
        self.assertEqual(result["frames_processed"], 0)
        self.capture.release.assert_called_once()

    def test_safe_cli_default_and_explicit_fifteen_minute_full_mode(self):
        defaults = launcher.arguments([])
        self.assertEqual(defaults.max_frames, 10)
        self.assertFalse(defaults.full_video)
        self.assertFalse(defaults.save_video)
        full = launcher.arguments(["--full-video", "--save-video", "--timeout", "900"])
        self.assertIsNone(full.max_frames)
        self.assertTrue(full.full_video)
        self.assertEqual(full.timeout, 900)

    def test_output_ignore_failure_stops_before_creating_directory_or_media(self):
        import subprocess
        with mock.patch.object(launcher, "ROOT", self.directory), mock.patch.object(
                launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "")):
            with self.assertRaisesRegex(RuntimeError, "must be Git-ignored"):
                launcher.prepare_output()
        self.assertFalse((self.directory / "demo-output").exists())

    def test_diagnostics_warm_up_sequentially_and_save_only_requested_interval(self):
        from collections import deque
        from src.demo_diagnostics import diagnose_video
        self.capture.get.side_effect = lambda field: {
            cv2.CAP_PROP_FPS: 2.0, cv2.CAP_PROP_FRAME_COUNT: 4.0}.get(field, 0.0)
        for name, value in {"min_confidence": 0.4, "smoke_confidence": 0.4,
                            "iou_threshold": 0.2, "flicker_threshold": 15.0,
                            "target_height": 480, "flicker_buffer": deque(maxlen=10)}.items():
            setattr(self.detector, name, value)

        def process(frame):
            self.detector.last_decision = {"candidates": [], "selected_candidate": None,
                "selected_status": None, "fake_suppressed_eligible_candidate": False}
            return frame, None

        self.detector.process_frame.side_effect = process
        result = diagnose_video(self.video, self.model, self.screen, 1, 2,
                                self.directory / "diagnostic.mp4", self.directory / "diagnostic.json",
                                capture_factory=self.capture_factory, detector_factory=self.factory)
        self.assertEqual(result["frames_processed_including_warmup"], 4)
        self.assertEqual(result["frames_written"], 2)
        self.assertEqual(result["output_verification"]["decoded_frames"], 2)
        self.capture.set.assert_not_called()
        self.capture.release.assert_called_once()
        self.assertEqual(self.detector.process_frame.call_count, 4)
        self.assertTrue(self.factory.call_args.kwargs["capture_decisions"])
        self.assertTrue(self.factory.call_args.kwargs["uncertain_labels"])
        failed_report = self.directory / "verification-failure.json"
        with mock.patch("src.demo_diagnostics.verify_video",
                        side_effect=RuntimeError("synthetic verification failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic verification failure"):
                diagnose_video(self.video, self.model, self.screen, 1, 2,
                               self.directory / "verification-failure.mp4", failed_report,
                               capture_factory=self.capture_factory, detector_factory=self.factory)
        import json
        self.assertFalse(json.loads(failed_report.read_text())["completed"])

    def candidate_detector(self, rows, *, uncertain_labels=False):
        from src.services import fire_detection_service as service

        class Array:
            def __init__(self, data):
                self.data = data

            def cpu(self):
                return self

            def numpy(self):
                return self.data

        class Boxes:
            def __init__(self, values):
                values = np.asarray(values, dtype=np.float32)
                self.xyxy, self.conf, self.cls = Array(values[:, :4]), Array(values[:, 4]), Array(values[:, 5])
                self.count = len(values)

            def __len__(self):
                return self.count

        primary, screen = mock.Mock(), mock.Mock()
        primary.model.names = {0: "Fire", 1: "Smoke"}
        primary.return_value = [types.SimpleNamespace(boxes=Boxes(rows))]
        screen.return_value = []
        with mock.patch.object(service, "YOLO", side_effect=[primary, screen]):
            return service.Detector(self.model, screen_model_path=self.screen,
                                    device="cpu", raise_errors=True, capture_decisions=True,
                                    uncertain_labels=uncertain_labels)

    def test_uncertain_rejections_keep_legacy_notification_gates_closed(self):
        import ast
        from src.demo_diagnostics import diagnostic_frame, summarize
        for reason, label, code in (
                ("No Flicker", "Fire candidate - low temporal change", "low_temporal_change"),
                ("Screen", "Fire candidate - screen overlap", "screen_overlap")):
            with self.subTest(reason=reason):
                detector = self.candidate_detector([[20, 20, 160, 160, 0.95, 0]], uncertain_labels=True)
                with mock.patch.object(detector, "_check_screen_overlap", return_value=reason == "Screen"), \
                        mock.patch.object(detector, "_is_fire_flickering", return_value=False) as flicker, \
                        mock.patch.object(detector, "draw_detection") as draw, \
                        mock.patch.object(detector, "_add_frame_info") as footer:
                    frame, legacy = detector.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
                self.assertEqual(legacy, f"Fake Fire ({reason})")
                self.assertEqual(draw.call_args.args[2], label)
                self.assertAlmostEqual(float(draw.call_args.args[3]), 0.95, places=6)
                self.assertEqual(footer.call_args.args[1], label)
                result = detector.last_result
                self.assertEqual(result["display_label"], label)
                self.assertEqual(result["rejection_reason"], reason)
                self.assertEqual(result["reason_code"], code)
                self.assertTrue(result["rejected"])
                self.assertFalse(result["alert_eligible"])
                self.assertEqual(result["authenticity"], "unverified")
                candidate = detector.last_decision["candidates"][0]
                self.assertFalse(candidate["passes_safeguards"])
                self.assertTrue(candidate["eligible"])  # confidence only; never an alert gate
                self.assertFalse(candidate["assessment"]["alert_eligible"])
                if reason == "Screen":
                    flicker.assert_not_called()
                # Evaluate only the existing, call-free AST predicate, never
                # import notification/camera services or instantiate clients.
                for filename in ("video_service.py", "camera_system.py"):
                    tree = ast.parse((launcher.ROOT / "src/services" / filename).read_text())
                    gates = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == "is_real_detection" for t in node.targets)]
                    self.assertTrue(gates)
                    for gate in gates:
                        self.assertFalse(any(isinstance(n, (ast.Call, ast.Attribute)) for n in ast.walk(gate)))
                        self.assertFalse(eval(compile(ast.Expression(gate), filename, "eval"),
                                              {"__builtins__": {}}, {"detection": legacy}))
                self.detector.last_result = result
                self.detector.process_frame.side_effect = lambda frame: (frame, legacy)
                summary = self.run_fixture(max_frames=1)
                self.assertEqual(summary["label_frame_counts"][label], 1)
                self.assertEqual(summary["detections"], {label: 1})
                record = detector.last_decision
                record["source_seconds"] = 15.0
                self.assertEqual(summarize([record])["label_frame_counts"], {label: 1})
                with mock.patch.object(cv2, "putText", wraps=cv2.putText) as text:
                    diagnostic_frame(frame, record)
                self.assertIn("Selected: " + label, [c.args[1] for c in text.call_args_list])

    def test_uncertain_presentation_preserves_accepted_fire_smoke_selection(self):
        for class_id, expected in ((0, "Fire"), (1, "Smoke")):
            with self.subTest(accepted_class=expected):
                records = []
                statuses = []
                for uncertain in (False, True):
                    detector = self.candidate_detector([
                        [20, 20, 160, 160, 0.95, 0], [300, 200, 420, 320, 0.80, class_id]],
                        uncertain_labels=uncertain)
                    with mock.patch.object(detector, "_is_fire_flickering",
                                           side_effect=[False, True] if class_id == 0 else [False]):
                        _, status = detector.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
                    statuses.append(status)
                    records.append(detector.last_decision["candidates"])
                    self.assertEqual(detector.last_decision["selected_candidate"], 1)
                    self.assertTrue(status.startswith(expected))
                    self.assertEqual(detector.last_result["display_label"], expected)
                    self.assertEqual(detector.last_result["display_status"], status)
                    self.assertFalse(detector.last_result["rejected"])
                    self.assertIsNone(detector.last_result["reason_code"])
                    self.assertTrue(detector.last_result["alert_eligible"])
                    self.assertFalse(records[-1][0]["assessment"]["alert_eligible"])
                self.assertEqual(statuses[0], statuses[1])
                self.assertEqual(records[0], records[1])

    def test_summary_prefers_structured_assessment_over_legacy_display_strings(self):
        from src.detection_status import assessment
        self.detector.last_result = assessment("Fire", eligible=True, rejection_reason="Screen")
        self.detector.process_frame.side_effect = lambda frame: (frame, "Fire")
        summary = self.run_fixture(max_frames=1)
        self.assertEqual(summary["label_frame_counts"]["Fire candidate - screen overlap"], 1)
        self.assertEqual(summary["label_frame_counts"]["Fire"], 0)

    def test_rejected_high_confidence_fire_does_not_suppress_accepted_candidate(self):
        for class_id, expected in ((0, "Fire"), (1, "Smoke")):
            with self.subTest(accepted_class=expected):
                detector = self.candidate_detector([
                    [20, 20, 160, 160, 0.95, 0], [300, 200, 420, 320, 0.80, class_id]])
                outcomes = [False, True] if class_id == 0 else [False]
                with mock.patch.object(detector, "_is_fire_flickering", side_effect=outcomes):
                    _, status = detector.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
                self.assertTrue(status.startswith(expected), status)
                self.assertEqual(detector.last_decision["selected_candidate"], 1)
                self.assertFalse(detector.last_decision["fake_suppressed_eligible_candidate"])
                self.assertEqual(detector.last_decision["candidates"][0]["rejection_reason"], "No Flicker")

    def test_all_rejected_candidates_remain_fake(self):
        detector = self.candidate_detector([
            [20, 20, 160, 160, 0.95, 0], [300, 200, 420, 320, 0.80, 0]])
        with mock.patch.object(detector, "_is_fire_flickering", return_value=False):
            _, status = detector.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(status, "Fake Fire (No Flicker)")
        self.assertEqual(detector.last_decision["selected_candidate"], 0)

    def test_flicker_uses_unannotated_pixels_for_overlapping_candidates(self):
        detector = self.candidate_detector([
            [20, 20, 220, 220, 0.95, 0], [60, 60, 160, 160, 0.80, 0]])
        samples = []

        def flicker(roi):
            samples.append(roi.copy())
            return True

        with mock.patch.object(detector, "_is_fire_flickering", side_effect=flicker):
            detector.process_frame(np.full((480, 640, 3), 80, dtype=np.uint8))
        self.assertEqual(len(samples), 2)
        np.testing.assert_array_equal(samples[1], np.full((100, 100, 3), 80, dtype=np.uint8))
        self.assertFalse(detector.last_decision["candidates"][1]["roi_contains_prior_annotations"])

    def test_low_motion_safeguard_and_warmup_are_preserved(self):
        detector = self.candidate_detector([[20, 20, 160, 160, 0.95, 0]])
        stationary = np.full((100, 100, 3), 80, dtype=np.uint8)
        self.assertTrue(detector._is_fire_flickering(stationary))
        self.assertTrue(detector._is_fire_flickering(stationary))
        self.assertFalse(detector._is_fire_flickering(stationary))
        self.assertEqual(detector.last_motion_score, 0)
        self.assertEqual(detector.flicker_threshold, 15)
        self.assertEqual(detector.flicker_buffer.maxlen, 10)
