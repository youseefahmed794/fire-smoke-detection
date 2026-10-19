import cv2
import logging
import sys
import os
import time
from pathlib import Path
from src.config import Config, setup_logging
from src.services.database_service import DatabaseService
from src.services.notification_service import NotificationService
from src.performance_monitor import PerformanceMonitor
from src.services.fire_detection_service import Detector
from src.background_workers import VideoStream, NotificationWorker, RecordingWorker

class CameraSystem:
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.frame = None
        self.latest_detection = None

        # Initialize components
        self.db = DatabaseService() # Ensure DB is init
        self.notification_service = NotificationService(Config)
        self.detector = Detector(Config.MODEL_PATH)
        self.monitor = PerformanceMonitor()

        # Workers
        self.notification_worker = NotificationWorker(self.notification_service)
        self.recording_worker = RecordingWorker(max_queue_size=30)
        self.video_stream = None # Created on start

        # State
        self.alert_cooldown = Config.ALERT_COOLDOWN
        self.last_alert_time = 0
        self.next_detection_to_report = "any"
        self.out_path = Path("detect") / f"cam_output_{time.strftime('%Y%m%d_%H%M%S')}.avi"

        Path("detect").mkdir(exist_ok=True)

    def start(self):
        self.logger.info("🚀 Starting Camera System")

        # Start workers
        self.notification_worker.start()
        self.recording_worker.start()

        # Start Stream
        rtsp_url = Config.get_rtsp_url()
        if not rtsp_url:
            raise ValueError("RTSP URL not configured")

        self.video_stream = VideoStream(rtsp_url)
        self.running = True
        self.logger.info(f"✅ System Started. Connected to {rtsp_url[:15]}...")

    def stop(self):
        self.logger.info("🛑 Stopping Camera System details...")
        self.running = False

        if self.video_stream:
            self.video_stream.stop()

        self.notification_worker.stop()
        self.recording_worker.stop()
        self.logger.info("✅ Camera System Stopped")

    def get_latest_frame(self):
        """
        Returns (frame, detection_info)
        Also handles the processing logic that was previously in the main loop.
        """
        if not self.running or not self.video_stream:
            return None, None

        ret, frame = self.video_stream.read()
        if not ret:
            return None, None

        # Resize
        frame = cv2.resize(frame, (640, 480))

        # Process
        processed_frame, detection = self.detector.process_frame(frame)

        # Record
        self.recording_worker.write_frame(processed_frame, self.out_path)

        # Alert Logic
        if detection:
             self._handle_detection(processed_frame, detection)

        # Monitor
        self.monitor.log_stats()

        self.frame = processed_frame
        return processed_frame, detection

    def _handle_detection(self, frame, detection):
        current_time = time.time()
        # Only alert if it's a real detection (doesn't contain "Fake")
        is_real_detection = "Fake" not in detection

        if is_real_detection and (self.next_detection_to_report == "any" or self.next_detection_to_report in detection) \
                and (current_time - self.last_alert_time) > self.alert_cooldown:

            self.notification_worker.send_alert(frame, detection)
            self.last_alert_time = current_time
            self.next_detection_to_report = "Smoke" if "Fire" in detection else "Fire"
            self.logger.info(f"🔥 Detection: {detection} (queued for notification)")
