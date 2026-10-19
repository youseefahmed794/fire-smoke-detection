"""
MAIN2 VID.PY
------------
Video file testing script using shared background workers.
Clean implementation with all threading logic in background_workers.py
"""
import cv2
import logging
import sys
from pathlib import Path
from src.config import Config, setup_logging
from .fire_detection_service import Detector
from .notification_service import NotificationService
import time
from src.performance_monitor import PerformanceMonitor
from src.background_workers import NotificationWorker, RecordingWorker


def main(video_source=None):
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Fire Detection System (Video File)")

    try:
        # Validate configuration
        Config.validate()
        logger.debug("Configuration validation successful")

        # Initialize services
        notification_service = NotificationService(Config)
        if notification_service.telegram_token:
            print("✅ Telegram Service ACTIVE")
            notification_service.send_message("🚀 Fire Detection System Started")
        else:
            print("❌ Telegram Service INACTIVE (Check logs)")
        logger.info("Initialized notification services")

        # Initialize detection components
        detector = Detector(Config.MODEL_PATH)
        logger.info(f"Loaded detection model: {Config.MODEL_PATH.name}")

        # Initialize performance monitor
        monitor = PerformanceMonitor()
        frame_count = 0

        # Initialize background workers
        notification_worker = NotificationWorker(notification_service)
        recording_worker = RecordingWorker(max_queue_size=30)

        notification_worker.start()
        recording_worker.start()

        # Video processing setup
        source = video_source if video_source else Config.VIDEO_SOURCE
        cap = cv2.VideoCapture(str(source))

        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            sys.exit(1)
        logger.info(f"Processing video source: {source}")

        # Alert state management
        alert_cooldown = Config.ALERT_COOLDOWN
        last_alert_time = 0
        next_detection_to_report = "any"

        # Generate output path
        out_path = Config.DETECTED_FIRES_DIR / f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

        logger.info("🎬 Starting video processing loop...")

        # Main processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("✅ Video processing completed")
                break

            # Performance optimization: Skip frames (process every 3rd frame)
            if frame_count % 3 != 0:
                frame_count += 1
                continue

            # Run detection
            processed_frame, detection = detector.process_frame(frame)

            # Background video recording
            recording_worker.write_frame(processed_frame, out_path)

            # Handle detections with cooldown
            if detection:
                current_time = time.time()
                # Only alert if it's a real detection (doesn't contain "Fake")
                is_real_detection = "Fake" not in detection

                if is_real_detection and (next_detection_to_report == "any" or next_detection_to_report in detection) \
                        and (current_time - last_alert_time) > alert_cooldown:

                    logger.warning(f"🐦‍🔥 {detection} Detected! Queueing alert")

                    # Background notification
                    notification_worker.send_alert(processed_frame, detection)

                    last_alert_time = current_time
                    next_detection_to_report = "Smoke" if "Fire" in detection else "Fire"

            # Performance monitoring
            frame_count += 1
            if frame_count % 100 == 0:
                monitor.log_stats()

            # Display output
            cv2.imshow("Fire Detection System", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("🛑 User initiated shutdown")
                break

    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.critical(f"🚨 Critical system failure: {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1)
    finally:
        # Cleanup resources
        logger.info("Shutting down...")

        if 'notification_worker' in locals():
            notification_worker.stop()

        if 'recording_worker' in locals():
            recording_worker.stop()

        if 'cap' in locals():
            cap.release()

        cv2.destroyAllWindows()
        logger.info("🛑 System shutdown complete")


if __name__ == "__main__":
    main()
