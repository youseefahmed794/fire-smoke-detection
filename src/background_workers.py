"""
BACKGROUND_WORKERS.PY
---------------------
Shared background worker threads for video recording and notifications.
These workers handle slow I/O operations without blocking the main detection loop.
"""
import logging
from queue import Queue
from threading import Thread
from pathlib import Path
import cv2

class NotificationWorker:
    """
    Background thread that handles sending notifications without blocking detection.
    """
    def __init__(self, notification_service):
        self.notification_service = notification_service
        self.queue = Queue()
        self.logger = logging.getLogger(__name__)
        self.thread = Thread(target=self._worker, daemon=True)
        self.running = False

    def start(self):
        """Start the background worker thread"""
        self.running = True
        self.thread.start()
        self.logger.info("✅ Notification worker started")

    def _worker(self):
        """Main worker loop - processes notification queue"""
        while self.running:
            item = self.queue.get()
            if item is None:
                break

            frame, detection = item
            try:
                self.notification_service.send_alert(frame, detection)
                self.logger.info(f"📤 Notification sent: {detection}")
            except Exception as e:
                self.logger.error(f"Failed to send notification: {e}")

            self.queue.task_done()

    def send_alert(self, frame, detection):
        """Queue a notification to be sent in background"""
        self.queue.put((frame.copy(), detection))

    def stop(self):
        """Stop the worker thread gracefully"""
        self.running = False
        self.queue.put(None)
        self.thread.join(timeout=5)
        self.logger.info("✅ Notification worker stopped")


class RecordingWorker:
    """
    Background thread that handles video recording without blocking detection.
    """
    def __init__(self, max_queue_size=30):
        self.queue = Queue(maxsize=max_queue_size)
        self.logger = logging.getLogger(__name__)
        self.thread = Thread(target=self._worker, daemon=True)
        self.running = False
        self.writer = None
        self.current_path = None

    def start(self):
        """Start the background worker thread"""
        self.running = True
        self.thread.start()
        self.logger.info("✅ Recording worker started")

    def _worker(self):
        """Main worker loop - processes recording queue"""
        while self.running:
            item = self.queue.get()
            if item is None:
                self._release_writer()
                break

            frame, path = item

            # Initialize or reinitialize writer if path changed
            if self.writer is None or self.current_path != path:
                self._release_writer()
                self._initialize_writer(frame, path)

            # Write frame
            if self.writer:
                self.writer.write(frame)

            self.queue.task_done()

    def _initialize_writer(self, frame, path):
        """Initialize video writer with frame dimensions"""
        try:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(path), fourcc, 20.0, (w, h))
            self.current_path = path
            self.logger.info(f"🎥 Recording video to: {path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize video writer: {e}")
            self.writer = None

    def _release_writer(self):
        """Release current video writer"""
        if self.writer:
            self.writer.release()
            self.logger.info(f"✅ Video recording stopped: {self.current_path}")
            self.writer = None
            self.current_path = None

    def write_frame(self, frame, output_path):
        """Queue a frame to be written to video"""
        if not self.queue.full():
            self.queue.put((frame.copy(), output_path))
        else:
            self.logger.warning("Recording queue full - dropping frame")

    def stop(self):
        """Stop the worker thread gracefully"""
        self.running = False
        self.queue.put(None)
        self.thread.join(timeout=5)
        self.logger.info("✅ Recording worker stopped")


class VideoStream:
    """
    Threaded video stream reader for IP cameras.
    Reads frames in background to prevent blocking on slow camera reads.
    """
    def __init__(self, src):
        self.logger = logging.getLogger(__name__)
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open stream: {src}")

        # Minimize buffer to reduce delay
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.stopped = False

        # Start background thread
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        self.logger.info("✅ Video stream thread started")

    def _update(self):
        """Background thread - continuously reads frames"""
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        """Get the most recent frame"""
        return self.ret, self.frame

    def stop(self):
        """Stop the stream thread gracefully"""
        self.stopped = True
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=2)
        except Exception:
            pass

        if self.cap.isOpened():
            self.cap.release()

        self.logger.info("✅ Video stream stopped")
