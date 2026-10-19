import sqlite3
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List
from src.config import Config

class DatabaseService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = Config.DB_PATH
        self.image_folder = Config.DETECTED_FIRES_DIR # Reusing the existing detect folder

        self.init_db()

    def init_db(self):
        """Creates the table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        detection_time TEXT NOT NULL,
                        detection_type TEXT NOT NULL,
                        alert_channels TEXT NOT NULL,
                        image_path TEXT NOT NULL
                    )
                ''')
            self.logger.info("✅ Database initialized")
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")

    def log_detection(self, detection_type: str, triggers_list: List[str], source_image_path: Path):
        """
        Logs the detection event to the database.\x20
        Note: We assume the image is already saved in DETECTED_FIRES_DIR by NotificationService,
        so we store that path directly.
        """
        try:
            # Format triggers list to string
            triggers_str = ", ".join(triggers_list)

            # Ensure path is string for DB
            image_path_str = str(source_image_path)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO detections (detection_time, detection_type, alert_channels, image_path)
                    VALUES (?, ?, ?, ?)
                ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), detection_type, triggers_str, image_path_str))

            self.logger.info(f"💾 Logged to DB: {detection_type}")

        except Exception as e:
            self.logger.error(f"❌ Failed to log to database: {e}")

    def get_recent_logs(self, limit=5):
        """Fetch recent logs for display"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,))
                return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Failed to fetch logs: {e}")
            return []
