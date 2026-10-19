"""
NOTIFICATION_SERVICE.PY
-----------------------
This file manages all outgoing alerts when fire or smoke is detected.
It handles:
1. Connecting to Telegram to send text messages and photos.
2. Connecting to WhatsApp (via TextMeBot/CallMeBot) for mobile alerts.
3. Uploading captured frames to Cloudinary so they can be viewed via a web link.
4. Using a 'ThreadPool' to send alerts in the background without slowing down the video.
"""
import os
import cv2
import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor
import cloudinary
import cloudinary.uploader

logger = logging.getLogger(__name__)

# =================================================


from .database_service import DatabaseService

class NotificationService:
    def __init__(self, config):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.db = DatabaseService() # Initialize DB

        # Use attributes from the config object
        self.telegram_token = config.TELEGRAM_TOKEN
        self.telegram_chat_id = config.TELEGRAM_CHAT_ID

        # WhatsApp / TextMeBot
        self.textmebot_api_key = config.TEXTMEBOT_API_KEY
        # self.callmebot_api_key = config.CALLMEBOT_API_KEY # Assuming it's in config
        self.whatsapp_api_key = self.textmebot_api_key # or config.CALLMEBOT_API_KEY
        self.whatsapp_number = config.RECEIVER_WHATSAPP_NUMBER
        if self.whatsapp_number:
            self.whatsapp_number = str(self.whatsapp_number).lstrip('+')

        # Cloudinary
        self.cloudinary_cloud_name = config.CLOUDINARY_CLOUD_NAME
        self.cloudinary_api_key = config.CLOUDINARY_API_KEY
        self.cloudinary_api_secret = config.CLOUDINARY_API_SECRET

        # Status Flags
        self.telegram_enabled = bool(self.telegram_token and self.telegram_chat_id)
        self.whatsapp_enabled = bool(self.whatsapp_api_key and self.whatsapp_number)
        self.cloudinary_enabled = all([
            self.cloudinary_cloud_name,
            self.cloudinary_api_key,
            self.cloudinary_api_secret
        ])

        # Logging Presence
        logger.info(f"Telegram Config: {'Found' if self.telegram_enabled else 'MISSING'}")
        logger.info(f"WhatsApp Config: {'Found' if self.whatsapp_enabled else 'MISSING'} (Service: {'TextMeBot' if self.textmebot_api_key else 'CallMeBot'})")
        logger.info(f"Cloudinary Config: {'Found' if self.cloudinary_enabled else 'MISSING'}")

        if not self.cloudinary_enabled:
            logger.warning(f"Cloudinary Details: Name={'Set' if self.cloudinary_cloud_name else 'NONE'}, Key={'Set' if self.cloudinary_api_key else 'NONE'}, Secret={'Set' if self.cloudinary_api_secret else 'NONE'}")

        # Service Setup
        if self.whatsapp_enabled:
            if self.textmebot_api_key:
                self.whatsapp_base_url = "https://api.textmebot.com/send.php"
            else:
                self.whatsapp_base_url = "https://api.callmebot.com/whatsapp.php"

        if self.cloudinary_enabled:
            cloudinary.config(
                cloud_name=self.cloudinary_cloud_name,
                api_key=self.cloudinary_api_key,
                api_secret=self.cloudinary_api_secret,
                secure=True
            )

    # =================================================

    def save_frame(self, frame) -> Path:
        """Save detection frame with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self.config.DETECTED_FIRES_DIR / f"alert_{timestamp}.jpg"
        cv2.imwrite(str(path), frame)
        return path

    # =================================================

    def send_message(self, message: str) -> bool:
        """Send a text-only message to all enabled platforms"""
        if self.telegram_enabled:
            self.executor.submit(self._send_telegram_message, message)

        if self.whatsapp_enabled:
            self.executor.submit(self._send_whatsapp_message, message)

        return True

    def _send_telegram_message(self, message: str) -> bool:
        """Internal method to send text message via Telegram API"""
        try:
            logger.info(f"Sending Telegram message: {message[:30]}...")
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message
            }
            response = requests.post(url, data=data, timeout=15)
            if response.status_code == 200:
                logger.debug("✅ Telegram message sent")
                return True
            logger.error(f"❌ Telegram failed [{response.status_code}]: {response.text}")
        except Exception as e:
            logger.error(f"Telegram communication error: {e}")
        return False

    def _send_whatsapp_message(self, message: str, image_url: str = None) -> bool:
        """Internal method to send text or image message via WhatsApp API"""
        try:
            logger.info(f"Sending WhatsApp message: {message[:30]}...")

            # TextMeBot uses different parameters than CallMeBot
            if self.textmebot_api_key:
                params = {
                    "recipient": self.whatsapp_number,
                    "apikey": self.whatsapp_api_key,
                    "text": message
                }
                if image_url:
                    params["file"] = image_url
            else:
                # CallMeBot
                params = {
                    "phone": self.whatsapp_number,
                    "apikey": self.whatsapp_api_key,
                    "text": message
                }
                if image_url:
                    # CallMeBot sometimes supports image links in text or via specific endpoint
                    # but mostly it's just text with a link.
                    pass

            response = requests.get(self.whatsapp_base_url, params=params, timeout=15)

            if response.status_code == 200:
                logger.debug("✅ WhatsApp message sent")
                return True
            logger.error(f"❌ WhatsApp failed [{response.status_code}]: {response.text}")
        except Exception as e:
            logger.error(f"WhatsApp communication error: {e}")
        return False

    # =================================================

    def send_alert(self, frame, detection="Fire") -> bool:
        """Save frame and dispatch alerts in background"""
        image_path = self.save_frame(frame)

        self.executor.submit(
            self._send_alerts_worker,
            image_path,
            detection
        )
        return True

    def _send_alerts_worker(self, image_path: Path, detection: str):
        """Worker thread for sending multiple notifications"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            caption = f"🚨 {detection} Detected at {timestamp}!"

            logger.info(f"Processing alerts for {detection}")

            triggers = []

            # Telegram (Direct photo upload)
            if self.telegram_enabled:
                if self._send_telegram_photo(image_path, caption):
                    triggers.append("Telegram")

            # WhatsApp (Requires URL, so upload to Cloudinary first)
            if self.whatsapp_enabled:
                if self.cloudinary_enabled:
                    image_url = self._upload_to_cloudinary(image_path)
                    if image_url:
                        # For TextMeBot, we can pass image_url directly
                        if self._send_whatsapp_message(caption, image_url=image_url):
                             triggers.append("WhatsApp")
                    else:
                        logger.error("WhatsApp skipped: Image upload failed")
                        # Fallback to text only if image fails
                        self._send_whatsapp_message(f"{caption} (Image upload failed)")
                else:
                    logger.warning("WhatsApp prompt: Set CLOUDINARY credentials for photo alerts")
                    self._send_whatsapp_message(caption)

            # Log to Database
            self.db.log_detection(detection, triggers, image_path)

        except Exception as e:
            logger.exception(f"Error in notification worker: {e}")

    # =================================================

    def _send_telegram_photo(self, image_path: Path, caption: str) -> bool:
        """Send photo to Telegram chat"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
            with open(image_path, "rb") as img:
                files = {"photo": img}
                data = {
                    "chat_id": self.telegram_chat_id,
                    "caption": caption
                }
                response = requests.post(url, data=data, files=files, timeout=20)

            if response.status_code == 200:
                logger.info("✅ Telegram alert sent")
                return True
            logger.error(f"❌ Telegram photo failed [{response.status_code}]: {response.text}")
        except Exception as e:
            logger.error(f"Telegram photo error: {e}")
        return False

    # =================================================

    def _upload_to_cloudinary(self, image_path: Path) -> str | None:
        """Upload image to Cloudinary and return the direct link"""
        try:
            logger.debug(f"Uploading {image_path.name} to Cloudinary...")
            result = cloudinary.uploader.upload(
                str(image_path),
                folder="fire_detections",
                resource_type="image"
            )
            return result.get("secure_url")
        except Exception as e:
            logger.error(f"Cloudinary upload failed: {e}")
            return None

    # =================================================

    def cleanup(self):
        """Shutdown the executor gracefully"""
        logger.info("Shutting down NotificationService...")
        self.executor.shutdown(wait=True)
