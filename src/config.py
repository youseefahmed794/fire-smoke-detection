"""
CONFIG.PY
----------
This file is the central configuration hub for the project.
It handles:
1. Loading environment variables from the root .env file.
2. Setting up the global logging system (saving logs to /logs/fire_detection.log).
3. Defining file paths for the AI model, video sources, and detection directories.
4. Storing constant values like alert cooldowns and API keys.
"""
import os
import logging
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / '.env'

# The isolated demo supplies a clean environment and must never inspect .env.
# Check this before importing dotenv or probing ENV_PATH.
if os.environ.get('FIRE_DEMO_ISOLATED') != '1':
    from dotenv import load_dotenv

    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)
    else:
        print(f"⚠️ Warning: .env file not found at {ENV_PATH}")
    load_dotenv(override=True)


def setup_logging():
    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        filename=log_dir / 'fire_detection.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

# Environment variables configuration


class Config:
    RECEIVER_WHATSAPP_NUMBER = os.getenv('RECEIVER_WHATSAPP_NUMBER')
    IMGUR_CLIENT_ID = os.getenv('IMGUR_CLIENT_ID')
    IMGUR_CLIENT_SECRET = os.getenv('IMGUR_CLIENT_SECRET')
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    CALLMEBOT_API_KEY = os.getenv('CALLMEBOT_API_KEY')
    TEXTMEBOT_API_KEY = os.getenv('TEXTMEBOT_API_KEY')
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')

    CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
    CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')





    PROJECT_ROOT = Path(__file__).parent.parent

    _model_path_str = os.getenv('MODEL_PATH', 'models_weights/firedetect-11x.pt')
    MODEL_PATH = PROJECT_ROOT / Path(_model_path_str)

    _video_source_str = os.getenv('VIDEO_SOURCE', 'assets/test samples/bucket11.mp4')
    VIDEO_SOURCE = PROJECT_ROOT / Path(_video_source_str)

    _detect_dir_str = os.getenv('DETECTED_FIRES_DIR', 'src/detect')
    DETECTED_FIRES_DIR = PROJECT_ROOT / Path(_detect_dir_str)

    ALERT_COOLDOWN = int(os.getenv('ALERT_COOLDOWN', 45))  # Seconds between alerts

    # Database Configuration
    DB_NAME = os.getenv('DB_NAME', 'fire_system.db')
    DB_PATH = PROJECT_ROOT / DB_NAME

    # Detection Configuration
    TARGET_HEIGHT = int(os.getenv('TARGET_HEIGHT', 480))
    IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.2))
    MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', 0.40))
    SMOKE_CONFIDENCE = float(os.getenv('SMOKE_CONFIDENCE', 0.40))

    # Anti-Spoofing Configuration
    FLICKER_THRESHOLD = float(os.getenv('FLICKER_THRESHOLD', 15.0))
    FLICKER_BUFFER_SIZE = int(os.getenv('FLICKER_BUFFER_SIZE', 10))

    # RTSP Configuration
    PROTOCOL = os.getenv('PROTOCOL', 'rtsp')
    USERNAME = os.getenv('USERNAME')
    PASSWORD = os.getenv('PASSWORD')
    IP = os.getenv('IP')
    PORT = os.getenv('PORT', '554')
    CHANNEL = os.getenv('CHANNEL', '')

    @classmethod
    def get_rtsp_url(cls):
        """Constructs the RTSP URL from components or returns the direct RTSP_URL if provided."""
        direct_url = os.getenv('RTSP_URL')
        if direct_url:
            return direct_url

        if all([cls.USERNAME, cls.PASSWORD, cls.IP]):
            # Use raw password if it doesn't seem to be URL encoded already
            # If it has '%' followed by hex, it might be encoded.
            # To be safe, we unquote then quote to ensure single encoding.
            from urllib.parse import unquote
            safe_password = quote(unquote(cls.PASSWORD))
            return f"{cls.PROTOCOL}://{cls.USERNAME}:{safe_password}@{cls.IP}:{cls.PORT}/{cls.CHANNEL.lstrip('/')}"

        return None

    @classmethod
    def validate(cls):
        # Essential credentials (Telegram)
        essential_vars = ["TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"]
        missing_essential = [var for var in essential_vars if getattr(cls, var) is None]

        if missing_essential:
            raise ValueError(
                f"Missing essential environment variables: {', '.join(missing_essential)}")

        # Create necessary directories
        cls.DETECTED_FIRES_DIR.mkdir(exist_ok=True)

        if not cls.VIDEO_SOURCE.exists():
            raise FileNotFoundError(
                f"Video source missing: {cls.VIDEO_SOURCE}")
