import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add src to the system path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.config import Config
from src.services.fire_detection_service import Detector

def test_headless():
    print("🧪 Starting Headless Anti-Spoofing Test...")

    # Initialize detector
    try:
        detector = Detector(Config.MODEL_PATH)
        print("✅ Detector initialized with two models (Fire + Screens)")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Create a dummy frame (black image)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Run process_frame
    print("🏃 Running process_frame on a dummy frame...")
    processed_frame, detection = detector.process_frame(frame)

    print(f"✅ Detection status: {detection}")
    print("✨ Test completed without crashes!")

if __name__ == "__main__":
    test_headless()
