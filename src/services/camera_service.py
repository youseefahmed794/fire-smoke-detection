
"""
MAIN3 CAM.PY (Wrapper)
----------------------
Maintains backward compatibility for standalone execution using CameraSystem.
"""
import cv2
import logging
import sys
import os
import time
from src.services.camera_system import CameraSystem

def main():
    system = CameraSystem()

    try:
        system.start()

        logger = logging.getLogger(__name__)
        logger.info("🎬 Starting main display loop...")

        while True:
            frame, detection = system.get_latest_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            # Display video if not headless
            if os.environ.get('HEADLESS', 'false').lower() != 'true':
                cv2.imshow("Fire Detection System", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.critical(f"Error: {e}")
    finally:
        system.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
