from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import threading
import time
from src.services.camera_system import CameraSystem
import uvicorn
import os

app = FastAPI(title="Fire Detection API")

# Global system instance
camera_system = CameraSystem()

@app.on_event("startup")
async def startup_event():
    # Start the camera system in a separate thread to avoid blocking the API
    # However, CameraSystem.start() just starts workers and opens stream, it does NOT block.
    # But get_latest_frame() is what drives the logic.
    # We need a background thread that continuously "pumps" the camera system
    # so that detection and recording happen even if no one is watching the stream.
    camera_system.start()
    threading.Thread(target=run_processing_loop, daemon=True).start()

@app.on_event("shutdown")
async def shutdown_event():
    camera_system.stop()

def run_processing_loop():
    """Continuously process frames so detection and recording works in background"""
    while camera_system.running:
        camera_system.get_latest_frame()
        # limit max FPS to avoid burning CPU equivalent to camera FPS
        # actually get_latest_frame calls read() which blocks until new frame, so no sleep needed typically
        # but if stream is down, we sleep
        if not camera_system.video_stream or not camera_system.video_stream.ret:
             time.sleep(0.1)

def generate_frames():
    """Generator for MJPEG stream"""
    while True:
        if camera_system.frame is not None:
            ret, buffer = cv2.imencode('.jpg', camera_system.frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
             # Yield a placeholder or wait
             time.sleep(0.1)

        # Limit stream FPS to save bandwidth
        time.sleep(0.04) # ~25 FPS

@app.get("/")
def read_root():
    return {"status": "online", "system": "Fire Detection System"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def get_status():
    return {
        "running": camera_system.running,
        "rtsp_url": os.environ.get("RTSP_URL", "Not Set")
    }
