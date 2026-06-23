# vision/camera.py
"""Camera capture thread – pushes frames to the shared queue."""

import cv2
import threading
import logging
import time
from utils.safe_queue import SafeQueue

class CameraThread:
    def __init__(self, camera_index=0, width=640, height=480, fps=30, queue=None):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.queue = queue or SafeQueue()
        self._running = False
        self._thread = None
        self._latest_frame = None
        self._lock = threading.Lock()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logging.info("Camera thread started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logging.info("Camera thread stopped.")

    def get_latest_frame(self):
        """Return the most recent frame (can be called from any thread)."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            logging.error(f"Cannot open camera {self.camera_index}")
            self._running = False
            return

        while self._running:
            ret, frame = cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame
                # Push frame to queue (GUI and agent can read it)
                self.queue.put({"type": "camera_frame", "data": frame.copy()})
            else:
                time.sleep(0.01)
        cap.release()