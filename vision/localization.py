# vision/localization.py
import cv2
import numpy as np
import threading
import time
from vision.aruco_detector import detect_aruco

class MarkerLocalizer:
    def __init__(self, marker_size_mm=50.0):
        from config import config
        self.marker_size_mm = marker_size_mm
        # Load directly from config (no file)
        self.markers = config.vision.marker_positions

    def get_object_xy(self, marker_id, marker_corners_px, object_pixel):
      """
      marker_id: int
      marker_corners_px: 4x2 array of corner pixel coords (from ArUco)
      object_pixel: (px_x, px_y) of the object in the image
      Returns (x_mm, y_mm) in robot coordinates.
      """
      if marker_id not in self.markers:
          print(f"Marker {marker_id} not in config")
          return None

      # Reference pixel: corner 0
      ref_px = marker_corners_px[0]

      # Scale: average side length in pixels (still computed from the whole marker)
      corners = marker_corners_px
      edge1 = np.linalg.norm(corners[0] - corners[1])
      edge2 = np.linalg.norm(corners[2] - corners[3])
      pixel_size = (edge1 + edge2) / 2.0
      mm_per_px = self.marker_size_mm / pixel_size if pixel_size > 0 else 0.5

      # Pixel offset from corner 0 to object
      offset_px = np.array(object_pixel) - ref_px

      # Convert to mm, keeping the same sign convention you used before
      offset_mm_x =  offset_px[0] * mm_per_px
      offset_mm_y = -offset_px[1] * mm_per_px   # flip Y if camera looks down

      # Add to the stored physical corner position
      marker_pos = self.markers[marker_id]
      obj_x = marker_pos["x"] + offset_mm_x
      obj_y = marker_pos["y"] + offset_mm_y

      return (obj_x, obj_y)
    def compute_global_homography(self, detected_markers):
        """
        Compute a single homography matrix from all detected markers.
        detected_markers: dict {marker_id: corners (4x2 numpy array)}
        Returns: homography matrix (3x3) or None if insufficient points.
        """
        pixel_points = []
        world_points = []

        for marker_id, corners_px in detected_markers.items():
            # Get physical position of corner 0 from config
            if marker_id not in self.markers:
                continue
            pos = self.markers[marker_id]
            x0 = pos["x"]
            y0 = pos["y"]
            size = self.marker_size_mm

            # Physical corners (assume marker is axis‑aligned)
            # corner 0: (x0, y0)
            # corner 1: (x0 + size, y0)
            # corner 2: (x0, y0 - size)   [since y increases downward in image, but robot y is up]
            # corner 3: (x0 + size, y0 - size)
            physical_corners = np.array([
                [x0, y0],
                [x0 + size, y0],
                [x0, y0 - size],
                [x0 + size, y0 - size]
            ], dtype=np.float32)

            # Pixel corners (from detection)
            px_corners = corners_px.astype(np.float32)

            # Append all 4 corners to the point lists
            pixel_points.append(px_corners)
            world_points.append(physical_corners)

        if not pixel_points:
            return None

        # Stack all points into single arrays
        pixel_points = np.vstack(pixel_points)
        world_points = np.vstack(world_points)

        # Compute homography (at least 4 points needed)
        H, _ = cv2.findHomography(pixel_points, world_points, cv2.RANSAC, 3.0)
        return H

class MarkerCalibrator:
    def __init__(self, camera, marker_ids=[0,2,3], num_frames=100):
        self.camera = camera
        self.marker_ids = marker_ids
        self.num_frames = num_frames
        self.calibrated_corners = {}   # id → (4,2) numpy array
        self.is_calibrated = False
        self._lock = threading.Lock()

    def calibrate(self, callback=None):
        """Start calibration in a background thread. callback(is_success) called when done."""
        t = threading.Thread(target=self._calibrate_thread, args=(callback,), daemon=True)
        t.start()

    def _calibrate_thread(self, callback):
        accum = {id: [] for id in self.marker_ids}
        collected = 0
        attempts = 0
        max_attempts = self.num_frames * 3

        while collected < self.num_frames and attempts < max_attempts:
            frame = self.camera.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                attempts += 1
                continue
            markers = detect_aruco(frame)
            detected = {m["id"]: m["corners"] for m in markers}
            if all(id in detected for id in self.marker_ids):
                for id in self.marker_ids:
                    accum[id].append(detected[id])
                collected += 1
            attempts += 1
            time.sleep(0.03)

        with self._lock:
            if collected > 0:
                for id in self.marker_ids:
                    if accum[id]:
                        self.calibrated_corners[id] = np.mean(accum[id], axis=0).astype(np.float32)
                self.is_calibrated = True
            else:
                self.is_calibrated = False

        if callback:
            callback(self.is_calibrated)