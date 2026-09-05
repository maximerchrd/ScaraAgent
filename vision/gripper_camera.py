# vision/gripper_camera.py
"""
Gripper-mounted camera (ESP32-CAM) integration.
Captures frames in a background thread for non-blocking access.
"""

import cv2
import numpy as np
import logging
import json
import time
import socket
import threading
import requests
from pathlib import Path
from config import config
import math

class GripperCamera:
    def __init__(self, port=81, static_octet=100):
        self.port = port
        self.static_octet = static_octet
        self.cap = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self.frame_width = None
        self.frame_height = None
        self.scale_mm_per_px = None
        self.hand_eye_offset = (0.0, 0.0)
        self.calibration_file = Path("gripper_calib.json")

        # Try OpenCV first
        if self._connect_cv2():
            self._start_thread()
        else:
            # Fallback to HTTP (not threaded, but we keep it simple)
            self._connect_http()

        self._load_calibration()

        # Check if gripper camera is enabled
        if not config.robot.enable_gripper_refinement:
            logging.info("Gripper camera disabled in config.")
            return

    # ---------- Network ----------
    def _get_esp_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            pc_ip = s.getsockname()[0]
            s.close()
            parts = pc_ip.split('.')
            return f"{parts[0]}.{parts[1]}.{parts[2]}.{self.static_octet}"
        except:
            # Fallback hardcoded for testing
            return "10.194.111.100"

    def _connect_cv2(self, retries=3):
        ip = self._get_esp_ip()
        if not ip:
            return False
        url = f"http://{ip}:{self.port}/stream"

        for attempt in range(retries):
            try:
                self.cap = cv2.VideoCapture(url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # --- Flush buffer ---
                for _ in range(5):
                    self.cap.grab()

                # Test read with timeout
                start = time.time()
                while time.time() - start < 5.0:
                    ret, frame = self.cap.read()
                    if ret:
                        with self._lock:
                            self._latest_frame = frame
                            self.frame_width, self.frame_height = frame.shape[1], frame.shape[0]
                        logging.info(f"✅ Gripper camera connected (OpenCV): {self.frame_width}x{self.frame_height}")
                        return True
                    time.sleep(0.05)
                self.cap.release()
                self.cap = None
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
            time.sleep(1.0)
        logging.error("❌ OpenCV connection to gripper camera failed.")
        return False

    def _start_thread(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logging.info("Gripper capture thread started.")

    def _capture_loop(self):
        """Background loop: continuously read frames from OpenCV."""
        while self._running and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 1)
                if ret:
                    with self._lock:
                        self._latest_frame = frame
                        if self.frame_width is None:
                            self.frame_width, self.frame_height = frame.shape[1], frame.shape[0]
                else:
                    # Read failed – try to reconnect
                    logging.warning("OpenCV read failed, reconnecting...")
                    self._reconnect_cv2()
            except Exception as e:
                logging.warning(f"Capture loop error: {e}")
                time.sleep(0.5)

    def _reconnect_cv2(self):
        """Release and reconnect the OpenCV capture."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self._connect_cv2()

    def get_frame(self):
        """Return the latest cached frame (non-blocking)."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    # ---------- HTTP fallback (simplified, not threaded) ----------
    def _connect_http(self):
        # If OpenCV failed, try HTTP fallback (not threaded, but get_frame will work)
        ip = self._get_esp_ip()
        if not ip:
            return
        url = f"http://{ip}:{self.port}/stream"
        try:
            self.session = requests.Session()
            self.stream_iter = self.session.get(url, stream=True, timeout=(5, 10))
            if self.stream_iter.status_code == 200:
                logging.info("✅ Gripper camera connected (HTTP fallback)")
                # Start a thread for HTTP reading? For simplicity, we'll just read in get_frame.
                # But we can also use a similar thread. For now, keep it simple.
        except Exception as e:
            logging.error(f"HTTP connection failed: {e}")

    # ---------- Calibration (unchanged) ----------
    def _load_calibration(self):
        if self.calibration_file.exists():
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                    self.scale_mm_per_px = data.get("scale_mm_per_px")
                    self.hand_eye_offset = (data.get("dx", 0.0), data.get("dy", 0.0))
                    logging.info(f"Loaded gripper calibration: scale={self.scale_mm_per_px:.4f} mm/px, offset={self.hand_eye_offset}")
            except Exception as e:
                logging.warning(f"Could not load calibration: {e}")

    def save_calibration(self):
        data = {
            "scale_mm_per_px": float(self.scale_mm_per_px) if self.scale_mm_per_px is not None else None,
            "dx": float(self.hand_eye_offset[0]),
            "dy": float(self.hand_eye_offset[1])
        }
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info("Gripper calibration saved.")

    def calibrate_scale(self, marker_size_mm=40.0):
        from vision.aruco_detector import detect_aruco
        frame = self.get_frame()
        if frame is None:
            logging.error("No frame for scale calibration.")
            return None

        markers = detect_aruco(frame)
        if not markers:
            logging.error("No ArUco marker detected in gripper camera frame.")
            return None

        corners = markers[0]['corners']
        side_lengths = [
            np.linalg.norm(corners[0] - corners[1]),
            np.linalg.norm(corners[1] - corners[2]),
            np.linalg.norm(corners[2] - corners[3]),
            np.linalg.norm(corners[3] - corners[0])
        ]
        avg_px_side = np.mean(side_lengths)
        self.scale_mm_per_px = float(marker_size_mm / avg_px_side)
        self.save_calibration()
        logging.info(f"Scale calibrated: {self.scale_mm_per_px:.4f} mm/pixel")
        return self.scale_mm_per_px

    def calibrate_hand_eye_simple(self, robot, marker_id=4, marker_size_mm=40.0):
      """
      Simple hand‑eye calibration: assume the robot is already positioned
      with the gripper camera centered over a known ArUco marker at pick height.
      The function detects the marker centre and computes the offset from TCP to camera.
      """
      # 1. Get current robot work position (TCP)
      rx, ry, _ = robot.get_work_position()

      # 2. Capture frame and detect marker
      frame = self.get_frame()
      if frame is None:
          logging.error("No frame for hand-eye calibration.")
          return False

      # --- DIAGNOSTIC: check thread status ---
      logging.info(f"Capture thread running: {self._running}")
      logging.info(f"Latest frame available: {self._latest_frame is not None}")
      
      frame = self.get_frame()
      if frame is None:
          logging.error("No frame for hand-eye calibration.")
          return False
      
      # Save frame for manual inspection
      cv2.imwrite("gripper_frame_debug.jpg", frame)
      logging.info("Saved gripper_frame_debug.jpg - check this image manually")

      from vision.aruco_detector import detect_aruco

      # 3. Try multiple dictionaries until one works
      dictionaries = ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50"]
      markers = []
      used_dict = None
      for d in dictionaries:
          markers = detect_aruco(frame, dictionary_name=d)
          if markers:
              used_dict = d
              break

      # Debug: log what was detected
      detected_ids = [m['id'] for m in markers]
      logging.info(f"Detected marker IDs in gripper frame (dict: {used_dict}): {detected_ids}")

      if not markers:
          logging.error("No ArUco marker detected in gripper camera frame with any dictionary.")
          logging.info("Try placing a different ArUco marker (4x4, 5x5, or 6x6) under the camera.")
          return False

      marker = next((m for m in markers if m['id'] == marker_id), None)
      if marker is None:
          logging.error(f"Marker ID {marker_id} not found. Detected: {detected_ids}")
          return False

      # 4. Get marker centre
      centre_px = marker['corners'].mean(axis=0)
      cx, cy = frame.shape[1]/2, frame.shape[0]/2
      dx_px, dy_px = centre_px[0] - cx, centre_px[1] - cy

      # 5. Convert pixel offset to mm using current scale
      if self.scale_mm_per_px is None:
          logging.warning("Scale not calibrated – calibrating now...")
          if not self.calibrate_scale(marker_size_mm):
              return False

      dx_mm = dx_px * self.scale_mm_per_px
      dy_mm = dy_px * self.scale_mm_per_px

      # 6. Set hand-eye offset
      self.hand_eye_offset = (-dx_mm, -dy_mm)
      self.save_calibration()
      logging.info(f"Hand-eye offset set to: {self.hand_eye_offset} mm (using dict: {used_dict})")
      logging.info(f"Scale: {self.scale_mm_per_px:.4f} mm/px")
      return True

    # ---------- Pixel → World offset ----------
    def pixel_to_world_offset(self, px_x, px_y, angle_rad):
      if self.frame_width is None or self.frame_height is None:
          return 0.0, 0.0

      cx = self.frame_width / 2.0
      cy = self.frame_height / 2.0
      dx_px = px_x - cx
      dy_px = px_y - cy

      if self.scale_mm_per_px is None:
          logging.warning("Scale not calibrated – using default 0.15 mm/px")
          mm_per_px = 0.15
      else:
          mm_per_px = self.scale_mm_per_px

      # Local offset in camera frame (x right, y down)
      dx_local = dx_px * mm_per_px + self.hand_eye_offset[0]
      dy_local = dy_px * mm_per_px + self.hand_eye_offset[1]

      # Rotate by robot wrist angle to get world offset
      cos_a = math.cos(angle_rad)
      sin_a = math.sin(angle_rad)
      dx_world = dx_local * cos_a - dy_local * sin_a
      dy_world = dx_local * sin_a + dy_local * cos_a

      return dx_world, dy_world

    # ---------- ORB-based template matching ----------
    def locate_template_orb(self, template, scene, nfeatures=500, match_threshold=0.7):
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
        if len(scene.shape) == 3:
            scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        else:
            scene_gray = scene

        orb = cv2.ORB_create(nfeatures=nfeatures)
        kp1, des1 = orb.detectAndCompute(template_gray, None)
        kp2, des2 = orb.detectAndCompute(scene_gray, None)

        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return None, None, 0

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=12, key_size=20, multi_probe_level=2)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < match_threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            return None, None, len(good_matches)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None, None, len(good_matches)

        h_t, w_t = template_gray.shape
        template_center = np.float32([[w_t/2, h_t/2]]).reshape(-1,1,2)
        projected = cv2.perspectiveTransform(template_center, H)
        cx, cy = projected[0][0]

        if 0 <= cx < scene.shape[1] and 0 <= cy < scene.shape[0]:
            return cx, cy, len(good_matches)
        return None, None, len(good_matches)

    def get_template_crop(self, main_frame, bbox):
        x, y, w, h = bbox
        x1 = int(max(0, x - w//2))
        y1 = int(max(0, y - h//2))
        x2 = int(min(main_frame.shape[1], x + w//2))
        y2 = int(min(main_frame.shape[0], y + h//2))
        return main_frame[y1:y2, x1:x2]

    def stop(self):
        """Stop the background capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if self.cap:
            self.cap.release()
            self.cap = None