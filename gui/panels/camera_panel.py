# gui/panels/camera_panel.py
"""Camera panel that displays the live webcam feed with ArUco, VLM, and calibrated marker overlays."""

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import logging
import numpy as np

from vision.aruco_detector import detect_aruco
from vision.image_utils import draw_markers, resize_frame


class CameraPanel(ctk.CTkFrame):
    def __init__(self, parent, camera=None, queue=None, width=640, height=480,
                 show_aruco_var=None, show_vlm_var=None,
                 calibrator=None, show_calibrated_var=None):
        super().__init__(parent)
        self.camera = camera
        self.queue = queue
        self.display_width = width
        self.display_height = height

        # Use external BooleanVars if provided, otherwise create local ones
        self.show_aruco = show_aruco_var or ctk.BooleanVar(value=True)
        self.show_vlm = show_vlm_var or ctk.BooleanVar(value=True)
        self.show_calibrated_var = show_calibrated_var or ctk.BooleanVar(value=True)
        self.calibrator = calibrator

        # Store latest VLM objects
        self.vlm_objects = []

        # Title
        ctk.CTkLabel(self, text="Camera Feed", font=("Arial", 14, "bold")).pack(pady=(5, 0))

        # Video label
        self.video_label = ctk.CTkLabel(self, text="Waiting for camera...", width=width, height=height)
        self.video_label.pack(padx=10, pady=5)

        # Status line
        self.status_label = ctk.CTkLabel(self, text="No frame", text_color="gray")
        self.status_label.pack(pady=2)

        # Start polling
        self._poll_frame()

    def update_vlm_objects(self, objects):
        """Receive list of dicts: {"point": [y_norm, x_norm], "label": ...} from VLM."""
        self.vlm_objects = objects

    def _poll_frame(self):
        """Grab the latest camera frame directly (no queue drain)."""
        if self.camera:
            frame = self.camera.get_latest_frame()
            if frame is not None:
                self._display_frame(frame)
        self.after(50, self._poll_frame)

    def _display_frame(self, frame):
        """Convert a BGR frame to PhotoImage and show it with optional overlays."""
        try:
            status_parts = []

            # ArUco overlay
            if self.show_aruco.get():
                markers = detect_aruco(frame)
                draw_markers(frame, markers)
                marker_ids = [m["id"] for m in markers] if markers else []
                status_parts.append(f"Markers: {marker_ids}" if marker_ids else "No markers")
            else:
                status_parts.append("ArUco off")

            # VLM objects overlay
            if self.show_vlm.get() and self.vlm_objects:
                self._draw_vlm_objects(frame)
                status_parts.append("VLM on")
            elif self.show_vlm.get():
                status_parts.append("VLM: no objects")

            # Calibrated markers overlay
            if self.show_calibrated_var.get() and self.calibrator and self.calibrator.is_calibrated:
                self._draw_calibrated_markers(frame)
                status_parts.append("Calib ✓")

            # Resize for display
            display_frame = resize_frame(frame, width=self.display_width, height=self.display_height)
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.configure(image=imgtk, text="")
            self.video_label.image = imgtk

            self.status_label.configure(text=" | ".join(status_parts) if status_parts else "No frame")

        except Exception as e:
            logging.error(f"Camera display error: {e}")
            self.status_label.configure(text=f"Error: {e}")

    def _draw_vlm_objects(self, frame):
        """Draw circles and labels for VLM objects (normalised coords 0-1000)."""
        h, w = frame.shape[:2]
        for obj in self.vlm_objects:
            try:
                norm_y, norm_x = obj["point"]
                label = obj.get("label", "?")
            except (KeyError, ValueError):
                continue

            px_x = int((norm_x / 1000.0) * w)
            px_y = int((norm_y / 1000.0) * h)

            cv2.circle(frame, (px_x, px_y), 12, (0, 255, 0), 2)
            cv2.circle(frame, (px_x, px_y), 3, (0, 0, 255), -1)
            cv2.putText(frame, label, (px_x + 15, px_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    def _draw_calibrated_markers(self, frame):
        """Draw yellow outlines and IDs for the calibrated marker positions."""
        for marker_id, corners in self.calibrator.calibrated_corners.items():
            pts = corners.astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(frame, str(marker_id), (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)