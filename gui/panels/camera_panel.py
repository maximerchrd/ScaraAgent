# gui/panels/camera_panel.py
"""Camera panel that displays the live webcam feed with ArUco overlays."""

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import logging

from vision.aruco_detector import detect_aruco
from vision.image_utils import draw_markers, resize_frame


class CameraPanel(ctk.CTkFrame):
    def __init__(self, parent, camera=None, queue=None, width=640, height=480):
        super().__init__(parent)
        self.camera = camera
        self.queue = queue
        self.display_width = width
        self.display_height = height

        # Title
        ctk.CTkLabel(self, text="Camera Feed", font=("Arial", 14, "bold")).pack(pady=5)

        # Label that will hold the video frame
        self.video_label = ctk.CTkLabel(self, text="Waiting for camera...", width=width, height=height)
        self.video_label.pack(padx=10, pady=5)

        # Status line
        self.status_label = ctk.CTkLabel(self, text="No frame", text_color="gray")
        self.status_label.pack(pady=2)

        # Checkbox to show/hide ArUco overlay
        self.show_aruco = ctk.BooleanVar(value=True)
        self.aruco_check = ctk.CTkCheckBox(self, text="Show ArUco markers", variable=self.show_aruco)
        self.aruco_check.pack(pady=5)

        # Start polling the queue for frames
        self._poll_frame()

    def _poll_frame(self):
        """Check the queue for new camera frames and display the latest."""
        if self.queue:
            frame = None
            # Drain all camera frames from the queue, keeping only the latest
            while not self.queue.empty():
                msg = self.queue.get_nowait()
                if msg.get("type") == "camera_frame":
                    frame = msg["data"]

            if frame is not None:
                self._display_frame(frame)
            else:
                # Try the camera directly as fallback
                if self.camera:
                    frame = self.camera.get_latest_frame()
                    if frame is not None:
                        self._display_frame(frame)

        # Poll again in 50ms (~20fps)
        self.after(50, self._poll_frame)

    def _display_frame(self, frame):
        """Convert a BGR frame to PhotoImage and show it."""
        try:
            # Detect ArUco markers if enabled
            if self.show_aruco.get():
                markers = detect_aruco(frame)
                draw_markers(frame, markers)
                marker_ids = [m["id"] for m in markers] if markers else []
                self.status_label.configure(text=f"Markers: {marker_ids}" if marker_ids else "No markers detected")
            else:
                self.status_label.configure(text="ArUco overlay off")

            # Resize for display
            display_frame = resize_frame(frame, width=self.display_width, height=self.display_height)

            # Convert BGR to RGB for PIL
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the label
            self.video_label.configure(image=imgtk, text="")
            self.video_label.image = imgtk  # Keep reference to prevent garbage collection

        except Exception as e:
            logging.error(f"Camera display error: {e}")
            self.status_label.configure(text=f"Error: {e}")