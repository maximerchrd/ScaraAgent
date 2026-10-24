# gui/panels/camera_panel.py
"""Camera panel that displays the live webcam feed with ArUco, VLM, and calibrated marker overlays."""

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import logging
import numpy as np

from vision.aruco_detector import detect_aruco
from vision.image_utils import draw_markers, resize_frame


class CameraPanel(ctk.CTkFrame):
    def __init__(self, parent, camera=None, queue=None, on_crop_changed=None, width=640, height=480,
                 show_aruco_var=None, show_vlm_var=None,
                 calibrator=None, show_calibrated_var=None):
        super().__init__(parent)
        self.camera = camera
        self.queue = queue
        self.display_width = width
        self.display_height = height

        self.raw_perceptions = []

        # Use external BooleanVars if provided, otherwise create local ones
        self.show_aruco = show_aruco_var or ctk.BooleanVar(value=True)
        self.show_vlm = show_vlm_var or ctk.BooleanVar(value=True)
        self.show_calibrated_var = show_calibrated_var or ctk.BooleanVar(value=True)
        self.calibrator = calibrator
        self.on_crop_changed = on_crop_changed
        

        # Store latest VLM objects
        self.vlm_objects = []

        # Title
        ctk.CTkLabel(self, text="Camera Feed", font=("Arial", 14, "bold")).pack(pady=(5, 0))

        # Video canvas (replaces the old CTkLabel)
        self.canvas = tk.Canvas(
            self, width=width, height=height,
            highlightthickness=0, bg="black"
        )
        self.canvas.pack(padx=10, pady=5)
        self.canvas.create_text(
            width // 2, height // 2,
            text="Waiting for camera...", fill="white"
        )

        # Crop state
        self.crop_rect   = None     # canvas coords (x0, y0, x1, y1) or None
        self.crop_mode   = False    # editing mode on/off
        self.drag_mode   = None     # None | "draw" | "move" | "resize"
        self.drag_handle = None     # "tl" | "tr" | "bl" | "br"
        self.drag_start  = None     # (cx, cy, rect_at_start)

        # Mouse bindings
        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_down)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        # Status line
        self.status_label = ctk.CTkLabel(self, text="No frame", text_color="gray")
        self.status_label.pack(pady=2)

        # (px_x, px_y, label)
        self.placement_marker = None

        self.perception_points = []   # list of {"point": [y_norm, x_norm], "label": str}

        # Start polling
        self._poll_frame()

    def _emit_crop(self):
        if self.on_crop_changed is None:
            return
        if self.crop_rect is None:
            self.on_crop_changed(None)
        else:
            x0, y0, x1, y1 = self.crop_rect
            W, H = self.display_width, self.display_height
            self.on_crop_changed((x0 / W, y0 / H, x1 / W, y1 / H))

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
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
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

            # Perceive overlays (magenta)
            if self.perception_points:
                self._draw_perception_points(frame)

            # Calibrated markers overlay
            if self.show_calibrated_var.get() and self.calibrator and self.calibrator.is_calibrated:
                self._draw_calibrated_markers(frame)
                status_parts.append("Calib ✓")

            h, w = frame.shape[:2]
            for raw_data in self.raw_perceptions:
                self._draw_raw_perception(frame, raw_data, "", h, w)

            if self.placement_marker:
                px, py, lbl = self.placement_marker
                cv2.drawMarker(frame, (int(px), int(py)), (255, 0, 255), 
                            markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.putText(frame, lbl, (int(px)+10, int(py)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Resize for display
            display_frame = resize_frame(frame, width=self.display_width, height=self.display_height)
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Draw to canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.image = imgtk    # keep reference

            # Redraw crop overlay on top
            self._redraw_canvas()

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
            
    def set_placement_marker(self, px_x, px_y, label):
        """Store a placement refinement marker to overlay on the next frame."""
        self.placement_marker = (px_x, px_y, label)

    def set_perception_points(self, points):
        """Receive list of dicts and add them to the current overlay set."""
        self.perception_points.extend(points)

    def clear_perception_points(self):
        """Clear all perceived point overlays."""
        self.perception_points = []

    def _draw_perception_points(self, frame):
        """Draw perceived points (normalised [y,x] 0-1000) in magenta."""
        h, w = frame.shape[:2]
        for obj in self.perception_points:
            try:
                norm_y, norm_x = obj["point"]
                label = obj.get("label", "?")
            except (KeyError, ValueError):
                continue

            px_x = int((norm_x / 1000.0) * w)
            px_y = int((norm_y / 1000.0) * h)

            cv2.circle(frame, (px_x, px_y), 10, (255, 0, 255), 2)
            cv2.circle(frame, (px_x, px_y), 2, (255, 255, 255), -1)
            cv2.putText(frame, label, (px_x + 12, px_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

    def update_raw_perception(self, data):
        """Store the raw VLM response for overlay."""
        if data is not None:
            self.raw_perceptions.append(data)
        else:
            self.raw_perceptions.clear()

    def clear_perception_points(self):
        self.perception_points = []
        self.raw_perceptions.clear()

    def _draw_raw_perception(self, frame, data, prefix, h, w):
        """Recursively draw any point-like lists from raw VLM output."""
            # First, check if this is a point (list/tuple of two numbers)
        if isinstance(data, (list, tuple)) and len(data) == 2 and all(isinstance(x, (int, float)) for x in data):
            norm_y, norm_x = data[0], data[1]
            if 0 <= norm_y <= 1000 and 0 <= norm_x <= 1000:
                px_x = int((norm_x / 1000.0) * w)
                px_y = int((norm_y / 1000.0) * h)
                cv2.circle(frame, (px_x, px_y), 8, (255, 255, 0), -1)
                cv2.circle(frame, (px_x, px_y), 8, (255, 255, 255), 1)
                label = prefix[-12:] if prefix else "pt"
                cv2.putText(frame, label, (px_x + 12, px_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            return  # stop recursion for this branch

        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._draw_raw_perception(frame, value, new_prefix, h, w)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                self._draw_raw_perception(frame, item, f"{prefix}[{idx}]", h, w)
        else:
            # Check if this is a point: a list/tuple of two numbers
            if isinstance(data, (list, tuple)) and len(data) == 2 and all(isinstance(x, (int, float)) for x in data):
                norm_y, norm_x = data[0], data[1]
                # Assume coordinates are normalised 0-1000
                if 0 <= norm_y <= 1000 and 0 <= norm_x <= 1000:
                    px_x = int((norm_x / 1000.0) * w)
                    px_y = int((norm_y / 1000.0) * h)
                    # Draw a bright cyan circle with white border
                    cv2.circle(frame, (px_x, px_y), 8, (255, 255, 0), -1)
                    cv2.circle(frame, (px_x, px_y), 8, (255, 255, 255), 1)
                    # Show the key path as label (shortened)
                    label = prefix[-12:] if prefix else "pt"
                    cv2.putText(frame, label, (px_x + 12, px_y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)


    def _which_handle(self, x, y, rect, tol=10):
        x0, y0, x1, y1 = rect
        for name, (hx, hy) in {
            "tl": (x0, y0), "tr": (x1, y0),
            "bl": (x0, y1), "br": (x1, y1),
        }.items():
            if abs(x - hx) <= tol and abs(y - hy) <= tol:
                return name
        return None

    def _on_mouse_down(self, event):
        if not self.crop_mode:
            return
        x, y = event.x, event.y
        if self.crop_rect:
            handle = self._which_handle(x, y, self.crop_rect)
            if handle:
                self.drag_mode = "resize"
                self.drag_handle = handle
                self.drag_start = (x, y, self.crop_rect)
                return
            x0, y0, x1, y1 = self.crop_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.drag_mode = "move"
                self.drag_start = (x, y, self.crop_rect)
                return
        # Fresh draw
        self.drag_mode = "draw"
        self.crop_rect = (x, y, x, y)
        self.drag_start = (x, y, None)

    def _on_mouse_drag(self, event):
        if self.drag_mode is None:
            return
        W, H = self.display_width, self.display_height
        x = max(0, min(W, event.x))
        y = max(0, min(H, event.y))

        if self.drag_mode == "draw":
            sx, sy, _ = self.drag_start
            self.crop_rect = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))

        elif self.drag_mode == "move":
            sx, sy, old = self.drag_start
            ox0, oy0, ox1, oy1 = old
            w, h = ox1 - ox0, oy1 - oy0
            nx0 = max(0, min(W - w, ox0 + (x - sx)))
            ny0 = max(0, min(H - h, oy0 + (y - sy)))
            self.crop_rect = (nx0, ny0, nx0 + w, ny0 + h)

        elif self.drag_mode == "resize":
            ox0, oy0, ox1, oy1 = self.crop_rect
            if self.drag_handle == "tl": ox0, oy0 = x, y
            elif self.drag_handle == "tr": ox1, oy0 = x, y
            elif self.drag_handle == "bl": ox0, oy1 = x, y
            elif self.drag_handle == "br": ox1, oy1 = x, y
            self.crop_rect = (min(ox0, ox1), min(oy0, oy1),
                              max(ox0, ox1), max(oy0, oy1))

        self._redraw_canvas()

    def _on_mouse_up(self, event):
        if self.drag_mode is None:
            return
        if self.crop_rect:
            x0, y0, x1, y1 = self.crop_rect
            if (x1 - x0) < 30 or (y1 - y0) < 30:
                self.crop_rect = None   # discard tiny rectangles
        self.drag_mode = None
        self.drag_handle = None
        self._emit_crop()

    def _redraw_canvas(self):
        """Redraw the crop overlay on top of the current frame image.
        Cheap: deletes only rect/handle items, keeps the image."""
        self.canvas.delete("crop")
        W, H = self.display_width, self.display_height
        if self.crop_rect is None:
            if self.crop_mode:
                self.canvas.create_text(
                    W // 2, 20,
                    text="Click + drag to set crop",
                    fill="cyan", font=("Arial", 12, "bold"),
                    tags="crop",
                )
            return
        x0, y0, x1, y1 = self.crop_rect

        # Dim outside crop
        for r in [(0, 0, W, y0), (0, y1, W, H),
                  (0, y0, x0, y1), (x1, y0, W, y1)]:
            self.canvas.create_rectangle(
                *r, fill="black", stipple="gray50", outline="", tags="crop"
            )

        # Crop rectangle
        self.canvas.create_rectangle(x0, y0, x1, y1,
                                     outline="cyan", width=2, tags="crop")

        # Corner handles (only in edit mode)
        if self.crop_mode:
            for hx, hy in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
                self.canvas.create_rectangle(
                    hx - 6, hy - 6, hx + 6, hy + 6,
                    fill="cyan", outline="white", tags="crop"
                )

    def _emit_crop(self):
        if self.on_crop_changed is None:
            return
        rect = getattr(self, "crop_rect", None)   # safe even if not initialized
        if rect is None:
            self.on_crop_changed(None)
        else:
            x0, y0, x1, y1 = rect
            W, H = self.display_width, self.display_height
            self.on_crop_changed((x0 / W, y0 / H, x1 / W, y1 / H))