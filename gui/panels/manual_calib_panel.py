# gui/panels/manual_calib_panel.py
import customtkinter as ctk
import logging
from config import config

class ManualCalibPanel(ctk.CTkFrame):
    def __init__(self, parent, robot, camera=None):
        super().__init__(parent)
        self.robot = robot
        self.camera = camera  # kept only for future use, not used in recording
        self.calib = robot.pixel_calib if robot else None

        ctk.CTkLabel(self, text="ArUco → Robot Coordinate Recording", font=("Arial", 14, "bold")).pack(pady=5)

        # Dropdown for marker ID
        self.marker_var = ctk.StringVar()
        marker_ids = [str(mid) for mid in config.manual_pixel_calib.marker_corners.keys()]
        if not marker_ids:
            marker_ids = ["No IDs defined"]
        self.marker_dropdown = ctk.CTkOptionMenu(self, variable=self.marker_var, values=marker_ids)
        self.marker_dropdown.pack(pady=5)

        # Show which corner is used for the selected marker
        self.corner_label = ctk.CTkLabel(self, text="Corner: -", text_color="gray")
        self.corner_label.pack(pady=2)
        self.marker_dropdown.bind("<<ComboboxSelected>>", self._update_corner_label)

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=5)

        self.record_btn = ctk.CTkButton(btn_frame, text="Record Current Pose", command=self._record_point)
        self.record_btn.pack(side="left", padx=5)

        # Status
        self.status_label = ctk.CTkLabel(self, text="Ready", text_color="gray")
        self.status_label.pack(pady=5)

        self._update_corner_label(None)

    def _update_corner_label(self, _):
        selected_id = int(self.marker_var.get()) if self.marker_var.get().isdigit() else None
        if selected_id is not None and selected_id in config.manual_pixel_calib.marker_corners:
            corner_idx = config.manual_pixel_calib.marker_corners[selected_id]
            corner_names = {-1: "Center", 0: "Top-Left", 1: "Top-Right", 2: "Bottom-Right", 3: "Bottom-Left"}
            self.corner_label.configure(text=f"Corner: {corner_names.get(corner_idx, 'Unknown')}")
        else:
            self.corner_label.configure(text="Corner: -")

    def _record_point(self):
        if self.robot is None or self.calib is None:
            self.status_label.configure(text="Robot not available", text_color="red")
            return

        # Get current robot position (native coordinates, no offset)
        rx, ry, _ = self.robot.get_work_position()

        # Get selected marker and corner
        selected_id = int(self.marker_var.get())
        corner_idx = config.manual_pixel_calib.marker_corners.get(selected_id, -1)

        # Save the fixed point
        self.calib.commit_fixed_point(selected_id, corner_idx, rx, ry)
        self.status_label.configure(
            text=f"Recorded: Marker {selected_id} corner {corner_idx} at ({rx:.1f}, {ry:.1f})",
            text_color="lightgreen"
        )

    def _clear_points(self):
        if self.calib:
            self.calib.fixed_points = []
            self.calib._save_fixed_points()
            self.status_label.configure(text="All fixed points cleared", text_color="orange")