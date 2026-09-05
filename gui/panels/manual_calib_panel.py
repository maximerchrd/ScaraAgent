# gui/panels/manual_calib_panel.py
import customtkinter as ctk
import logging
from config import config

class ManualCalibPanel(ctk.CTkFrame):
    def __init__(self, parent, robot, camera=None, orchestrator=None):
        """
        Panel for manual calibration:
        - Recording ArUco marker fixed points (main camera)
        - Gripper camera calibration (hand-eye + scale)
        """
        super().__init__(parent)
        self.robot = robot
        self.camera = camera
        self.orchestrator = orchestrator   # <-- store orchestrator
        self.calib = robot.pixel_calib if robot else None

        # ========== Header ==========
        ctk.CTkLabel(self, text="ArUco → Robot Coordinate Recording", font=("Arial", 14, "bold")).pack(pady=5)

        # ========== Marker selection (main camera) ==========
        self.marker_var = ctk.StringVar()
        marker_ids = [str(mid) for mid in config.manual_pixel_calib.marker_corners.keys()]
        if not marker_ids:
            marker_ids = ["No IDs defined"]
        self.marker_dropdown = ctk.CTkOptionMenu(self, variable=self.marker_var, values=marker_ids)
        self.marker_dropdown.pack(pady=5)

        self.corner_label = ctk.CTkLabel(self, text="Corner: -", text_color="gray")
        self.corner_label.pack(pady=2)
        self.marker_dropdown.bind("<<ComboboxSelected>>", self._update_corner_label)

        # ========== Main camera recording buttons ==========
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=5)

        self.record_btn = ctk.CTkButton(btn_frame, text="Record Current Pose", command=self._record_point)
        self.record_btn.pack(side="left", padx=5)

        self.clear_btn = ctk.CTkButton(btn_frame, text="Clear All Points", command=self._clear_points, fg_color="#E67E22")
        self.clear_btn.pack(side="left", padx=5)

        self.status_label = ctk.CTkLabel(self, text="Ready", text_color="gray")
        self.status_label.pack(pady=5)

        # ========== Gripper Camera Calibration ==========
        self.calib_gripper_btn = ctk.CTkButton(
            self,
            text="Calibrate Gripper Camera",
            command=self._calibrate_gripper
        )
        self.calib_gripper_btn.pack(pady=5)

        self.gripper_calib_status = ctk.CTkLabel(
            self, text="Gripper calib: idle", text_color="gray"
        )
        self.gripper_calib_status.pack(pady=2)

        # Initial update
        self._update_corner_label(None)

    # ---------- Helper to update corner label ----------
    def _update_corner_label(self, _):
        selected_id = int(self.marker_var.get()) if self.marker_var.get().isdigit() else None
        if selected_id is not None and selected_id in config.manual_pixel_calib.marker_corners:
            corner_idx = config.manual_pixel_calib.marker_corners[selected_id]
            corner_names = {-1: "Center", 0: "Top-Left", 1: "Top-Right", 2: "Bottom-Right", 3: "Bottom-Left"}
            self.corner_label.configure(text=f"Corner: {corner_names.get(corner_idx, 'Unknown')}")
        else:
            self.corner_label.configure(text="Corner: -")

    # ---------- Record point (main camera) ----------
    def _record_point(self):
        if self.robot is None or self.calib is None:
            self.status_label.configure(text="Robot not available", text_color="red")
            return

        rx, ry, _ = self.robot.get_work_position()
        selected_id = int(self.marker_var.get())
        corner_idx = config.manual_pixel_calib.marker_corners.get(selected_id, -1)

        self.calib.commit_fixed_point(selected_id, corner_idx, rx, ry)
        self.status_label.configure(
            text=f"Recorded: Marker {selected_id} corner {corner_idx} at ({rx:.1f}, {ry:.1f})",
            text_color="lightgreen"
        )

    # ---------- Clear points ----------
    def _clear_points(self):
        if self.calib:
            self.calib.fixed_points = []
            self.calib._save_fixed_points()
            self.status_label.configure(text="All fixed points cleared", text_color="orange")

    # ---------- Gripper Camera Calibration ----------
    def _calibrate_gripper(self):
        """Trigger gripper camera calibration via orchestrator."""
        if self.orchestrator is None:
            return

        # Read selected ID from the existing dropdown
        try:
            marker_id = int(self.marker_var.get())
        except ValueError:
            self.gripper_calib_status.configure(text="Select a valid marker ID", text_color="red")
            return

        self.calib_gripper_btn.configure(state="disabled", text="Calibrating...")
        self.gripper_calib_status.configure(text="Calibrating...", text_color="orange")

        # Pass the selected ID to the orchestrator
        self.orchestrator.calibrate_gripper(marker_id=marker_id)