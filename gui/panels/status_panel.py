# gui/panels/status_panel.py
"""Displays live axis positions, endswitch states, homing/estop, zeroing."""

import customtkinter as ctk
import logging

class StatusPanel(ctk.CTkFrame):
    def __init__(self, parent, robot=None):
        super().__init__(parent, width=430)
        self.parent_app = parent
        self.robot = robot

        # Homing & E‑Stop
        ctk.CTkButton(self, text="🏠 HOME ALL AXES (G28)", font=("Arial", 13, "bold"),
                       fg_color="#D35400", hover_color="#A04000", height=40,
                       command=self._home).pack(pady=15, fill="x", padx=20)

        ctk.CTkButton(self, text="🛑 EMERGENCY STOP", font=("Arial", 13, "bold"),
                       fg_color="#C0392B", hover_color="#922B21", height=40,
                       command=self.parent_app._emergency_stop).pack(pady=10, fill="x", padx=20)

        # Status text line
        self.status_text = ctk.CTkLabel(self, text="Ready", font=("Arial", 12), text_color="lightgreen")
        self.status_text.pack(pady=5)

        # Live Axis Positions
        ctk.CTkLabel(self, text="Live Axis Positions", font=("Arial", 16, "bold")).pack(pady=5)

        # XYZ label (global, updated from app)
        self.xyz_label = ctk.CTkLabel(self, text="X: 0.0 mm   Y: 0.0 mm   Z: 0.0 mm",
                                      font=("Courier New", 14, "bold"), text_color="#2ECC71")
        self.xyz_label.pack(pady=5)

        # Individual axes
        self.pos_frame = ctk.CTkFrame(self, fg_color="#1E272C")
        self.pos_frame.pack(pady=5, fill="x", padx=20)

        self.pos_labels = {}
        axes_units = [("Z-Axis", "steps"), ("Joint 1", "steps"), ("Joint 2", "steps"), ("Wrist Yaw", "°")]
        for idx, (axis, unit) in enumerate(axes_units):
            lbl = ctk.CTkLabel(self.pos_frame, text=f"{axis}: 0 {unit}",
                               font=("Courier New", 14, "bold"), text_color="#3498DB")
            lbl.grid(row=idx // 2, column=idx % 2, padx=25, pady=12, sticky="w")
            self.pos_labels[axis] = (lbl, unit)

        # Endswitches
        ctk.CTkLabel(self, text="Hardware Endswitches", font=("Arial", 16, "bold")).pack(pady=15)
        self.sw_frame = ctk.CTkFrame(self)
        self.sw_frame.pack(pady=5, fill="x", padx=20)

        self.sw_labels = {}
        for idx, axis in enumerate(["Pitch", "Z-Axis", "Joint 1", "Joint 2"]):
            lbl = ctk.CTkLabel(self.sw_frame, text=f"⚪ {axis} (Open)", text_color="white")
            lbl.grid(row=idx // 2, column=idx % 2, padx=25, pady=10, sticky="w")
            self.sw_labels[axis] = lbl

        # Zero setting
        ctk.CTkButton(self, text="📍 Set Current as Zero", command=self._set_zero).pack(pady=10, fill="x", padx=20)

        # Record measurement
        ctk.CTkButton(self, text="📝 Record Measurement", command=self._record_measurement).pack(pady=5)

    def update_positions(self, steps_z, steps_j1, steps_j2, yaw):
        """Update relative step displays; zero offsets are handled by robot."""
        rel_z = steps_z
        rel_j1 = steps_j1
        rel_j2 = steps_j2
        if self.robot:
            rel_z -= self.robot.zero_offset_z
            rel_j1 -= self.robot.zero_offset_j1
            rel_j2 -= self.robot.zero_offset_j2

        self.pos_labels["Z-Axis"][0].configure(text=f"Z-Axis: {rel_z} steps")
        self.pos_labels["Joint 1"][0].configure(text=f"Joint 1: {rel_j1} steps")
        self.pos_labels["Joint 2"][0].configure(text=f"Joint 2: {rel_j2} steps")
        self.pos_labels["Wrist Yaw"][0].configure(text=f"Wrist Yaw: {yaw}°")

    def set_xyz(self, x, y, z):
        self.xyz_label.configure(text=f"X: {x:.1f} mm   Y: {y:.1f} mm   Z: {z:.1f} mm")

    def update_switches(self, states):
        axes = ["Pitch", "Z-Axis", "Joint 1", "Joint 2"]
        for idx, state in enumerate(states):
            if idx >= len(axes): break
            axis = axes[idx]
            if state == 1:
                self.sw_labels[axis].configure(text=f"🔴 {axis} (TRIGGERED)", text_color="red")
            else:
                self.sw_labels[axis].configure(text=f"🟢 {axis} (Open)", text_color="lightgreen")

    def set_homing_status(self, status):
        if status == "complete":
            self.status_text.configure(text="Homing Complete", text_color="lightgreen")
        elif status == "aborted":
            self.status_text.configure(text="Homing Aborted", text_color="orange")

    def set_status_text(self, text, color="white"):
        self.status_text.configure(text=text, text_color=color)

    def _home(self):
        if self.robot:
            self.robot.home()
        else:
            print("[Sim] Homing all axes")
            self.set_homing_status("complete")

    def _set_zero(self):
        if self.robot:
            self.robot.set_zero()
            self.status_text.configure(text="Zero position set")
        else:
            # In simulation, just store current steps as zero
            if hasattr(self.parent_app, 'last_steps') and self.parent_app.last_steps:
                steps_z, steps_j1, steps_j2 = self.parent_app.last_steps
                # create a dummy zero if not present
                self.parent_app.robot.zero_offset_z = steps_z
                self.parent_app.robot.zero_offset_j1 = steps_j1
                self.parent_app.robot.zero_offset_j2 = steps_j2
                self.parent_app.robot.has_zero = True
                print("[Sim] Zero set")

    def _record_measurement(self):
        """Simplistic CSV recording; real version will use the calibrator."""
        import tkinter.simpledialog as sd
        x_str = sd.askstring("Measurement", "Enter measured X (mm):")
        y_str = sd.askstring("Measurement", "Enter measured Y (mm):")
        z_str = sd.askstring("Measurement", "Enter measured Z (mm):")
        if not (x_str and y_str and z_str):
            return
        try:
            x_meas = float(x_str)
            y_meas = float(y_str)
            z_meas = float(z_str)
        except ValueError:
            return

        if not hasattr(self.parent_app, 'last_steps') or self.parent_app.last_steps is None:
            return
        steps_z, steps_j1, steps_j2 = self.parent_app.last_steps
        with open("measurements.csv", "a") as f:
            f.write(f"{steps_j1},{steps_j2},{steps_z},{x_meas},{y_meas},{z_meas}\n")
        print("Measurement recorded")