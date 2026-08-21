# gui/panels/status_panel.py
"""Displays live axis positions, endswitch states, homing/estop, zeroing."""

import customtkinter as ctk
import logging
import threading
import time
import numpy as np

class StatusPanel(ctk.CTkFrame):
    def __init__(self, parent, robot=None):
        super().__init__(parent, width=430)
        self.parent_app = parent
        self.robot = robot

        # Homing & E‑Stop
        ctk.CTkButton(self, text="🏠 HOME ALL AXES (G28)", font=("Arial", 13, "bold"),
                       fg_color="#D35400", hover_color="#A04000", height=40,
                       command=self._home).pack(pady=15, fill="x", padx=20)

                # Add Z calibration button (after the homing button, before estop)
        self.z_calib_btn = ctk.CTkButton(
            self,
            text="📏 Calibrate Z",
            font=("Arial", 13, "bold"),
            fg_color="#2980B9",
            hover_color="#1F618D",
            command=self._start_z_calibration
        )
        self.z_calib_btn.pack(pady=5, fill="x", padx=20)

        self.z_calib_status = ctk.CTkLabel(
            self, text="Z calib: idle", text_color="gray"
        )
        self.z_calib_status.pack(pady=2)

        # Status text line
        self.status_text = ctk.CTkLabel(self, text="Ready", font=("Arial", 12), text_color="lightgreen")
        self.status_text.pack(pady=5)

        # Live Axis Positions
        ctk.CTkLabel(self, text="Live Axis Positions", font=("Arial", 16, "bold")).pack(pady=5)

        # XYZ label (global, updated from app)
        self.xyz_label = ctk.CTkLabel(self, text="X: 0.0 mm   Y: 0.0 mm   Z: 0.0 mm",
                                      font=("Courier New", 14, "bold"), text_color="#2ECC71")
        self.xyz_label.pack(pady=5)

        # Distance label
        self.distance_label = ctk.CTkLabel(
            self,
            text="Distance: -- mm",
            font=("Courier New", 14, "bold"),
            text_color="#2ECC71"
        )
        self.distance_label.pack(pady=5)

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

    def _start_z_calibration(self):
        if not self.robot:
            self.z_calib_status.configure(text="Robot not connected", text_color="red")
            return
        self.z_calib_btn.configure(state="disabled", text="Running…")
        t = threading.Thread(target=self._run_z_calibration, daemon=True)
        t.start()

    def _run_z_calibration(self):
        try:
            data = self._collect_z_data()
            if len(data) < 6:
                self.z_calib_status.configure(text="Not enough valid points", text_color="red")
                return

            coeffs = self._fit_polynomial(data)
            a, b, c, d, e, f = coeffs

            print("\n=== Fitted Z-correction coefficients ===")
            print(f"a = {a:.6f}")
            print(f"b = {b:.6f}")
            print(f"c = {c:.6f}")
            print(f"d = {d:.6f}")
            print(f"e = {e:.6f}")
            print(f"f = {f:.6f}")
            print("\nCopy this into config.py:")
            print(f"({a:.6f}, {b:.6f}, {c:.6f}, {d:.6f}, {e:.6f}, {f:.6f})")

            # Save coefficients to config (or print them)
            from config import config
            config.robot.z_correction_coeffs = (a, b, c, d, e, f)
            logging.info(f"New Z coefficients: {coeffs}")

            self.z_calib_status.configure(
                text=f"Z done. Mean err: {self._mean_error(data, coeffs):.2f} mm",
                text_color="lightgreen"
            )
        except Exception as e:
            logging.error(f"Z calibration error: {e}")
            self.z_calib_status.configure(text="Z calibration failed", text_color="red")
        finally:
            self.z_calib_btn.configure(state="normal", text="📏 Calibrate Z")

    def _collect_z_data(self):
        # Same grid as CAMERA_CALIB_GRID (you can import it)
        grid_points = [
            (350, -420), (350, -350), (350, -275), (350, -200), (350, -125), (350, -50), (350, 50),
            (390, -420), (390, -350), (390, -275), (390, -200), (390, -125), (390, -50), (390, 50),
            (465, -420), (465, -350), (465, -275), (465, -200), (465, -125), (465, -50), (465, 50),
            (520, -420), (520, -350), (520, -275), (520, -200), (520, -125), (520, -50), (520, 50),
            (580, -350), (580, -275), (580, -200), (580, -125), (580, -50), (580, 50),
            (650, -200), (650, -125), (650, -50),
        ]

        S0 = 269.0
        Z_COMMAND = 10.0
        SETTLE_TIME = 10.0

        data = []
        # Move to safe Z first
        cur_x, cur_y, cur_z = self.robot.get_work_position()
        self.robot.move_to_xyz(cur_x, cur_y, 60.0, use_z_correction=False, block=False)
        time.sleep(3.0)

        for i, (x, y) in enumerate(grid_points):
            self.z_calib_status.configure(text=f"Point {i+1}/{len(grid_points)}")
            logging.info(f"Z calib point {i+1}: X={x} Y={y}")

            self.robot.move_to_xyz(x, y, Z_COMMAND, use_z_correction=False, block=False)
            time.sleep(3.0)          # motion wait (adjust)
            time.sleep(SETTLE_TIME)

            distance = self.robot.get_distance_mm()
            if distance is None:
                logging.warning(f"No distance at X={x} Y={y}")
                continue

            z_touch = Z_COMMAND - (distance - S0)
            data.append((x, y, z_touch))
            logging.info(f"  distance={distance:.1f}, z_touch={z_touch:.3f}")

        return data

    def _fit_polynomial(self, data):
        X = np.array([d[0] for d in data])
        Y = np.array([d[1] for d in data])
        Z = np.array([d[2] for d in data])
        A = np.column_stack([np.ones_like(X), X, Y, X**2, Y**2, X*Y])
        coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        return coeffs

    def _mean_error(self, data, coeffs):
        errors = []
        for x, y, z in data:
            z_fit = coeffs[0] + coeffs[1]*x + coeffs[2]*y + coeffs[3]*x**2 + coeffs[4]*y**2 + coeffs[5]*x*y
            errors.append(abs(z - z_fit))
        return np.mean(errors)
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

    def update_distance(self, distance_mm):
        if distance_mm is None:
            self.distance_label.configure(text="Distance: -- mm")
        else:
            self.distance_label.configure(text=f"Distance: {distance_mm:.1f} mm")

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