# gui/app.py
"""
Main application window.
Arranges panels, sets up keyboard bindings, and processes the shared queue.
"""

import tkinter as tk
import customtkinter as ctk
import logging
import threading
import numpy as np

from config import config

from gui.styles import DARK_BG
from gui.panels.connection_panel import ConnectionPanel
from gui.panels.jog_panel import JogPanel
from gui.panels.status_panel import StatusPanel
from gui.widgets.arm_canvas import ArmCanvas
from robot.kinematics import joints_to_xyz
from gui.panels.camera_panel import CameraPanel

# Shared calibration logic (imported from calibration folder)
from calibration.utils import (
    collect_calibration_data,
    fit_homography,
    evaluate_homography,
    save_homography,
)

try:
    from gui.panels.agent_panel import AgentPanel
except ImportError:
    AgentPanel = None


# ================================================================
#  Camera calibration parameters (edit as needed)
# ================================================================
CAMERA_CALIB_GRID = [
    (350, -420, 10.0),
    (350, -350, 10.0),
    (350, -275, 10.0),
    (350, -200, 10.0),
    (350, -125, 10.0),
    (350,  -50, 10.0),
    (350,  50, 10.0),

    (390, -420, 10.0),
    (390, -350, 10.0),
    (390, -275, 10.0),
    (390, -200, 10.0),
    (390, -125, 10.0),
    (390,  -50, 10.0),
    (390,  50, 10.0),

    (465, -420, 10.0),
    (465, -350, 10.0),
    (465, -275, 10.0),
    (465, -200, 10.0),
    (465, -125, 10.0),
    (465,  -50, 10.0),
    (465,  50, 10.0),

    (520, -420, 10.0),
    (520, -350, 10.0),
    (520, -275, 10.0),
    (520, -200, 10.0),
    (520, -125, 10.0),
    (520,  -50, 10.0),
    (520,  50, 10.0),

    (580, -350, 10.0),
    (580, -275, 10.0),
    (580, -200, 10.0),
    (580, -125, 10.0),
    (580,  -50, 10.0),
    (580,  50, 10.0),

    (650, -200, 10.0),
    (650, -125, 10.0),
    (650,  -50, 10.0),
]

MARKER_OFFSET_ALONG = 64.0          # mm from TCP along second link to marker centre
CALIB_MARKER_ID = 5                 # ArUco ID of gripper marker
CALIB_SETTLE_TIME = 1.0             # seconds after each move
CALIB_MAX_RETRIES = 3
CALIB_OUTPUT_FILE = "calib_homography.npy"


class ScaraAgentApp(ctk.CTk):
    def __init__(self, queue, agent_queue, robot=None, camera=None, orchestrator=None, calibrator=None):
        super().__init__()
        self.queue = queue
        self.agent_queue = agent_queue
        self.robot = robot
        self.camera = camera
        self.orchestrator = orchestrator
        self.calibrator = calibrator

        self.title("SCARA Agent Control Dashboard")
        self.geometry("1450x850")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Configuration
        self.jog_step_linear = 50
        self.jog_step_yaw = 5
        self.current_yaw = 90  # updated by position messages

        # Internal state
        self.last_steps = None   # (z_steps, j1_steps, j2_steps)

        # ---------- Layout ----------
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=0)   # connection bar
        self.grid_rowconfigure(1, weight=1)   # main content
        self.grid_rowconfigure(2, weight=0)   # agent panel
        self.grid_rowconfigure(2, minsize=90)

        # --- Top: Connection ---
        self.conn_panel = ConnectionPanel(self, robot)
        self.conn_panel.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 0), sticky="ew")

        # --- Left: Jogging ---
        self.jog_panel = JogPanel(
            self, robot,
            step_linear=self.jog_step_linear,
            step_yaw=self.jog_step_yaw
        )
        self.jog_panel.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        # --- Center: Status ---
        self.status_panel = StatusPanel(self, robot)
        self.status_panel.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")

        # --- Right: Arm canvas + checkboxes + camera ---
        right_frame = ctk.CTkFrame(self, fg_color="transparent")
        right_frame.grid(row=1, column=2, padx=20, pady=20, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=0)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Arm canvas and toggles row
        arm_row = ctk.CTkFrame(right_frame, fg_color="transparent")
        arm_row.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        arm_row.grid_columnconfigure(0, weight=0)
        arm_row.grid_columnconfigure(1, weight=1)

        self.arm_canvas = ArmCanvas(arm_row, width=160, height=160, bg=DARK_BG)
        self.arm_canvas.grid(row=0, column=0)
        self.arm_canvas.update_joints(0, 0)

        toggle_frame = ctk.CTkFrame(arm_row, fg_color="transparent")
        toggle_frame.grid(row=0, column=1, padx=20, sticky="w")

        self.show_aruco = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(toggle_frame, text="Show ArUco", variable=self.show_aruco).pack(anchor="w", pady=3)

        self.show_vlm = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(toggle_frame, text="Show VLM Objects", variable=self.show_vlm).pack(anchor="w", pady=3)

        self.show_calibrated = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(toggle_frame, text="Show Calibrated Markers", variable=self.show_calibrated).pack(anchor="w", pady=3)

        # Marker calibration button (original)
        self.calib_btn = ctk.CTkButton(toggle_frame, text="Recalibrate", width=100,
                                       command=self._start_calibration)
        self.calib_btn.pack(anchor="w", pady=2)

        # Camera calibration button (new)
        self.calib_camera_btn = ctk.CTkButton(toggle_frame, text="Calibrate Camera", width=120,
                                              command=self._start_camera_calibration)
        self.calib_camera_btn.pack(anchor="w", pady=2)

        self.calib_status = ctk.CTkLabel(toggle_frame, text="Not calibrated", text_color="gray")
        self.calib_status.pack(anchor="w")

        self.camera_panel = CameraPanel(
            right_frame,
            camera=self.camera,
            queue=self.queue,
            width=640,
            height=360,
            show_aruco_var=self.show_aruco,
            show_vlm_var=self.show_vlm,
            calibrator=self.calibrator,
            show_calibrated_var=self.show_calibrated
        )
        self.camera_panel.grid(row=1, column=0, sticky="nsew")

        # --- Agent panel (bottom) ---
        if AgentPanel:
            self.agent_panel = AgentPanel(
                self,
                orchestrator=self.orchestrator,
                agent_queue=self.agent_queue
            )
            self.agent_panel.grid(row=2, column=0, columnspan=3, padx=20, pady=(0,10), sticky="ew")

        # --- Keyboard bindings (disabled by default) ---
        # self.bind_keyboard()

        # --- Start queue processing ---
        self.after(50, self.process_queue)

        # Clean exit
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ---------- Marker calibration (unchanged) ----------
    def _start_calibration(self):
        if not self.calibrator:
            return
        self.calib_btn.configure(state="disabled", text="Calibrating…")
        self.calib_status.configure(text="Capturing 100 frames…", text_color="orange")
        self.calibrator.calibrate(callback=self._on_calibration_done)

    def _on_calibration_done(self, success):
        self.calib_btn.configure(state="normal", text="Recalibrate")
        if success:
            self.calib_status.configure(text="Calibrated ✓", text_color="lightgreen")
        else:
            self.calib_status.configure(text="Calibration failed", text_color="red")

    # ---------- Camera calibration (new) ----------
    def _start_camera_calibration(self):
        if not self.robot or not self.camera:
            self.queue.put({"type": "agent_error", "data": "Robot or camera not available"})
            return
        self.calib_camera_btn.configure(state="disabled", text="Running…")
        thread = threading.Thread(target=self._run_camera_calibration, daemon=True)
        thread.start()

    def _run_camera_calibration(self):
        try:
            data = collect_calibration_data(
                self.robot,
                self.camera,
                CAMERA_CALIB_GRID,
                marker_offset_along=MARKER_OFFSET_ALONG,
                marker_id=CALIB_MARKER_ID,
                settle_time=CALIB_SETTLE_TIME,
                max_retries=CALIB_MAX_RETRIES
            )
            if data is None or len(data) < 4:
                self.queue.put({"type": "agent_error", "data": "Not enough calibration points"})
                return

            H, mask = fit_homography(data)
            mean_err, max_err, errors = evaluate_homography(H, data)
            save_homography(H, CALIB_OUTPUT_FILE)

            msg = f"Camera calibration done. Mean error: {mean_err:.2f} mm, Max: {max_err:.2f} mm"
            self.queue.put({"type": "calib_done", "data": msg})
            logging.info(msg)

        except Exception as e:
            logging.error(f"Camera calibration error: {e}")
            self.queue.put({"type": "agent_error", "data": f"Calibration failed: {e}"})
        finally:
            self.calib_camera_btn.configure(state="normal", text="Calibrate Camera")

    # ---------- Keyboard bindings (commented out, kept for reference) ----------
    def bind_keyboard(self):
        """Bind all hotkeys to the main window."""
        # Jogging (continuous for Z, A, B)
        self.bind("<KeyPress-w>", lambda e: self._jog_start("Z", "+"))
        self.bind("<KeyRelease-w>", lambda e: self._jog_stop("Z"))
        self.bind("<KeyPress-s>", lambda e: self._jog_start("Z", "-"))
        self.bind("<KeyRelease-s>", lambda e: self._jog_stop("Z"))
        self.bind("<KeyPress-a>", lambda e: self._jog_start("A", "-"))
        self.bind("<KeyRelease-a>", lambda e: self._jog_stop("A"))
        self.bind("<KeyPress-d>", lambda e: self._jog_start("A", "+"))
        self.bind("<KeyRelease-d>", lambda e: self._jog_stop("A"))
        self.bind("<KeyPress-Left>", lambda e: self._jog_start("B", "-"))
        self.bind("<KeyRelease-Left>", lambda e: self._jog_stop("B"))
        self.bind("<KeyPress-Right>", lambda e: self._jog_start("B", "+"))
        self.bind("<KeyRelease-Right>", lambda e: self._jog_stop("B"))

        # Yaw: discrete step
        self.bind("<i>", lambda e: self._jog_yaw(self.jog_step_yaw))
        self.bind("<k>", lambda e: self._jog_yaw(-self.jog_step_yaw))

        # Pitch continuous
        self.bind("<KeyPress-r>", lambda e: self._pitch_up())
        self.bind("<KeyRelease-r>", lambda e: self._pitch_stop())
        self.bind("<KeyPress-f>", lambda e: self._pitch_down())
        self.bind("<KeyRelease-f>", lambda e: self._pitch_stop())

        # Gripper
        self.bind("<q>", lambda e: self._gripper_open())
        self.bind("<e>", lambda e: self._gripper_close())

        # Emergency stop
        self.bind("<space>", lambda e: self._emergency_stop())

        # Cartesian move (Enter)
        self.bind("<Return>", lambda e: self.jog_panel.move_to_xyz())

    # ---------- Keyboard action helpers ----------
    def _jog_start(self, axis, direction):
        if self.robot:
            self.robot.send_raw(f"JOG_START {axis} {direction}")
        else:
            print(f"[Sim] JOG_START {axis} {direction}")

    def _jog_stop(self, axis):
        if self.robot:
            self.robot.send_raw(f"JOG_STOP {axis}")
        else:
            print(f"[Sim] JOG_STOP {axis}")

    def _jog_yaw(self, amount):
        if self.robot:
            new_angle = self.current_yaw + amount
            new_angle = max(0, min(180, new_angle))
            self.robot.send_raw(f"G0 Y{new_angle}")
        else:
            print(f"[Sim] Jog yaw by {amount}")

    def _pitch_up(self):
        self._send("PITCH_UP")
    def _pitch_down(self):
        self._send("PITCH_DOWN")
    def _pitch_stop(self):
        self._send("PITCH_STOP")

    def _gripper_open(self):
        if self.robot:
            self.robot.gripper_open()
        else:
            print("[Sim] Gripper open")

    def _gripper_close(self):
        if self.robot:
            self.robot.gripper_close()
        else:
            print("[Sim] Gripper close")

    def _emergency_stop(self):
        if self.robot:
            self.robot.estop()
        else:
            print("[Sim] EMERGENCY STOP")

    def _send(self, cmd):
        if self.robot:
            self.robot.send_raw(cmd)
        else:
            print(f"[Sim] {cmd}")

    # ---------- Queue processing ----------
    def process_queue(self):
        """Read all messages from the shared queue and update GUI panels."""
        while not self.queue.empty():
            try:
                msg = self.queue.get_nowait()
                msg_type = msg.get("type")
                data = msg.get("data")

                if msg_type == "robot_position":
                    self.handle_position_update(data)
                elif msg_type == "robot_switches":
                    self.status_panel.update_switches(data)
                elif msg_type == "robot_homing_complete":
                    self.status_panel.set_homing_status("complete")
                    if self.robot:
                        self.robot.reset_zero()
                        self.robot.move_to_xyz(
                            config.robot.park_x_mm,
                            config.robot.park_y_mm,
                            config.robot.park_z_mm
                        )
                elif msg_type == "robot_homing_aborted":
                    self.status_panel.set_homing_status("aborted")
                elif msg_type == "robot_estop":
                    self.status_panel.set_status_text("EMERGENCY STOP", "red")
                elif msg_type == "agent_response":
                    if hasattr(self, 'agent_panel') and self.agent_panel:
                        self.agent_panel.display_response(data)
                elif msg_type == "agent_error":
                    if hasattr(self, 'agent_panel') and self.agent_panel:
                        self.agent_panel.display_error(data)
                elif msg_type == "vlm_objects":
                    if hasattr(self, 'camera_panel') and self.camera_panel:
                        self.camera_panel.update_vlm_objects(data)
                elif msg_type == "agent_reasoning":
                    if hasattr(self, 'agent_panel') and self.agent_panel:
                        self.agent_panel.display_reasoning(data)
                elif msg_type == "calib_done":
                    if hasattr(self, 'agent_panel') and self.agent_panel:
                        self.agent_panel.display_response(data)
                    logging.info(f"GUI: {data}")
            except Exception as e:
                logging.error(f"Queue processing error: {e}")

        self.after(50, self.process_queue)

    def handle_position_update(self, data):
        """data is a list of strings: [z_steps, j1_steps, j2_steps, yaw_angle]"""
        if len(data) < 3:
            return

        steps_z = int(data[0])
        steps_j1 = int(data[1])
        steps_j2 = int(data[2])
        self.current_yaw = int(data[3]) if len(data) > 3 else 90

        if self.robot:
            self.robot._last_z = steps_z
            self.robot._last_j1 = steps_j1
            self.robot._last_j2 = steps_j2
            self.robot._last_yaw = self.current_yaw

        self.status_panel.update_positions(steps_z, steps_j1, steps_j2, self.current_yaw)
        self.last_steps = (steps_z, steps_j1, steps_j2)
        self.arm_canvas.update_joints(steps_j1, steps_j2)

        try:
            x_native, y_native, z_native = joints_to_xyz(steps_z, steps_j1, steps_j2)
            if self.robot and self.robot.has_zero:
                zero_mm = self.robot.zero_mm
                x_work = x_native - zero_mm[0]
                y_work = y_native - zero_mm[1]
                z_work = z_native - zero_mm[2]
            else:
                x_work, y_work, z_work = x_native, y_native, z_native
            self.status_panel.set_xyz(x_work, y_work, z_work)
        except Exception as e:
            logging.warning(f"Kinematics error: {e}")

    def on_closing(self):
        logging.info("Shutting down GUI...")
        if self.robot:
            self.robot.shutdown()
        self.destroy()