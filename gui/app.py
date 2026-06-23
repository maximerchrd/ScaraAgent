# gui/app.py
"""
Main application window.
Arranges panels, sets up keyboard bindings, and processes the shared queue.
"""

import tkinter as tk
import customtkinter as ctk
import logging

from gui.styles import DARK_BG
from gui.panels.connection_panel import ConnectionPanel
from gui.panels.jog_panel import JogPanel
from gui.panels.status_panel import StatusPanel
from gui.widgets.arm_canvas import ArmCanvas
from robot.kinematics import joints_to_xyz
from gui.panels.camera_panel import CameraPanel

try:
    from gui.panels.agent_panel import AgentPanel
except ImportError:
    AgentPanel = None


class ScaraAgentApp(ctk.CTk):
    def __init__(self, queue, robot=None, camera=None, orchestrator=None):
        super().__init__()
        self.queue = queue
        self.robot = robot
        self.camera = camera
        self.orchestrator = orchestrator

        self.title("SCARA Agent Control Dashboard")
        self.geometry("1450x850")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Configuration
        self.jog_step_linear = 50
        self.jog_step_yaw = 5
        self.current_yaw = 90  # will be updated from position messages

        # Internal state
        self.last_steps = None   # (z_steps, j1_steps, j2_steps)

                # ---------- Layout ----------
        # Three columns: left (jog), center (status), right (arm + camera)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=1)         # right side expands
        self.grid_rowconfigure(0, weight=0)             # connection bar
        self.grid_rowconfigure(1, weight=1)             # main content fills rest

        # --- Top: Connection ---
        self.conn_panel = ConnectionPanel(self, robot)
        self.conn_panel.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 0), sticky="ew")

        # --- Left: Jogging ---
        self.jog_panel = JogPanel(
            self,
            robot,
            step_linear=self.jog_step_linear,
            step_yaw=self.jog_step_yaw
        )
        self.jog_panel.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        # --- Center: Status ---
        self.status_panel = StatusPanel(self, robot)
        self.status_panel.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")

        # --- Right: Arm canvas + camera stacked vertically ---
        right_frame = ctk.CTkFrame(self, fg_color="transparent")
        right_frame.grid(row=1, column=2, padx=20, pady=20, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=0)      # arm canvas (fixed height)
        right_frame.grid_rowconfigure(1, weight=1)      # camera fills remaining space
        right_frame.grid_columnconfigure(0, weight=1)

        self.arm_canvas = ArmCanvas(right_frame, width=240, height=240, bg=DARK_BG)
        self.arm_canvas.grid(row=0, column=0, pady=(0, 10))

        self.camera_panel = CameraPanel(
            right_frame,
            camera=self.camera,
            queue=self.queue,
            width=640,
            height=360            # smaller to fit below the arm canvas
        )
        self.camera_panel.grid(row=1, column=0, sticky="nsew")

        # --- Agent ---
        

        # --- Keyboard bindings ---
        self.bind_keyboard()

        # --- Start queue processing loop ---
        self.after(50, self.process_queue)

        # Clean exit
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

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

        # Cartesian move (Enter in XYZ entry fields)
        # We'll bind to the Entry widgets via the jog panel; see jog_panel.py
        # Also bind <Return> globally for convenience:
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
                    # Reset zero offsets because homing changes the coordinate system
                    if self.robot:
                        self.robot.reset_zero()
                elif msg_type == "robot_homing_aborted":
                    self.status_panel.set_homing_status("aborted")
                elif msg_type == "robot_estop":
                    self.status_panel.set_status_text("EMERGENCY STOP", "red")
                # Future message types (camera frame, agent output) will be handled here
            except Exception as e:
                logging.error(f"Queue processing error: {e}")

        # Continue polling
        self.after(50, self.process_queue)

    def handle_position_update(self, data):
        """data is a list of strings: [z_steps, j1_steps, j2_steps, yaw_angle]"""
        if len(data) < 3:
            return

        # Convert to ints
        steps_z = int(data[0])
        steps_j1 = int(data[1])
        steps_j2 = int(data[2])
        self.current_yaw = int(data[3]) if len(data) > 3 else 90

        # Update status panel labels (relative steps)
        self.status_panel.update_positions(steps_z, steps_j1, steps_j2, self.current_yaw)

        # Store for zero / measurement
        self.last_steps = (steps_z, steps_j1, steps_j2)

        # Update arm graphic (native angles)
        self.arm_canvas.update_joints(steps_j1, steps_j2)

        # Update XYZ label
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