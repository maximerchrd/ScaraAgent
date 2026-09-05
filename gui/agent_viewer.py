# gui/agent_viewer.py
import customtkinter as ctk
from gui.panels.camera_panel import CameraPanel
from gui.panels.agent_panel import AgentPanel

class AgentViewerWindow(ctk.CTkToplevel):
    def __init__(self, master, camera, orchestrator, agent_queue, queue,
                 show_aruco_var, show_vlm_var, show_calibrated_var, calibrator=None):
        super().__init__(master)
        self.title("Agent Viewer - Main & Gripper Cameras")
        self.geometry("1200x750")
        self.minsize(900, 600)

        self.orchestrator = orchestrator
        self.gripper_cam = orchestrator.gripper_cam if orchestrator else None

        # Grid: 2 columns for cameras, 1 row for agent panel
        self.grid_rowconfigure(0, weight=1)   # cameras
        self.grid_rowconfigure(1, weight=0)   # agent panel
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ------ Main camera panel (left) ------
        self.main_camera_panel = CameraPanel(
            self,
            camera=camera,
            queue=queue,
            width=480,
            height=360,
            show_aruco_var=show_aruco_var,
            show_vlm_var=show_vlm_var,
            calibrator=calibrator,
            show_calibrated_var=show_calibrated_var
        )
        self.main_camera_panel.grid(row=0, column=0, padx=10, pady=(10,5), sticky="nsew")

        # ------ Gripper camera panel (right) ------
        # We pass camera=None and override the frame source
        self.gripper_camera_panel = CameraPanel(
            self,
            camera=None,          # no internal camera thread
            queue=queue,
            width=480,
            height=360,
            show_aruco_var=show_aruco_var,
            show_vlm_var=show_vlm_var,
            calibrator=None,
            show_calibrated_var=show_calibrated_var
        )
        self.gripper_camera_panel.grid(row=0, column=1, padx=10, pady=(10,5), sticky="nsew")

        # Override the polling method to use the gripper camera
        self.gripper_camera_panel._poll_frame = self._poll_gripper_frame

        # ------ Agent panel (bottom, spans both columns) ------
        self.agent_panel = AgentPanel(
            self,
            orchestrator=orchestrator,
            agent_queue=agent_queue
        )
        self.agent_panel.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        # ------ Close button ------
        self.close_btn = ctk.CTkButton(self, text="Close Viewer", command=self.destroy)
        self.close_btn.grid(row=2, column=0, columnspan=2, pady=10)

        self.protocol("WM_DELETE_WINDOW", self.destroy)

        # Start polling the gripper camera
        self._poll_gripper_frame()

    def _poll_gripper_frame(self):
        """Get a frame from the gripper camera and display it."""
        if self.gripper_cam:
            frame = self.gripper_cam.get_frame()
            if frame is not None:
                # Display the frame in the gripper panel
                self.gripper_camera_panel._display_frame(frame)
        # Keep polling
        self.after(50, self._poll_gripper_frame)