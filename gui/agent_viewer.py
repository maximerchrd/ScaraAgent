# gui/agent_viewer.py
import customtkinter as ctk
from gui.panels.camera_panel import CameraPanel
from gui.panels.agent_panel import AgentPanel

class AgentViewerWindow(ctk.CTkToplevel):
    def __init__(self, master, camera, orchestrator, agent_queue, queue,
                 show_aruco_var, show_vlm_var, show_calibrated_var, calibrator=None):
        super().__init__(master)
        self.title("Agent Viewer")
        self.geometry("900x750")
        self.minsize(600, 500)

        # Make it stay on top? Optional – user can toggle.
        # self.attributes('-topmost', True)

        # Configure grid
        self.grid_rowconfigure(0, weight=1)   # camera
        self.grid_rowconfigure(1, weight=0)   # agent panel (fixed height)
        self.grid_rowconfigure(2, weight=0)   # close button
        self.grid_columnconfigure(0, weight=1)

        # Camera panel (same as main window)
        self.camera_panel = CameraPanel(
            self,
            camera=camera,
            queue=queue,
            width=640,
            height=480,
            show_aruco_var=show_aruco_var,
            show_vlm_var=show_vlm_var,
            calibrator=calibrator,
            show_calibrated_var=show_calibrated_var
        )
        self.camera_panel.grid(row=0, column=0, padx=10, pady=(10,5), sticky="nsew")

        # Agent panel (same as main window)
        self.agent_panel = AgentPanel(
            self,
            orchestrator=orchestrator,
            agent_queue=agent_queue
        )
        self.agent_panel.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Close button
        self.close_btn = ctk.CTkButton(self, text="Close Viewer", command=self.destroy)
        self.close_btn.grid(row=2, column=0, pady=10)

        # If the main window is closed, close this too
        self.protocol("WM_DELETE_WINDOW", self.destroy)