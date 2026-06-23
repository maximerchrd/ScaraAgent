# gui/panels/agent_panel.py
import customtkinter as ctk

class AgentPanel(ctk.CTkFrame):
    def __init__(self, parent, orchestrator=None):
        super().__init__(parent)
        ctk.CTkLabel(self, text="Agent prompt & response area").pack(padx=20, pady=20)
        self.orchestrator = orchestrator