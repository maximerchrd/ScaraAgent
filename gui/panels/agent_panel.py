# gui/panels/agent_panel.py
import customtkinter as ctk
import tkinter as tk

class AgentPanel(ctk.CTkFrame):
    def __init__(self, parent, orchestrator=None, agent_queue=None):
        super().__init__(parent, height=120)  # increased height for log box
        self.orchestrator = orchestrator
        self.agent_queue = agent_queue

        # Top row: prompt entry + submit + status
        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.pack(fill="x", padx=10, pady=(5,0))

        self.prompt_entry = ctk.CTkEntry(top_row, width=300, placeholder_text="Describe the task...")
        self.prompt_entry.pack(side="left", padx=5)

        self.submit_btn = ctk.CTkButton(
            top_row, text="Submit Task", command=self._submit_task,
            fg_color="#2ECC71", hover_color="#27AE60", width=100
        )
        self.submit_btn.pack(side="left", padx=5)

        self.status_label = ctk.CTkLabel(top_row, text="Idle", text_color="gray", width=80)
        self.status_label.pack(side="left", padx=5)

        # Bottom: scrollable log for LLM responses
        self.log_text = ctk.CTkTextbox(self, height=80, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(5,10))
        self.log_text.insert("end", "Agent ready.\n")
        self.log_text.configure(state="disabled")  # read-only

    def _submit_task(self):
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            return
        if self.orchestrator:
            self.orchestrator.submit_task(prompt)
            self.status_label.configure(text="Processing...", text_color="orange")
            self._append_log(f"Task: {prompt}")
            self.prompt_entry.delete(0, "end")
        else:
            self._append_log("Error: Orchestrator not available")

    def display_response(self, text):
        self.status_label.configure(text="Done", text_color="lightgreen")
        self._append_log(f"Response: {text}")

    def display_error(self, text):
        self.status_label.configure(text="Error", text_color="red")
        self._append_log(f"Error: {text}")

    def _append_log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")      # auto-scroll
        self.log_text.configure(state="disabled")

    def display_reasoning(self, text):
        self._append_log(f"🧠 Reasoning: {text}")