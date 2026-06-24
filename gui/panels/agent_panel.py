# gui/panels/agent_panel.py
import customtkinter as ctk

class AgentPanel(ctk.CTkFrame):
    def __init__(self, parent, orchestrator=None, queue=None):
        super().__init__(parent, height=50)
        self.orchestrator = orchestrator
        self.queue = queue

        # Prompt entry
        self.prompt_entry = ctk.CTkEntry(self, width=300, placeholder_text="Describe the task...")
        self.prompt_entry.pack(side="left", padx=10, pady=10)

        # Submit button
        self.submit_btn = ctk.CTkButton(
            self, text="Submit Task", command=self._submit_task,
            fg_color="#2ECC71", hover_color="#27AE60", width=100
        )
        self.submit_btn.pack(side="left", padx=5)

        # Status label
        self.status_label = ctk.CTkLabel(self, text="Idle", text_color="gray", width=120)
        self.status_label.pack(side="left", padx=10)

        # Response label (shows latest result inline, truncated)
        self.response_label = ctk.CTkLabel(self, text="Ready for tasks", text_color="gray", width=300, anchor="w")
        self.response_label.pack(side="left", padx=10)

    def _submit_task(self):
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            return
        if self.orchestrator:
            self.orchestrator.submit_task(prompt)
            self.status_label.configure(text="Processing...", text_color="orange")
            self.prompt_entry.delete(0, "end")
        else:
            self.response_label.configure(text="Error: Orchestrator not available", text_color="red")

    def display_response(self, text):
        self.status_label.configure(text="Done", text_color="lightgreen")
        # Show first 80 chars of response inline
        short = text[:80] + "..." if len(text) > 80 else text
        self.response_label.configure(text=short, text_color="lightgreen")

    def display_error(self, text):
        self.status_label.configure(text="Error", text_color="red")
        short = text[:80] + "..." if len(text) > 80 else text
        self.response_label.configure(text=short, text_color="red")