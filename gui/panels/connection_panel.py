# gui/panels/connection_panel.py
"""Serial port selection and connection controls."""

import customtkinter as ctk
import serial.tools.list_ports
import logging

class ConnectionPanel(ctk.CTkFrame):
    def __init__(self, parent, robot=None):
        super().__init__(parent, height=60)
        self.robot = robot
        self.parent_app = parent  # reference to the main app for callbacks

        # Port dropdown
        self.port_var = ctk.StringVar()
        self.port_dropdown = ctk.CTkOptionMenu(self, variable=self.port_var, values=["No Ports Found"])
        self.port_dropdown.pack(side="left", padx=10, pady=10)

        # Refresh button
        self.refresh_btn = ctk.CTkButton(self, text="🔄 Refresh", width=80, command=self.refresh_ports)
        self.refresh_btn.pack(side="left", padx=5)

        # Connect / Disconnect button
        self.connect_btn = ctk.CTkButton(self, text="Connect", width=100, fg_color="green",
                                         hover_color="darkgreen", command=self.toggle_connection)
        self.connect_btn.pack(side="left", padx=5)

        # Status label
        self.status_label = ctk.CTkLabel(self, text="Disconnected", text_color="yellow")
        self.status_label.pack(side="right", padx=15)

        self.refresh_ports()

    def refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()][::-1]
        if ports:
            self.port_dropdown.configure(values=ports)
            self.port_var.set(ports[0])
        else:
            self.port_dropdown.configure(values=["No Ports Found"])
            self.port_var.set("No Ports Found")

    def toggle_connection(self):
        if self.robot and self.robot.is_connected():
            # Disconnect
            self.robot.disconnect()
            self.connect_btn.configure(text="Connect", fg_color="green")
            self.status_label.configure(text="Disconnected", text_color="yellow")
        else:
            port = self.port_var.get()
            if not port or port == "No Ports Found":
                return
            if self.robot:
                try:
                    self.robot.connect(port)
                    self.connect_btn.configure(text="Disconnect", fg_color="red")
                    self.status_label.configure(text="Connected!", text_color="lightgreen")
                    # Send initial speed
                    self.robot.set_speed(self.parent_app.jog_panel.speed_slider.get())
                except Exception as e:
                    logging.error(f"Connection failed: {e}")
                    self.status_label.configure(text="Connection Error", text_color="red")
            else:
                # Simulation mode: pretend to connect
                self.connect_btn.configure(text="Disconnect", fg_color="red")
                self.status_label.configure(text="Simulated", text_color="lightgreen")