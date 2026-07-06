# gui/panels/jog_panel.py
"""Jog controls (buttons, speed slider) and Cartesian XYZ move."""

import customtkinter as ctk

class JogPanel(ctk.CTkFrame):
    def __init__(self, parent, robot, step_linear=50, step_yaw=5):
        super().__init__(parent, width=430)
        self.parent_app = parent
        self.robot = robot
        self.step_linear = step_linear
        self.step_yaw = step_yaw

        # Title
        ctk.CTkLabel(self, text="Live Jog Controls", font=("Arial", 16, "bold")).pack(pady=10)

        # Speed slider
        ctk.CTkLabel(self, text="Max Speed (Steps/sec)").pack()
        self.speed_slider = ctk.CTkSlider(self, from_=100, to=2000, command=self._on_speed_change)
        self.speed_slider.set(1000)
        self.speed_slider.pack(pady=5)

        # Jog buttons (grid inside self)
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(pady=10)

        self._create_jog_row("Z-Axis Height (W / S)", 'Z', 0)
        self._create_jog_row("Joint 1 Base (A / D)", 'A', 1)
        self._create_jog_row("Joint 2 Elbow (Left / Right)", 'B', 2)
        self._create_jog_row("Wrist Yaw Angle (I / K)", 'Y', 3)

        # Pitch continuous buttons
        row4 = ctk.CTkFrame(self.btn_frame, fg_color="transparent")
        row4.grid(row=4, column=0, columnspan=3, pady=5)
        ctk.CTkLabel(row4, text="Pitch Continuous (R / F)", width=180, anchor="w").pack(side="left", padx=10)
        self.pitch_down_btn = ctk.CTkButton(row4, text="-", width=45)
        self.pitch_down_btn.pack(side="left", padx=5)
        self.pitch_down_btn.bind("<ButtonPress-1>", lambda e: self.parent_app._pitch_down())
        self.pitch_down_btn.bind("<ButtonRelease-1>", lambda e: self.parent_app._pitch_stop())
        self.pitch_up_btn = ctk.CTkButton(row4, text="+", width=45)
        self.pitch_up_btn.pack(side="left", padx=5)
        self.pitch_up_btn.bind("<ButtonPress-1>", lambda e: self.parent_app._pitch_up())
        self.pitch_up_btn.bind("<ButtonRelease-1>", lambda e: self.parent_app._pitch_stop())

        # Gripper – compact row
        gripper_row = ctk.CTkFrame(self.btn_frame, fg_color="transparent")
        gripper_row.grid(row=5, column=0, columnspan=3, pady=5)
        ctk.CTkLabel(gripper_row, text="Gripper", width=180, anchor="w").pack(side="left", padx=10)
        ctk.CTkButton(gripper_row, text="Open", width=45,
                      fg_color="#E67E22", hover_color="#D35400",
                      command=self.parent_app._gripper_open).pack(side="left", padx=5)
        ctk.CTkButton(gripper_row, text="Close", width=45,
                      fg_color="#2C3E50", hover_color="#1A252F",
                      command=self.parent_app._gripper_close).pack(side="left", padx=5)

        #ctk.CTkLabel(self, text="Keyboard shortcuts active when window focused.",
        #            text_color="gray", font=("Arial", 10)).pack(side="bottom", pady=10)

        # Cartesian Move
        self._create_cartesian_move()

    def _create_jog_row(self, label, axis, row):
        """Create a row with two jog buttons ( - , + ) and a label."""
        frame = ctk.CTkFrame(self.btn_frame, fg_color="transparent")
        frame.grid(row=row, column=0, columnspan=3, pady=5)

        ctk.CTkLabel(frame, text=label, width=180, anchor="w").pack(side="left", padx=10)

        btn_minus = ctk.CTkButton(frame, text="-", width=45)
        btn_minus.pack(side="left", padx=5)
        btn_plus = ctk.CTkButton(frame, text="+", width=45)
        btn_plus.pack(side="left", padx=5)

        if axis in ['Z', 'A', 'B']:
            # Continuous jog
            btn_minus.bind("<ButtonPress-1>", lambda e, ax=axis: self.parent_app._jog_start(ax, "-"))
            btn_minus.bind("<ButtonRelease-1>", lambda e, ax=axis: self.parent_app._jog_stop(ax))
            btn_plus.bind("<ButtonPress-1>", lambda e, ax=axis: self.parent_app._jog_start(ax, "+"))
            btn_plus.bind("<ButtonRelease-1>", lambda e, ax=axis: self.parent_app._jog_stop(ax))
        else:  # Yaw
            btn_minus.configure(command=lambda: self.parent_app._jog_yaw(-self.step_yaw))
            btn_plus.configure(command=lambda: self.parent_app._jog_yaw(self.step_yaw))

    def _create_cartesian_move(self):
        cart_frame = ctk.CTkFrame(self)
        cart_frame.pack(pady=(10, 5), fill="x", padx=10)

        ctk.CTkLabel(cart_frame, text="Cartesian Move (XYZ)", font=("Arial", 13, "bold")).pack(pady=5)

        # X
        row_x = ctk.CTkFrame(cart_frame, fg_color="transparent")
        row_x.pack(pady=2)
        ctk.CTkLabel(row_x, text="X (mm):", width=60).pack(side="left")
        self.entry_x = ctk.CTkEntry(row_x, width=80)
        self.entry_x.pack(side="left", padx=5)
        self.entry_x.insert(0, "0.0")

        # Y
        row_y = ctk.CTkFrame(cart_frame, fg_color="transparent")
        row_y.pack(pady=2)
        ctk.CTkLabel(row_y, text="Y (mm):", width=60).pack(side="left")
        self.entry_y = ctk.CTkEntry(row_y, width=80)
        self.entry_y.pack(side="left", padx=5)
        self.entry_y.insert(0, "0.0")

        # Z
        row_z = ctk.CTkFrame(cart_frame, fg_color="transparent")
        row_z.pack(pady=2)
        ctk.CTkLabel(row_z, text="Z (mm):", width=60).pack(side="left")
        self.entry_z = ctk.CTkEntry(row_z, width=80)
        self.entry_z.pack(side="left", padx=5)
        self.entry_z.insert(0, "0.0")

        self.btn_move = ctk.CTkButton(cart_frame, text="Move to XYZ", command=self.move_to_xyz)
        self.btn_move.pack(pady=5)

    def _on_speed_change(self, val):
        speed = int(float(val))
        if self.robot:
            self.robot.set_speed(speed)
        else:
            print(f"[Sim] Speed set to {speed}")

    def move_to_xyz(self):
        try:
            x_mm = float(self.entry_x.get())
            y_mm = float(self.entry_y.get())
            z_mm = float(self.entry_z.get())
        except ValueError:
            print("Invalid XYZ input")
            return

        if self.robot:
            self.robot.move_to_xyz(x_mm, y_mm, z_mm)
        else:
            # In simulation, just print and simulate update
            print(f"[Sim] Move to X={x_mm} Y={y_mm} Z={z_mm}")
            # If kinematics import available, we could compute steps and simulate position update
            try:
                from robot.kinematics import xyz_to_joints
                steps_z, steps_j1, steps_j2 = xyz_to_joints(x_mm, y_mm, z_mm)
                self.parent_app.handle_position_update([steps_z, steps_j1, steps_j2, self.parent_app.current_yaw])
            except ImportError:
                pass