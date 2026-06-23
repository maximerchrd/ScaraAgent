# gui/widgets/arm_canvas.py
"""Canvas widget that draws the SCARA arm in top‑down view."""

import tkinter as tk
import math

# Kinematics constants (should be taken from config later)
try:
    from robot.kinematics import STEPS_PER_DEG_J1, STEPS_PER_DEG_J2, L1, L2
except ImportError:
    # fallback constants
    STEPS_PER_DEG_J1 = 11.3778
    STEPS_PER_DEG_J2 = 11.3778
    L1 = 200.0
    L2 = 150.0

class ArmCanvas(tk.Canvas):
    def __init__(self, parent, width=400, height=400, bg="#1E272C"):
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0)
        self.width = width
        self.height = height

    def update_joints(self, steps_j1, steps_j2):
        """Redraw the arm using native joint angles."""
        self.delete("all")

        # Convert steps to degrees
        deg1 = steps_j1 / STEPS_PER_DEG_J1
        deg2 = steps_j2 / STEPS_PER_DEG_J2
        theta1 = math.radians(deg1)
        theta2 = math.radians(deg2)

        max_reach = L1 + L2
        scale = (self.width * 0.4) / max_reach
        cx = self.width / 2
        cy = self.height / 2

        x0, y0 = cx, cy
        x1 = x0 + scale * L1 * math.cos(theta1)
        y1 = y0 - scale * L1 * math.sin(theta1)
        x2 = x1 + scale * L2 * math.cos(theta1 + theta2)
        y2 = y1 - scale * L2 * math.sin(theta1 + theta2)

        # Draw links
        self.create_line(x0, y0, x1, y1, fill="#3498DB", width=4)
        self.create_line(x1, y1, x2, y2, fill="#E67E22", width=4)

        # Draw joints
        r = 5
        self.create_oval(x0 - r, y0 - r, x0 + r, y0 + r, fill="#2ECC71", outline="")
        self.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, fill="#F1C40F", outline="")
        self.create_oval(x2 - r, y2 - r, x2 + r, y2 + r, fill="#E74C3C", outline="")

        # Crosshair at base
        self.create_line(cx - 10, cy, cx + 10, cy, fill="gray")
        self.create_line(cx, cy - 10, cx, cy + 10, fill="gray")