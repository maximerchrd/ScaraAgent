# robot/calibrator.py
"""
Calibration utilities: zero-point management and measurement recording.
"""

import csv
import logging

class Calibrator:
    def __init__(self, controller, csv_file="measurements.csv"):
        self.controller = controller
        self.csv_file = csv_file

    def record_measurement(self, measured_x, measured_y, measured_z):
        """
        Record a calibration point: current joint steps + measured XYZ.
        Appends to CSV: j1_steps, j2_steps, z_steps, measured_x, measured_y, measured_z
        """
        # Get current steps
        try:
            steps_z = self.controller._last_z
            steps_j1 = self.controller._last_j1
            steps_j2 = self.controller._last_j2
        except AttributeError:
            logging.error("No position data available")
            return

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([steps_j1, steps_j2, steps_z, measured_x, measured_y, measured_z])
        logging.info(f"Measurement recorded: {steps_j1}, {steps_j2}, {steps_z} -> X={measured_x} Y={measured_y} Z={measured_z}")