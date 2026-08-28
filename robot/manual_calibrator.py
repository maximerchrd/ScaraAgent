# robot/manual_calibrator.py
import logging
import json
from pathlib import Path

class ManualPixelCalibrator:
    def __init__(self, fixed_points_path="fixed_robot_points.json"):
        self.fixed_points_path = Path(fixed_points_path)
        self.fixed_points = []   # list of {"marker_id": int, "corner_index": int, "robot_x": float, "robot_y": float}
        self.load_fixed_points()

    def commit_fixed_point(self, marker_id, corner_index, robot_x, robot_y):
        """Permanently store the physical robot coordinate for a specific marker corner."""
        # Check if already exists, update if so
        for fp in self.fixed_points:
            if fp["marker_id"] == marker_id and fp["corner_index"] == corner_index:
                fp["robot_x"] = robot_x
                fp["robot_y"] = robot_y
                self._save_fixed_points()
                logging.info(f"Updated fixed point: Marker {marker_id} corner {corner_index} -> ({robot_x:.1f},{robot_y:.1f})")
                return
        # Otherwise append new
        self.fixed_points.append({
            "marker_id": marker_id,
            "corner_index": corner_index,
            "robot_x": robot_x,
            "robot_y": robot_y
        })
        self._save_fixed_points()
        logging.info(f"Committed fixed point: Marker {marker_id} corner {corner_index} -> ({robot_x:.1f},{robot_y:.1f})")

    def _save_fixed_points(self):
        with open(self.fixed_points_path, 'w') as f:
            json.dump(self.fixed_points, f, indent=2)

    def load_fixed_points(self):
        if self.fixed_points_path.exists():
            try:
                with open(self.fixed_points_path, 'r') as f:
                    self.fixed_points = json.load(f)
                logging.info(f"Loaded {len(self.fixed_points)} fixed robot coordinates.")
            except Exception as e:
                logging.warning(f"Could not load fixed points: {e}")
                self.fixed_points = []
        else:
            self.fixed_points = []

    def get_fixed_robot_coord(self, marker_id, corner_index):
        """Return (rx, ry) for a specific marker corner, or None if not found."""
        for fp in self.fixed_points:
            if fp["marker_id"] == marker_id and fp["corner_index"] == corner_index:
                return fp["robot_x"], fp["robot_y"]
        return None