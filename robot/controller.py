# robot/controller.py
"""
High‑level robot command interface.
Translates GUI/agent calls into G-code, handles zero offsets, and manages gripper timers.
"""

import logging
import math
import time
import threading
from robot.kinematics import xyz_to_joints, joints_to_xyz
from utils.safe_queue import SafeQueue
from config import config
from robot.manual_calibrator import ManualPixelCalibrator
from robot.kinematics import STEPS_PER_MM_Z

class RobotController:
    def __init__(self, serial_comm, queue=None):
        self.serial_comm = serial_comm
        self.queue = queue or SafeQueue()

        # Zero position offsets (native steps at which "zero" was set)
        self.zero_offset_z = 0
        self.zero_offset_j1 = 0
        self.zero_offset_j2 = 0
        self.zero_mm = (0.0, 0.0, 0.0)   # native mm at zero
        self.has_zero = False

        self.pixel_calib = ManualPixelCalibrator()   # auto-loads existing file

        # Gripper timing
        self._gripper_timer = None
        self.gripper_open_duration = 1500   # ms
        self.gripper_close_duration = 800   # ms

        # Store last known steps for get_position()
        self._last_z = 0
        self._last_j1 = 0
        self._last_j2 = 0
        self._last_yaw = 90

        # Polling thread for position cache
        self._poll_thread = None
        self._stop_poll = False

        self._z_coeffs = config.robot.z_correction_coeffs

        self._last_yaw = 90
        self._last_distance = None   # distance sensor mm
        self._last_pitch = None      # IMU pitch (deg), None if not present

        self.current_speed = config.robot.default_speed

    def set_speed(self, speed):
        self.current_speed = int(speed)
        self.send_raw(f"SPEED {int(speed)}")

    # ---------- Connection helpers ----------
    def connect(self, port):
        self.serial_comm.connect(port)

    def disconnect(self):
        self._stop_polling()
        self.serial_comm.disconnect()

    def is_connected(self):
        return self.serial_comm.is_connected()

    def send_raw(self, cmd):
        self.serial_comm.send(cmd)

    # ---------- Polling for internal position cache ----------
    def _start_polling(self):
        self._stop_poll = False
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _stop_polling(self):
        self._stop_poll = True
        if self._poll_thread:
            self._poll_thread.join(timeout=1)
        self._poll_thread = None

    def _poll_loop(self):
        """Continuously read position messages from queue to update cache."""
        while not self._stop_poll:
            try:
                msg = self.queue.get(timeout=0.1)
                if msg["type"] == "robot_position":
                    data = msg["data"]
                    if len(data) >= 3:
                        self._last_z = int(data[0])
                        self._last_j1 = int(data[1])
                        self._last_j2 = int(data[2])
                        self._last_yaw = int(data[3]) if len(data) > 3 else self._last_yaw
                    if len(data) > 4:
                        try:
                            self._last_distance = float(data[4])
                        except (ValueError, TypeError):
                            self._last_distance = None
                    if len(data) > 5:
                        try:
                            v = float(data[5])
                            # reject NaN ("nan" parses to float('nan'))
                            if v == v:
                                self._last_pitch = v - config.robot.pitch_vertical_deg
                            else:
                                self._last_pitch = None
                        except (ValueError, TypeError):
                            self._last_pitch = None
            except Exception as e:
                # Ignore empty queue or key errors
                pass

    # ---------- Motion commands ----------
    def home(self):
        self.send_raw("G28")

    def estop(self):
        self.send_raw("ESTOP")
        self.queue.put({"type": "robot_estop", "data": None})

    def set_speed(self, speed):
        self.send_raw(f"SPEED {int(speed)}")

    def move_to_xyz(self, x_mm, y_mm, z_mm=None, use_z_correction=True, block=False, pitch_deg=None):
        """
        Move the tool to work coordinates (relative to zero).
        If block=True, wait until the move is likely finished (simulated).
        """
        if pitch_deg is None:
            pitch_deg = self._last_pitch if self._last_pitch is not None else 0.0
        pitch_rad = math.radians(pitch_deg)
        if use_z_correction:
            z_mm = self.get_table_z(x_mm, y_mm) + z_mm
        
        # Convert to native mm
        if self.has_zero:
            x_native = x_mm + self.zero_mm[0]
            y_native = y_mm + self.zero_mm[1]
            z_native = z_mm + self.zero_mm[2]
        else:
            x_native, y_native, z_native = x_mm, y_mm, z_mm

        # Convert to steps
        steps_z, steps_j1, steps_j2 = xyz_to_joints(x_native, y_native, z_native, pitch_rad)
        cmd = f"G0 Z{steps_z} A{steps_j1} B{steps_j2}"
        self.send_raw(cmd)

        if block:
            self._wait_for_position((steps_z, steps_j1, steps_j2))
            time.sleep(0.15)   # settle

    def move_joints(self, z_steps, j1_steps, j2_steps, yaw=None):
        """Direct step move."""
        cmd = f"G0 Z{z_steps} A{j1_steps} B{j2_steps}"
        if yaw is not None:
            cmd += f" Y{yaw}"
        self.send_raw(cmd)

    # ---------- Gripper ----------
    def gripper_open(self):
        self.send_raw("GRIP_OPEN")
        self._gripper_schedule_stop(self.gripper_open_duration)

    def gripper_close(self):
        self.send_raw("GRIP_CLOSE")
        self._gripper_schedule_stop(self.gripper_close_duration)

    def _gripper_schedule_stop(self, duration_ms):
        """Schedule GRIP_STOP after duration_ms."""
        if self._gripper_timer:
            self._gripper_timer.cancel()
        self._gripper_timer = threading.Timer(duration_ms / 1000.0, self._gripper_stop)
        self._gripper_timer.start()

    def _gripper_stop(self):
        self.send_raw("GRIP_STOP")

    # ---------- Zero / Calibration ----------
    def set_zero(self):
        """Store current position as the work coordinate zero."""
        self.zero_offset_z = self._last_z
        self.zero_offset_j1 = self._last_j1
        self.zero_offset_j2 = self._last_j2
        x, y, z = joints_to_xyz(
            self._last_z, self._last_j1, self._last_j2,
            math.radians(self._last_pitch or 0.0))
        self.zero_mm = (x, y, z)
        self.has_zero = True
        logging.info(f"Zero set at native steps Z={self._last_z} J1={self._last_j1} J2={self._last_j2} -> mm ({x:.1f}, {y:.1f}, {z:.1f})")

    def reset_zero(self):
        """Clear the zero offset (e.g. after homing)."""
        self.zero_offset_z = 0
        self.zero_offset_j1 = 0
        self.zero_offset_j2 = 0
        self.zero_mm = (0.0, 0.0, 0.0)
        self.has_zero = False

    def get_work_position(self):
        """Return (x_mm, y_mm, z_mm) in work coordinates."""
        pitch_deg = self._last_pitch or 0.0
        pitch_rad = math.radians(pitch_deg)
        x_nat, y_nat, z_nat = joints_to_xyz(self._last_z, self._last_j1, self._last_j2, pitch_rad)
        if self.has_zero:
            return (x_nat - self.zero_mm[0], y_nat - self.zero_mm[1], z_nat - self.zero_mm[2])
        return (x_nat, y_nat, z_nat)

    def shutdown(self):
        self._stop_polling()
        if self.is_connected():
            self.disconnect()

    def get_table_z(self, x_mm, y_mm):
        """Return Z (work coords) to touch the table at (x_mm, y_mm)."""
        c = self._z_coeffs
        return c[0] + c[1]*x_mm + c[2]*y_mm + c[3]*x_mm**2 + c[4]*y_mm**2 + c[5]*x_mm*y_mm

    def get_distance_mm(self):
        return self._last_distance

    def get_pitch_deg(self):
        return self._last_pitch

    def move_to_native(self, x_native, y_native, z_mm, use_z_correction=True, block=False, pitch_deg=None):
        """
        Move to coordinates that are already in the robot's native work frame
        (i.e., the same frame returned by get_work_position()).
        This bypasses the pixel calibration mapping.
        """
        if pitch_deg is None:
            pitch_deg = (self._last_pitch or 0.0)   # work pitch already
        pitch_rad = math.radians(pitch_deg)

        if use_z_correction:
            z_mm = self.get_table_z(x_native, y_native) + z_mm

        if self.has_zero:
            x_abs = x_native + self.zero_mm[0]
            y_abs = y_native + self.zero_mm[1]
            z_abs = z_mm + self.zero_mm[2]
        else:
            x_abs, y_abs, z_abs = x_native, y_native, z_mm

        steps_z, steps_j1, steps_j2 = xyz_to_joints(x_abs, y_abs, z_abs, pitch_rad)
        cmd = f"G0 Z{steps_z} A{steps_j1} B{steps_j2}"
        self.send_raw(cmd)

        if block:
            self._wait_for_position((steps_z, steps_j1, steps_j2))
            time.sleep(0.15)   # settle

    def set_pitch(self, work_deg, timeout=4.0):
        """
        Command the pitch motor to a work pitch (0 = vertical) and wait for arrival.
        Sends raw = work + pitch_vertical_deg because the firmware speaks raw.
        """
        raw_target = work_deg + config.robot.pitch_vertical_deg
        self.send_raw(f"PITCH_SET {raw_target:.2f}")

        start = time.time()
        while time.time() - start < timeout:
            if self._last_pitch is not None and abs(self._last_pitch - work_deg) < 3.0:
                time.sleep(0.3)   # let the mechanical settle
                return True
            time.sleep(0.05)

        logging.warning(f"set_pitch({work_deg}) timed out, last={self._last_pitch}")
        return False

    def move_z(self, z_mm, block=False, use_z_correction=True):
        """
        Move ONLY the Z axis, keeping current J1/J2 (and therefore XY) fixed.
        z_mm is in work coordinates, same convention as move_to_native().
        """
        # Get current XY so table sag correction stays consistent
        x_work, y_work, _ = self.get_work_position()

        if use_z_correction:
            z_work = self.get_table_z(x_work, y_work) + z_mm
        else:
            z_work = z_mm

        if self.has_zero:
            z_abs = z_work + self.zero_mm[2]
        else:
            z_abs = z_work

        steps_z = round(z_abs * STEPS_PER_MM_Z)
        # Send current A/B as no-ops in case the firmware requires all words.
        self.send_raw(f"G0 Z{steps_z} A{self._last_j1} B{self._last_j2}")

        logging.info(
            f"move_z: in={z_mm}  x_work={x_work:.1f} y_work={y_work:.1f}  "
            f"table_z={self.get_table_z(x_work, y_work):.1f}  "
            f"z_work={z_work:.1f}  steps_z={steps_z}"
        )

        if block:
            self._wait_for_position((steps_z, None, None))
            time.sleep(0.15)   # settle

    def _wait_for_position(self, target_steps, timeout=45.0, tol=8, poll=0.03):
        """
        target_steps: (steps_z, steps_j1, steps_j2). Use None to skip an axis.
        tol is in steps.
        """
        tz, tj1, tj2 = target_steps
        start = time.time()
        while time.time() - start < timeout:
            ok = True
            if tz  is not None and abs(self._last_z  - tz ) > tol: ok = False
            if tj1 is not None and abs(self._last_j1 - tj1) > tol: ok = False
            if tj2 is not None and abs(self._last_j2 - tj2) > tol: ok = False
            if ok:
                return True
            time.sleep(poll)
        logging.warning(f"_wait_for_position timed out (target={target_steps}, "
                        f"last=({self._last_z},{self._last_j1},{self._last_j2}))")
        return False