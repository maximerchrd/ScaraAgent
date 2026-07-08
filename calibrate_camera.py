#!/usr/bin/env python3
"""
Standalone automatic camera‑to‑robot calibration.
Assumes the robot is ALREADY at a known work coordinate (set via GUI).
Skips homing – just connects, infers the zero offset, and collects data.

Run this script directly:   python calibrate_camera.py
"""

import time
import logging
import numpy as np
from pathlib import Path

from config import config
from robot.controller import RobotController
from robot.serial_comm import SerialComm
from vision.camera import CameraThread
from utils.safe_queue import SafeQueue
from utils.logger import setup_logging
from robot.kinematics import joints_to_xyz

# Shared calibration logic
from calibration.utils import (
    collect_calibration_data,
    fit_homography,
    evaluate_homography,
    save_homography,
)

# ======================================================================
# HARDCODED PARAMETERS – EDIT THESE BEFORE RUNNING
# ======================================================================

# The exact work coordinates the robot is currently at (must match GUI setting)
INITIAL_WORK_POSITION = (400.0, -400.0, 10.0)   # X, Y, Z (mm)

# Calibration grid – marker world positions (x, y, z) in mm
GRID_POINTS = [
    (350, -350, 10.0),
    (350, -275, 10.0),
    (350, -200, 10.0),
    (350, -125, 10.0),
    (350,  -50, 10.0),

    (390, -350, 10.0),
    (390, -275, 10.0),
    (390, -200, 10.0),
    (390, -125, 10.0),
    (390,  -50, 10.0),

    (465, -350, 10.0),
    (465, -275, 10.0),
    (465, -200, 10.0),
    (465, -125, 10.0),
    (465,  -50, 10.0),

    (520, -350, 10.0),
    (520, -275, 10.0),
    (520, -200, 10.0),
    (520, -125, 10.0),
    (520,  -50, 10.0),

    (580, -350, 10.0),
    (580, -275, 10.0),
    (580, -200, 10.0),
    (580, -125, 10.0),
    (580,  -50, 10.0),
]

MARKER_OFFSET_ALONG = 62.0   # mm along second link from TCP to marker centre
MARKER_ID = 5                # ArUco ID on the gripper

SETTLE_TIME = 1.0            # seconds to wait after each move
MAX_RETRIES = 3
MAX_FAILURE_RATIO = 0.3

OUTPUT_FILE = "calib_homography.npy"

SERIAL_PORT = "/dev/cu.usbserial110"   # your robot's serial port
CAMERA_INDEX = config.vision.camera_index
CAMERA_WARMUP = 3

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    setup_logging(level=logging.INFO)

    print("\n" + "="*60)
    print("  SCARA ROBOT – AUTOMATIC CAMERA CALIBRATION")
    print("  (Robot must already be at known position via GUI)")
    print("="*60)
    print(f"  Assumed work pos:  {INITIAL_WORK_POSITION}")
    print(f"  Grid points:       {len(GRID_POINTS)}")
    print(f"  Marker offset:     {MARKER_OFFSET_ALONG} mm")
    print(f"  Marker ID:         {MARKER_ID}")
    print(f"  Serial port:       {SERIAL_PORT}")
    print(f"  Camera:            {CAMERA_INDEX}")
    print(f"  Output file:       {OUTPUT_FILE}")
    print("="*60 + "\n")

    # --- Port auto‑detection ---
    import serial.tools.list_ports

    port_to_use = SERIAL_PORT
    try:
        test_ser = serial.Serial(port_to_use, config.robot.baudrate, timeout=0.1)
        test_ser.close()
        print(f"Port {port_to_use} OK.\n")
    except Exception:
        print(f"Port {port_to_use} not available – scanning...")
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if not ports:
            print("❌ No serial ports found.")
            exit(1)
        print("Available ports:")
        for i, p in enumerate(ports):
            print(f"  [{i}] {p}")
        if len(ports) == 1:
            port_to_use = ports[0]
            print(f"Auto‑selected: {port_to_use}\n")
        else:
            choice = input("Choose number (or 'q'): ")
            if choice.lower() == 'q':
                exit(0)
            try:
                port_to_use = ports[int(choice)]
            except (ValueError, IndexError):
                print("Invalid choice.")
                exit(1)

    # --- Robot connection ---
    queue = SafeQueue()
    serial_comm = SerialComm(port=port_to_use, baudrate=config.robot.baudrate, queue=queue)
    robot = RobotController(serial_comm, queue)

    print(f"Connecting to {port_to_use} ...")
    robot.connect(port_to_use)
    time.sleep(2)

    # --- Read current position and infer zero offset ---
    print("Reading current robot position...")
    pos_received = False
    start = time.time()
    while time.time() - start < 10:
        try:
            msg = queue.get(timeout=0.5)
            if msg.get("type") == "robot_position":
                data = msg["data"]
                robot._last_z = int(data[0])
                robot._last_j1 = int(data[1])
                robot._last_j2 = int(data[2])
                if len(data) > 3:
                    robot._last_yaw = int(data[3])
                pos_received = True
                break
        except:
            pass

    if not pos_received:
        print("❌ No position received from robot. Aborting.")
        robot.disconnect()
        exit(1)

    # Compute native mm and set zero
    native_x, native_y, native_z = joints_to_xyz(robot._last_z, robot._last_j1, robot._last_j2)
    wx, wy, wz = INITIAL_WORK_POSITION
    robot.zero_mm = (native_x - wx, native_y - wy, native_z - wz)
    robot.has_zero = True
    robot.zero_offset_z = robot._last_z
    robot.zero_offset_j1 = robot._last_j1
    robot.zero_offset_j2 = robot._last_j2

    print(f"Current native pos:   ({native_x:.1f}, {native_y:.1f}, {native_z:.1f}) mm")
    print(f"Work zero stored at:   {robot.zero_mm}")
    print(f"Work pos verification: {robot.get_work_position()}")
    print("(Should match INITIAL_WORK_POSITION)\n")

    # --- Camera setup ---
    print("Starting camera...")
    camera = CameraThread(camera_index=CAMERA_INDEX,
                          width=config.vision.frame_width,
                          height=config.vision.frame_height,
                          fps=config.vision.fps, queue=queue)
    camera.start()
    time.sleep(CAMERA_WARMUP)

    if camera.get_latest_frame() is None:
        print("❌ Camera not delivering frames. Check URL.")
        input("Press ENTER to continue anyway (or Ctrl+C to abort).")

    print("Camera ready.\n")

    # --- Show grid and confirm ---
    print(f"Calibration grid: {len(GRID_POINTS)} points")
    for p in GRID_POINTS:
        print(f"  ({p[0]:6.0f}, {p[1]:6.0f}, {p[2]:4.0f})")

    print(f"\n⚠️  Robot will move to {len(GRID_POINTS)} positions.")
    response = input("Press ENTER to start, or 'q' to quit: ")
    if response.lower() == 'q':
        print("Aborted.")
        camera.stop()
        robot.disconnect()
        exit(0)

    # --- Run calibration ---
    data = collect_calibration_data(robot, camera, GRID_POINTS,
                                    marker_offset_along=MARKER_OFFSET_ALONG,
                                    marker_id=MARKER_ID,
                                    settle_time=SETTLE_TIME,
                                    max_retries=MAX_RETRIES,
                                    max_failure_ratio=MAX_FAILURE_RATIO)

    if data is None or len(data) < 4:
        print("❌ Not enough valid calibration points.")
        camera.stop()
        robot.disconnect()
        exit(1)

    H, mask = fit_homography(data)
    mean_err, max_err, errors = evaluate_homography(H, data)
    save_homography(H, OUTPUT_FILE)

    # --- Print results ---
    print(f"\n{'='*60}")
    print("CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"Points used:     {len(data)}")
    print(f"Mean error:      {mean_err:.2f} mm")
    print(f"Max error:       {max_err:.2f} mm")
    print(f"RANSAC inliers:  {np.sum(mask)} / {len(data)}")

    if mean_err > 5.0:
        print("⚠️  WARNING: Mean error > 5 mm.")
    elif mean_err > 2.0:
        print("⚠️  Acceptable but could be better.")
    else:
        print("✅ Good accuracy (< 2 mm).")

    print("\nPer‑point errors (mm):")
    for i, (d, err) in enumerate(zip(data, errors)):
        flag = " ⚠️" if err > 3.0 else ""
        print(f"  {i+1:2d}: world ({d[2]:6.0f}, {d[3]:6.0f}) → pixel ({d[0]:6.0f}, {d[1]:6.0f})  error = {err:.2f} mm{flag}")

    # --- Park and disconnect ---
    print("\nReturning to safe position...")
    robot.move_to_xyz(config.robot.park_x_mm, config.robot.park_y_mm, config.robot.safe_z_mm, block=True)
    time.sleep(1)

    camera.stop()
    robot.disconnect()

    print(f"\n✅ Homography saved to {OUTPUT_FILE}")
    print(f"Matrix:\n{H}")