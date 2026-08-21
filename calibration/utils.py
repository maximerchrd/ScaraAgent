"""
Reusable calibration utilities for camera‑to‑robot homography.
Used by both the standalone script (calibrate_camera.py) and the GUI.
"""

import time
import logging
import math
import numpy as np
import cv2
from vision.aruco_detector import detect_aruco
from robot.kinematics import L1, L2, xyz_to_joints, STEPS_PER_DEG_J1, STEPS_PER_DEG_J2


def marker_to_tcp(marker_x, marker_y, offset_along=62.0):
    """
    Given the desired world position of the marker (which is mounted
    'offset_along' mm beyond the TCP along the second link),
    return the TCP world position (xtcp, ytcp) that the robot must
    be commanded to reach.
    """
    L2_virtual = L2 + offset_along

    # Inverse kinematics for the virtual endpoint (the marker)
    D = (marker_x**2 + marker_y**2 - L1**2 - L2_virtual**2) / (2 * L1 * L2_virtual)
    D = max(-1.0, min(1.0, D))
    theta2_virtual = math.atan2(math.sqrt(1 - D**2), D)   # elbow‑up
    theta1 = math.atan2(marker_y, marker_x) - math.atan2(
        L2_virtual * math.sin(theta2_virtual),
        L1 + L2_virtual * math.cos(theta2_virtual)
    )

    # Absolute angle of the (virtual) second link
    phi = theta1 + theta2_virtual

    # TCP is backward along the link
    xtcp = marker_x - offset_along * math.cos(phi)
    ytcp = marker_y - offset_along * math.sin(phi)

    return xtcp, ytcp


def collect_calibration_data(robot, camera, grid_points,
                             marker_offset_along=62.0,
                             marker_id=5,
                             settle_time=1.0,
                             max_retries=3,
                             max_failure_ratio=0.3):
    """
    Move to each grid point, detect the ArUco marker, and return a list of
    (pixel_x, pixel_y, world_x, world_y) tuples.

    Returns None if too many points fail.
    """
    data = []
    failures = 0
    total = len(grid_points)

    logging.info(f"Starting data collection: {total} grid points")
    logging.info(f"Marker offset along arm: {marker_offset_along} mm")

    for i, (xm, ym, zm) in enumerate(grid_points):
        xtcp, ytcp = marker_to_tcp(xm, ym, offset_along=marker_offset_along)
        gz = zm

        logging.info(f"Point {i+1}/{total}: marker ({xm:.0f},{ym:.0f}) → TCP ({xtcp:.0f},{ytcp:.0f})")

        robot.move_to_xyz(xtcp, ytcp, gz, block=True)
        time.sleep(settle_time)

        detected = False
        for _ in range(max_retries):
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.3)
                continue
            markers = detect_aruco(frame)
            for m in markers:
                if m['id'] == marker_id:
                    centre = m['corners'].mean(axis=0)   # (x, y) in pixels
                    data.append((centre[0], centre[1], xm, ym))
                    detected = True
                    break
            if detected:
                break
            time.sleep(0.5)

        if detected:
            logging.info(f"  -> pixel ({centre[0]:.0f}, {centre[1]:.0f})")
        else:
            failures += 1
            logging.warning(f"  -> marker NOT detected")
            if failures > total * max_failure_ratio:
                logging.error(f"Too many failures ({failures}/{total}) – aborting.")
                return None

    logging.info(f"Data collection done: {len(data)} valid, {failures} missed")
    return data


def fit_homography(data):
    """Fit a homography from pixel→world coordinates using RANSAC."""
    pixel_pts = np.float32([(d[0], d[1]) for d in data])
    world_pts = np.float32([(d[2], d[3]) for d in data])
    H, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC, 3.0)
    return H, mask


def evaluate_homography(H, data):
    """Return (mean_error, max_error, per_point_errors)."""
    pixel_pts = np.float32([(d[0], d[1]) for d in data]).reshape(-1, 1, 2)
    reproj = cv2.perspectiveTransform(pixel_pts, H).reshape(-1, 2)
    world_pts = np.float32([(d[2], d[3]) for d in data])
    errors = np.linalg.norm(reproj - world_pts, axis=1)
    return np.mean(errors), np.max(errors), errors


def save_homography(H, filepath="calib_homography.npy"):
    np.save(filepath, H)
    logging.info(f"Homography saved to {filepath}")

def collect_calibration_data_multi_marker(
        robot, camera, grid_points,
        marker_ids_offsets,
        settle_time=1.0,
        max_retries=3,
        max_failure_ratio=0.3):
    """
    Collect calibration data using multiple ArUco markers on the gripper.

    Parameters:
        robot: RobotController instance
        camera: CameraThread instance
        grid_points: list of (x, y, z) TCP target coordinates (work coords)
        marker_ids_offsets: dict {
            11: {"offset": 103.5, "direction": "front"},
            10: {"offset": 102.5, "direction": "left"},
            12: {"offset": 101.5, "direction": "right"}
        }
        settle_time: seconds to wait after each move
        max_retries: number of detection attempts per point
        max_failure_ratio: abort if too many failures

    Returns:
        data: list of (pixel_x, pixel_y, world_x, world_y)
        or None if too many failures.
    """
    data = []
    failures = 0
    total = len(grid_points)

    logging.info(f"Starting multi-marker calibration: {total} TCP grid points")

    for i, (x_tcp, y_tcp, z_tcp) in enumerate(grid_points):
        logging.info(f"Point {i+1}/{total}: TCP ({x_tcp:.0f}, {y_tcp:.0f})")

        # Move TCP directly to grid point
        robot.move_to_xyz(x_tcp, y_tcp, z_tcp, block=True)
        time.sleep(settle_time)

        # Compute joint angles for forearm direction (using commanded TCP)
        steps_z, steps_j1, steps_j2 = xyz_to_joints(x_tcp, y_tcp, z_tcp)
        theta1 = math.radians(steps_j1 / STEPS_PER_DEG_J1)
        theta2 = math.radians(steps_j2 / STEPS_PER_DEG_J2)

        # Forearm unit vector
        phi = theta1 + theta2
        u_x = math.cos(phi)
        u_y = math.sin(phi)

        # Perpendicular vectors
        v_left_x = -u_y
        v_left_y =  u_x
        v_right_x =  u_y
        v_right_y = -u_x

        detected_any = False
        for _ in range(max_retries):
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.3)
                continue
            markers = detect_aruco(frame)
            for m in markers:
                mid = int(m['id'])
                if mid in marker_ids_offsets:
                    # Pixel center
                    centre = m['corners'].mean(axis=0)
                    px, py = centre[0], centre[1]

                    # Compute world coordinate from offset
                    offset_info = marker_ids_offsets[mid]
                    offset = offset_info["offset"]
                    direction = offset_info["direction"]

                    if direction == "front":
                        wx = x_tcp + offset * u_x
                        wy = y_tcp + offset * u_y
                    elif direction == "left":
                        wx = x_tcp + offset * v_left_x
                        wy = y_tcp + offset * v_left_y
                    elif direction == "right":
                        wx = x_tcp + offset * v_right_x
                        wy = y_tcp + offset * v_right_y
                    else:
                        continue

                    data.append((px, py, wx, wy))
                    detected_any = True
                    logging.info(f"  -> marker {mid} at pixel ({px:.0f}, {py:.0f}), "
                                 f"world ({wx:.0f}, {wy:.0f})")
            if detected_any:
                break
            time.sleep(0.5)

        if not detected_any:
            failures += 1
            logging.warning(f"  -> no markers detected")
            if failures > total * max_failure_ratio:
                logging.error(f"Too many failures ({failures}/{total}) – aborting.")
                return None

    logging.info(f"Data collection done: {len(data)} correspondences, {failures} missed points")
    return data