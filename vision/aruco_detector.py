# vision/aruco_detector.py
"""ArUco marker detection – only requires OpenCV."""

import cv2
import numpy as np

def detect_aruco(frame, dictionary_name="DICT_4X4_50", camera_matrix=None, dist_coeffs=None, marker_length=50):
    """
    Detect ArUco markers in a BGR frame.
    Returns a list of dicts: {id, corners (pixel coords), tvec, rvec (optional)}.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    markers = []
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            marker = {"id": int(marker_id), "corners": corners[i][0]}
            if camera_matrix is not None and dist_coeffs is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], marker_length, camera_matrix, dist_coeffs)
                marker["rvec"] = rvecs[0][0]
                marker["tvec"] = tvecs[0][0]
            markers.append(marker)
    return markers