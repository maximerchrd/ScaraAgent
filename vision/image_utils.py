# vision/image_utils.py
"""
Image processing helpers: resizing, drawing overlays, converting for Tkinter.
"""

import cv2
import numpy as np


def resize_frame(frame, width=None, height=None):
    """Resize keeping aspect ratio if only one dimension given."""
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    if width and height:
        return cv2.resize(frame, (width, height))
    if width:
        ratio = width / w
        return cv2.resize(frame, (width, int(h * ratio)))
    if height:
        ratio = height / h
        return cv2.resize(frame, (int(w * ratio), height))
    return frame


def draw_markers(frame, markers):
    """Draw detected ArUco markers on the frame (modifies in place)."""
    if frame is None:
        return frame
    for marker in markers:
        corners = marker["corners"].astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        # Put ID text at the center
        center = corners.mean(axis=0).astype(int)
        cv2.putText(frame, str(marker["id"]), tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


def convert_for_tk(frame, target_width=640, target_height=480):
    """Convert OpenCV BGR frame to RGB PIL ImageTk for tkinter display."""
    if frame is None:
        return None
    try:
        from PIL import Image, ImageTk
    except ImportError:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img)