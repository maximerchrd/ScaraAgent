# vision/localization.py
import numpy as np

class MarkerLocalizer:
    def __init__(self, marker_size_mm=50.0):
        from config import config
        self.marker_size_mm = marker_size_mm
        # Load directly from config (no file)
        self.markers = config.vision.marker_positions

    def get_object_xy(self, marker_id, marker_corners_px, object_pixel):
      """
      marker_id: int
      marker_corners_px: 4x2 array of corner pixel coords (from ArUco)
      object_pixel: (px_x, px_y) of the object in the image
      Returns (x_mm, y_mm) in robot coordinates.
      """
      if marker_id not in self.markers:
          print(f"Marker {marker_id} not in config")
          return None

      # Reference pixel: corner 0
      ref_px = marker_corners_px[0]

      # Scale: average side length in pixels (still computed from the whole marker)
      corners = marker_corners_px
      edge1 = np.linalg.norm(corners[0] - corners[1])
      edge2 = np.linalg.norm(corners[2] - corners[3])
      pixel_size = (edge1 + edge2) / 2.0
      mm_per_px = self.marker_size_mm / pixel_size if pixel_size > 0 else 0.5

      # Pixel offset from corner 0 to object
      offset_px = np.array(object_pixel) - ref_px

      # Convert to mm, keeping the same sign convention you used before
      offset_mm_x =  offset_px[0] * mm_per_px
      offset_mm_y = -offset_px[1] * mm_per_px   # flip Y if camera looks down

      # Add to the stored physical corner position
      marker_pos = self.markers[marker_id]
      obj_x = marker_pos["x"] + offset_mm_x
      obj_y = marker_pos["y"] + offset_mm_y

      return (obj_x, obj_y)