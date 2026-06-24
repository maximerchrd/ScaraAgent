# agent/gemini_er.py
"""
Interface to Google Gemini Robotics ER (Vision‑Language Model).
Sends an image + prompt and returns a structured scene description.
"""

import google.generativeai as genai
from config import config
import json
import re
import logging

class GeminiER:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = config.llm.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.llm.gemini_model)

    def describe_scene(self, image, prompt=None):
        pil_img = self._to_pil(image)   # ← use the helper
        default_prompt = (
            "You are a spatial reasoning assistant for a SCARA robot. "
            "Describe the scene briefly: list all objects, their approximate positions "
            "relative to the robot (left/right/front), any markers (ArUco) and their IDs. "
            "Focus only on actionable items for a pick‑and‑place task."
        )
        prompt = prompt or default_prompt
        response = self.model.generate_content([prompt, pil_img])
        return response.text
    def localize_objects(self, image, image_width=1280, image_height=720):
        """
        Ask Gemini to find objects and report pixel coordinates + nearest marker.
        Returns a list of dicts, e.g.:
        [{'label': 'red cube', 'pixel_x': 340, 'pixel_y': 220, 'marker_id': 3}, ...]
        """
        prompt = f"""
You are a robot vision system. The image size is {image_width}x{image_height} pixels.
List every movable object (cubes, boxes, tools) and for each object provide:
- A short label (e.g., "red cube")
- Its approximate pixel coordinates (x, y) of its center
- The ID of the nearest ArUco marker in the image (if a marker is close enough to be the "nearest", give its numeric ID; otherwise use null).

Return ONLY a JSON array. Do not include any other text. Example:
[{{"label": "red cube", "pixel_x": 200, "pixel_y": 150, "marker_id": 5}}]
""".strip()  # ← strip leading/trailing whitespace
        response_text = self.model.generate_content([prompt, self._to_pil(image)]).text
        # Try to extract JSON
        try:
            # Clean up possible markdown code fences
            json_str = re.sub(r'```json|```', '', response_text).strip()
            objects = json.loads(json_str)
            return objects if isinstance(objects, list) else []
        except Exception as e:
            logging.warning(f"Gemini JSON parse failed: {e}\nResponse: {response_text}")
            return []

    def _to_pil(self, image):
        from PIL import Image
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = image[..., ::-1]  # BGR->RGB
        else:
            image_rgb = image
        return Image.fromarray(image_rgb)