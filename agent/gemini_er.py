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
    def localize_objects(self, image):
        """
        Ask Gemini Robotics-ER to point to all objects in the scene.
        Returns a list of dicts with 'point' (normalized [y, x] 0-1000) and 'label'.
        """
        prompt = """Look carefully at this image of a robot workspace. 
Identify and point to up to 15 distinct physical objects you can see (tools, parts, blocks, boxes, etc.).
For each object, provide a short descriptive name (e.g., "red cube", "blue screwdriver", "metal bracket", "black box").
Ignore ArUco markers — only point to actual objects that could be manipulated.
Return ONLY a JSON array like: [{"point": [y, x], "label": "descriptive name"}, ...]
Points are [y, x] normalized 0-1000."""
        
        response_text = self.model.generate_content([prompt, self._to_pil(image)]).text

        logging.info(f"Gemini VLM raw response:\n{response_text}")
        try:
            import re, json
            json_str = re.sub(r'```json|```', '', response_text).strip()
            objects = json.loads(json_str)
            if isinstance(objects, list):
                # Extract 'point' and 'label' only
                return [{"point": obj["point"], "label": obj["label"]} for obj in objects if "point" in obj and "label" in obj]
        except Exception as e:
            logging.warning(f"Gemini Robotics-ER parse failed: {e}\nResponse: {response_text}")
        return []

    def _to_pil(self, image):
        from PIL import Image
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = image[..., ::-1]  # BGR->RGB
        else:
            image_rgb = image
        return Image.fromarray(image_rgb)