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
    
    def localize_objects(self, image, prompt=None):
        """
        Ask Gemini to detect objects. If prompt is provided, use it;
        otherwise use the default generic prompt.
        """
        if prompt is None:
            prompt = (
                "Look carefully at this image of a robot workspace. "
                "Identify and point to up to 15 distinct physical objects you can see "
                "(tools, parts, blocks, boxes, etc.). "
                "For each object, provide a short descriptive name (e.g., \"red cube\", "
                "\"blue screwdriver\"). "
                "Ignore ArUco markers — only point to actual objects that could be manipulated. "
                "Return ONLY a JSON array like: [{\"point\": [y, x], \"label\": \"descriptive name\"}, ...] "
                "Points are [y, x] normalized 0-1000. "
                "Do NOT include any other text, markdown, or formatting. Output ONLY the JSON array."
            )
        response_text = self.model.generate_content([prompt, self._to_pil(image)]).text

        logging.info(f"Gemini VLM raw response:\n{response_text}")
        
        # Remove code fences
        json_str = re.sub(r'```json|```', '', response_text).strip()

        # Fix common JSON errors: missing closing braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        # Fix missing closing brackets
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)

        # Try full parse first
        try:
            objects = json.loads(json_str)
            if isinstance(objects, list):
                parsed = []
                for obj in objects:
                    # Handle both "point" and "box_2d" formats
                    if "point" in obj and "label" in obj:
                        parsed.append({"point": obj["point"], "label": obj["label"]})
                    elif "box_2d" in obj and "label" in obj:
                        # box_2d is [x1, y1, x2, y2] — use center, convert to [y, x]
                        x1, y1, x2, y2 = obj["box_2d"]
                        x = (x1 + x2) / 2
                        y = (y1 + y2) / 2
                        parsed.append({"point": [y, x], "label": obj["label"]})
                if parsed:
                    return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: extract individual objects with regex
        objects = []
        # Matches {"point": [...], "label": "..."}
        pattern = r'\{\s*"point"\s*:\s*\[([^\]]*)\]\s*,\s*"label"\s*:\s*"([^"]*)"\s*\}'
        for match in re.finditer(pattern, response_text):
            point_str = match.group(1)
            label = match.group(2)
            try:
                coords = [float(x.strip()) for x in point_str.split(",")]
                if len(coords) >= 2:
                    if len(coords) == 4:
                        y = (coords[0] + coords[2]) / 2
                        x = (coords[1] + coords[3]) / 2
                    else:
                        y, x = coords[0], coords[1]
                    objects.append({"point": [y, x], "label": label})
            except ValueError:
                continue

        # Also try box_2d pattern in regex fallback
        box_pattern = r'\{\s*"box_2d"\s*:\s*\[([^\]]*)\]\s*,\s*"label"\s*:\s*"([^"]*)"\s*\}'
        for match in re.finditer(box_pattern, response_text):
            box_str = match.group(1)
            label = match.group(2)
            try:
                coords = [float(x.strip()) for x in box_str.split(",")]
                if len(coords) == 4:
                    x = (coords[0] + coords[2]) / 2
                    y = (coords[1] + coords[3]) / 2
                    # Check if we already have something very close
                    duplicate = False
                    for obj in objects:
                        dist = ((obj["point"][0] - y)**2 + (obj["point"][1] - x)**2)**0.5
                        if dist < 40:
                            duplicate = True
                            break
                    if not duplicate:
                        objects.append({"point": [y, x], "label": label})
            except ValueError:
                continue

        logging.info(f"Extracted {len(objects)} objects via regex fallback")
        return objects

    def _to_pil(self, image):
        from PIL import Image
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = image[..., ::-1]  # BGR->RGB
        else:
            image_rgb = image
        return Image.fromarray(image_rgb)