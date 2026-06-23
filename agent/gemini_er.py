# agent/gemini_er.py
"""
Interface to Google Gemini Robotics ER (Vision‑Language Model).
Sends an image + prompt and returns a structured scene description.
"""

import google.generativeai as genai
from config import config

class GeminiER:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = config.llm.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.llm.gemini_model)

    def describe_scene(self, image, prompt=None):
        """
        image: numpy array (BGR or RGB) – will be converted to PIL.
        prompt: optional additional instructions.
        Returns the model’s text response.
        """
        from PIL import Image
        # Ensure RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = image[..., ::-1]  # BGR -> RGB
        else:
            image_rgb = image
        pil_img = Image.fromarray(image_rgb)

        default_prompt = (
            "You are a spatial reasoning assistant for a SCARA robot. "
            "Describe the scene briefly: list all objects, their approximate positions "
            "relative to the robot (left/right/front), any markers (ArUco) and their IDs. "
            "Focus only on actionable items for a pick‑and‑place task."
        )
        prompt = prompt or default_prompt

        response = self.model.generate_content([prompt, pil_img])
        return response.text