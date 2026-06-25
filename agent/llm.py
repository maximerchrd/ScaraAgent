# agent/llm.py
"""
Interface to ChatGPT OSS 120B (or any OpenAI‑compatible API).
Used for reasoning and action planning.
"""

import requests
import json
import logging
from config import config

class ChatGPTOSS:
    def __init__(self, endpoint=None, model=None):
        self.endpoint = endpoint or config.llm.chatgpt_endpoint
        self.model = model or config.llm.chatgpt_model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.llm.chatgpt_api_key}"
        }

    def chat(self, messages, temperature=0.3):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024
        }
        try:
            resp = requests.post(
                self.endpoint,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=config.llm.request_timeout
            )
            resp.raise_for_status()
            logging.info(f"LLM raw response:\n{resp.json()["choices"][0]["message"]["content"]}")
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM request failed: {e}")
            return None