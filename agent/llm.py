# agent/llm.py
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

    def chat(self, messages, temperature=0.6):    # recommended 0.5‑0.7
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": 4096,        # increased for reasoning
            "top_p": 0.95,
            "include_reasoning": True             # explicit, though default is true
        }
        try:
            resp = requests.post(
                self.endpoint,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=config.llm.request_timeout
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]["message"]
            content = choice.get("content", "")
            reasoning = choice.get("reasoning", "")   # ← the chain‑of‑thought

            if reasoning:
                logging.info(f"LLM reasoning:\n{reasoning}")
            logging.info(f"LLM raw response:\n{content}")

            return {
                "content": content,
                "reasoning": reasoning
            }
        except Exception as e:
            logging.error(f"LLM request failed: {e}")
            return None