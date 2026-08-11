# agent/perception.py
"""
PerceptionManager: iterative VLM‑LLM‑VLM loop.
Improves scene understanding by letting the LLM ask clarifying questions.
Provides live feedback to the GUI via the shared queue.
"""

import logging
import json
from agent.gemini_er import GeminiER
from agent.llm import ChatGPTOSS
from config import config


class PerceptionManager:
    def __init__(self, vlm: GeminiER, critic_llm: ChatGPTOSS, queue=None):
        self.vlm = vlm
        self.critic_llm = critic_llm 
        self.queue = queue          # SafeQueue for GUI messages
        self.max_iterations = getattr(config, 'max_vlm_iterations', 2)
        self.critic_prompt_template = (
          "You are a perception critic for a SCARA robot.\n"
          "You are given a list of detected objects (with normalised coordinates 0‑1000) and a user task.\n"
          "Your job is to decide whether the detected objects are sufficient to accomplish the task.\n\n"
          "IMPORTANT — Semantic matching:\n"
          "The vision model often describes objects with different words than the user.\n"
          "You MUST judge whether two labels plausibly refer to the same physical object.\n"
          "Consider: synonyms, category membership, material vs function descriptions,\n"
          "and different levels of specificity (e.g., 'container' vs 'box' vs 'plastic bin').\n"
          "If an object label could reasonably be what the user meant, treat it as a MATCH.\n"
          "Only mark 'sufficient': false if NO object in the list could plausibly be\n"
          "the one the user is asking for.\n\n"
          "If all needed objects are present (even under different names), output {\"sufficient\": true}.\n\n"
          "If an object is truly missing, output {\"sufficient\": false, \"question\": \"...\"}.\n"
          "The question must be an instruction for a vision model to locate the missing object.\n"
          "The question MUST end with: 'Return ONLY a JSON array. No text. Format: [{\"point\": [y,x], \"label\": \"name\"}]'\n\n"
          "Your entire answer must be ONLY a valid JSON object. No markdown, no explanation."
      )

    def _send_reasoning(self, text):
        """Push a reasoning message to the GUI queue if available."""
        if self.queue:
            self.queue.put({"type": "agent_reasoning", "data": text})

    def perceive(self, image, user_task: str) -> list:
        """
        Run the iterative perception loop and return a list of objects
        in the same format as GeminiER.localize_objects (dicts with 'point', 'label').
        """
        # 1. Initial detection
        all_objects = self.vlm.localize_objects(image)
        logging.info(f"Initial VLM objects: {len(all_objects)}")
        self._send_reasoning(f"🔍 Initial VLM found {len(all_objects)} objects: {', '.join(obj['label'] for obj in all_objects)}")

        for iteration in range(self.max_iterations):
            # 2. Ask LLM to critique
            critique = self._ask_critic(all_objects, user_task, iteration)
            if critique is None:
                self._send_reasoning("⚠️  Failed to get a valid critique from LLM – stopping perception loop.")
                break

            if critique.get("sufficient", False):
                logging.info("Perception sufficient according to LLM.")
                self._send_reasoning("✅ LLM critic: scene description is sufficient.")
                break

            question = critique.get("question")
            if not question:
                logging.warning("LLM said insufficient but provided no question.")
                self._send_reasoning("❓ LLM critic: insufficient, but no question provided.")
                break

            # 3. Ask VLM the follow‑up question
            logging.info(f"Iteration {iteration+1}: asking VLM: {question}")
            self._send_reasoning(f"📸 Asking VLM: {question}")
            new_objects = self.vlm.localize_objects(image, prompt=question)
            self._log_objects(new_objects)
            self._send_reasoning(f"   ↳ VLM returned {len(new_objects)} objects: {', '.join(obj['label'] for obj in new_objects)}")

            # 4. Merge
            all_objects = self._merge_objects(all_objects, new_objects)
            logging.info(f"Objects after iteration {iteration+1}: {len(all_objects)}")

        return all_objects

    def _ask_critic(self, objects, task, iteration):
        """Send the objects and task to the LLM; return parsed JSON or a fallback question."""
        obj_desc = json.dumps(objects, indent=2)
        user_msg = (
            f"Detected objects (iteration {iteration}):\n{obj_desc}\n\n"
            f"User task: {task}\n\n"
            "Is this enough to accomplish the task? Reply with JSON."
        )

        messages = [
            {"role": "system", "content": self.critic_prompt_template},
            {"role": "user", "content": user_msg},
        ]

        self._send_reasoning(f"🤔 Asking LLM critic...")
        
        try:
            response = self.critic_llm.chat(messages, temperature=0.1, include_reasoning=False)
        except Exception as e:
            logging.warning(f"Critic LLM call failed: {e}")
            self._send_reasoning(f"⚠️  Critic call failed ({e}) — using fallback question.")
            return self._fallback_question(task)

        if response is None:
            self._send_reasoning(f"⚠️  Critic returned None — using fallback question.")
            return self._fallback_question(task)

        # Extract content, fallback to reasoning if content is empty
        content = response.get("content", "").strip()
        reasoning = response.get("reasoning", "").strip()
        if not content and reasoning:
            content = reasoning

        if not content:
            self._send_reasoning(f"⚠️  Critic returned empty response — using fallback question.")
            return self._fallback_question(task)

        # Try direct parse first (the LLM should output valid JSON)
        try:
            critique = json.loads(content)
            self._send_reasoning(f"   ↳ Critic: {json.dumps(critique, indent=2)}")
            return critique
        except json.JSONDecodeError:
            pass   # fall through to regex as a safety net

        # Fallback: regex extraction (kept for malformed responses)
        import re
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)

        if json_match:
            try:
                critique = json.loads(json_match.group(0))
                self._send_reasoning(f"   ↳ Critic (regex): {json.dumps(critique, indent=2)}")
                return critique
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse LLM critique JSON even with regex: {content[:300]}")
                self._send_reasoning("⚠️  Couldn't parse critic response — using fallback question.")
                return self._fallback_question(task)

        # No JSON found at all
        logging.warning(f"No JSON found in critic response: {content[:300]}")
        self._send_reasoning("⚠️  No JSON in critic response — using fallback question.")
        return self._fallback_question(task)


    def _fallback_question(self, task):
        """Generate a reasonable default question when the critic fails."""
        question = (
            "Look carefully at this image of a robot workspace. "
            "List all objects that could be relevant to this task: "
            f"'{task}'. "
            "For each object, provide a short descriptive name. "
            "Return ONLY a JSON array like: [{\"point\": [y, x], \"label\": \"descriptive name\"}, ...] "
            "Points are [y, x] normalized 0-1000. "
            "Do NOT include any other text, markdown, or formatting. Output ONLY the JSON array."
        )
        logging.info(f"Using fallback question: {question}")
        return {"sufficient": False, "question": question}

    def _merge_objects(self, existing, new, pixel_threshold=40):
        merged = existing.copy()
        for new_obj in new:
            new_pt = new_obj.get("point")
            if not new_pt:
                continue
            duplicate = False
            for i, ex_obj in enumerate(merged):
                ex_pt = ex_obj.get("point")
                if not ex_pt:
                    continue
                dist = ((new_pt[0] - ex_pt[0])**2 + (new_pt[1] - ex_pt[1])**2)**0.5
                if dist < pixel_threshold:
                    merged[i] = new_obj        # replace with newer label
                    duplicate = True
                    break
            if not duplicate:
                merged.append(new_obj)
        return merged

    def _log_objects(self, objects):
        for obj in objects:
            logging.debug(f"  {obj['label']} @ {obj['point']}")