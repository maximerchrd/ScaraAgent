# agent/orchestrator.py
"""
AgentOrchestrator ties vision, Gemini, LLM and robot together.
Runs in a separate thread to avoid blocking the GUI.
"""

import threading
import time
import json
import logging

from agent.gemini_er import GeminiER
from agent.llm import ChatGPTOSS
from agent.prompt_templates import SYSTEM_PROMPT, FEW_SHOT_USER, FEW_SHOT_ASSISTANT

class AgentOrchestrator:
    def __init__(self, robot, camera, queue, gemini_api_key=None, chatgpt_endpoint=None):
        self.robot = robot
        self.camera = camera
        self.queue = queue
        self.gemini = GeminiER(api_key=gemini_api_key)
        self.llm = ChatGPTOSS(endpoint=chatgpt_endpoint)

        self.running = False
        self.thread = None

    def start(self):
        if self.thread is not None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logging.info("Agent orchestrator started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _run_loop(self):
        """Listen for user prompts from the queue."""
        while self.running:
            # Non‑blocking check for a prompt
            if not self.queue.empty():
                msg = self.queue.get_nowait()
                if msg.get("type") == "agent_prompt":
                    user_prompt = msg["data"]
                    self._process_task(user_prompt)
            time.sleep(0.5)

    def submit_task(self, user_prompt):
        """External method to enqueue a new prompt (called from GUI)."""
        self.queue.put({"type": "agent_prompt", "data": user_prompt})

    def _process_task(self, user_prompt):
        """Full pipeline: see → describe → reason → act."""
        try:
            # 1. Capture current frame from camera
            frame = None
            if self.camera:
                frame = self.camera.get_latest_frame()
            if frame is None:
                logging.warning("No camera frame available for scene description.")
                self.queue.put({"type": "agent_error", "data": "No camera frame"})
                return

            # 2. Get scene understanding from Gemini ER
            scene_desc = self.gemini.describe_scene(frame)
            logging.info(f"Scene description: {scene_desc}")

            # 3. Build LLM messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": FEW_SHOT_USER},
                {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
                {"role": "user", "content": f"Scene: {scene_desc}\nTask: {user_prompt}"}
            ]

            # 4. Request action plan from LLM
            plan_text = self.llm.chat(messages)
            logging.info(f"LLM plan: {plan_text}")
            if plan_text is None:
                self.queue.put({"type": "agent_error", "data": "LLM request failed"})
                return

            # 5. Parse JSON plan
            try:
                plan = json.loads(plan_text)
            except json.JSONDecodeError:
                # If LLM wraps in code block, try to extract
                import re
                json_match = re.search(r'\[.*\]', plan_text, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group(0))
                else:
                    self.queue.put({"type": "agent_error", "data": f"Could not parse plan: {plan_text}"})
                    return

            # 6. Execute actions one by one
            for step in plan:
                action = step.get("action")
                if action == "move_to":
                    x = step["x"]
                    y = step["y"]
                    z = step["z"]
                    self.robot.move_to_xyz(x, y, z, block=True)   # block until done
                elif action == "grip_close":
                    self.robot.gripper_close()
                    time.sleep(0.5)
                elif action == "grip_open":
                    self.robot.gripper_open()
                    time.sleep(0.5)
                elif action == "wait":
                    time.sleep(step.get("seconds", 1))
                else:
                    logging.warning(f"Unknown action: {action}")

            self.queue.put({"type": "agent_response", "data": "Task completed."})

        except Exception as e:
            logging.error(f"Agent processing error: {e}")
            self.queue.put({"type": "agent_error", "data": str(e)})