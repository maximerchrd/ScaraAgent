# agent/orchestrator.py
"""
AgentOrchestrator ties vision, Gemini, LLM and robot together.
Runs in a separate thread to avoid blocking the GUI.
"""

import threading
import time
import json
import logging
import numpy as np

from agent.gemini_er import GeminiER
from agent.llm import ChatGPTOSS
from agent.prompt_templates import SYSTEM_PROMPT, FEW_SHOT_USER, FEW_SHOT_ASSISTANT
from vision.localization import MarkerLocalizer
from vision.aruco_detector import detect_aruco
from config import config
from agent.perception import PerceptionManager


class AgentOrchestrator:
    def __init__(self, robot, camera, queue, agent_queue, gemini_api_key=None,
                 chatgpt_endpoint=None, calibrator=None):
        self.robot = robot
        self.camera = camera
        self.queue = queue
        self.agent_queue = agent_queue
        self.gemini = GeminiER(api_key=gemini_api_key)
        self.llm = ChatGPTOSS(endpoint=chatgpt_endpoint, model=config.llm.planner_model)
        critic_llm = ChatGPTOSS(endpoint=chatgpt_endpoint, model=config.llm.critic_model)
        self.calibrator = calibrator
        self.perception = PerceptionManager(self.gemini, critic_llm, queue=self.queue)

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
            if not self.agent_queue.empty():
                msg = self.agent_queue.get_nowait()
                if msg.get("type") == "agent_prompt":
                    user_prompt = msg["data"]
                    self._process_task(user_prompt)
            time.sleep(0.5)

    def submit_task(self, user_prompt):
        """External method to enqueue a new prompt (called from GUI)."""
        self.agent_queue.put({"type": "agent_prompt", "data": user_prompt})

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

            h_img, w_img = frame.shape[:2]

            # 2. Detect ArUco markers
            if self.calibrator and self.calibrator.is_calibrated:
                detected_markers = self.calibrator.calibrated_corners
                logging.info("Using calibrated marker positions.")
            else:
                markers = detect_aruco(frame)
                detected_markers = {m["id"]: m["corners"] for m in markers}

            # 3. Compute global homography from all visible markers
            localizer = MarkerLocalizer()
            H = localizer.compute_global_homography(detected_markers)
            if H is None:
                logging.error("Failed to compute homography from visible markers.")
                self.queue.put({"type": "agent_error", "data": "Homography computation failed"})
                return

            # 4. Iterative VLM‑LLM perception loop
            objects = self.perception.perceive(frame, user_prompt)

            # Send final VLM object list to GUI for overlay
            self.queue.put({"type": "vlm_objects", "data": objects})

            located_objects = []
            for obj in objects:
                # Gemini returns point as [y, x] normalized 0-1000
                norm_y, norm_x = obj["point"]
                # Convert to pixel coordinates (x = column, y = row)
                px_x = (norm_x / 1000.0) * w_img
                px_y = (norm_y / 1000.0) * h_img

                # Apply homography to get world mm coordinates
                pt_pixel = np.array([px_x, px_y, 1.0])
                pt_world = H @ pt_pixel
                pt_world = pt_world / pt_world[2]  # homogeneous normalization
                x_mm, y_mm = pt_world[0], pt_world[1]

                # Filter out homography outliers
                if not (config.vision.workspace_x_min <= x_mm <= config.vision.workspace_x_max and
                        config.vision.workspace_y_min <= y_mm <= config.vision.workspace_y_max):
                    logging.warning(f"Outlier rejected: {obj['label']} at ({x_mm:.1f}, {y_mm:.1f}) "
                                    f"— outside workspace bounds")
                    continue

                located_objects.append({
                    "label": obj["label"],
                    "x_mm": round(x_mm, 1),
                    "y_mm": round(y_mm, 1),
                })

            # Build scene description for the LLM
            scene_desc = f"Objects detected: {json.dumps(located_objects)}\n"
            logging.info(f"Scene description: {scene_desc}")

            # 5. Build LLM messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": FEW_SHOT_USER},
                {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
                {"role": "user", "content": f"Scene: {scene_desc}\nTask: {user_prompt}"}
            ]

            logging.info(f"LLM full conversation:\n{json.dumps(messages, indent=2)}")

            # 6. Request action plan from LLM
            if config.skip_llm:
                plan_text = "[]"
                llm_reasoning = ""
                logging.info("LLM skipped (testing mode)")
            else:
                llm_response = self.llm.chat(messages)
                if llm_response is None:
                    self.queue.put({"type": "agent_error", "data": "LLM request failed"})
                    return
                plan_text = llm_response.get("content", "")
                llm_reasoning = llm_response.get("reasoning", "")
                
                if llm_reasoning:
                    self.queue.put({"type": "agent_reasoning", "data": llm_reasoning})
                
                logging.info(f"LLM plan: {plan_text}")

            # 7. Parse JSON plan
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

            # 8. Execute actions one by one
            for step in plan:
                action = step.get("action")
                if action == "move_to":
                    x = step["x"]
                    y = step["y"]
                    # Move to XY at safe travel height
                    self.robot.move_to_xyz(x, y, config.robot.safe_z_mm, block=True)
                elif action == "move_safe":
                    self.robot.move_to_xyz(
                        config.robot.park_x_mm,
                        config.robot.park_y_mm,
                        config.robot.safe_z_mm,    # park at safe height
                        block=True
                    )
                elif action == "pick":
                    # Lower to pick height
                    self.robot.move_to_xyz(
                        self.robot.get_work_position()[0],
                        self.robot.get_work_position()[1],
                        config.robot.pick_z_mm,
                        block=True
                    )
                    # Close gripper
                    self.robot.gripper_close()
                    time.sleep(0.5)
                    # Raise back to safe height
                    self.robot.move_to_xyz(
                        self.robot.get_work_position()[0],
                        self.robot.get_work_position()[1],
                        config.robot.safe_z_mm,
                        block=True
                    )
                elif action == "place":
                    self.robot.move_to_xyz(
                        self.robot.get_work_position()[0],
                        self.robot.get_work_position()[1],
                        config.robot.pick_z_mm,
                        block=True
                    )
                    self.robot.gripper_open()
                    time.sleep(0.5)
                    self.robot.move_to_xyz(
                        self.robot.get_work_position()[0],
                        self.robot.get_work_position()[1],
                        config.robot.safe_z_mm,
                        block=True
                    )
                else:
                    logging.warning(f"Unknown action: {action}")

            self.queue.put({"type": "agent_response", "data": "Task completed."})

        except Exception as e:
            logging.error(f"Agent processing error: {e}")
            self.queue.put({"type": "agent_error", "data": str(e)})