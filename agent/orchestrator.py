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
import re

from agent.gemini_er import GeminiER
from agent.llm import ChatGPTOSS
from agent.prompt_templates import SYSTEM_PROMPT, FEW_SHOT_USER, FEW_SHOT_ASSISTANT, PLACEMENT_REFINEMENT_PROMPT
from vision.localization import MarkerLocalizer
from vision.aruco_detector import detect_aruco
from config import config
from agent.perception import PerceptionManager
from agent.skills.skill_manager import SkillManager


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
        self.skill_decision_llm = ChatGPTOSS(endpoint=chatgpt_endpoint, model=config.llm.skill_decision_model)
        self.calibrator = calibrator
        self.perception = PerceptionManager(self.gemini, critic_llm, queue=self.queue)
        self.skill_manager = SkillManager()

        self.skill_state = {}
        self.running = False
        self.thread = None
        # Track the last commanded move_to target for pick/place actions
        self._last_move_target = None

        # store the last homography for placement refinement
        self._last_homography = None
        self._last_frame_shape = None

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

    def _refine_placement(self, target_label, target_x, target_y):
        """
        Ask the VLM to find the best drop point on/inside the target.
        Only called when the target appears to be a container.
        Returns (refined_x_mm, refined_y_mm) or (None, None) on failure.
        """
        if self._last_homography is None or self._last_frame_shape is None:
            return None, None

        # Capture a fresh frame
        frame = self.camera.get_latest_frame() if self.camera else None
        if frame is None:
            return None, None

        h_img, w_img = frame.shape[:2]

        prompt = (
            "You are helping a SCARA robot place an object accurately.\n"
            f"The robot will place an object onto/into the target '{target_label}', "
            f"currently near world coordinates ({target_x:.0f}, {target_y:.0f}) mm.\n\n"
            "Look at this image of the workspace. If the target has an opening, cavity, "
            "or hollow area (like a box, bin, slot, tray with walls), point to the best "
            "location *inside* that cavity where the object should be dropped.\n"
            "If the target is a flat surface without an opening, point to its centre.\n"
            "Return ONLY a JSON array with exactly one object:\n"
            "[{\"point\": [y, x], \"label\": \"placement_target\"}]\n"
            "Points are [y, x] normalized 0-1000.\n"
            "Do NOT include any other text, markdown, or formatting."
        )

        try:
            result = self.gemini.localize_objects(frame, prompt=prompt)
            if result and len(result) > 0:
                refined_point = result[0]["point"]   # [y, x] normalized 0-1000
                px_x = (refined_point[1] / 1000.0) * w_img
                px_y = (refined_point[0] / 1000.0) * h_img

                # After computing refined_point and px_x, px_y:
                if self.queue:
                    self.queue.put({
                        "type": "placement_marker",
                        "data": {"pixel": [px_x, px_y], "label": f"drop {target_label}"}
                    })

                H = self._last_homography
                pt_pixel = np.array([px_x, px_y, 1.0])
                pt_world = H @ pt_pixel
                pt_world = pt_world / pt_world[2]
                x_mm, y_mm = pt_world[0], pt_world[1]

                if (config.vision.workspace_x_min <= x_mm <= config.vision.workspace_x_max and
                    config.vision.workspace_y_min <= y_mm <= config.vision.workspace_y_max):
                    logging.info(f"Placement refined for '{target_label}': "
                                 f"({target_x:.1f}, {target_y:.1f}) → ({x_mm:.1f}, {y_mm:.1f})")
                    self.queue.put({"type": "agent_reasoning",
                                    "data": f"🎯 Refined placement for '{target_label}' → inside cavity"})
                    return x_mm, y_mm
        except Exception as e:
            logging.warning(f"Placement refinement failed: {e}")

        return None, None

    def _is_container(self, label):
        """Return True if the label suggests a container with an opening."""
        container_keywords = ["box", "bin", "tray", "cup", "slot", "holder", "container", "basket"]
        label_lower = label.lower()
        return any(kw in label_lower for kw in container_keywords)

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

            # store for later use in placement refinement
            self._last_homography = H
            self._last_frame_shape = (h_img, w_img)

            # 4. Iterative VLM‑LLM perception loop
            objects = self.perception.perceive(frame, user_prompt)

            # Send final VLM object list to GUI for overlay
            self.queue.put({"type": "vlm_objects", "data": objects})

            located_objects = []
            for obj in objects:
                norm_y, norm_x = obj["point"]
                px_x = (norm_x / 1000.0) * w_img
                px_y = (norm_y / 1000.0) * h_img

                pt_pixel = np.array([px_x, px_y, 1.0])
                pt_world = H @ pt_pixel
                pt_world = pt_world / pt_world[2]
                x_mm, y_mm = pt_world[0], pt_world[1]

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

            # Store initial scene objects for later planning
            self.skill_state["scene_objects"] = located_objects

            # Build scene description for the LLM
            scene_desc = f"Objects detected: {json.dumps(located_objects)}\n"
            logging.info(f"Scene description: {scene_desc}")

            # ──────────────────────────────────────────────────
            # 5. Build LLM messages with skill awareness
            # ──────────────────────────────────────────────────
            skill_names = self.skill_manager.get_skill_names()
            
            system_content = SYSTEM_PROMPT + "\n\n" + skill_names + "\n\n"
            system_content += (
                "You are a fast decision-making model for a robot control system.\n"
                "Decide how to handle the user's task:\n\n"
                "1. If the task is simple (e.g., pick one object, move to a coordinate, place one object) "
                "and you can reliably plan it yourself, output ONLY the JSON array of actions.\n\n"
                "2. If the task matches one of the available skills above, output ONLY:\n"
                '{"skill_request": "<skill_name>"}\n\n'
                "3. If the task is complex, requires multi-step reasoning, or you are uncertain, "
                "output ONLY:\n"
                '{"escalate": true, "reason": "<brief reason>"}\n\n'
                "Do NOT output any commentary. Choose exactly one of the three output formats.\n"
                "You can also use the action \"perceive\" to ask the vision model for more information. "
                "Example: {\"action\": \"perceive\", \"prompt\": \"Find the red cube\", \"store_as\": \"red_cube_location\"}"
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": FEW_SHOT_USER},
                {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
                {"role": "user", "content": f"Scene: {scene_desc}\nTask: {user_prompt}"}
            ]

            logging.info(f"LLM full conversation:\n{json.dumps(messages, indent=2)}")

            # ──────────────────────────────────────────────────
            # 6. First LLM call — decide plan or request skill
            # ──────────────────────────────────────────────────
            plan = []
            if config.skip_llm:
                plan_text = "[]"
                llm_reasoning = ""
                logging.info("LLM skipped (testing mode)")
            else:
                # Use skill decision model for the first call
                llm_response = self._chat_with_retry(self.skill_decision_llm, messages)
                if llm_response is None:
                    self.queue.put({"type": "agent_error", "data": "LLM request failed (skill decision)"})
                    return
                plan_text = llm_response.get("content", "")
                llm_reasoning = llm_response.get("reasoning", "")

                if llm_reasoning:
                    self.queue.put({"type": "agent_reasoning", "data": llm_reasoning})

                # Parse decision output
                decision_type, decision_data = self._parse_decision_output(plan_text)

                if decision_type == 'plan':
                    # Simple task – use plan directly, no second call
                    logging.info("Decision model produced a direct plan.")
                    plan = decision_data

                elif decision_type == 'skill':
                    # Skill requested – fetch skill text and call main planner
                    skill_name = decision_data.get("skill_request")
                    logging.info(f"LLM requested skill: {skill_name}")
                    self.queue.put({"type": "agent_reasoning",
                                    "data": f"🔧 Using skill: {skill_name}"})

                    skill_text = self.skill_manager.get_skill_text(skill_name)
                    if skill_text is None:
                        self.queue.put({"type": "agent_error", "data": f"Skill '{skill_name}' not found"})
                        return

                    # Append skill request + skill details to conversation
                    messages.append({"role": "assistant", "content": json.dumps(decision_data)})
                    messages.append({"role": "user", "content": f"{skill_text}\n\nNow produce the action plan."})

                    messages[0]["content"] = SYSTEM_PROMPT 
                    llm_response = self._chat_with_retry(self.llm, messages)
                    if llm_response is None:
                        self.queue.put({"type": "agent_error", "data": "LLM request failed after skill injection"})
                        return
                    plan_text = llm_response.get("content", "")
                    llm_reasoning = llm_response.get("reasoning", "")

                    if llm_reasoning:
                        self.queue.put({"type": "agent_reasoning", "data": llm_reasoning})

                    # Parse final plan
                    plan = self._parse_plan(plan_text)
                    if plan is None:
                        self.queue.put({"type": "agent_error", "data": f"Could not parse plan: {plan_text}"})
                        return

                elif decision_type == 'escalate':
                    # Complex task – use main planner with original prompt
                    reason = decision_data.get("reason", "unknown")
                    logging.info(f"Decision model escalated to main planner: {reason}")
                    self.queue.put({"type": "agent_reasoning",
                                    "data": f"🧠 Escalating to planner: {reason}"})

                    messages[0]["content"] = SYSTEM_PROMPT 
                    llm_response = self._chat_with_retry(self.llm, messages)
                    if llm_response is None:
                        self.queue.put({"type": "agent_error", "data": "LLM request failed during escalation"})
                        return
                    plan_text = llm_response.get("content", "")
                    llm_reasoning = llm_response.get("reasoning", "")

                    if llm_reasoning:
                        self.queue.put({"type": "agent_reasoning", "data": llm_reasoning})

                    plan = self._parse_plan(plan_text)
                    if plan is None:
                        self.queue.put({"type": "agent_error", "data": f"Could not parse plan: {plan_text}"})
                        return

                else:
                    # Parsing failed – fall back to main planner as a safe default
                    logging.warning("Could not parse decision model output. Falling back to main planner.")
                    messages[0]["content"] = SYSTEM_PROMPT 
                    llm_response = self._chat_with_retry(self.llm, messages)
                    if llm_response is None:
                        self.queue.put({"type": "agent_error", "data": "LLM request failed during fallback"})
                        return
                    plan_text = llm_response.get("content", "")
                    plan = self._parse_plan(plan_text)
                    if plan is None:
                        self.queue.put({"type": "agent_error", "data": f"Could not parse plan: {plan_text}"})
                        return

                logging.info(f"Final plan: {plan}")

            # 8. Execute actions, re-plan after perceives
            max_replans = 5
            replan_count = 0

            self.queue.put({"type": "clear_perception_overlay", "data": None})

            while True:
                if plan is None:
                    self.queue.put({"type": "agent_error", "data": "Plan is None"})
                    return

                executed_perceive = False

                # Execute all consecutive perceive actions at the beginning of the plan
                for i, step in enumerate(plan):
                    if step.get("action") == "perceive":
                        prompt = step.get("prompt")
                        store_as = step.get("store_as", "last_perception")

                        if not prompt:
                            self.queue.put({"type": "agent_error", "data": "Perceive action missing prompt"})
                            return

                        frame = self.camera.get_latest_frame()
                        if frame is None:
                            self.queue.put({"type": "agent_error", "data": "No camera frame for perceive"})
                            return

                        logging.info(f"👁️ Perceive: {prompt[:80]}...")

                        result = self.gemini.query_json(frame, prompt=prompt)

                        if result is None:
                            logging.warning(f"Perceive '{store_as}' returned no JSON. Prompt was: {prompt}")
                            self.queue.put({
                                "type": "agent_error",
                                "data": f"Perceive '{store_as}' returned no usable result. Prompt may need stricter JSON formatting."
                            })
                            return

                        logging.info(f"Raw perception '{store_as}': {json.dumps(result)}")

                        # overlay raw perceived points on the camera feed
                        overlay_points = self._extract_perception_points(result, label=store_as)

                        # Auto-convert any normalised coordinates to world mm
                        result = self._convert_perceived_coordinates(result)

                        self.skill_state[store_as] = result

                        logging.info(f"Stored perception '{store_as}': {json.dumps(result)}")

                        # Log any converted point outside the robot workspace
                        self._log_workspace_violations(result, store_as)

                        self.queue.put({"type": "agent_reasoning",
                                        "data": f"👁️ Perception ({store_as}): {json.dumps(result)}"})

                        # Append the perception result to the conversation
                        messages.append({"role": "assistant", "content": json.dumps(step)})
                        messages.append({"role": "user", "content": f"Perception result for '{store_as}': {json.dumps(result)}"})

                        executed_perceive = True
                    else:
                        # Stop at the first non‑perceive action
                        break

                if executed_perceive:
                    # Re‑plan once with all perception results
                    known_objects = json.dumps(self.skill_state.get("scene_objects", []))

                    messages.append({"role": "user", "content": 
                        f"Known object positions from earlier scene detection: {known_objects}\n\n"
                        "The perceives have been executed. Their results are stored above. "
                        "Use BOTH the stored perception results AND the known object positions. "
                        "Remember: board_state.board_pieces are pieces already on the board. "
                        "Available playing pieces may be in the known object positions, off the board. "
                        "Output concrete numeric x and y values for all move_to actions. "
                        "Do NOT use $ref for computed coordinates. "
                        "Do NOT include any more perceive actions. "
                        "Output ONLY a JSON array of actions."})
                    messages[0]["content"] = SYSTEM_PROMPT

                    # Optional delay to avoid rate limits
                    time.sleep(2.0)

                    llm_response = self._chat_with_retry(self.llm, messages)
                    if llm_response is None:
                        self.queue.put({"type": "agent_error", "data": "LLM failed after perceive batch"})
                        return

                    plan_text = llm_response.get("content", "")
                    llm_reasoning = llm_response.get("reasoning", "")
                    if llm_reasoning:
                        self.queue.put({"type": "agent_reasoning", "data": llm_reasoning})

                    plan = self._parse_plan(plan_text)
                    if plan is None:
                        dec_type, dec_data = self._parse_decision_output(plan_text)
                        if dec_type == 'escalate':
                            self.queue.put({"type": "agent_error",
                                            "data": f"Planner escalated after perceive batch: {dec_data.get('reason', 'unknown')}"})
                        else:
                            self.queue.put({"type": "agent_error", "data": f"Could not parse plan after re-plan: {plan_text}"})
                        return

                    replan_count += 1
                    if replan_count >= max_replans:
                        self.queue.put({"type": "agent_error", "data": "Too many perceive/plan cycles"})
                        return

                    # Loop again; the new plan should contain no perceives, but if it does, it will be handled
                    continue

                else:
                    # No perceives: execute the plan normally
                    resolved_plan = self._resolve_references(plan)
                    if resolved_plan is None:
                        self.queue.put({
                            "type": "agent_error",
                            "data": "Could not resolve $ref references in plan"
                        })
                        return

                    for step in resolved_plan:
                        self._execute_action(step)
                    break

            self.queue.put({"type": "agent_response", "data": "Task completed."})

        except Exception as e:
            logging.error(f"Agent processing error: {e}")
            self.queue.put({"type": "agent_error", "data": str(e)})

    def _parse_decision_output(self, text: str):
        """
        Parse the output of the skill-decision model.
        Returns:
        - ('plan', list)          if a direct JSON array was returned
        - ('skill', dict)         if a skill_request was returned
        - ('escalate', dict)      if an escalate object was returned
        - (None, None)            if parsing fails
        """
        import re
        clean = re.sub(r'```(?:json)?\s*', '', text).strip()

        # Try to parse as JSON array directly
        try:
            data = json.loads(clean)
            if isinstance(data, list):
                return ('plan', data)
            if isinstance(data, dict):
                if "skill_request" in data:
                    return ('skill', data)
                if "escalate" in data:
                    return ('escalate', data)
        except json.JSONDecodeError:
            pass

        # Fallback: find array
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if match:
            try:
                plan = json.loads(match.group(0))
                return ('plan', plan)
            except json.JSONDecodeError:
                pass

        # Fallback: find skill_request object
        match = re.search(r'\{[^}]*"skill_request"[^}]*\}', clean, re.DOTALL)
        if match:
            try:
                return ('skill', json.loads(match.group(0)))
            except json.JSONDecodeError:
                pass

        # Fallback: find escalate object
        match = re.search(r'\{[^}]*"escalate"[^}]*\}', clean, re.DOTALL)
        if match:
            try:
                return ('escalate', json.loads(match.group(0)))
            except json.JSONDecodeError:
                pass

        return (None, None)

    def _chat_with_retry(self, llm_client, messages, max_retries=3, initial_delay=10.0):
        """
        Call the LLM with retry and exponential backoff.
        Returns the response dict or None if all attempts fail.
        """
        for attempt in range(max_retries):
            try:
                response = llm_client.chat(messages)
                if response is not None:
                    return response
                logging.warning(f"LLM returned None (attempt {attempt+1}/{max_retries})")
            except Exception as e:
                logging.warning(f"LLM call exception (attempt {attempt+1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                delay = initial_delay * (1.8 ** attempt)  # e.g. 1s, 2s, 4s...
                logging.info(f"Retrying LLM call in {delay:.1f}s...")
                time.sleep(delay)
        return None

    def _parse_plan(self, text: str):
        """Parse a JSON array of actions from the LLM output."""
        import re
        clean = re.sub(r'```(?:json)?\s*', '', text).strip()
        try:
            plan = json.loads(clean)
            if isinstance(plan, list):
                return plan
        except json.JSONDecodeError:
            pass
        # Extract first array
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if match:
            try:
                plan = json.loads(match.group(0))
                return plan
            except json.JSONDecodeError:
                pass
        return None

    def _execute_action(self, step):
        """Execute a single non‑perceive action."""
        action = step.get("action")
        if action == "move_to":
            x = step["x"]
            y = step["y"]

            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                self.queue.put({"type": "agent_error", "data": f"Invalid non-numeric coordinate in action: {step}"})
                return

            if not (config.vision.workspace_x_min <= x <= config.vision.workspace_x_max and
                    config.vision.workspace_y_min <= y <= config.vision.workspace_y_max):
                logging.error(f"Move out of workspace: ({x}, {y})")
                self.queue.put({"type": "agent_error", "data": f"Plan contains out-of-workspace move: ({x}, {y})"})
                return

            self._last_move_target = (x, y)
            self.robot.move_to_xyz(x, y, config.robot.safe_z_mm, block=True)
        elif action == "move_safe":
            self._last_move_target = None
            self.robot.move_to_xyz(
                config.robot.park_x_mm,
                config.robot.park_y_mm,
                config.robot.safe_z_mm,
                block=True
            )
        elif action == "pick":
            pick_x = step.get("x", self._last_move_target[0] if self._last_move_target else None)
            pick_y = step.get("y", self._last_move_target[1] if self._last_move_target else None)
            if pick_x is None or pick_y is None:
                pick_x, pick_y = self.robot.get_work_position()[0], self.robot.get_work_position()[1]
                logging.warning("pick() without coordinates or preceding move_to")

            # Lower to pick height at the stored XY
            self.robot.move_to_xyz(pick_x, pick_y, config.robot.pick_z_mm, block=True)
            self.robot.gripper_close()
            time.sleep(max(0.5, self.robot.gripper_close_duration / 1000.0 + 0.2))
            # Raise back at the same XY
            self.robot.move_to_xyz(pick_x, pick_y, config.robot.safe_z_mm, block=True)
        elif action == "place":
            place_x = step.get("x", self._last_move_target[0] if self._last_move_target else None)
            place_y = step.get("y", self._last_move_target[1] if self._last_move_target else None)
            if place_x is None or place_y is None:
                place_x, place_y = self.robot.get_work_position()[0], self.robot.get_work_position()[1]
                logging.warning("place() without coordinates or preceding move_to")

            orig_x, orig_y = place_x, place_y

            target_label = step.get("target", "the target")
            if self._is_container(target_label):
                refined_x, refined_y = self._refine_placement(
                    target_label, place_x, place_y
                )
                if refined_x is not None and refined_y is not None:
                    place_x, place_y = refined_x, refined_y
                    logging.info(f"📍 Placement shift for '{target_label}': "
                                f"Δx={place_x - orig_x:.1f}mm, Δy={place_y - orig_y:.1f}mm")
                else:
                    logging.info(f"⚠️  Refinement failed for '{target_label}' — using original coordinates")

            self.robot.move_to_xyz(place_x, place_y, config.robot.place_z_mm, block=True)
            self.robot.gripper_open()
            time.sleep(max(0.5, self.robot.gripper_open_duration / 1000.0 + 0.2))
            self.robot.move_to_xyz(place_x, place_y, config.robot.safe_z_mm, block=True)
        else:
            logging.warning(f"Unknown action: {action}")

    def _resolve_references(self, plan):
        """
        Replace $ref placeholders in plan actions with actual values from skill_state.
        Returns a new plan with concrete values, or None if any reference cannot be resolved.
        """
        import re
        resolved_plan = []
        for step in plan:
            new_step = {}
            for key, value in step.items():
                if isinstance(value, str) and value.startswith("$ref:"):
                    ref_path = value[5:].strip()  # e.g., "target_cell.cell_center[1]"
                    resolved = self._resolve_ref_path(ref_path)
                    if resolved is None:
                        logging.warning(f"Could not resolve reference: {ref_path}")
                        return None
                    new_step[key] = resolved
                else:
                    new_step[key] = value
            resolved_plan.append(new_step)
        return resolved_plan

    def _resolve_ref_path(self, ref_path):
        """
        Resolve a dot/bracket path like 'target_cell.cell_center[1]' against skill_state.
        Returns the value or None if not found.
        """
        parts = re.split(r'[\.\[\]]+', ref_path)
        parts = [p for p in parts if p != '']
        current = self.skill_state
        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    return None
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    if idx >= len(current):
                        return None
                    current = current[idx]
                except ValueError:
                    return None
            else:
                return None
        return current


    def _normalized_to_world(self, norm_x, norm_y):
        """
        Convert normalised 0-1000 image coordinates to robot world mm.
        Uses the current homography and frame size.
        """
        if self._last_homography is None or self._last_frame_shape is None:
            return None, None

        h_img, w_img = self._last_frame_shape
        px_x = (norm_x / 1000.0) * w_img
        px_y = (norm_y / 1000.0) * h_img

        pt_pixel = np.array([px_x, px_y, 1.0])
        pt_world = self._last_homography @ pt_pixel
        pt_world = pt_world / pt_world[2]
        return float(pt_world[0]), float(pt_world[1])

    def _convert_perceived_coordinates(self, data):
        """
        Recursively convert normalised coordinate fields in a perception result
        to robot world mm. Recognises:
        - {"point": [y, x]}
        - {"x": ..., "y": ...} where both are normalised 0-1000
        - {"top_left": [y,x], ...} corner dictionaries
        - Lists of the above
        Returns a new object with converted coordinates.
        """
        if isinstance(data, dict):
            converted = {}

            for key, value in data.items():
                # Convert a point array [y, x]
                if key == "point" and isinstance(value, (list, tuple)) and len(value) == 2:
                    y_norm, x_norm = float(value[0]), float(value[1])
                    wx, wy = self._normalized_to_world(x_norm, y_norm)
                    if wx is not None:
                        converted[key] = [wx, wy]   # store [x, y] world mm consistently
                    else:
                        converted[key] = value

                # Convert corner keys (top_left, etc.)
                elif key in ("top_left", "top_right", "bottom_left", "bottom_right") and isinstance(value, (list, tuple)) and len(value) == 2:
                    y_norm, x_norm = float(value[0]), float(value[1])
                    wx, wy = self._normalized_to_world(x_norm, y_norm)
                    if wx is not None:
                        converted[key] = [wx, wy]   # store [x, y] world mm consistently
                    else:
                        converted[key] = value

                # Recurse into nested dictionaries/lists
                else:
                    converted[key] = self._convert_perceived_coordinates(value)

            # Special case: dict with "x" and "y" keys (both normalised)
            if "x" in data and "y" in data:
                try:
                    x_val = float(data["x"])
                    y_val = float(data["y"])
                    if 0 <= x_val <= 1000 and 0 <= y_val <= 1000:
                        wx, wy = self._normalized_to_world(x_val, y_val)
                        if wx is not None:
                            converted["x"] = wx
                            converted["y"] = wy
                except (ValueError, TypeError):
                    pass

            return converted

        if isinstance(data, list):
            return [self._convert_perceived_coordinates(item) for item in data]

        return data

    def _extract_perception_points(self, data, label=None):
        """
        Recursively extract normalised [y, x] points (0-1000) from a perceive result.
        Returns a list of dicts: {"point": [y, x], "label": str}
        """
        points = []

        if isinstance(data, dict):
            # Direct "point" field
            if "point" in data and isinstance(data["point"], (list, tuple)) and len(data["point"]) == 2:
                try:
                    y, x = float(data["point"][0]), float(data["point"][1])
                    lbl = data.get("label", label or "perceive")
                    points.append({"point": [y, x], "label": lbl})
                except (ValueError, TypeError):
                    pass

            # Corners often returned by grid perceive
            for key in ("top_left", "top_right", "bottom_left", "bottom_right"):
                if key in data and isinstance(data[key], (list, tuple)) and len(data[key]) == 2:
                    try:
                        y, x = float(data[key][0]), float(data[key][1])
                        points.append({"point": [y, x], "label": key})
                    except (ValueError, TypeError):
                        pass

            # Direct x,y fields (some prompts may return this)
            if "x" in data and "y" in data:
                try:
                    x_val = float(data["x"])
                    y_val = float(data["y"])
                    if 0 <= x_val <= 1000 and 0 <= y_val <= 1000:
                        points.append({"point": [y_val, x_val], "label": label or "perceive"})
                except (ValueError, TypeError):
                    pass

            # Recurse into other values (skip fields we've already handled)
            for key, value in data.items():
                if key in ("point", "top_left", "top_right", "bottom_left", "bottom_right", "x", "y", "label"):
                    continue
                points.extend(self._extract_perception_points(value, label=key))

        elif isinstance(data, list):
            for item in data:
                points.extend(self._extract_perception_points(item, label=label))

        return points

    def _log_workspace_violations(self, data, store_as, path=""):
        """
        Recursively check converted perception data for world coordinates
        that fall outside the configured workspace and log a warning.
        """
        if isinstance(data, dict):
            # Check a point array [x, y]
            if "point" in data and isinstance(data["point"], (list, tuple)) and len(data["point"]) == 2:
                try:
                    x, y = float(data["point"][0]), float(data["point"][1])
                    if not (config.vision.workspace_x_min <= x <= config.vision.workspace_x_max and
                            config.vision.workspace_y_min <= y <= config.vision.workspace_y_max):
                        logging.warning(
                            f"Perception '{store_as}' point '{path}' outside workspace: "
                            f"({x:.1f}, {y:.1f})"
                        )
                except (ValueError, TypeError):
                    pass

            # Check direct x/y fields
            if "x" in data and "y" in data:
                try:
                    x = float(data["x"])
                    y = float(data["y"])
                    if not (config.vision.workspace_x_min <= x <= config.vision.workspace_x_max and
                            config.vision.workspace_y_min <= y <= config.vision.workspace_y_max):
                        logging.warning(
                            f"Perception '{store_as}' x/y '{path}' outside workspace: "
                            f"({x:.1f}, {y:.1f})"
                        )
                except (ValueError, TypeError):
                    pass

            # Recurse into nested structures
            for key, value in data.items():
                if key in ("point", "x", "y"):
                    continue
                self._log_workspace_violations(value, store_as, f"{path}.{key}")

        elif isinstance(data, list):
            for idx, item in enumerate(data):
                self._log_workspace_violations(item, store_as, f"{path}[{idx}]")