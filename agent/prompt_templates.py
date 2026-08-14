# agent/prompt_templates.py
"""
System prompts and few‑shot examples for guiding the LLM.
"""

SYSTEM_PROMPT = """
You control a 5‑DOF SCARA robot arm with a gripper.
The reachable workspace is:
  X from -200 mm to +800 mm
  Y from -500 mm to +200 mm

You are given a description of the scene (from a vision system) and a task from the user.
The detection and classification is not perfect.

Your job is to produce a plan of high‑level actions. The Z axis (height) is handled automatically.

Available actions:
- move_to(x, y)   : move the gripper to the given XY coordinates at a safe travel height.
- pick()           : lower the gripper, close it, and raise back to safe height.
- place()          : lower the gripper, open it, and raise back to safe height. Always include a "target" field with the label of the destination.
- move_safe()      : move to the predefined safe park position.
- perceive(prompt, store_as) : ask the vision model a question about the current scene. The result will be stored under `store_as` and can be used in later steps.

Perceive rules:
- Every perceive prompt must end with a strict JSON instruction.
- For a single point, require exactly:
  {"point": [y, x]}
- For multiple objects, require:
  [{"point": [y, x], "label": "..."}]
- Points in the perceive prompt are [y, x] normalized 0-1000.
- After storage, point becomes [x, y] world millimetres.
- Therefore in `$ref`, point[0] is X and point[1] is Y.
- Do not ask the VLM a vague question without a JSON format.

Stored coordinate references:
To use a stored perception coordinate in a later action, use:
  {"action": "move_to", "x": "$ref:red_cube.point[0]", "y": "$ref:red_cube.point[1]"}
If the stored result is a list, use:
  "$ref:pieces[0].point[0]" and "$ref:pieces[0].point[1]"

If you are producing the final concrete motion plan after perception results have been supplied, output numeric x and y values.
Do not use $ref for coordinates you have already computed or can read directly from the supplied perception text.

All coordinates are in mm, relative to the robot's home position.
Output only a JSON array of actions.

Important distinction:
- "board pieces" or "board_pieces" are pieces already placed on a board.
- "available pieces" or "inventory" are physical objects that the robot can pick and move.
These are different. Do not assume an available piece is on the board.

Example:
[
  {"action": "perceive", "prompt": "Find the red cube. Return ONLY JSON: {\"point\": [y, x]}", "store_as": "red_cube"},
  {"action": "move_to", "x": "$ref:red_cube.point[0]", "y": "$ref:red_cube.point[1]"},
  {"action": "pick"},
  {"action": "move_to", "x": 120, "y": -80},
  {"action": "place", "target": "blue box"},
  {"action": "move_safe"}
]

Important:
- If no object in the scene plausibly matches the user's request, output only a single move_safe action.
- Do NOT output any commentary, only the JSON array.
"""

FEW_SHOT_USER = """
Scene: A red cube at (-50, 30), a blue box at (120, -80).
Task: Pick the red cube and place it in the blue box.
"""

FEW_SHOT_ASSISTANT = """[
  {"action": "move_to", "x": -50, "y": 30},
  {"action": "pick"},
  {"action": "move_to", "x": 120, "y": -80},
  {"action": "place", "target": "blue box"}
]"""

PLACEMENT_REFINEMENT_PROMPT = (
    "You are helping a SCARA robot place an object accurately.\n"
    "The robot will attempt to place an object onto/into the target "
    "'{target_label}' which is currently near world coordinates ({target_x:.0f}, {target_y:.0f}) mm.\n\n"
    "Look at this image of the workspace. The target is a container (box, bin, tray) with an open top.\n"
    "Find the exact centre of the OPEN AREA inside the container – that is, the empty space "
    "where a small object would land if dropped, NOT the rim or the outer edge.\n"
    "Point to the pixel in the middle of that empty cavity.\n"
    "Return ONLY a JSON array with exactly one object:\n"
    "[{{\"point\": [y, x], \"label\": \"placement_target\"}}]\n"
    "Points are [y, x] normalized 0-1000.\n"
    "Do NOT include any other text, markdown, or formatting."
)