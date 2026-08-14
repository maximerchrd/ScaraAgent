# pack_project.py
"""
Pack all relevant project files into a single text file for AI assistance.
Run this file directly and it creates project_dump.txt.
"""

import os
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_FILE = "project_dump.txt"

PROJECT_DESCRIPTION = """
# SCARA Agent — Project Overview
#
# This is a 5‑DOF SCARA robot control application evolving into an agentic system.
# The robot uses G‑code over serial, with a Python GUI (customtkinter).
#
# Current capabilities:
#   - Manual jog control (joints + Cartesian XYZ)
#   - Live position feedback: endswitch monitoring, homing, emergency stop
#   - Gripper control (open/close with timed stops)
#   - Top‑down arm visualization on canvas
#   - Serial communication via pyserial (queue‑based)
#   - Webcam feed with ArUco marker detection (fallback) and pre‑calibrated homography from file
#   - Agent pipeline: VLM scene description → LLM skill decision → planner → execution
#   - Skill library (JSON) with LLM‑driven selection and escalation to main planner
#   - Mid‑plan VLM querying via "perceive" actions (batched, auto‑converted to world mm — NOT YET TESTED)
#   - Retry/backoff for LLM calls
#
# Agentic enhancements (phased):
#   Phase 1 – Iterative VLM questioning (implemented):
#       The LLM critic examines the VLM’s first scene description, identifies missing/ambiguous
#       objects, and formulates targeted follow‑up questions to the VLM. This yields a more
#       reliable world model before planning.
#
#   Phase 2 – Closed‑loop verification & recovery (planned, not yet implemented):
#       After every pick/place, a verification image is taken and the VLM is asked to
#       confirm the expected state change. If verification fails, the LLM generates a
#       repair plan.
#
#   Phase 3 – Escalation to Gemini Robotics ER (planned, not yet implemented):
#       When confidence is low (empty results, contradictions, or repeated verification
#       failures), the full image is sent to a more powerful VLM for high‑accuracy perception.
#       Currently, the system escalates to the main planner LLM when the skill decision model
#       is uncertain, but not yet to a specialised VLM.
#
#   Phase 4 – Persistent spatial memory & hierarchical planning (planned, not yet implemented):
#       A scene graph (object → coordinate) is maintained and updated after each action.
#       The LLM uses it for multi‑step planning without re‑detection. Complex tasks are
#       broken into sub‑goals by a hierarchical planner, which simulates geometric
#       constraints before committing to actions.
#
#   Skill Library (initial implementation):
#       A JSON file (agent/skills/library.json) contains predefined skills, e.g.:
#         - play_tic_tac_toe_on_3x3_grid
#         - tidy_table_by_sorting_objects_into_containers
#         - build_tower_by_stacking_objects
#       A lightweight LLM (skill_decision_model) decides whether to use a skill, output a
#       direct plan, or escalate to the main planner. When a skill is selected, its full
#       instructions and geometry are injected into the planner’s context.
#       The planner can issue "perceive" actions to query the VLM mid‑plan; consecutive
#       perceives are batched and re‑planned once. Perceived coordinates are automatically
#       converted from normalised image space to robot world mm (untested).
#
# Architecture:
#   - gui/        — all UI panels and widgets
#   - robot/      — kinematics, serial comm, high‑level controller, calibration
#   - vision/     — camera thread, ArUco detection, image utilities, marker localisation
#   - agent/      — Gemini VLM interface, LLM interface, orchestrator, prompt templates,
#                   perception manager, skill library
#   - utils/      — thread‑safe queue, logging setup
#   - config.py   — all tunable parameters in dataclasses (including marker positions)
#   - main.py     — entry point, wires everything together
#
# The system uses a SafeQueue for inter‑thread communication.
# The GUI runs in the main thread; robot serial, camera, and agent each run in daemon threads.
"""

# Which file extensions to include
INCLUDE_EXTENSIONS = {'.py', '.txt', '.md', '.yaml', '.yml', '.json', '.env.example'}

# Directories to exclude entirely
EXCLUDE_DIRS = {'__pycache__', '.git', '.venv', 'venv', 'weights', '.idea', '.vscode'}

# Files to exclude
EXCLUDE_FILES = {'pack_project.py', OUTPUT_FILE}


def should_include(filepath: Path) -> bool:
    if filepath.suffix not in INCLUDE_EXTENSIONS:
        return False
    for part in filepath.parts:
        if part in EXCLUDE_DIRS:
            return False
    if filepath.name in EXCLUDE_FILES:
        return False
    return True


def pack_project(root_dir: str = ".") -> None:
    root = Path(root_dir).resolve()
    output_path = root / OUTPUT_FILE

    files_found = []
    for filepath in sorted(root.rglob("*")):
        if filepath.is_file() and should_include(filepath):
            files_found.append(filepath)

    with open(output_path, 'w', encoding='utf-8') as out:
        # Project description first
        out.write(PROJECT_DESCRIPTION.strip())
        out.write(f"\n# Packed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        out.write(f"\n# Total files: {len(files_found)}")
        out.write(f"\n# {'=' * 60}\n\n")

        for filepath in files_found:
            rel_path = filepath.relative_to(root)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                content = f"# ERROR reading file: {e}"

            out.write(f"# {'=' * 60}\n")
            out.write(f"# FILE: {rel_path}\n")
            out.write(f"# {'=' * 60}\n\n")
            out.write(content)
            out.write("\n\n\n")

        out.write(f"# Total files packed: {len(files_found)}\n")

    print(f"✅ Packed {len(files_found)} files into {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    pack_project()