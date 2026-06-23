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
# This is a 5-DOF SCARA robot control application evolving into an agentic system.
# The robot uses G-code over serial, with a Python GUI (customtkinter).
#
# Current state:
#   - Manual jog control (joints + Cartesian XYZ)
#   - Live position feedback, endswitch monitoring, homing, emergency stop
#   - Gripper control (open/close with timed stops)
#   - Top-down arm visualization on canvas
#   - Serial communication via pyserial
#
# Goal / work-in-progress:
#   - Add webcam feed with ArUco marker detection (vision/camera.py, vision/aruco_detector.py)
#   - Integrate Gemini Robotics ER (VLM) to describe the scene
#   - Integrate ChatGPT-OSS 120B (LLM) to reason about the scene and produce action plans
#   - Orchestrator agent converts LLM plans into robot G-code commands
#   - GUI panels for camera view and agent prompt/response
#
# Architecture:
#   - gui/        — all UI panels and widgets
#   - robot/      — kinematics, serial comm, high-level controller, calibration
#   - vision/     — camera thread, ArUco detection, image utilities
#   - agent/      — Gemini VLM interface, LLM interface, orchestrator, prompt templates
#   - utils/      — thread-safe queue, logging setup
#   - config.py   — all tunable parameters in dataclasses
#   - main.py     — entry point, wires everything together
#
# The system uses a SafeQueue for inter-thread communication.
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