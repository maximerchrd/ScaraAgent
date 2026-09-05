# config.py
"""
Central configuration for the SCARA agent.
All tunable parameters live here; environment variables or a .env file should supply secrets.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path

# Load .env from the project root (next to main.py, config.py)
try:
    from dotenv import load_dotenv, find_dotenv
    # find_dotenv searches upward from this file's location
    env_path = find_dotenv(usecwd=False)
    if env_path:
        load_dotenv(env_path)
    else:
        # fallback: explicit path relative to config.py
        project_root = Path(__file__).parent
        dotenv_file = project_root / ".env"
        if dotenv_file.exists():
            load_dotenv(dotenv_file)
except ImportError:
    pass

@dataclass
class RobotConfig:
    # Serial
    serial_port: str = "/dev/cu.usbserial-110"
    baudrate: int = 115200
    serial_timeout: float = 0.05

    # Kinematics (mm & steps)
    link1_length: float = 325.5       # upper arm length (mm)
    link2_length: float = 327.5       # forearm length (mm)
    steps_per_deg_j1: float = 139.31
    steps_per_deg_j2: float = 63.83
    z_steps_per_mm: float = 200.218

    # Default joint limits
    j1_min: float = -150.0
    j1_max: float = 150.0
    j2_min: float = -150.0
    j2_max: float = 150.0
    z_min: float = -50.0
    z_max: float = 50.0

    # Jog defaults
    jog_step_linear: int = 50         # steps for Z, J1, J2 discrete jog
    jog_step_yaw: int = 5             # degrees for wrist yaw
    max_speed: int = 2000
    default_speed: int = 1000

    # Gripper timings (ms)
    gripper_open_duration: int = 1500
    gripper_close_duration: int = 800

    # Safe park position in work coordinates (used when no object found)
    park_x_mm: float = 300.0
    park_y_mm: float = -460.0
    park_z_mm: float = 10.0

    safe_z_mm: float = 60.0      # travel height (clears all objects)
    pick_z_mm: float = -3.0      # height for picking
    place_z_mm: float = 10.0    # height for releasing

    # Z sag correction coefficients (2nd order polynomial)
    # Z = a + b*X + c*Y + d*X^2 + e*Y^2 + f*X*Y
    z_correction_coeffs: tuple = (-68.977513, 0.270916, -0.090421, -0.000234, -0.000050, 0.000139)

    enable_gripper_refinement: bool = False

@dataclass
class VisionConfig:
    camera_index: Any = "http://10.86.51.32:8080/video"
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30

    # ArUco
    aruco_dict_name: str = "DICT_4X4_50"
    marker_size_mm: float = 100.0      # side length of physical marker
    camera_matrix: Any = None         # fill with your calibration
    dist_coeffs: Any = None

    marker_positions: dict = field(default_factory=lambda: {
        0: {"x": 644.1, "y": -105.8, "z": 0.0},    #corner 0 of marker 0 (master zero point)
        2: {"x": 453.4, "y": -376.2, "z": 0.0}, 
        3: {"x": 122.1, "y": -389.2, "z": 0.0}
    })

    # Workspace bounds for outlier rejection (mm, in robot coordinates)
    workspace_x_min: float = -200.0
    workspace_x_max: float = 800.0
    workspace_y_min: float = -500.0
    workspace_y_max: float = 200.0

    gripper_camera_scale_mm_per_px: float = 0.0   # will be loaded from calibration
    gripper_camera_offset_x: float = 0.0
    gripper_camera_offset_y: float = 0.0
    show_gripper_feed: bool = False

@dataclass
class LLMConfig:
    # Gemini VLM models
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = "gemini-3.1-flash-lite"
    #gemini_model: str = "gemini-robotics-er-1.6-preview"

    # Groq (ChatGPT-OSS 120B)
    chatgpt_endpoint: str = "https://api.groq.com/openai/v1/chat/completions"
    chatgpt_model: str = "openai/gpt-oss-120b"
    chatgpt_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # Model for the perception critic (cheaper, fast)
    critic_model: str = "llama-3.3-70b-versatile"
    planner_model: str = "openai/gpt-oss-120b"
    skill_decision_model: str = "openai/gpt-oss-20b"

    # Timeout for API calls (seconds)
    request_timeout: float = 15.0

    max_vlm_iterations: int = 2   # max number of VLM calls in perception loop

@dataclass
class ManualPixelCalibConfig:
    # Map ArUco ID to the corner you will jog to.
    # -1 = center, 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
    marker_corners: dict = field(default_factory=lambda: {
        #top-left: 0; top-right: 1; bottom-right: 2; bottom-left: 3; center: -1
        0: 2,   
        1: 2,
        2: 3,
        3: 2,
        4: 2,
        5: 3,
        6: 2,
        7: 3,
    })

@dataclass
class Config:
    robot: RobotConfig = field(default_factory=RobotConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    manual_pixel_calib: ManualPixelCalibConfig = field(default_factory=ManualPixelCalibConfig)
    
    # File paths
    measurements_csv: str = "measurements.csv"
    log_file: str = "scara_agent.log"

    # Agent behaviour
    agent_loop_delay: float = 0.5     # seconds between action checks
    skip_llm: bool = False

# Create a singleton instance for easy import
config = Config()