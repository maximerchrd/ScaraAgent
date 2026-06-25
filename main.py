# main.py
"""
Entry point for the SCARA Agent.
Launches the GUI, starts optional robot/vision/agent threads, and runs the main loop.
"""

import sys
import logging
from utils.logger import setup_logging
from config import config

# ----------------------------------------------------------------------
# Import components gracefully – if a module is missing, the app still works manually.
# ----------------------------------------------------------------------
try:
    from robot.serial_comm import SerialComm
except ImportError:
    SerialComm = None

try:
    from robot.controller import RobotController
except ImportError:
    RobotController = None

try:
    from vision.camera import CameraThread
except ImportError:
    CameraThread = None

try:
    from agent.orchestrator import AgentOrchestrator
except ImportError:
    AgentOrchestrator = None

from utils.safe_queue import SafeQueue
from gui.app import ScaraAgentApp

def main():
    # --- Logging ---
    setup_logging(level=logging.INFO, log_file=config.log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting SCARA Agent...")

    # --- Shared queue for inter-thread communication ---
    queue = SafeQueue()
    agent_queue = SafeQueue()

    # --- Optional: Serial & Robot Controller ---
    robot = None
    if SerialComm and RobotController:
        try:
            serial_comm = SerialComm(
                port=config.robot.serial_port,
                baudrate=config.robot.baudrate,
                queue=queue
            )
            robot = RobotController(serial_comm, queue)
            logger.info("Robot controller initialised.")
        except Exception as e:
            logger.error(f"Could not initialise robot: {e}")
    else:
        logger.warning("Robot modules not found – running in simulation mode.")

    # --- Optional: Vision ---
    camera = None
    if CameraThread:
        try:
            camera = CameraThread(
                camera_index=config.vision.camera_index,
                width=config.vision.frame_width,
                height=config.vision.frame_height,
                fps=config.vision.fps,
                queue=queue
            )
            camera.start()
            logger.info("Camera thread started.")
        except Exception as e:
            logger.error(f"Could not start camera: {e}")
    else:
        logger.info("Vision module not found – no camera feed.")

    # --- Optional: Agent Orchestrator ---
    orchestrator = None
    if AgentOrchestrator and robot:
        try:
            orchestrator = AgentOrchestrator(
                robot=robot,
                camera=camera,
                queue=queue,
                agent_queue=agent_queue,
                gemini_api_key=config.llm.gemini_api_key,
                chatgpt_endpoint=config.llm.chatgpt_endpoint
            )
            orchestrator.start()
            logger.info("Agent orchestrator ready.")
        except Exception as e:
            logger.error(f"Could not initialise orchestrator: {e}")

    # --- Launch GUI ---
    app = ScaraAgentApp(
        queue=queue,
        agent_queue=agent_queue,
        robot=robot,
        camera=camera,
        orchestrator=orchestrator
    )

    try:
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Shutting down by user request.")
    finally:
        # Cleanup
        if camera:
            camera.stop()
        if robot and robot.serial_comm and robot.serial_comm.is_open():
            robot.serial_comm.close()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()