# kinematics.py
import math
from config import config as cfg

L1 = cfg.robot.link1_length
L2 = cfg.robot.link2_length
STEPS_PER_MM_Z = cfg.robot.z_steps_per_mm
STEPS_PER_DEG_J1 = cfg.robot.steps_per_deg_j1
STEPS_PER_DEG_J2 = cfg.robot.steps_per_deg_j2
GRIPPER_LENGTH = cfg.robot.gripper_length_mm

def xyz_to_joints_closest(x_mm, y_mm, z_mm, current_j1_steps, current_j2_steps):
    """
    Choose the elbow configuration that minimises total joint movement.
    """
    steps_z = round(z_mm * STEPS_PER_MM_Z)
    
    # Both solutions
    _, j1_up, j2_up = xyz_to_joints(x_mm, y_mm, z_mm, elbow_up=True)
    _, j1_down, j2_down = xyz_to_joints(x_mm, y_mm, z_mm, elbow_up=False)
    
    # Movement in degrees
    cur_j1 = current_j1_steps / STEPS_PER_DEG_J1
    cur_j2 = current_j2_steps / STEPS_PER_DEG_J2
    
    move_up = abs(j1_up / STEPS_PER_DEG_J1 - cur_j1) + abs(j2_up / STEPS_PER_DEG_J2 - cur_j2)
    move_down = abs(j1_down / STEPS_PER_DEG_J1 - cur_j1) + abs(j2_down / STEPS_PER_DEG_J2 - cur_j2)
    
    if move_up <= move_down:
        return steps_z, j1_up, j2_up
    else:
        return steps_z, j1_down, j2_down

def _wrist_to_joint_angles(x_wrist, y_wrist, elbow_up=True):
    D = (x_wrist**2 + y_wrist**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = max(-1.0, min(1.0, D))
    if elbow_up:
        theta2 = math.atan2(math.sqrt(1 - D**2), D)
    else:
        theta2 = math.atan2(-math.sqrt(1 - D**2), D)
    theta1 = math.atan2(y_wrist, x_wrist) - math.atan2(
        L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    return theta1, theta2


# ============================================================
# INVERSE KINEMATICS (XYZ → step targets)
# ============================================================
def _ik_two_link(x, y, L2_eff, elbow_up=True):
    D = (x*x + y*y - L1*L1 - L2_eff*L2_eff) / (2 * L1 * L2_eff)
    D = max(-1.0, min(1.0, D))
    if elbow_up:
        theta2 = math.atan2(math.sqrt(1 - D*D), D)
    else:
        theta2 = math.atan2(-math.sqrt(1 - D*D), D)
    theta1 = math.atan2(y, x) - math.atan2(
        L2_eff * math.sin(theta2),
        L1 + L2_eff * math.cos(theta2))
    return theta1, theta2


def xyz_to_joints(x_mm, y_mm, z_mm, pitch_rad=0.0, elbow_up=True):
    L2_eff = L2 + GRIPPER_LENGTH * math.sin(pitch_rad)
    theta1, theta2 = _ik_two_link(x_mm, y_mm, L2_eff, elbow_up)

    steps_j1 = round(math.degrees(theta1) * STEPS_PER_DEG_J1)
    steps_j2 = round(math.degrees(theta2) * STEPS_PER_DEG_J2)
    steps_z  = round(z_mm * STEPS_PER_MM_Z)
    return steps_z, steps_j1, steps_j2

# ============================================================
# FORWARD KINEMATICS (steps → real‑world XYZ)
# ============================================================
def joints_to_xyz(steps_z, steps_j1, steps_j2, pitch_rad=0.0):
    """Return TCP (gripper tip) position."""
    z_wrist = steps_z / STEPS_PER_MM_Z
    theta1 = math.radians(steps_j1 / STEPS_PER_DEG_J1)
    theta2 = math.radians(steps_j2 / STEPS_PER_DEG_J2)
    phi = theta1 + theta2

    x_wrist = L1 * math.cos(theta1) + L2 * math.cos(phi)
    y_wrist = L1 * math.sin(theta1) + L2 * math.sin(phi)

    x_tcp = x_wrist + GRIPPER_LENGTH * math.sin(pitch_rad) * math.cos(phi)
    y_tcp = y_wrist + GRIPPER_LENGTH * math.sin(pitch_rad) * math.sin(phi)
    z_tcp = z_wrist
    return x_tcp, y_tcp, z_tcp