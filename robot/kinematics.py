# kinematics.py
import math

# ============================================================
# CONFIGURATION – EDIT AFTER CALIBRATION
# ============================================================
STEPS_PER_MM_Z = 226.63          # adjust: steps per mm vertical
STEPS_PER_DEG_J1 = 139.31       # adjust: steps per degree J1
STEPS_PER_DEG_J2 = 63.83       # adjust: steps per degree J2
L1 = 325.5                     # upper arm length (mm)
L2 = 327.5                     # forearm length (mm)

# ============================================================
# FORWARD KINEMATICS (steps → real‑world XYZ)
# ============================================================
def joints_to_xyz(steps_z, steps_j1, steps_j2):
    """
    Convert step positions to millimetre coordinates.
    Returns (x_mm, y_mm, z_mm)
    """
    z_mm = steps_z / STEPS_PER_MM_Z
    theta1 = math.radians(steps_j1 / STEPS_PER_DEG_J1)
    theta2 = math.radians(steps_j2 / STEPS_PER_DEG_J2)
    
    x = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
    y = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
    return x, y, z_mm


# ============================================================
# INVERSE KINEMATICS (XYZ → step targets)
# ============================================================
def xyz_to_joints(x_mm, y_mm, z_mm, elbow_up=True):
    """
    Return (steps_z, steps_j1, steps_j2) for a given XYZ.
    elbow_up = True gives the "elbow up" solution (default).
    """
    D = (x_mm**2 + y_mm**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = max(-1.0, min(1.0, D))   # clamp for safety
    
    if elbow_up:
        theta2 = math.atan2(math.sqrt(1 - D**2), D)
    else:
        theta2 = math.atan2(-math.sqrt(1 - D**2), D)
    
    theta1 = math.atan2(y_mm, x_mm) - math.atan2(L2 * math.sin(theta2), 
                                                   L1 + L2 * math.cos(theta2))
    
    steps_j1 = round(math.degrees(theta1) * STEPS_PER_DEG_J1)
    steps_j2 = round(math.degrees(theta2) * STEPS_PER_DEG_J2)
    steps_z = round(z_mm * STEPS_PER_MM_Z)
    
    return steps_z, steps_j1, steps_j2