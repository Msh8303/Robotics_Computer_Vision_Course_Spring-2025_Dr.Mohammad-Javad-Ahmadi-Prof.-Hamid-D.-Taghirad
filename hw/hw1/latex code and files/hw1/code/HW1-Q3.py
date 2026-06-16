import numpy as np
import math


def rotation_matrix_to_euler(R, epsilon=1e-6):
    """
    Convert a 3x3 rotation matrix into Euler angles (alpha, beta, gamma)
    following the decision flowchart logic.

    Parameters:
        R (numpy.ndarray): 3x3 rotation matrix
        epsilon (float): tolerance value for detecting Gimbal Lock

    Returns:
        tuple: (alpha, beta, gamma, mode)
            mode = "Normal" or "Gimbal Lock (...)"
    """

    # Extract matrix elements
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    # Clamp r31 to [-1, 1] to avoid numerical errors in arcsin
    r31_clamped = max(-1.0, min(1.0, r31))

    # Step 1: compute beta
    beta = -math.asin(r31_clamped)

    # Step 2: compute cos(beta) to determine Gimbal Lock
    cb = math.cos(beta)

    # Check for Gimbal Lock
    if abs(cb) >= epsilon:
        # Normal case
        alpha = math.atan2(r21, r11)
        gamma = math.atan2(r32, r33)
        mode = "Normal"

    else:
        # Gimbal Lock case
        gamma = 0.0  # convention: set gamma to zero

        # Check whether beta = +pi/2 or -pi/2
        if r31 <= -1.0 + epsilon:
            # beta = +pi/2
            beta = math.pi / 2.0
            alpha = math.atan2(r12, r22)
            mode = "Gimbal Lock (alpha-gamma relation)"
        else:
            # beta = -pi/2
            beta = -math.pi / 2.0
            alpha = math.atan2(-r12, r22)
            mode = "Gimbal Lock (alpha+gamma relation)"

    return alpha, beta, gamma, mode


# ==========================================
# Test section
# ==========================================
if __name__ == "__main__":

    # 1. Normal rotation matrix example
    R_normal = np.array([
        [ 0.81379768, -0.44096961,  0.37852231],
        [ 0.46984631,  0.88256412,  0.01802831],
        [-0.34202014,  0.16317591,  0.92541658]
    ])

    # 2. Gimbal lock example (beta = +pi/2 → r31 = -1)
    R_gimbal_pos = np.array([
        [ 0.0,  0.0,  1.0],
        [ 0.0,  1.0,  0.0],
        [-1.0,  0.0,  0.0]
    ])

    for name, R in [
        ("Normal", R_normal),
        ("Gimbal Lock (+pi/2)", R_gimbal_pos)
    ]:
        alpha, beta, gamma, mode = rotation_matrix_to_euler(R)

        print(f"--- Test Case: {name} ---")
        print(f"Mode:  {mode}")
        print(f"Alpha: {math.degrees(alpha):.2f} deg")
        print(f"Beta:  {math.degrees(beta):.2f} deg")
        print(f"Gamma: {math.degrees(gamma):.2f} deg\n")
