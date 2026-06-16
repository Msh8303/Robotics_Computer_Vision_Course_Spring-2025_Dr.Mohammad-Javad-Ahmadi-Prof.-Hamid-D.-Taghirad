#!/usr/bin/env python
# coding: utf-8

# # Implementation Code with Modified DH Table

# In[12]:


import numpy as np

def mdh_transformation_matrix(alpha, a, d, theta):
    """
    Calculates the homogeneous transformation matrix for a single link 
    using the Modified Denavit-Hartenberg (MDH / Craig's) convention.
    
    Formula: T = Rot_x(alpha) * Trans_x(a) * Rot_z(theta) * Trans_z(d)
    
    Parameters:
    alpha (float): Link twist angle in radians (alpha_{i-1})
    a (float): Link length in meters (a_{i-1})
    d (float): Link offset in meters (d_i)
    theta (float): Joint angle in radians (theta_i)
    
    Returns:
    numpy.ndarray: 4x4 Homogeneous Transformation Matrix (T_{i-1}^i)
    """
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    
    # MDH Transformation Matrix elements
    T = np.array([
        [c_theta,            -s_theta,            0,        a],
        [s_theta * c_alpha,   c_theta * c_alpha, -s_alpha, -d * s_alpha],
        [s_theta * s_alpha,   c_theta * s_alpha,  c_alpha,  d * c_alpha],
        [0,                   0,                  0,        1]
    ])
    
    return T

def forward_kinematics(thetas, d_params):
    """
    Calculates the forward kinematics of the 7-DOF robot using the MDH table.
    
    Parameters:
    thetas (list or array): List of 7 joint angles [theta1, theta2, ..., theta7] in radians.
    d_params (list or array): List of d offsets [d1, d3, d5, d7] in meters. 
                              (d2, d4, d6 are zero based on the table).
                              
    Returns:
    numpy.ndarray: Final 4x4 Transformation Matrix from Base (Frame 0) to End-Effector (Frame 7).
    """
    # Unpack joint variables
    t1, t2, t3, t4, t5, t6, t7 = thetas
    
    # Unpack link offsets (d parameters)
    d1, d3, d5, d7 = d_params
    
    # Define the MDH table parameters: [alpha_{i-1}, a_{i-1}, d_i, theta_i]
    mdh_table = [
        [0,           0,  d1, t1],  # Link 1
        [-np.pi/2,    0,  0,  t2],  # Link 2
        [np.pi/2,     0,  d3, t3],  # Link 3
        [np.pi/2,     0,  0,  t4],  # Link 4
        [-np.pi/2,    0,  d5, t5],  # Link 5
        [-np.pi/2,    0,  0,  t6],  # Link 6
        [np.pi/2,     0,  d7, t7]   # Link 7
    ]
    
    # Initialize the total transformation matrix as an Identity matrix
    T_0_to_7 = np.eye(4)
    
    # Iteratively multiply the transformation matrices
    for i in range(7):
        alpha, a, d, theta = mdh_table[i]
        # Get transformation matrix for the current link
        T_i = mdh_transformation_matrix(alpha, a, d, theta)
        
        # Multiply with the cumulative transformation matrix
        # T_0^n = T_0^1 * T_1^2 * ... * T_{n-1}^n
        T_0_to_7 = np.dot(T_0_to_7, T_i)
        
    return T_0_to_7

# ==========================================
# Example Usage and Testing
# ==========================================
if __name__ == "__main__":
    # Example joint angles in radians (e.g., all joints at zero position)
    example_thetas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Example link offsets in meters (replace with actual robot dimensions from datasheet)
    d1, d3, d5, d7 = 0.34, 0.40, 0.40, 0.126
    example_d_params = [0.34, 0.40, 0.40, 0.126] 
    
    # Calculate Forward Kinematics
    final_transformation = forward_kinematics(example_thetas, example_d_params)
    
    # Extract position (X, Y, Z) and rotation matrix
    position = final_transformation[0:3, 3]
    rotation_matrix = final_transformation[0:3, 0:3]
    
    print("Final Homogeneous Transformation Matrix (T_0^7):")
    # Using np.round to clean up very small floating-point errors (like 1e-17)
    print(np.round(final_transformation, decimals=4))
    
    print("\nEnd-Effector Position (X, Y, Z) in meters:")
    print(np.round(position, decimals=4))
    
    print("--- Scenario 2: Joint 2 bent at 90 degrees ---")
    # Joint 2 with angle pi/2
    thetas_2 = [0, np.pi/2, 0, 0, 0, 0, 0]
    T2 = forward_kinematics(thetas_2,   example_d_params)
    print("Expected Z:", d1, "| Expected X:", d3 + d5 + d7)
    print("Calculated Pos (X, Y, Z):", np.round(T2[0:3, 3], 4), "\n")

    print("--- Scenario 3: Joint 4 bent at 90 degrees ---")
    # joint 4 with angle pi/2
    thetas_3 = [0, 0, 0, np.pi/2, 0, 0, 0]
    T3 = forward_kinematics(thetas_3, example_d_params)
    print("Expected Z:", d1 + d3, "| Expected X or Y extension:", d5 + d7)
    print("Calculated Pos (X, Y, Z):", np.round(T3[0:3, 3], 4), "\n")


# # Implementation Code with Standard DH Table

# In[14]:


import numpy as np

def dh_transform(theta, d, a, alpha):
    """Calculates the standard DH transformation matrix."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def forward_kinematics_kuka_iiwa(q, link_offsets):
    """
    Calculates the forward kinematics of the KUKA LBR iiwa robot using Standard DH parameters.
    q: List of joint angles [theta1, ..., theta7] in radians.
    link_offsets: List of Z-axis offsets [d1, d3, d5, d7].
    """
    # Assign offsets to d variables
    d1, d3, d5, d7 = link_offsets
    
    # DH table: [theta, d, a, alpha]
    dh_table = [
        [q[0], d1, 0, -np.pi/2],
        [q[1], 0,  0,  np.pi/2],
        [q[2], d3, 0,  np.pi/2],
        [q[3], 0,  0, -np.pi/2],
        [q[4], d5, 0, -np.pi/2],
        [q[5], 0,  0,  np.pi/2],
        [q[6], d7, 0,  0]
    ]
    
    # Initialize base transformation matrix as identity
    T_end_effector = np.eye(4)
    
    # Multiply transformation matrices to find the final pose
    for params in dh_table:
        A_i = dh_transform(params[0], params[1], params[2], params[3])
        T_end_effector = np.dot(T_end_effector, A_i)
        
    return T_end_effector

# --- Numerical test example ---
if __name__ == "__main__":
    example_thetas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    d1, d3, d5, d7 = 0.34, 0.40, 0.40, 0.126
    example_d_params = [d1, d3, d5, d7] 
    
    final_transformation = forward_kinematics_kuka_iiwa(example_thetas, example_d_params)
    position = final_transformation[0:3, 3]
    
    print("Final Homogeneous Transformation Matrix (T_0^7):")
    print(np.round(final_transformation, decimals=4))
    
    print("\nEnd-Effector Position (X, Y, Z) in meters:")
    print(np.round(position, decimals=4))
    
    print("\n--- Scenario 2: Joint 2 bent at 90 degrees ---")
    thetas_2 = [0, np.pi/2, 0, 0, 0, 0, 0]
    T2 = forward_kinematics_kuka_iiwa(thetas_2, example_d_params)
    print("Expected Z:", d1, "| Expected X/Y:", d3 + d5 + d7)
    print("Calculated Pos (X, Y, Z):", np.round(T2[0:3, 3], 4))

    print("\n--- Scenario 3: Joint 4 bent at 90 degrees ---")
    thetas_3 = [0, 0, 0, np.pi/2, 0, 0, 0]
    T3 = forward_kinematics_kuka_iiwa(thetas_3, example_d_params)
    print("Expected Z:", d1 + d3, "| Expected X/Y:", d5 + d7)
    print("Calculated Pos (X, Y, Z):", np.round(T3[0:3, 3], 4))

