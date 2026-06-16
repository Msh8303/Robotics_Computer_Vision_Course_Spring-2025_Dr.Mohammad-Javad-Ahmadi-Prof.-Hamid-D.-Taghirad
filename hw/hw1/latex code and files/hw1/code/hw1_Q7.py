import numpy as np


def random_rotation_matrix_small(angle_scale=0.1):
    """
    Generate a small random rotation matrix using Rodrigues' rotation formula.
    The rotation is defined by a random axis and a small random angle.
    """

    # Generate a random axis in 3D
    axis = np.random.randn(3)

    # Normalize the axis to unit length
    axis = axis / np.linalg.norm(axis)

    # Generate a small random rotation angle (radians)
    theta = angle_scale * np.random.randn()

    # Construct the skew-symmetric cross-product matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # Rodrigues rotation formula
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    return R


def random_translation(scale=0.01):
    """
    Generate a small random translation vector.
    """
    return scale * np.random.randn(3)


def create_homogeneous_transform(R, t):
    """
    Construct a 4x4 homogeneous transformation matrix
    from a rotation matrix R and translation vector t.
    """

    T = np.eye(4)

    T[:3, :3] = R   # rotation part
    T[:3, 3] = t    # translation part

    return T


def extract_rotation_translation(T):
    """
    Extract rotation matrix and translation vector
    from a homogeneous transformation matrix.
    """

    R = T[:3, :3]
    t = T[:3, 3]

    return R, t


if __name__ == "__main__":

    print("\n Incompetence Score Calculation\n")

    N = int(input("Enter number of frames (N): "))
    if N < 2:
        raise ValueError("At least 2 frames are required.")

    homogeneous_transforms = []

    # Starting point
    current_R = np.eye(3)
    current_t = np.zeros(3)

    for i in range(N):

        # Generate small incremental motion
        dR = random_rotation_matrix_small()
        dt = random_translation()

        # Update current pose
        current_R = current_R @ dR
        current_t = current_t + dt

        # Construct homogeneous transform
        T = create_homogeneous_transform(current_R, current_t)

        homogeneous_transforms.append(T)

    # Print generated homogeneous matrices
    print("\nGenerated Homogeneous Transformation Matrices:\n")

    for i, T in enumerate(homogeneous_transforms):
        print(f"Frame {i}:\n{T}\n")

    # Extract rotation matrices from transforms
    rotations = []

    for T in homogeneous_transforms:
        R, _ = extract_rotation_translation(T)
        rotations.append(R)



    total_score = 0.0
    angles = np.zeros(N - 1)
    pair_scores = np.zeros(N - 1)

    # Compute frame-to-frame rotation angles
    for i in range(N - 1):

        R_i = rotations[i]
        R_next = rotations[i + 1]

        # Compute relative rotation between two frames
        R_rel = R_i.T @ R_next

        # Compute the trace of the relative rotation matrix
        tr = np.trace(R_rel)

        value = (tr - 1) / 2

        # Clip value to avoid numerical errors in arccos
        value = np.clip(value, -1.0, 1.0)

        
        theta = np.arccos(value)

        angles[i] = theta
        pair_scores[i] = theta ** 2

        # Accumulate total incompetence score
        total_score += pair_scores[i]

    
    print("\nFrame-to-Frame Rotation Angles (radians):\n")
    print(angles)

    print("\nSquared Angle Scores:\n")
    print(pair_scores)

    print("\nTotal Incompetence Score:\n")
    print(total_score)
