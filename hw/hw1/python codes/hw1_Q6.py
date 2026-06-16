import numpy as np 

def extract_rotation_translation(T: np.ndarray):
    """
    Extracts the rotation matrix and translation vector from a 4x4
    homogeneous transformation matrix.

    Parameters
    ----------
    T : np.ndarray
        A 4x4 homogeneous transformation matrix.

    Returns
    -------
    R : np.ndarray
        A 3x3 matrix representing the (possibly corrupted) rotation part.
    P : np.ndarray
        A 3x1 translation vector.
    """
    
    if T.shape != (4,4):
        raise ValueError("Input matrix must be 4x4.")
    
    R = T[0:3, 0:3]
    P = T[0:3, 3]
    
    return R, P


def closest_rotation_matrix_least_squares(R: np.ndarray):
    """
    Computes the closest valid rotation matrix to a given 3x3 matrix
    using a least-squares formulation (Orthogonal Procrustes problem).

    The optimization problem solved is:
        min ||R - Q||_F^2
        subject to Q^T Q = I

    Parameters
    ----------
    R : np.ndarray
        A 3x3 matrix representing a corrupted rotation matrix.

    Returns
    -------
    R_valid : np.ndarray
        The closest valid rotation matrix in the least-squares sense.

    Raises
    ------
    ValueError
        If the resulting rotation matrix represents a reflection
        (determinant < 0).
    """
    
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    
    U, _, Vt = np.linalg.svd(R)
    
    R_valid = U @ Vt
    
    if np.linalg.det(R_valid) < 0:
        raise ValueError(
            "Reflection detected: a valid rigid-body rotation cannot be reconstructed."
        )

    return R_valid



def reconstruct_homogeneous_transform(R: np.ndarray, P: np.ndarray):
    """
    Reconstructs a 4x4 homogeneous transformation matrix from a
    rotation matrix and a translation vector.

    Parameters
    ----------
    R : np.ndarray
        A valid 3x3 rotation matrix.
    P : np.ndarray
        A 3-element translation vector.

    Returns
    -------
    T : np.ndarray
        A 4x4 homogeneous transformation matrix.
    """
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    if P.shape not in [(3,), (3, 1)]:
        raise ValueError("Translation vector must have 3 elements.")

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = P.reshape(3)

    return T


def validate_homogeneous_transform(T_input: np.ndarray):
    """
    Validates and reconstructs the closest geometrically valid homogeneous
    transformation matrix from a corrupted input matrix using least squares.

    The function:
    1. Extracts rotation and translation.
    2. Computes the closest valid rotation using SVD-based least squares.
    3. Detects reflection and raises an error if present.
    4. Reconstructs the final homogeneous transformation matrix.

    Parameters
    ----------
    T_input : np.ndarray
        A 4x4 corrupted homogeneous transformation matrix.

    Returns
    -------
    T_valid : np.ndarray
        A 4x4 geometrically valid homogeneous transformation matrix.

    Raises
    ------
    ValueError
        If reflection is detected or input dimensions are invalid.
    """
    
    R_input, P_input = extract_rotation_translation(T_input)

    R_valid = closest_rotation_matrix_least_squares(R_input)

    T_valid = reconstruct_homogeneous_transform(R_valid, P_input)

    return T_valid


if __name__ == "__main__":
    print("\n TESTING HOMOGENEOUS TRANSFORM VALIDATOR\n")

    R_corrupted = np.array([
        [0.96, -0.30,  0.05],
        [0.28,  0.92, -0.27],
        [0.12,  0.24,  0.94]
    ])

    # Add noise so it is not orthogonal
    R_corrupted += np.random.normal(0, 0.2, size=(3,3))

    # Translation vector
    P = np.array([1.2, -0.4, 2.5])

    # Build corrupted T
    T_corrupted = np.eye(4)
    T_corrupted[0:3, 0:3] = R_corrupted
    T_corrupted[0:3, 3] = P

    print("Input corrupted T:\n", T_corrupted)
    print("\nDeterminant of R (should NOT be 1):", np.linalg.det(R_corrupted))

    try:
        T_valid = validate_homogeneous_transform(T_corrupted)

        print("\n--- RESULT ---")
        print("Valid (corrected) homogeneous matrix T:\n", T_valid)
        print("\nDeterminant of corrected R:", np.linalg.det(T_valid[0:3, 0:3]))
        print("\nIs R orthogonal? (R^T R):\n", T_valid[0:3, 0:3].T @ T_valid[0:3, 0:3])

    except ValueError as e:
        print("\nERROR:", e)