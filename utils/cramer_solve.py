import numpy as np


def cramer_solve(A, b):
    """Решение системы 2x2 методом Крамера."""
    det = np.linalg.det(A)
    if np.abs(det) < 1e-10:
        return np.zeros(2)
    det1 = np.linalg.det(np.column_stack([b, A[:, 1]]))
    det2 = np.linalg.det(np.column_stack([A[:, 0], b]))
    return np.array([det1 / det, det2 / det])