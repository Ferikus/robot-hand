from utils.config import *


def forward_kinematics(q):
    """Расчёт прямой кинематики Fx, Fy"""
    q1, q2, q3, q4 = q
    x = l1 * np.cos(q1) + (q3 + l2 + l3) * np.cos(q1 + q2) + l4 * np.cos(q1 + q2 + q4)
    y = l1 * np.sin(q1) + (q3 + l2 + l3) * np.sin(q1 + q2) + l4 * np.sin(q1 + q2 + q4)
    return x, y
