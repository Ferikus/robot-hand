import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils.forward_kinematics import *
from utils.cramer_solve import *
from utils.trajectory import *


def jacobian(q):
    """Якобиан для q1, q2, q3, q4 по x, y"""
    q1, q2, q3, q4 = q
    J = np.zeros((2, 4))

    # Производные по q1
    J[0, 0] = -l1 * np.sin(q1) - (q3 + l2 + l3) * np.sin(q1 + q2) - l4 * np.sin(q1 + q2 + q4)
    J[1, 0] = l1 * np.cos(q1) + (q3 + l2 + l3) * np.cos(q1 + q2) + l4 * np.cos(q1 + q2 + q4)

    # Производные по q2
    J[0, 1] = -(q3 + l2 + l3) * np.sin(q1 + q2) - l4 * np.sin(q1 + q2 + q4)
    J[1, 1] = (q3 + l2 + l3) * np.cos(q1 + q2) + l4 * np.cos(q1 + q2 + q4)

    # Производные по q3 (поступательное движение)
    J[0, 2] = np.cos(q1 + q2)
    J[1, 2] = np.sin(q1 + q2)

    # Производные по q4
    J[0, 3] = -l4 * np.sin(q1 + q2 + q4)
    J[1, 3] = l4 * np.cos(q1 + q2 + q4)

    return J


def update_angles(t, q_prev):
    """Обновление углов и растяжения"""
    q = q_prev.copy()
    x_target, y_target = lemniscate(t)

    for _ in range(30):  # Увеличено число итераций
        x_current, y_current = forward_kinematics(q)
        error = np.array([x_target - x_current, y_target - y_current])

        if np.linalg.norm(error) < 1e-5:
            break

        J = jacobian(q)
        J_reduced = J[:, [0, 2]]

        # Регуляризация для устойчивости
        A = J_reduced.T @ J_reduced + 0.1 * np.eye(2)
        b = J_reduced.T @ error
        delta_q = cramer_solve(A, b)

        q[[0, 2]] += delta_q

    return q


if __name__ == "__main__":
    # Генерация траектории (2 оборота)
    t_values = np.linspace(0, 4 * np.pi, 200)  # Два периода
    theta_values = [theta_initial.copy()]

    # Предварительный расчет траектории
    for t in t_values[1:]:
        theta_values.append(update_angles(t, theta_values[-1]))

    theta_values += theta_values[:50]

    # Визуализация
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, 2)
    ax.set_ylim(-0.5, 1.5)

    x_lem, y_lem = lemniscate(t_values)
    ax.plot(x_lem, y_lem, 'r--', label='Лемниската')

    links, = ax.plot([], [], 'bo-', lw=2)
    ax.legend()


    def animate(i):
        q = theta_values[i]
        x = [
            0,
            l1 * np.cos(q[0]),
            l1 * np.cos(q[0]) + (q[2] + l2 + l3) * np.cos(q[0] + q[1]),
            l1 * np.cos(q[0]) + (q[2] + l2 + l3) * np.cos(q[0] + q[1]) + l4 * np.cos(q[0] + q[1] + q[3])
        ]
        y = [
            0,
            l1 * np.sin(q[0]),
            l1 * np.sin(q[0]) + (q[2] + l2 + l3) * np.sin(q[0] + q[1]),
            l1 * np.sin(q[0]) + (q[2] + l2 + l3) * np.sin(q[0] + q[1]) + l4 * np.sin(q[0] + q[1] + q[3])
        ]
        links.set_data(x, y)
        return links,


    ani = FuncAnimation(fig, animate, frames=len(t_values), interval=50, blit=True)
    plt.title('Траектория манипулятора')
    plt.show()