import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.config import l1, l2, l3, l4, theta_initial
from utils.forward_kinematics import forward_kinematics
from utils.trajectory import lemniscate


def full_forward_kinematics(q):
    """Возвращает положения узлов манипулятора"""
    q1, q2, q3, q4 = q
    p0 = np.array([0.0, 0.0])
    p1 = p0 + np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    p2 = p1 + np.array([(q3 + l2 + l3) * np.cos(q1 + q2), (q3 + l2 + l3) * np.sin(q1 + q2)])
    p3 = p2 + np.array([l4 * np.cos(q1 + q2 + q4), l4 * np.sin(q1 + q2 + q4)])
    return [p0, p1, p2, p3]


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


def solve_ik(px, py, q_init, max_iter=50, tol=1e-3):
    """Решение обратной кинематики"""
    q = q_init.copy()
    for _ in range(max_iter):
        x, y = forward_kinematics(q)
        err = np.array([px - x, py - y])
        if np.linalg.norm(err) < tol: break
        J = jacobian(q)
        dq = np.linalg.pinv(J).dot(err)
        q += 0.5 * dq
        q[3] = max(0, q[3])
    return q


if __name__ == '__main__':
    # Инициализация параметров
    q0 = theta_initial.copy()
    t_values = np.linspace(0, 2 * np.pi, 200)
    traj = np.array([lemniscate(t) for t in t_values])

    # Расчет конфигураций
    configs = []
    q_cur = q0.copy()
    for pt in traj:
        q_sol = solve_ik(pt[0], pt[1], q_cur)
        configs.append(q_sol)
        q_cur = q_sol

    # Настройка анимации
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 1.2)

    # Отрисовка лемнискаты
    x_lem, y_lem = lemniscate(t_values)
    ax.plot(x_lem, y_lem, 'r--', lw=1.5, label='Траектория')

    lines = [ax.plot([], [], 'bo-', lw=3)[0] for _ in range(3)]


    def init():
        for ln in lines: ln.set_data([], [])
        return lines


    def animate(i):
        pts = full_forward_kinematics(configs[i])
        for k in range(3):
            x0, y0 = pts[k]
            x1, y1 = pts[k + 1]
            lines[k].set_data([x0, x1], [y0, y1])
        return lines


    ani = FuncAnimation(fig, animate, init_func=init,
                        frames=len(configs), interval=25, blit=True)
    ani.save('movement_without_obstacle.gif', writer='pillow', fps=20)
    plt.title('Движение манипулятора без препятствий')
    plt.show()
