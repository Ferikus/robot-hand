import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from utils.config import l1, l2, l3, l4, theta_initial
from utils.forward_kinematics import forward_kinematics
from utils.trajectory import x0, y0, lemniscate


# 1. Прямая кинематика (импортирована из forward_kinematics.py)
def full_forward_kinematics(q):
    """Возвращает точки манипулятора: [p0, p1, p2, p3] (3 звена)"""
    q1, q2, q3, q4 = q

    # Первое звено (вращательное)
    p0 = np.array([0.0, 0.0])
    p1 = p0 + np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    p2 = p1 + np.array([(q3 + l2 + l3) * np.cos(q1 + q2), (q3 + l2 + l3) * np.sin(q1 + q2)])
    p3 = p2 + np.array([l4 * np.cos(q1 + q2 + q4), l4 * np.sin(q1 + q2 + q4)])

    return [p0, p1, p2, p3]


# 2. Аналитический якобиан
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


# Решение обратной кинематики методом псевдообратного якобиана
def solve_ik(px, py, q_init, max_iter=50, tol=1e-3):
    q = q_init.copy()
    for _ in range(max_iter):
        x_current, y_current = forward_kinematics(q)
        err = np.array([px - x_current, py - y_current])
        if np.linalg.norm(err) < tol:
            break
        J = jacobian(q)
        dq = np.linalg.pinv(J).dot(err)
        q += dq
        q[3] = max(0, q[3])  # ограничение q4
    return q


if __name__ == '__main__':
    # Параметры препятствия
    rect_x, rect_y = -1.05, 1.3
    rect_w, rect_h = 1.04, 1.0
    safety_radius = 0.2
    zone_center = np.array([rect_x + rect_w, rect_y - rect_h])

    # Инициализация
    q0 = theta_initial.copy()

    # Генерация траектории
    t_values = np.linspace(0, 2 * np.pi, 200)
    traj = np.array([lemniscate(t) for t in t_values])

    # Основной цикл с random restarts
    configs = []
    colors = []
    q_cur = q0.copy()

    for pt in traj:
        q_sol = solve_ik(pt[0], pt[1], q_cur)
        p1 = full_forward_kinematics(q_sol)[1]
        dist1 = np.linalg.norm(p1 - zone_center)

        if dist1 >= safety_radius:
            configs.append(q_sol)
            colors.append('blue')
            q_cur = q_sol
            continue

        best_q = q_sol.copy()
        best_d = dist1
        safe = False
        for _ in range(5):
            q_try = q_cur.copy()
            q_try[:3] += (np.random.rand(3) - 0.5) * 0.1
            q_try[3] = np.random.rand() * 0.5
            q_candidate = solve_ik(pt[0], pt[1], q_try)
            p1c = forward_kinematics(q_candidate)[1]
            d1c = np.linalg.norm(p1c - zone_center)
            if d1c > best_d:
                best_d = d1c
                best_q = q_candidate.copy()
            if d1c >= safety_radius:
                safe = True
                break

        configs.append(best_q)
        colors.append('blue' if safe else 'red')
        q_cur = best_q

    # Отрисовка анимации
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 1.2)

    # Препятствие и безопасная зона
    ax.add_patch(Rectangle((rect_x, rect_y - rect_h), rect_w, rect_h, color='red', alpha=0.4))
    ax.add_patch(Circle(zone_center, safety_radius, color='red', alpha=0.5))
    ax.plot(zone_center[0], zone_center[1], 'ro')

    # Лемниската
    x_lem, y_lem = lemniscate(t_values)
    ax.plot(x_lem, y_lem, 'r--', lw=1.5, label='Траектория')

    # Манипулятор
    lines = [ax.plot([], [], 'o-', lw=3)[0] for _ in range(3)]
    ax.legend()


    def init():
        for ln in lines:
            ln.set_data([], [])
        return lines


    def animate(i):
        q = configs[i]
        pts = full_forward_kinematics(q)
        for k in range(3):  # Обновляем только 3 звена
            x0, y0 = pts[k]
            x1, y1 = pts[k + 1]
            lines[k].set_data([x0, x1], [y0, y1])
            lines[k].set_color(colors[i])
        return lines


    ani = FuncAnimation(fig, animate, init_func=init, frames=len(configs), interval=25, blit=True)
    ani.save('movement_with_obstacle.gif', writer='pillow', fps=20)
    plt.title('Движение манипулятора с препятствием')
    plt.show()
