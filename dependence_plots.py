import numpy as np
import matplotlib.pyplot as plt

from utils.forward_kinematics import *
from utils.cramer_solve import *
from utils.trajectory import *

from movement_wo_obstacle import jacobian

# Параметры манипулятора
l1 = l4 = 0.5
l2 = l3 = 0.25
theta_initial = np.deg2rad([45, -45, 0.0, 45])


def update_angles(t, q_prev, max_iter=100, tolerance=1e-5):
    """Обновление углов и растяжения в зависимости от итераций и ошибки"""
    q = q_prev.copy()
    x_target, y_target = lemniscate(t)
    iterations = 0

    for _ in range(max_iter):
        x_current, y_current = forward_kinematics(q)
        error = np.array([x_target - x_current, y_target - y_current])
        iterations += 1

        if np.linalg.norm(error) < tolerance:
            break

        J = jacobian(q)
        J_reduced = J[:, [0, 2]]
        A = J_reduced.T @ J_reduced + 0.1 * np.eye(2)
        b = J_reduced.T @ error
        delta_q = cramer_solve(A, b)
        q[[0, 2]] += delta_q * 0.5

    return q, iterations, np.linalg.norm(error)


if __name__ == "__main__":
    # Параметры исследования
    n_points = 50
    tolerance = 1e-4
    max_iter = 100

    # Сбор данных
    errors = {}
    iter_counts = {}

    t_values = np.linspace(0, 2 * np.pi, n_points)
    theta_values = [theta_initial.copy()]
    current_errors = []
    current_iter_counts = []

    for t in t_values[1:]:
        q_new, iters, err = update_angles(t, theta_values[-1], max_iter, tolerance)
        theta_values.append(q_new)
        current_errors.append(err)
        current_iter_counts.append(iters)

    errors[n_points] = current_errors
    iter_counts[n_points] = current_iter_counts

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # График ошибок
    plt.subplot(1, 2, 1)
    plt.plot(errors[n_points], label=f'N={n_points}')
    plt.xlabel('Точка траектории')
    plt.ylabel('Ошибка позиционирования')
    plt.yscale('log')
    plt.legend()
    plt.title('Зависимость ошибки от количества точек')

    # График итераций
    plt.subplot(1, 2, 2)
    plt.plot(iter_counts[n_points], label=f'N={n_points}')
    plt.xlabel('Точка траектории')
    plt.ylabel('Количество итераций')
    plt.legend()
    plt.title('Зависимость итераций от количества точек')

    plt.tight_layout()
    plt.show()