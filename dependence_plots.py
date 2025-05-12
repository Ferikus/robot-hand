import numpy as np
import matplotlib.pyplot as plt

from utils.forward_kinematics import *
from utils.cramer_solve import *
from utils.trajectory import *

from movement_wo_obstacle import jacobian


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
    n_points_list = [10, 20, 30, 40, 50, 75, 100, 150, 200]  # исследуемые значения количества точек
    tolerance = 1e-4
    max_iter = 100

    errors = {}    # средняя ошибка для каждого n_points
    iter_counts = {}  # среднее число итераций для каждого n_points

    for n_points in n_points_list:
        t_values = np.linspace(0, 2 * np.pi, n_points)
        theta_values = [theta_initial.copy()]
        current_errors = []
        current_iter_counts = []

        # Расчет для текущего n_points
        for t in t_values[1:]:
            q_new, iters, err = update_angles(t, theta_values[-1], max_iter, tolerance)
            theta_values.append(q_new)
            current_errors.append(err)
            current_iter_counts.append(iters)

        # Сохраняем средние значения
        errors[n_points] = np.mean(current_errors)
        iter_counts[n_points] = np.mean(current_iter_counts)

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # График средней ошибки
    plt.subplot(1, 2, 1)
    plt.plot(n_points_list, [errors[n] for n in n_points_list], 'bo-')
    plt.xlabel('Количество точек траектории')
    plt.ylabel('Средняя ошибка')
    plt.yscale('log')
    plt.title('Зависимость ошибки от числа точек')

    # График средних итераций
    plt.subplot(1, 2, 2)
    plt.plot(n_points_list, [iter_counts[n] for n in n_points_list], 'ro-')
    plt.xlabel('Количество точек траектории')
    plt.ylabel('Среднее число итераций')
    plt.title('Зависимость итераций от числа точек')

    plt.tight_layout()
    plt.show()