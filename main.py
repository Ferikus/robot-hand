import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize

# Параметры манипулятора
l1 = l4 = 0.5
l2 = l3 = 0.25
theta_initial = np.deg2rad([45, 45, 45, 45])
d_initial = 0.0  # начальное поступательное смещение

# Параметры лемнискаты
a = 0.6
x0 = 1.1
y0 = 0.7

def forward_kinematics(theta, d):
    """Прямая кинематика с поступательным смещением."""
    theta1, theta2, theta3, theta4 = theta
    x = l1 * np.cos(theta1) + (d + l2 + l3) * np.cos(theta1 + theta2) + l4 * np.cos(theta1 + theta2 + theta3)
    y = l1 * np.sin(theta1) + (d + l2 + l3) * np.sin(theta1 + theta2) + l4 * np.sin(theta1 + theta2 + theta3)
    return x, y

def error(params, t):
    """Функция ошибки для оптимизации."""
    theta1, theta2, theta3, theta4, d = params
    x, y = forward_kinematics([theta1, theta2, theta3, theta4], d)
    x_d = x0 + a * np.cos(t) / (1 + np.sin(t)**2)
    y_d = y0 + a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
    return np.sqrt((x - x_d)**2 + (y - y_d)**2)

# Оптимизация для каждого момента времени
t_values = np.linspace(0, 2*np.pi, 50)
theta_d_values = []
d_values = []

for t in t_values:
    res = minimize(
        error,
        x0=[*theta_initial, d_initial],
        args=(t,),
        method='L-BFGS-B',
        bounds=[(-np.pi, np.pi)]*4 + [(0, 1.0)]  # Ограничения для d
    )
    theta_d_values.append(res.x[:4])
    d_values.append(res.x[4])

# Визуализация

fig, ax = plt.subplots()
ax.set_xlim(-0.5, 2)
ax.set_ylim(-0.5, 1.5)

# Траектория лемнискаты
x_d = x0 + a * np.cos(t_values) / (1 + np.sin(t_values)**2)
y_d = y0 + a * np.sin(t_values) * np.cos(t_values) / (1 + np.sin(t_values)**2)
ax.plot(x_d, y_d, 'r--', label='Лемниската')

# Манипулятор
links, = ax.plot([], [], 'bo-')
ax.legend()

def animate(i):
    theta = theta_d_values[i]
    d = d_values[i]
    x = [
        0,
        l1 * np.cos(theta[0]),
        l1 * np.cos(theta[0]) + (d + l2 + l3) * np.cos(theta[0] + theta[1]),
        l1 * np.cos(theta[0]) + (d + l2 + l3) * np.cos(theta[0] + theta[1]) + l4 * np.cos(theta[0] + theta[1] + theta[2])
    ]
    y = [
        0,
        l1 * np.sin(theta[0]),
        l1 * np.sin(theta[0]) + (d + l2 + l3) * np.sin(theta[0] + theta[1]),
        l1 * np.sin(theta[0]) + (d + l2 + l3) * np.sin(theta[0] + theta[1]) + l4 * np.sin(theta[0] + theta[1] + theta[2])
    ]
    links.set_data(x, y)
    return links,

ani = FuncAnimation(fig, animate, frames=len(t_values), interval=100)
plt.title('Движение руки манипулятора')
plt.show()