import numpy as np

# Параметры лемнискаты Бернулли
a = 0.3
scale = 1.2
x0, y0 = 1.5, 0.6


def lemniscate(t):
    """Параметризация лемнискаты."""
    denominator = 1 + np.sin(t) ** 2
    x = x0 + scale * a * np.sqrt(2) * np.cos(t) / denominator
    y = y0 + scale * a * np.sqrt(2) * np.sin(t) * np.cos(t) / denominator
    return x, y