"""
Example models for use in the ODE lab.

Index:
    - linear_ode, a simple linear ODE
    - l63, the Lorenz 63 model
"""

import numpy as np


def linear_ode(
    state: np.ndarray, A: np.ndarray = np.array([[-1.0, 0.0], [0.0, -2.0]])
) -> np.ndarray:
    return A @ state


def l63(
    state: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8 / 3
) -> np.ndarray:
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])
