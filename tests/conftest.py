import numpy as np
import pytest
from scipy.linalg import expm


@pytest.fixture
def linear_ode_1d():
    """
    Fixture for a 1D linear ODE system.
    dx/dt = -a * x
    """

    def rhs(x, a=1.0):
        return -a * x

    def solution(t, x0=1.0, a=1.0):
        return x0 * np.exp(-a * t)

    initial_condition = np.array([1.0])  # Initial condition x(0) = 1.0
    parameters = {"a": 1.0}  # Parameter a

    return rhs, solution, initial_condition, parameters


@pytest.fixture
def linear_ode_2d():
    """
    Fixture for a 2D linear ODE system.
    dx/dt = A * x
    where A is a 2x2 matrix
    """
    A = np.array([[-1.0, 0.0], [0.0, -2.0]])

    def rhs(x, A=A):
        return A @ x

    def solution(t, x0=np.array([1.0, 1.0]), A=A):
        return expm(A * t) @ x0

    initial_condition = np.array([1.0, 1.0])  # Initial condition x(0) = [1.0, 1.0]
    parameters = {"A": A}  # Matrix A

    return rhs, solution, initial_condition, parameters
