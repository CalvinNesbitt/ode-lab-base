import numpy as np

from ode_lab.base.integrate import Integrator
from ode_lab.base.lab import ODELab
from ode_lab.base.observe import TrajectoryObserver


def test_lab_1d(linear_ode_1d):
    dt = 0.1
    rhs, solution, initial_condition, parameters = linear_ode_1d
    observable_names = ["X_0"]

    # Set up the lab
    integrator = Integrator(rhs, initial_condition, parameters)
    observer = TrajectoryObserver(observable_names=observable_names)
    lab = ODELab(integrator, observer)

    assert lab.integrator.ndim == 1

    # Check we pack things correctly
    assert lab.integrator.time == 0.0
    assert np.allclose(lab.integrator.state, initial_condition)
    assert isinstance(lab.integrator.state, np.ndarray)
    assert lab.integrator.parameters == parameters
    assert lab.integrator.rhs == rhs

    # Check we can integrate
    n1 = 10
    lab.make_observations(n1, dt)
    assert np.allclose(lab.observations.time[-1], dt * n1)
    for i in range(n1):
        observed_val = lab.observations.isel(time=i)["X_0"]
        observed_time = lab.observations.isel(time=i)["time"]
        assert np.allclose(observed_val, solution(observed_time, **parameters))

    n2 = 5
    lab.make_observations(n2, dt)
    assert np.allclose(lab.observations.time[-1], dt * (n1 + n2))
    for i in range(n1 + n2):
        observed_val = lab.observations.isel(time=i)["X_0"]
        observed_time = lab.observations.isel(time=i)["time"]
        assert np.allclose(observed_val, solution(observed_time, **parameters))


def test_lab_2d(linear_ode_2d):
    dt = 0.1
    rhs, solution, initial_condition, parameters = linear_ode_2d
    observable_names = ["X_0", "X_1"]

    # Set up the lab
    integrator = Integrator(rhs, initial_condition, parameters)
    observer = TrajectoryObserver(observable_names=observable_names)
    lab = ODELab(integrator, observer)

    assert lab.integrator.ndim == 2

    # Check we pack things correctly
    assert lab.integrator.time == 0.0
    assert np.allclose(lab.integrator.state, initial_condition)
    assert isinstance(lab.integrator.state, np.ndarray)
    assert lab.integrator.parameters == parameters
    assert lab.integrator.rhs == rhs

    # Check we can integrate
    n1 = 10
    lab.make_observations(n1, dt)
    assert np.allclose(lab.observations.time[-1], dt * n1)
    for i in range(n1):
        observed_val_0 = lab.observations.isel(time=i)["X_0"]
        observed_val_1 = lab.observations.isel(time=i)["X_1"]
        observed_time = lab.observations.isel(time=i)["time"].values.item()
        assert np.allclose(
            np.array([observed_val_0, observed_val_1]),
            solution(observed_time, **parameters),
        )

    n2 = 5
    lab.make_observations(n2, dt)
    assert np.allclose(lab.observations.time[-1], dt * (n1 + n2))
    for i in range(n1 + n2):
        observed_val_0 = lab.observations.isel(time=i)["X_0"]
        observed_val_1 = lab.observations.isel(time=i)["X_1"]
        observed_time = lab.observations.isel(time=i)["time"].values.item()
        assert np.allclose(
            np.array([observed_val_0, observed_val_1]),
            solution(observed_time, **parameters),
        )
