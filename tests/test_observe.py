import numpy as np

from ode_lab.observe import TrajectoryObserver


def test_observe_1d(linear_ode_1d):
    dt = 0.1
    rhs, solution, initial_condition, parameters = linear_ode_1d
    observer = TrajectoryObserver(rhs, initial_condition, parameters)
    assert observer.ndim == 1

    # Check we pack things correctly
    assert observer.time == 0.0
    assert np.allclose(observer.state, initial_condition)
    assert isinstance(observer.state, np.ndarray)
    assert observer.parameters == parameters
    assert observer.rhs == rhs

    # Check we can integrate
    n1 = 10
    observer.make_observations(n1, dt)
    assert np.allclose(observer.packaged_observations.time[-1], dt * n1)
    for i in range(n1):
        observed_val = observer.packaged_observations.isel(time=i)["X_0"]
        observed_time = observer.packaged_observations.isel(time=i)["time"]
        assert np.allclose(observed_val, solution(observed_time, **parameters))

    n2 = 5
    observer.make_observations(n2, dt)
    assert np.allclose(observer.packaged_observations.time[-1], dt * (n1 + n2))
    for i in range(n1 + n2):
        observed_val = observer.packaged_observations.isel(time=i)["X_0"]
        observed_time = observer.packaged_observations.isel(time=i)["time"]
        assert np.allclose(observed_val, solution(observed_time, **parameters))


def test_observe_2d(linear_ode_2d):
    dt = 0.1
    rhs, solution, initial_condition, parameters = linear_ode_2d
    observer = TrajectoryObserver(rhs, initial_condition, parameters)
    assert observer.ndim == 2

    # Check we pack things correctly
    assert observer.time == 0.0
    assert np.allclose(observer.state, initial_condition)
    assert isinstance(observer.state, np.ndarray)
    assert observer.parameters == parameters
    assert observer.rhs == rhs

    # Check we can integrate
    n1 = 10
    observer.make_observations(n1, dt)
    assert np.allclose(observer.packaged_observations.time[-1], dt * n1)
    for i in range(n1):
        observed_val_0 = observer.packaged_observations.isel(time=i)["X_0"]
        observed_val_1 = observer.packaged_observations.isel(time=i)["X_1"]
        observed_time = observer.packaged_observations.isel(time=i)[
            "time"
        ].values.item()
        assert np.allclose(
            np.array([observed_val_0, observed_val_1]),
            solution(observed_time, **parameters),
        )

    n2 = 5
    observer.make_observations(n2, dt)
    assert np.allclose(observer.packaged_observations.time[-1], dt * (n1 + n2))
    for i in range(n1 + n2):
        observed_val_0 = observer.packaged_observations.isel(time=i)["X_0"]
        observed_val_1 = observer.packaged_observations.isel(time=i)["X_1"]
        observed_time = observer.packaged_observations.isel(time=i)[
            "time"
        ].values.item()
        assert np.allclose(
            np.array([observed_val_0, observed_val_1]),
            solution(observed_time, **parameters),
        )
