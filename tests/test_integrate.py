import numpy as np

from ode_lab.base.integrate import Integrator


def test_integrator_1d(linear_ode_1d):
    rhs, solution, initial_condition, parameters = linear_ode_1d
    integrator = Integrator(rhs, initial_condition, parameters)
    assert integrator.ndim == 1

    # Check we pack things correctly
    assert integrator.time == 0.0
    assert np.allclose(integrator.state, initial_condition)
    assert isinstance(integrator.state, np.ndarray)
    assert integrator.parameters == parameters
    assert integrator.rhs == rhs

    # Check we can integrate
    t1 = 0.1
    integrator.run(t1)
    assert integrator.time == t1
    assert np.allclose(integrator.state, solution(t1, **parameters))
    assert integrator.parameters == parameters

    t2 = 0.5
    integrator.run(t2)
    assert integrator.time == t1 + t2
    assert np.allclose(integrator.state, solution(t1 + t2, **parameters))
    assert integrator.parameters == parameters
    assert isinstance(integrator.state, np.ndarray)


def test_integrator_2d(linear_ode_2d):
    rhs, solution, initial_condition, parameters = linear_ode_2d
    integrator = Integrator(rhs, initial_condition, parameters)
    assert integrator.ndim == 2

    # Check we pack things correctly
    assert integrator.time == 0.0
    assert np.allclose(integrator.state, initial_condition)
    assert isinstance(integrator.state, np.ndarray)
    assert integrator.parameters == parameters
    assert integrator.rhs == rhs

    # Check we can integrate
    t1 = 0.1
    integrator.run(t1)
    assert integrator.time == t1
    assert np.allclose(integrator.state, solution(t1, **parameters))
    assert integrator.parameters == parameters

    t2 = 0.5
    integrator.run(t2)
    assert integrator.time == t1 + t2
    assert np.allclose(integrator.state, solution(t1 + t2, **parameters))
    assert integrator.parameters == parameters
    assert isinstance(integrator.state, np.ndarray)
