# Imports
import inspect
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


class Integrator:
    """
    Integrate a system of ordinary diffferential equations dx/dt = f(x).
    This is a light wrapper of scipy.integrate.solve_ivp [1].

    ...

    Attributes
    ----------
    rhs : function
        The rhs of our ODE system.
    ic : np.array
        The ic for our IVP.
    state: np.array
        The current point we're at in our integration.
    time: float
        How long we've integrated for.
    method: string
        The scheme we use to integrate our IVP.

    Methods
    -------
    run(t)
        Integrate the ODEs for a length of time t.

    References
    -------
    """

    def __init__(
        self,
        rhs: Callable,
        ic: np.ndarray | list,
        parameters: dict | None = None,
        method: str = "DOP853",
    ):
        """
        Initialise an OdeIntegrator object.

        Parameters
        -------
        rhs : function
            The rhs of our ODE system.
            Expected to be of the form rhs(x, **parameters).
        ic : np.array| list
            The ic for our IVP.
        parameters: dict, optional
            Parameters of our ode.
            {'foo' : bar} will be passed as rhs(foo=bar).
        method: string, optional
            The scheme we use to integrate our IVP.
        """
        self.rhs = rhs
        if isinstance(ic, list):
            ic = np.array(ic)
        self.ic = ic
        self.state = ic

        # Assume first parameter is state, and rest are args/kwargs.
        sig = inspect.signature(rhs)
        param_keys = list(sig.parameters.keys())[1:]

        if parameters is None:
            parameters = {k: sig.parameters[k].default for k in param_keys}
        else:
            for k in param_keys:
                if k not in parameters:
                    parameters[k] = sig.parameters[k].default

        self.parameters = parameters

        self.time = 0
        self.method = method
        self.ndim = len(ic)

    def _rhs_dt(self, _: float, state: np.array) -> np.array:
        return self.rhs(state, **self.parameters)

    def run(self, t: float) -> None:
        """
        Integrate ODEs for a time t. This will update both .state and .time attributes.

        Parameters
        -------
        t : float
            How long we integrate.

        Returns
        -------
        None

        """

        # Integration, default uses RK45 with adaptive stepping.
        solver_return = solve_ivp(
            self._rhs_dt,
            (self.time, self.time + t),
            self.state,
            dense_output=True,
            method=self.method,
        )

        # Updating variables
        self.state = solver_return.y[:, -1]
        self.time = self.time + t


class EnsembleIntegrator:
    """
    Integrate a system of ordinary diffferential equations dx/dt = f(x) for
    an ensemble of initial conditions.
    """

    def __init__(
        self,
        rhs: Callable,
        ic_list: list,
        parameters: dict | None = None,
        method: str = "RK45",
    ):
        self.integrators = [Integrator(rhs, ic, parameters, method) for ic in ic_list]

    def run(self, t: float) -> None:
        for integrator in self.integrators:
            integrator.run(t)
