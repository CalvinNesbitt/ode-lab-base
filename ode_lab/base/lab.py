from tqdm import tqdm

from ode_lab.base.integrate import Integrator
from ode_lab.base.logger import logger
from ode_lab.base.observe import Observer


class ODELab:
    """
    High level class for integrating and observing ODEs.
    """

    def __init__(self, integrator: Integrator, observer: Observer):
        self.integrator = integrator
        self.observer = observer
        observable_names = observer.observable_names
        self.observer.observe_attributes(self.integrator)
        self.have_I_run_a_transient = False

        # Check that the observer is observing the integrator
        self.observable_output_dimension = len(
            self.observer.observing_function(self.integrator)
        )
        if observable_names is None:
            observable_names = [
                f"O_{i}" for i in range(self.observable_output_dimension)
            ]
        self.observable_names = observable_names
        try:
            assert len(self.observable_names) == self.observable_output_dimension
        except AssertionError:
            raise ValueError(
                "The number of observable names must match"
                + "the output dimension of the observable function."
            )

    def make_observations(
        self, number: int, frequency: float, transient: float = 0, timer: bool = False
    ) -> None:
        if isinstance(number, float):
            number = int(number)

        # Determine if we need to run a transient
        if self.have_I_run_a_transient and transient > 0:
            logger.warning(
                "I've already run a transient! I'm going to ignore this one."
            )
            transient = 0
        if transient > 0:
            self.integrator.run(transient)  # No observations for transient
            self.integrator.time = 0  # Reset time
            self.have_I_run_a_transient = True

        # Make observations
        self.observer.look(self.integrator)
        logger.info(f"Making {number} observations with frequency {frequency}.")
        for _ in tqdm(range(number), disable=not timer):
            self.integrator.run(frequency)
            self.observer.look(self.integrator)
        return

    @property
    def observations(self):
        return self.observer.observations
