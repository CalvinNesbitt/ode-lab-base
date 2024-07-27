import logging
from pathlib import Path
from typing import Union

import numpy as np
import tqdm
import xarray as xr

from ode_lab.integrate import Integrator
from ode_lab.logger import logger


class BaseObserver:
    def __init__(
        self,
        integrator: Integrator,
        observable_names: list["str"] | None = None,
        log_level: str = "INFO",
        log_file: str | None = None,
    ):
        """
        Class to observe an integrator object and store observations
        as xarray datasets.

        Parameters
        ----------
        integrator : Integrator
            An integrator object to observe.
        """

        # Needed knowledge of the integrator
        self.parameters = integrator.parameters
        self.integrator = integrator

        # Observable info
        self.observable_output_dimension = len(
            self.observing_function(integrator.state, integrator.time)
        )
        if observable_names is None:
            observable_names = [
                f"O_{i}" for i in range(self.observable_output_dimension)
            ]

        # Observation logs
        self._time_obs = []  # Times we've made observations
        self._observations = []
        self.number_of_saves = 0
        self.have_I_run_a_transient = False

        # Set up logging
        self.logger = logger
        self.set_logging_level(log_level)
        if log_file is not None:
            self.set_log_file(log_file)

    def observing_function(self, state: np.ndarray, time: float) -> np.ndarray:
        """
        Function to observe the state of the system at a given time.

        Parameters
        ----------
        state : np.ndarray
            The state of the system.
        time : float
            The time at which the observation is made.

        Returns
        -------
        np.ndarray
            The observation made at the given time.
        """
        raise NotImplementedError

    @property
    def packaged_observations(self) -> xr.Dataset:
        """
        Package the observations into an xarray dataset.

        Returns
        -------
        xr.Dataset
            An xarray dataset containing the observations.
        """

        time_dict = {"time": self._time_obs}
        obs_dict = {
            self.observable_names[i]: [obs[i] for obs in self._observations]
            for i in range(self.observable_output_dimension)
        }
        return xr.Dataset(
            data_vars={**time_dict, **obs_dict},
            coords={"time": self._time_obs},
            attrs=self.parameters,
        )

    def look(self, integrator: Integrator) -> None:
        """Look at the integrator and store observations"""
        self._time_obs.append(integrator.time)
        observation = self.observing_function(integrator.state, integrator.time)
        self._observations.append(observation)
        return

    def make_observations(
        self, number: int, frequency: float, transient: float = 0, timer: bool = False
    ) -> None:
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
        self.look(self.integrator)  # Initial observation
        logger.info(f"Making {number} observations with frequency {frequency}.")
        for _ in tqdm(range(number), disable=not timer):
            self.integrator.run(frequency)
            self.look(self.integrator)
        return

    def wipe_observations(self):
        """Erases observations"""
        self._time_obs = []
        self._observations = []

    def save_observations(self, file_path: Union[str, Path, None]) -> None:
        """
        Save the observations to a netcdf file.

        Parameters
        ----------
        file_path : str | Path | None
            The path to the file to write.
            If None, the file will be named
            observations_{number_of_saves}.nc.

        """

        if len(self._observations) == 0:
            logger.info("I have no observations to save.")
            return

        # Guarantee that the file path is a Path object
        if isinstance(file_path, None):
            file_path = Path(f"observations_{self.number_of_saves}.nc")

        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Ensure that the file ends in .nc
        if not file_path.suffix == ".nc":
            file_path = file_path.with_suffix(".nc")

        # Ensure that the directory exists
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
            logger.info(f"Making directory at {file_path.parent}")

        nc_file = self.packaged_observations
        nc_file.to_netcdf(file_path)
        logger.info(
            f"Wrote {len(self._observations)} written to {file_path}. "
            "Wiping observations."
        )
        self.wipe_observations()
        self.save_count += 1
        return

    def set_logging_level(self, level: str) -> None:
        """
        Set the logging level for the logger.

        Parameters:
        level (str): The logging level to set.
        Can be 'DEBUG', 'INFO', 'WARNING',
        'ERROR', 'CRITICAL', or 'NONE'.
        """
        if level.upper() == "NONE":
            numeric_level = logging.CRITICAL + 1
        else:
            numeric_level = getattr(logging, level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {level}")
        self.logger.setLevel(numeric_level)
        for handler in self.logger.handlers:
            handler.setLevel(numeric_level)

    def set_log_file(self, file_path: str) -> None:
        """
        Set the file to which the logger writes.

        Parameters:
        file_path (str): The path to the log file.
        """

        # Remove existing file handlers
        self.logger.handlers = [
            h for h in self.logger.handlers if not isinstance(h, logging.FileHandler)
        ]

        # Add new file handler
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(self.logger.level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
