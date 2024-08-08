import logging
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

from ode_lab.base.integrate import Integrator
from ode_lab.base.logger import logger


class Observer:
    def __init__(
        self,
        observable_names: list["str"] | None = None,
        log_level: str = "INFO",
        log_file: str | None = None,
    ):
        """
        Class to observe an integrator object and store observations
        as xarray datasets.

        Parameters
        ----------

        """
        self.observable_names = observable_names
        # Observation logs
        self._time_obs = []  # Times we've made observations
        self._observations = []
        self.number_of_saves = 0
        self.attrs = None

        # Set up logging
        self.logger = logger
        self.set_logging_level(log_level)
        if log_file is not None:
            self.set_log_file(log_file)

    def observing_function(self, integrator: Integrator) -> np.ndarray:
        raise NotImplementedError

    @property
    def observations(self) -> xr.Dataset:
        """
        Package the observations into an xarray dataset.

        Returns
        -------
        xr.Dataset
            An xarray dataset containing the observations.
        """

        data_var_dict = {}
        for i, obs_name in enumerate(self.observable_names):
            data_var_dict[obs_name] = xr.DataArray(
                np.stack(self._observations)[:, i],
                dims=["time"],
                coords={"time": self._time_obs},
            )
        return xr.Dataset(data_var_dict, attrs=self.attrs)

    def look(self, integrator: Integrator) -> None:
        """Look at the integrator state and store observations"""
        self._time_obs.append(integrator.time)
        observation = self.observing_function(integrator)
        self._observations.append(observation)
        return

    def observe_attributes(self, integrator: Integrator) -> None:
        self.attrs = integrator.parameters
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

        nc_file = self.observations
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


class TrajectoryObserver(Observer):
    def observing_function(self, integrator: Integrator) -> np.ndarray:
        return integrator.state
