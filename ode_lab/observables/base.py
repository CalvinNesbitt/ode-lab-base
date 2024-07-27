from abc import ABC, abstractmethod


class Observable(ABC):
    """
    An abstract baseclass for observables.
    """

    @abstractmethod
    def look(self):
        "Look at the integrator"

    @abstractmethod
    def make_observations(self):
        "Make many observations"

    @abstractmethod
    def observations(self):
        "Return your observations"

    @abstractmethod
    def save(self):
        "Save your observations"

    @abstractmethod
    def clear(self):
        "Clear your observations"
