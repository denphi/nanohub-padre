"""
Electrode definition for PADRE simulations.

Defines electrode locations on device boundaries.
"""

from typing import Optional, Union
from .base import PadreCommand


class Electrode(PadreCommand):
    """
    Define an electrode on the device.

    Electrodes can be defined by index bounds (rectangular mesh) or
    by spatial coordinates.

    Parameters
    ----------
    number : int
        Electrode number (1-10, using 0 for 10)

    Position by indices:
    ix_low, ix_high : int, optional
        X index bounds
    iy_low, iy_high : int, optional
        Y index bounds

    Position by coordinates:
    x_min, x_max : float, optional
        X coordinate bounds (microns)
    y_min, y_max : float, optional
        Y coordinate bounds (microns)
    z_min, z_max : float, optional
        Z coordinate bounds (microns)

    region : int, optional
        Restrict electrode to specified region number

    Special operations:
    clear : bool, optional
        Reinitialize/clear this electrode
    include : int, optional
        Include another electrode into this one
    dump : int, optional
        Print terminal info separately for included electrode

    Example
    -------
    >>> # Back contact
    >>> e1 = Electrode(number=1, ix_low=1, ix_high=40, iy_low=17, iy_high=17)
    >>>
    >>> # Gate electrode
    >>> e2 = Electrode(number=2, x_min=-1, x_max=1, y_min=-0.05, y_max=-0.05)
    """

    command_name = "ELEC"

    def __init__(
        self,
        number: int,
        # Index position
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
        iy_low: Optional[int] = None,
        iy_high: Optional[int] = None,
        # Coordinate position
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        region: Optional[int] = None,
        # Special
        clear: bool = False,
        include: Optional[int] = None,
        dump: Optional[int] = None,
    ):
        super().__init__()
        self.number = number
        self.ix_low = ix_low
        self.ix_high = ix_high
        self.iy_low = iy_low
        self.iy_high = iy_high
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.region = region
        self.clear = clear
        self.include = include
        self.dump = dump

    def to_padre(self) -> str:
        params = {"NUM": self.number}
        flags = []

        # Index position
        if self.ix_low is not None:
            params["IX.L"] = self.ix_low
        if self.ix_high is not None:
            params["IX.H"] = self.ix_high
        if self.iy_low is not None:
            params["IY.L"] = self.iy_low
        if self.iy_high is not None:
            params["IY.H"] = self.iy_high

        # Coordinate position
        if self.x_min is not None:
            params["X.MIN"] = self.x_min
        if self.x_max is not None:
            params["X.MAX"] = self.x_max
        if self.y_min is not None:
            params["Y.MIN"] = self.y_min
        if self.y_max is not None:
            params["Y.MAX"] = self.y_max
        if self.z_min is not None:
            params["Z.MIN"] = self.z_min
        if self.z_max is not None:
            params["Z.MAX"] = self.z_max

        if self.region is not None:
            params["REG"] = self.region

        # Special
        if self.clear:
            flags.append("CLEAR")
        if self.include is not None:
            params["INCLUDE"] = self.include
        if self.dump is not None:
            params["DUMP"] = self.dump

        return self._build_command(params, flags)

    @classmethod
    def by_indices(cls, number: int, ix_low: int, ix_high: int,
                   iy_low: int, iy_high: int) -> "Electrode":
        """Create electrode using index bounds."""
        return cls(number=number, ix_low=ix_low, ix_high=ix_high,
                   iy_low=iy_low, iy_high=iy_high)

    @classmethod
    def by_coords(cls, number: int, x_min: float, x_max: float,
                  y_min: float, y_max: float) -> "Electrode":
        """Create electrode using coordinate bounds."""
        return cls(number=number, x_min=x_min, x_max=x_max,
                   y_min=y_min, y_max=y_max)
