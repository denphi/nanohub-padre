"""
Region definition for PADRE simulations.

Defines material regions within the device mesh.
"""

from typing import Optional, Union
from .base import PadreCommand


class Region(PadreCommand):
    """
    Define a material region in the device mesh.

    Every element in the mesh must be assigned to some material region.

    Parameters
    ----------
    number : int
        Region number identifier
    material : str, optional
        Material name (e.g., "silicon", "gaas", "sio2", "si3n4")
    semiconductor : bool, optional
        Mark region as semiconductor (carrier transport computed)
    insulator : bool, optional
        Mark region as insulator (only displacement currents)

    Position by indices (for rectangular meshes):
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

    newnum : int, optional
        Assign a new region number to this subregion

    Predefined materials (for backward compatibility):
    silicon, gaas, germanium, oxide, nitride, sapphire

    Example
    -------
    >>> # Silicon substrate region
    >>> r1 = Region(number=1, material="silicon", semiconductor=True,
    ...             ix_low=1, ix_high=25, iy_low=4, iy_high=20)
    >>>
    >>> # Gate oxide
    >>> r2 = Region(number=2, material="sio2", insulator=True,
    ...             x_min=0, x_max=1, y_min=-0.04, y_max=0)
    """

    command_name = "REGION"

    # Known PADRE materials
    SEMICONDUCTORS = {"silicon", "gaas", "germanium", "poly", "sige", "iga53", "aga45"}
    INSULATORS = {"sio2", "oxide", "si3n4", "nitride", "sapphire"}

    def __init__(
        self,
        number: int,
        material: Optional[str] = None,
        semiconductor: Optional[bool] = None,
        insulator: Optional[bool] = None,
        # Index-based position
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
        iy_low: Optional[int] = None,
        iy_high: Optional[int] = None,
        # Coordinate-based position
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        # Special
        newnum: Optional[int] = None,
        # Legacy material flags
        silicon: bool = False,
        gaas: bool = False,
        germanium: bool = False,
        oxide: bool = False,
        nitride: bool = False,
        sapphire: bool = False,
    ):
        super().__init__()
        self.number = number
        self.material = material
        self.semiconductor = semiconductor
        self.insulator = insulator

        # Position
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
        self.newnum = newnum

        # Legacy flags
        self.silicon = silicon
        self.gaas = gaas
        self.germanium = germanium
        self.oxide = oxide
        self.nitride = nitride
        self.sapphire = sapphire

    def to_padre(self) -> str:
        params = {"NUM": self.number}
        flags = []

        # Position - indices first (matches original format)
        if self.ix_low is not None:
            params["IX.L"] = self.ix_low
        if self.ix_high is not None:
            params["IX.H"] = self.ix_high
        if self.iy_low is not None:
            params["IY.L"] = self.iy_low
        if self.iy_high is not None:
            params["IY.H"] = self.iy_high

        # Material
        if self.material:
            params["NAME"] = self.material

        # Type flags
        if self.semiconductor:
            flags.append("SEMI")
        if self.insulator:
            flags.append("INS")

        # Position - coordinates
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

        # Legacy material flags
        if self.silicon:
            flags.append("SILICON")
        if self.gaas:
            flags.append("GAAS")
        if self.germanium:
            flags.append("GERMANIUM")
        if self.oxide:
            flags.append("OXIDE")
        if self.nitride:
            flags.append("NITRIDE")
        if self.sapphire:
            flags.append("SAPPHIRE")

        # Special
        if self.newnum is not None:
            params["NEW"] = self.newnum

        return self._build_command(params, flags)

    @classmethod
    def silicon(cls, number: int, **kwargs) -> "Region":
        """Create a silicon semiconductor region."""
        return cls(number=number, material="silicon", semiconductor=True, **kwargs)

    @classmethod
    def oxide(cls, number: int, **kwargs) -> "Region":
        """Create an SiO2 insulator region."""
        return cls(number=number, material="sio2", insulator=True, **kwargs)

    @classmethod
    def gaas(cls, number: int, **kwargs) -> "Region":
        """Create a GaAs semiconductor region."""
        return cls(number=number, material="gaas", semiconductor=True, **kwargs)
