"""
Interface definition for PADRE simulations.

Defines surface recombination and fixed charges at interfaces.
"""

from typing import Optional, Union, List
from .base import PadreCommand


class Interface(PadreCommand):
    """
    Define interface properties (recombination, fixed charge).

    Parameters
    ----------
    number : int
        Interface number
    qf : float, optional
        Fixed charge density (/cm^2)
    s_n : float or list, optional
        Electron surface recombination velocity (cm/s)
    s_p : float or list, optional
        Hole surface recombination velocity (cm/s)
    trap_type : str, optional
        Trap types ("n", "p", "0" for each level)
    etrap : float or list, optional
        Trap energy levels (Et - Ei in eV)
    ntrap : float or list, optional
        Trap densities (/cm^2)

    Example
    -------
    >>> # Si-SiO2 interface with fixed charge and recombination
    >>> intf = Interface(number=1, qf=1e10, s_n=1e4, s_p=1e4)
    """

    command_name = "INTERFACE"

    def __init__(
        self,
        number: int,
        qf: Optional[float] = None,
        s_n: Optional[Union[float, List[float]]] = None,
        s_p: Optional[Union[float, List[float]]] = None,
        trap_type: Optional[str] = None,
        etrap: Optional[Union[float, List[float]]] = None,
        ntrap: Optional[Union[float, List[float]]] = None,
    ):
        super().__init__()
        self.number = number
        self.qf = qf
        self.s_n = s_n
        self.s_p = s_p
        self.trap_type = trap_type
        self.etrap = etrap
        self.ntrap = ntrap

    def _format_vector(self, value: Union[float, List[float]]) -> str:
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return str(value)

    def to_padre(self) -> str:
        params = {"NUM": self.number}
        flags = []

        if self.qf is not None:
            params["QF"] = self.qf
        if self.s_n is not None:
            params["S.N"] = self._format_vector(self.s_n)
        if self.s_p is not None:
            params["S.P"] = self._format_vector(self.s_p)
        if self.trap_type:
            params["TRAP.TYPE"] = self.trap_type
        if self.etrap is not None:
            params["ETRAP"] = self._format_vector(self.etrap)
        if self.ntrap is not None:
            params["NTRAP"] = self._format_vector(self.ntrap)

        return self._build_command(params, flags)


class Surface(PadreCommand):
    """
    Define surface/interface location for later reference.

    Parameters
    ----------
    number : int
        Surface/electrode number
    interface : bool
        Define as interface (vs electrode)
    electrode : bool
        Define as electrode
    x_min, x_max : float
        X bounds (microns)
    y_min, y_max : float
        Y bounds (microns)
    z_min, z_max : float
        Z bounds (microns)
    reg1, reg2 : int
        Region numbers for interface boundary

    Example
    -------
    >>> # Interface between regions 1 and 4
    >>> surf = Surface(number=1, interface=True, reg1=1, reg2=4,
    ...                x_min=-4, x_max=4, y_min=-0.5, y_max=4)
    """

    command_name = "SURFACE"

    def __init__(
        self,
        number: int,
        interface: bool = False,
        electrode: bool = False,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        reg1: Optional[int] = None,
        reg2: Optional[int] = None,
    ):
        super().__init__()
        self.number = number
        self.interface = interface
        self.electrode = electrode
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.reg1 = reg1
        self.reg2 = reg2

    def to_padre(self) -> str:
        params = {"NUM": self.number}
        flags = []

        if self.interface:
            flags.append("INTERFACE")
        if self.electrode:
            flags.append("ELECTRODE")

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
        if self.reg1 is not None:
            params["REG1"] = self.reg1
        if self.reg2 is not None:
            params["REG2"] = self.reg2

        return self._build_command(params, flags)
