"""
Contact boundary conditions for PADRE simulations.

Defines physical parameters for electrodes including work functions,
surface recombination, and lumped elements.
"""

from typing import Optional, Union, List
from .base import PadreCommand


class Contact(PadreCommand):
    """
    Define physical parameters for an electrode contact.

    Parameters
    ----------
    number : int, optional
        Electrode number to configure
    all_contacts : bool
        Apply to all electrodes

    Work function specification (one of):
    neutral : bool
        Charge-neutral ohmic contact (default)
    aluminum : bool
        Aluminum contact (4.17 V)
    p_polysilicon : bool
        P+ polysilicon (4.17 + Egap V)
    n_polysilicon : bool
        N+ polysilicon (4.17 V)
    molybdenum : bool
        Molybdenum (4.53 V)
    tungsten : bool
        Tungsten (4.63 V)
    mo_disilicide : bool
        Mo disilicide (4.80 V)
    tu_disilicide : bool
        W disilicide (4.80 V)
    workfunction : float
        Custom work function (V)
    min_workfunction : float
        Minority carrier barrier height (V)

    Surface recombination:
    surf_rec : bool
        Enable surface recombination for both carriers
    n_surf_rec : bool
        Enable electron surface recombination
    p_surf_rec : bool
        Enable hole surface recombination
    vsurfn : float
        Electron surface recombination velocity (cm/s)
    vsurfp : float
        Hole surface recombination velocity (cm/s)
    barrierl : bool
        Enable barrier lowering
    alpha : float
        Linear dipole barrier lowering coefficient

    Boundary conditions:
    current : float
        Current boundary condition (A/um)
    resistance : float
        Lumped resistance (Ohms)
    capacitance : float
        Lumped capacitance (F)
    inductance : float
        Lumped inductance (H)
    distributed : bool
        Distribute lumped element along contact
    con_resist : float
        Distributed contact resistance (Ohm-cm^2)

    Example
    -------
    >>> # Ohmic contacts for all electrodes
    >>> c_all = Contact(all_contacts=True, neutral=True)
    >>>
    >>> # Schottky contact with surface recombination
    >>> c2 = Contact(number=2, aluminum=True, surf_rec=True, barrierl=True)
    >>>
    >>> # Add lumped resistor
    >>> c3 = Contact(number=3, resistance=1e5)
    """

    command_name = "CONTACT"

    def __init__(
        self,
        number: Optional[int] = None,
        all_contacts: bool = False,
        # Work function
        neutral: bool = False,
        aluminum: bool = False,
        p_polysilicon: bool = False,
        n_polysilicon: bool = False,
        molybdenum: bool = False,
        tungsten: bool = False,
        mo_disilicide: bool = False,
        tu_disilicide: bool = False,
        workfunction: Optional[float] = None,
        min_workfunction: Optional[float] = None,
        # Surface recombination
        surf_rec: bool = False,
        n_surf_rec: bool = False,
        p_surf_rec: bool = False,
        vsurfn: Optional[float] = None,
        vsurfp: Optional[float] = None,
        barrierl: bool = False,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        # Boundary conditions
        current: Optional[float] = None,
        resistance: Optional[float] = None,
        capacitance: Optional[float] = None,
        inductance: Optional[float] = None,
        distributed: bool = False,
        con_resist: Optional[float] = None,
        r_table: Optional[str] = None,
        r_scale: Optional[float] = None,
        # Special
        one_d_base: bool = False,
    ):
        super().__init__()
        self.number = number
        self.all_contacts = all_contacts

        # Work function
        self.neutral = neutral
        self.aluminum = aluminum
        self.p_polysilicon = p_polysilicon
        self.n_polysilicon = n_polysilicon
        self.molybdenum = molybdenum
        self.tungsten = tungsten
        self.mo_disilicide = mo_disilicide
        self.tu_disilicide = tu_disilicide
        self.workfunction = workfunction
        self.min_workfunction = min_workfunction

        # Surface recombination
        self.surf_rec = surf_rec
        self.n_surf_rec = n_surf_rec
        self.p_surf_rec = p_surf_rec
        self.vsurfn = vsurfn
        self.vsurfp = vsurfp
        self.barrierl = barrierl
        self.alpha = alpha
        self.beta = beta

        # Boundary conditions
        self.current = current
        self.resistance = resistance
        self.capacitance = capacitance
        self.inductance = inductance
        self.distributed = distributed
        self.con_resist = con_resist
        self.r_table = r_table
        self.r_scale = r_scale

        # Special
        self.one_d_base = one_d_base

    def to_padre(self) -> str:
        params = {}
        flags = []

        # Number
        if self.all_contacts:
            flags.append("ALL")
        elif self.number is not None:
            params["NUM"] = self.number

        # Work function
        if self.neutral:
            flags.append("NEUTRAL")
        if self.aluminum:
            flags.append("ALUMINUM")
        if self.p_polysilicon:
            flags.append("P.POLYSILICON")
        if self.n_polysilicon:
            flags.append("N.POLYSILICON")
        if self.molybdenum:
            flags.append("MOLYBDENUM")
        if self.tungsten:
            flags.append("TUNGSTEN")
        if self.mo_disilicide:
            flags.append("MO.DISILICIDE")
        if self.tu_disilicide:
            flags.append("TU.DISILICIDE")
        if self.workfunction is not None:
            params["WORKFUNCTION"] = self.workfunction
        if self.min_workfunction is not None:
            params["MIN.WORKFUNCTION"] = self.min_workfunction

        # Surface recombination
        if self.surf_rec:
            flags.append("SURF.REC")
        if self.n_surf_rec:
            flags.append("N.SURF.REC")
        if self.p_surf_rec:
            flags.append("P.SURF.REC")
        if self.vsurfn is not None:
            params["VSURFN"] = self.vsurfn
        if self.vsurfp is not None:
            params["VSURFP"] = self.vsurfp
        if self.barrierl:
            flags.append("BARRIERL")
        if self.alpha is not None:
            params["ALPHA"] = self.alpha
        if self.beta is not None:
            params["BETA"] = self.beta

        # Boundary conditions
        if self.current is not None:
            params["CURRENT"] = self.current
        if self.resistance is not None:
            params["RESISTANCE"] = self.resistance
        if self.capacitance is not None:
            params["CAPACITANCE"] = self.capacitance
        if self.inductance is not None:
            params["INDUCTANCE"] = self.inductance
        if self.distributed:
            flags.append("DISTRIBUTED")
        if self.con_resist is not None:
            params["CON.RESIST"] = self.con_resist
        if self.r_table:
            params["R.TABLE"] = self.r_table
        if self.r_scale is not None:
            params["R.SCALE"] = self.r_scale

        # Special
        if self.one_d_base:
            flags.append("1D.BASE")

        return self._build_command(params, flags)

    @classmethod
    def ohmic(cls, number: Optional[int] = None,
              all_contacts: bool = False) -> "Contact":
        """Create an ohmic (charge-neutral) contact."""
        return cls(number=number, all_contacts=all_contacts, neutral=True)

    @classmethod
    def schottky(cls, number: int, workfunction: float,
                 surf_rec: bool = True, barrierl: bool = True) -> "Contact":
        """Create a Schottky contact with surface recombination."""
        return cls(number=number, workfunction=workfunction,
                   surf_rec=surf_rec, barrierl=barrierl)

    @classmethod
    def with_resistance(cls, number: int, resistance: float,
                        distributed: bool = False) -> "Contact":
        """Create a contact with lumped resistance."""
        return cls(number=number, resistance=resistance,
                   distributed=distributed)
