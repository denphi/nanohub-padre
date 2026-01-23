"""
3D plotting for PADRE simulations.

PLOT.3D outputs scatter files for 3D visualization.
"""

from typing import Optional, List, Union
from .base import PadreCommand


class Plot3D(PadreCommand):
    """
    Dump 3D scatter plot files for visualization.

    Parameters
    ----------
    Quantities (up to 5):
    potential : bool
        Mid-gap potential
    qfn : bool
        Electron quasi-Fermi level
    qfp : bool
        Hole quasi-Fermi level
    n_temp : bool
        Electron temperature
    p_temp : bool
        Hole temperature
    band_val : bool
        Valence band
    band_cond : bool
        Conduction band
    doping : bool
        Doping concentration
    electrons : bool
        Electron concentration
    holes : bool
        Hole concentration
    net_charge : bool
        Net charge
    net_carrier : bool
        Net carrier concentration
    e_field : bool
        Electric field
    recomb : bool
        Recombination rate

    Control:
    outfile : str
        Output scatter file name
    region : list
        Regions to include
    ign_region : list
        Regions to ignore
    semiconductor : bool
        Include semiconductor regions (default True)
    insulator : bool
        Include insulator regions (default True)
    logarithm : bool
        Logarithmic scale
    absolute : bool
        Absolute value
    x_compon, y_compon, z_compon : bool
        Vector components

    Example
    -------
    >>> # Save potential and carrier concentrations
    >>> p = Plot3D(potential=True, electrons=True, holes=True,
    ...            outfile="plt.wmc")
    """

    command_name = "PLOT.3D"

    def __init__(
        self,
        # Quantities
        potential: bool = False,
        qfn: bool = False,
        qfp: bool = False,
        n_temp: bool = False,
        p_temp: bool = False,
        band_val: bool = False,
        band_cond: bool = False,
        doping: bool = False,
        electrons: bool = False,
        holes: bool = False,
        net_charge: bool = False,
        net_carrier: bool = False,
        e_field: bool = False,
        recomb: bool = False,
        # Control
        outfile: Optional[str] = None,
        region: Optional[List[int]] = None,
        ign_region: Optional[List[int]] = None,
        semiconductor: bool = True,
        insulator: bool = True,
        absolute: bool = False,
        logarithm: bool = False,
        x_compon: bool = False,
        y_compon: bool = False,
        z_compon: bool = False,
        mix_mater: bool = False,
    ):
        super().__init__()
        self.potential = potential
        self.qfn = qfn
        self.qfp = qfp
        self.n_temp = n_temp
        self.p_temp = p_temp
        self.band_val = band_val
        self.band_cond = band_cond
        self.doping = doping
        self.electrons = electrons
        self.holes = holes
        self.net_charge = net_charge
        self.net_carrier = net_carrier
        self.e_field = e_field
        self.recomb = recomb
        self.outfile = outfile
        self.region = region
        self.ign_region = ign_region
        self.semiconductor = semiconductor
        self.insulator = insulator
        self.absolute = absolute
        self.logarithm = logarithm
        self.x_compon = x_compon
        self.y_compon = y_compon
        self.z_compon = z_compon
        self.mix_mater = mix_mater

    def to_padre(self) -> str:
        params = {}
        flags = []

        # Quantities
        if self.potential:
            flags.append("POTEN")
        if self.qfn:
            flags.append("QFN")
        if self.qfp:
            flags.append("QFP")
        if self.n_temp:
            flags.append("N.TEMP")
        if self.p_temp:
            flags.append("P.TEMP")
        if self.band_val:
            flags.append("BAND.VAL")
        if self.band_cond:
            flags.append("BAND.COND")
        if self.doping:
            flags.append("DOPING")
        if self.electrons:
            flags.append("ELECT")
        if self.holes:
            flags.append("HOLES")
        if self.net_charge:
            flags.append("NET.CH")
        if self.net_carrier:
            flags.append("NET.CA")
        if self.e_field:
            flags.append("E.FIELD")
        if self.recomb:
            flags.append("RECOMB")

        # Control
        if self.outfile:
            params["OUTF"] = self.outfile
        if self.region:
            params["REGION"] = ",".join(str(r) for r in self.region)
        if self.ign_region:
            params["IGN.REGION"] = ",".join(str(r) for r in self.ign_region)
        if not self.semiconductor:
            params["SEMI"] = False
        if not self.insulator:
            params["INS"] = False
        if self.absolute:
            flags.append("ABS")
        if self.logarithm:
            flags.append("LOG")
        if self.x_compon:
            flags.append("X.COMP")
        if self.y_compon:
            flags.append("Y.COMP")
        if self.z_compon:
            flags.append("Z.COMP")
        if self.mix_mater:
            flags.append("MIX.MATER")

        return self._build_command(params, flags)
