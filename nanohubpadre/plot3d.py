"""
3D plotting for PADRE simulations.

PLOT.3D outputs scatter files for 3D visualization.
This module also provides a parser for PADRE Plot3D scatter output files.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np
from .base import PadreCommand


@dataclass
class Plot3DData:
    """
    Parsed data from a PADRE Plot3D scatter file.

    Attributes
    ----------
    x : np.ndarray
        X coordinates of mesh nodes (in cm)
    y : np.ndarray
        Y coordinates of mesh nodes (in cm)
    values : np.ndarray
        Quantity values at each node
    label : str
        Label of the quantity (e.g., "Potential", "Doping")
    num_nodes : int
        Total number of nodes parsed
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    label: str = ""
    num_nodes: int = 0

    def to_grid(self, n_grid: int = 100, method: str = 'linear'):
        """
        Interpolate scattered data onto a regular grid.

        Uses scipy.interpolate.griddata (same approach as the
        Rappture MESFET Lab's MATLAB post-processing).

        Parameters
        ----------
        n_grid : int
            Number of grid points in each direction (default: 100)
        method : str
            Interpolation method: 'linear', 'cubic', or 'nearest'
            (default: 'linear' — avoids boundary overshoot)

        Returns
        -------
        tuple (xi, yi, zi)
            xi : 1D array of x grid coordinates (µm)
            yi : 1D array of y grid coordinates (µm)
            zi : 2D array (n_grid, n_grid) of interpolated values
        """
        from scipy.interpolate import griddata

        x_um = self.x * 1e4
        y_um = self.y * 1e4

        xi = np.linspace(x_um.min(), x_um.max(), n_grid)
        yi = np.linspace(y_um.min(), y_um.max(), n_grid)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        zi = griddata(
            (x_um, y_um), self.values,
            (xi_grid, yi_grid), method=method
        )

        return xi, yi, zi


def parse_plot3d_file(filepath: str) -> Plot3DData:
    """
    Parse a PADRE Plot3D scatter file.

    PADRE's ``PLOT.3D`` command writes scatter files with the following
    structure:

    - 11 header lines: ``begin scatter``, ``origin``, ``d``, ``title``,
      ``begin p``, ``td``, ``topology point``, ``m <ncols>``,
      ``total <N>``, ``label ...``, ``f``
    - *N* data lines (repeated twice for z=0 and z=1 planes):
      ``x  y  z  value``  (coordinates in cm)
    - ``end p`` marker
    - A second section (``begin P``) with mesh connectivity indices
      (prism topology) — **not** data; skipped entirely.

    For a 2D device, PADRE duplicates every node at z=0 and z=1.
    Only the z=0 plane is kept.

    Parameters
    ----------
    filepath : str
        Path to the Plot3D scatter file

    Returns
    -------
    Plot3DData
        Parsed mesh-node data with x, y coordinates (cm) and values

    Example
    -------
    >>> data = parse_plot3d_file("output/pot_eq")
    >>> print(f"{data.num_nodes} nodes, label={data.label}")
    >>> xi, yi, zi = data.to_grid(n_grid=100)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    xs, ys, vals = [], [], []
    in_data = False
    label = ""

    for line in lines:
        stripped = line.strip()

        # Extract the quantity label from the header
        if stripped.startswith('label '):
            # e.g. label "x-coord" "y-coord" "z-coord" "Potential"
            parts = stripped.split('"')
            if len(parts) >= 8:
                label = parts[7]  # 4th quoted string

        # Start reading after the "f" header line
        if stripped == 'f' and not in_data:
            in_data = True
            continue

        # Stop at "end p" — don't parse the connectivity section
        if stripped == 'end p':
            break

        if not in_data:
            continue

        parts = stripped.split()
        if len(parts) != 4:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            val = float(parts[3])
        except ValueError:
            continue

        # Keep only the z=0 plane (z=1 is a duplicate for 2D devices)
        if abs(z) > 0.5:
            continue

        xs.append(x)
        ys.append(y)
        vals.append(val)

    result = Plot3DData(
        x=np.array(xs),
        y=np.array(ys),
        values=np.array(vals),
        label=label,
        num_nodes=len(xs),
    )
    return result


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
