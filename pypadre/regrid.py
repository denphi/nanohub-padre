"""
Mesh refinement/coarsening for PADRE simulations.

REGRID performs one-time refinement, ADAPT enables automatic adaptation.
"""

from typing import Optional, Union, List
from .base import PadreCommand


class Regrid(PadreCommand):
    """
    Perform mesh refinement or coarsening.

    Parameters
    ----------
    Variable to refine on (one of):
    potential : bool
        Mid-gap potential
    qfn : bool
        Electron quasi-Fermi level
    qfp : bool
        Hole quasi-Fermi level
    doping : bool
        Net doping concentration
    electron : bool
        Electron concentration
    hole : bool
        Hole concentration
    error : bool
        Potential error estimate

    Criterion:
    r_step : float
        Refinement criterion (change threshold)
    c_step : float
        Coarsening criterion
    logarithm : bool
        Use logarithmic comparison
    absolute : bool
        Use absolute value
    relative : bool
        Use relative comparison
    refine : bool
        Force refinement
    coarsen : bool
        Force coarsening

    Location bounds:
    x_min, x_max : float
        X bounds (microns)
    y_min, y_max : float
        Y bounds (microns)
    z_min, z_max : float
        Z bounds (microns)
    region : int
        Region to refine
    ignore : int
        Region to ignore

    Control:
    max_level : int
        Maximum refinement level
    rel_level : int
        Relative level change allowed
    lev_ignore : bool
        Ignore level limits
    hmin : float
        Minimum edge length (microns)
    debye : float
        Minimum edge as multiple of Debye length
    smooth_k : int
        Smoothing key
    it_smooth : int
        Smoothing iterations

    Files:
    outfile : str
        Output mesh file
    dopfile : str
        Doping file for re-doping
    ascii : bool
        ASCII file format

    Example
    -------
    >>> # Refine on doping gradient
    >>> rg = Regrid(doping=True, logarithm=True, r_step=6,
    ...             outfile="grid1", dopfile="dop1")
    >>>
    >>> # Refine until minimum spacing reached
    >>> rg = Regrid(doping=True, logarithm=True, r_step=6,
    ...             lev_ignore=True, hmin=0.01, outfile="grid2")
    """

    command_name = "REGRID"

    def __init__(
        self,
        # Variable
        potential: bool = False,
        qfn: bool = False,
        qfp: bool = False,
        n_temp: bool = False,
        p_temp: bool = False,
        doping: bool = False,
        ion_imp: bool = False,
        electron: bool = False,
        hole: bool = False,
        net_chrg: bool = False,
        net_carr: bool = False,
        min_carr: bool = False,
        p_track: bool = False,
        hetero: bool = False,
        error: bool = False,
        # Criterion
        r_step: Optional[float] = None,
        c_step: Optional[float] = None,
        change: bool = True,
        relative: bool = False,
        dv_min: Optional[float] = None,
        r_threshold: Optional[float] = None,
        n_threshold: Optional[int] = None,
        f_threshold: Optional[float] = None,
        refine: bool = False,
        coarsen: bool = False,
        localdop: bool = False,
        logarithm: bool = False,
        absolute: bool = False,
        # Location
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        region: Optional[int] = None,
        ignore: Optional[int] = None,
        box_refine: bool = False,
        # Control
        max_level: Optional[int] = None,
        rel_level: Optional[int] = None,
        lev_ignore: bool = False,
        hmin: Optional[float] = None,
        hdir: bool = True,
        debye: Optional[float] = None,
        freeze: bool = False,
        smooth_k: Optional[int] = None,
        it_smooth: Optional[int] = None,
        condense: Optional[str] = None,
        reorder: bool = False,
        three_d_refine: bool = False,
        dz_level: Optional[int] = None,
        # Files
        outfile: Optional[str] = None,
        out_green: Optional[str] = None,
        in_green: Optional[str] = None,
        no_green: bool = False,
        dopfile: Optional[str] = None,
        ascii: bool = True,
        stats: bool = False,
        # Interpolation
        fem: bool = True,
    ):
        super().__init__()
        # Variable
        self.potential = potential
        self.qfn = qfn
        self.qfp = qfp
        self.n_temp = n_temp
        self.p_temp = p_temp
        self.doping = doping
        self.ion_imp = ion_imp
        self.electron = electron
        self.hole = hole
        self.net_chrg = net_chrg
        self.net_carr = net_carr
        self.min_carr = min_carr
        self.p_track = p_track
        self.hetero = hetero
        self.error = error

        # Criterion
        self.r_step = r_step
        self.c_step = c_step
        self.change = change
        self.relative = relative
        self.dv_min = dv_min
        self.r_threshold = r_threshold
        self.n_threshold = n_threshold
        self.f_threshold = f_threshold
        self.refine = refine
        self.coarsen = coarsen
        self.localdop = localdop
        self.logarithm = logarithm
        self.absolute = absolute

        # Location
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.region = region
        self.ignore = ignore
        self.box_refine = box_refine

        # Control
        self.max_level = max_level
        self.rel_level = rel_level
        self.lev_ignore = lev_ignore
        self.hmin = hmin
        self.hdir = hdir
        self.debye = debye
        self.freeze = freeze
        self.smooth_k = smooth_k
        self.it_smooth = it_smooth
        self.condense = condense
        self.reorder = reorder
        self.three_d_refine = three_d_refine
        self.dz_level = dz_level

        # Files
        self.outfile = outfile
        self.out_green = out_green
        self.in_green = in_green
        self.no_green = no_green
        self.dopfile = dopfile
        self.ascii = ascii
        self.stats = stats

        # Interpolation
        self.fem = fem

    def to_padre(self) -> str:
        params = {}
        flags = []

        # Variable
        if self.potential:
            flags.append("POTENTIAL")
        if self.qfn:
            flags.append("QFN")
        if self.qfp:
            flags.append("QFP")
        if self.n_temp:
            flags.append("N.TEMP")
        if self.p_temp:
            flags.append("P.TEMP")
        if self.doping:
            flags.append("DOPING")
        if self.ion_imp:
            flags.append("ION.IMP")
        if self.electron:
            flags.append("ELECTRON")
        if self.hole:
            flags.append("HOLE")
        if self.net_chrg:
            flags.append("NET.CHRG")
        if self.net_carr:
            flags.append("NET.CARR")
        if self.min_carr:
            flags.append("MIN.CARR")
        if self.p_track:
            flags.append("P.TRACK")
        if self.hetero:
            flags.append("HETERO")
        if self.error:
            flags.append("ERROR")

        # Criterion
        if self.r_step is not None:
            params["R.STEP"] = self.r_step
        if self.c_step is not None:
            params["C.STEP"] = self.c_step
        if not self.change:
            params["CHANGE"] = False
        if self.relative:
            flags.append("RELATIVE")
        if self.dv_min is not None:
            params["DV.MIN"] = self.dv_min
        if self.r_threshold is not None:
            params["R.THRESH"] = self.r_threshold
        if self.n_threshold is not None:
            params["N.THRESH"] = self.n_threshold
        if self.f_threshold is not None:
            params["F.THRESH"] = self.f_threshold
        if self.refine:
            flags.append("REFINE")
        if self.coarsen:
            flags.append("COARSEN")
        if self.localdop:
            flags.append("LOCALDOP")
        if self.logarithm:
            flags.append("LOG")
        if self.absolute:
            flags.append("ABS")

        # Location
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
            params["REGION"] = self.region
        if self.ignore is not None:
            params["IGNORE"] = self.ignore
        if self.box_refine:
            flags.append("BOX.REFINE")

        # Control
        if self.max_level is not None:
            params["MAX.LEVEL"] = self.max_level
        if self.rel_level is not None:
            params["REL.LEVEL"] = self.rel_level
        if self.lev_ignore:
            flags.append("LEV.IGN")
        if self.hmin is not None:
            params["HMIN"] = self.hmin
        if not self.hdir:
            params["HDIR"] = False
        if self.debye is not None:
            params["DEBYE"] = self.debye
        if self.freeze:
            flags.append("FREEZE")
        if self.smooth_k is not None:
            params["SMOOTH.K"] = self.smooth_k
        if self.it_smooth is not None:
            params["IT.SMOOTH"] = self.it_smooth
        if self.condense:
            params["CONDENSE"] = self.condense
        if self.reorder:
            flags.append("REORDER")
        if self.three_d_refine:
            flags.append("3D.REFINE")
        if self.dz_level is not None:
            params["DZ.LEVEL"] = self.dz_level

        # Files
        if self.outfile:
            params["OUTF"] = self.outfile
        if self.out_green:
            params["OUT.GREEN"] = self.out_green
        if self.in_green:
            params["IN.GREEN"] = self.in_green
        if self.no_green:
            flags.append("NO.GREEN")
        if self.dopfile:
            params["DOPF"] = self.dopfile
        if not self.ascii:
            params["ASCII"] = False
        if self.stats:
            flags.append("STATS")

        # Interpolation
        if not self.fem:
            params["FEM"] = False

        return self._build_command(params, flags)


class Adapt(Regrid):
    """
    Automatic mesh adaptation at each bias point.

    Same parameters as Regrid, plus:

    Parameters
    ----------
    it_resolve : int
        Maximum re-adaptation attempts per point
    n_resolve : float
        Minimum fraction of elements to refine
    off : bool
        Turn off previous ADAPT

    Example
    -------
    >>> # Adaptive refinement on error
    >>> adapt = Adapt(error=True, r_threshold=0.01, c_step=0.001,
    ...               n_threshold=500, outfile="rmesh_a")
    """

    command_name = "ADAPT"

    def __init__(
        self,
        it_resolve: Optional[int] = None,
        n_resolve: Optional[float] = None,
        off: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.it_resolve = it_resolve
        self.n_resolve = n_resolve
        self.off = off

    def to_padre(self) -> str:
        # Get base command
        base = super().to_padre()

        # Add ADAPT-specific params
        extra_params = []
        if self.it_resolve is not None:
            extra_params.append(f"IT.RESOLVE={self.it_resolve}")
        if self.n_resolve is not None:
            extra_params.append(f"N.RESOLVE={self.n_resolve}")
        if self.off:
            extra_params.append("OFF")

        if extra_params:
            base = base + "  " + "  ".join(extra_params)

        return base
