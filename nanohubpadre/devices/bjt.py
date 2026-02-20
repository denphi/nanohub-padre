"""
Bipolar Junction Transistor (BJT) factory function.
"""

from typing import Optional, Tuple, List
from ..simulation import Simulation
from ..mesh import Mesh
from ..region import Region
from ..electrode import Electrode
from ..doping import Doping
from ..contact import Contact
from ..models import Models
from ..solver import System, Solve
from ..log import Log
from ..plot3d import Plot3D


def create_bjt(
    # Geometry parameters
    emitter_width: float = 1.0,
    base_width: float = 0.5,
    collector_width: float = 2.0,
    device_depth: float = 1.0,
    # Mesh parameters
    nx: int = 100,
    ny: int = 30,
    # Doping parameters
    emitter_doping: float = 1e20,
    base_doping: float = 1e17,
    collector_doping: float = 1e16,
    device_type: str = "npn",
    # Physical models
    temperature: float = 300,
    srh: bool = True,
    auger: bool = True,
    bgn: bool = True,
    conmob: bool = True,
    fldmob: bool = True,
    # Simulation options
    title: Optional[str] = None,
    # Output logging options
    log_iv: bool = False,
    iv_file: str = "ic_vce.log",
    log_bands_eq: bool = False,
    # Voltage sweep options - Common-emitter output characteristics
    vbe: float = 0.0,
    vce_sweep: Optional[Tuple[float, float, float]] = None,
    # Gummel plot sweep
    gummel_sweep: Optional[Tuple[float, float, float]] = None,
    gummel_vce: float = 2.0,
    # 2D contour map options
    contour_maps: bool = False,
    contour_vbe: float = 0.7,
    contour_vce: float = 2.0,
    contour_quantities: Optional[List[str]] = None,
) -> Simulation:
    """
    Create a Bipolar Junction Transistor (BJT) simulation.

    Creates a 1D-like NPN or PNP transistor structure.

    Parameters
    ----------
    emitter_width : float
        Emitter region width in microns (default: 1.0)
    base_width : float
        Base region width in microns (default: 0.5)
    collector_width : float
        Collector region width in microns (default: 2.0)
    device_depth : float
        Device depth in microns (default: 1.0)
    nx : int
        Mesh points in x direction (default: 100)
    ny : int
        Mesh points in y direction (default: 30)
    emitter_doping : float
        Emitter doping concentration in cm^-3 (default: 1e20)
    base_doping : float
        Base doping concentration in cm^-3 (default: 1e17)
    collector_doping : float
        Collector doping concentration in cm^-3 (default: 1e16)
    device_type : str
        "npn" or "pnp" (default: "npn")
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    srh : bool
        Enable Shockley-Read-Hall recombination (default: True)
    auger : bool
        Enable Auger recombination (default: True)
    bgn : bool
        Enable band-gap narrowing (default: True)
    conmob : bool
        Enable concentration-dependent mobility (default: True)
    fldmob : bool
        Enable field-dependent mobility (default: True)
    title : str, optional
        Simulation title
    log_iv : bool
        If True, add I-V logging (default: False)
    iv_file : str
        Filename for I-V log (default: "ic_vce.log")
    log_bands_eq : bool
        If True, log band diagrams at equilibrium (default: False)
    vbe : float
        Base-emitter voltage for output characteristics (default: 0.0)
    vce_sweep : tuple (v_start, v_end, v_step), optional
        Collector-emitter voltage sweep for output characteristics (Ic vs Vce).
        Example: (0.0, 3.0, 0.1) sweeps Vce from 0V to 3V
    gummel_sweep : tuple (v_start, v_end, v_step), optional
        Base-emitter voltage sweep for Gummel plot (Ic, Ib vs Vbe).
        Example: (0.0, 0.8, 0.05) sweeps Vbe from 0V to 0.8V
    gummel_vce : float
        Fixed Vce for Gummel plot (default: 2.0)
    contour_maps : bool
        If True, add Plot3D scatter file dumps at equilibrium and under bias
        for 2D contour visualization (default: False)
    contour_vbe : float
        Base-emitter voltage for contour map bias point (default: 0.7)
    contour_vce : float
        Collector-emitter voltage for contour map bias point (default: 2.0)
    contour_quantities : list of str, optional
        Quantities to dump. Default: ["potential", "doping", "electrons",
        "holes", "e_field", "qfn", "qfp"]

    Returns
    -------
    Simulation
        Configured BJT simulation

    Example
    -------
    >>> # Basic BJT - add your own solve commands
    >>> sim = create_bjt(device_type="npn", base_width=0.3)
    >>> sim.add_solve(Solve(initial=True))
    >>> print(sim.generate_deck())
    >>>
    >>> # Common-emitter output characteristic
    >>> sim = create_bjt(
    ...     log_iv=True,
    ...     vbe=0.7,
    ...     vce_sweep=(0.0, 3.0, 0.1)
    ... )
    >>> result = sim.run()
    >>>
    >>> # Gummel plot
    >>> sim = create_bjt(
    ...     log_iv=True,
    ...     iv_file="gummel",
    ...     gummel_sweep=(0.0, 0.8, 0.05),
    ...     gummel_vce=2.0
    ... )
    >>>
    >>> # 2D contour maps
    >>> sim = create_bjt(contour_maps=True, contour_vbe=0.7, contour_vce=2.0)
    >>> result = sim.run()
    >>> sim.plot_contour("pot_eq", title="Potential — Equilibrium", cbar_title="V")
    """
    is_npn = device_type.lower() == "npn"
    sim = Simulation(title=title or f"{'NPN' if is_npn else 'PNP'} BJT")
    sim._device_type = "bjt"
    sim._device_kwargs = dict(
        emitter_width=emitter_width, base_width=base_width,
        collector_width=collector_width, device_depth=device_depth,
        nx=nx, ny=ny, emitter_doping=emitter_doping, base_doping=base_doping,
        collector_doping=collector_doping, device_type=device_type,
        temperature=temperature, srh=srh, auger=auger, bgn=bgn,
        conmob=conmob, fldmob=fldmob, title=title, log_iv=log_iv,
        iv_file=iv_file, log_bands_eq=log_bands_eq, vbe=vbe,
        vce_sweep=vce_sweep, gummel_sweep=gummel_sweep, gummel_vce=gummel_vce,
        contour_maps=contour_maps, contour_vbe=contour_vbe,
        contour_vce=contour_vce, contour_quantities=contour_quantities,
    )

    total_width = emitter_width + base_width + collector_width

    # Mesh point distribution
    nx_e = int(nx * emitter_width / total_width)
    nx_b = int(nx * base_width / total_width)
    nx_c = nx - nx_e - nx_b

    sim.mesh = Mesh(nx=nx, ny=ny)
    sim.mesh.add_x_mesh(1, 0, ratio=1.1)
    sim.mesh.add_x_mesh(nx_e, emitter_width, ratio=0.9)
    sim.mesh.add_x_mesh(nx_e + nx_b, emitter_width + base_width, ratio=1.1)
    sim.mesh.add_x_mesh(nx, total_width, ratio=0.9)
    sim.mesh.add_y_mesh(1, 0, ratio=1)
    sim.mesh.add_y_mesh(ny, device_depth, ratio=1)

    # Regions
    sim.add_region(Region(1, ix_low=1, ix_high=nx_e, iy_low=1, iy_high=ny, silicon=True))  # Emitter
    sim.add_region(Region(2, ix_low=nx_e, ix_high=nx_e + nx_b, iy_low=1, iy_high=ny, silicon=True))  # Base
    sim.add_region(Region(3, ix_low=nx_e + nx_b, ix_high=nx, iy_low=1, iy_high=ny, silicon=True))  # Collector

    # Electrodes
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=1, iy_low=1, iy_high=ny))  # Emitter contact
    sim.add_electrode(Electrode(2, ix_low=nx_e, ix_high=nx_e + nx_b, iy_low=ny, iy_high=ny))  # Base contact
    sim.add_electrode(Electrode(3, ix_low=nx, ix_high=nx, iy_low=1, iy_high=ny))  # Collector contact

    # Doping (NPN: n-emitter, p-base, n-collector)
    if is_npn:
        sim.add_doping(Doping(region=1, n_type=True, uniform=True, concentration=emitter_doping))
        sim.add_doping(Doping(region=2, p_type=True, uniform=True, concentration=base_doping))
        sim.add_doping(Doping(region=3, n_type=True, uniform=True, concentration=collector_doping))
    else:
        sim.add_doping(Doping(region=1, p_type=True, uniform=True, concentration=emitter_doping))
        sim.add_doping(Doping(region=2, n_type=True, uniform=True, concentration=base_doping))
        sim.add_doping(Doping(region=3, p_type=True, uniform=True, concentration=collector_doping))

    # Contacts
    sim.add_contact(Contact(all_contacts=True, neutral=True))

    # Models
    sim.models = Models(temperature=temperature, srh=srh, auger=auger, bgn=bgn,
                        conmob=conmob, fldmob=fldmob)
    sim.system = System(electrons=True, holes=True, newton=True)

    # I-V logging
    if log_iv:
        sim.add_log(Log(ivfile=iv_file))

    # Only add solve commands if sweeps or contour maps are specified
    if vce_sweep is not None or gummel_sweep is not None or log_bands_eq or contour_maps:
        # Always start with equilibrium solve
        sim.add_solve(Solve(initial=True, outfile="eq_sol"))

        # Log band diagram at equilibrium (horizontal cut along device from emitter to collector)
        if log_bands_eq:
            y_mid = device_depth / 2
            sim.log_band_diagram(
                outfile_prefix="eq",
                x_start=0.0, x_end=total_width,
                y_start=y_mid, y_end=y_mid
            )

        # Common-emitter output characteristics (Ic vs Vce at fixed Vbe)
        if vce_sweep is not None:
            v_start, v_end, v_step = vce_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))

            # Set base-emitter voltage first (only 1 prior solution: use previous=True)
            if abs(vbe) > 1e-10:
                sim.add_solve(Solve(previous=True, v2=vbe, electrode=2, outfile="vbe_set_sol"))

            # Sweep collector-emitter voltage (2 prior solutions now exist: use project=True)
            sim.add_solve(Solve(
                project=True,
                v3=v_start,
                vstep=v_step,
                nsteps=nsteps,
                electrode=3,
                outfile="ic_vce_sol"
            ))

        # Gummel plot (Ic, Ib vs Vbe at fixed Vce)
        if gummel_sweep is not None:
            v_start, v_end, v_step = gummel_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))

            # Set collector-emitter voltage first (only 1 prior solution: use previous=True)
            if abs(gummel_vce) > 1e-10:
                sim.add_solve(Solve(previous=True, v3=gummel_vce, electrode=3, outfile="vce_set_sol"))

            # Sweep base-emitter voltage (2 prior solutions now exist: use project=True)
            sim.add_solve(Solve(
                project=True,
                v2=v_start,
                vstep=v_step,
                nsteps=nsteps,
                electrode=2,
                outfile="gummel_sol"
            ))

        # 2D contour maps (Plot3D scatter files)
        if contour_maps:
            quantities = contour_quantities or [
                "potential", "doping", "electrons", "holes",
                "e_field", "qfn", "qfp",
            ]

            # Map quantity names to Plot3D kwargs and output file suffixes
            _qty_map = {
                "potential":  ("potential",  "pot"),
                "doping":     ("doping",     "dop"),
                "electrons":  ("electrons",  "el"),
                "holes":      ("holes",      "hh"),
                "e_field":    ("e_field",    "ef"),
                "qfn":        ("qfn",        "qfn"),
                "qfp":        ("qfp",        "qfp"),
                "band_val":   ("band_val",   "bv"),
                "band_cond":  ("band_cond",  "bc"),
                "net_charge": ("net_charge", "nch"),
                "recomb":     ("recomb",     "rec"),
            }

            # Equilibrium dumps
            for qty in quantities:
                if qty not in _qty_map:
                    continue
                kwarg, suffix = _qty_map[qty]
                extras = {"absolute": True} if qty == "doping" else {}
                sim.add_command(Plot3D(**{kwarg: True}, outfile=f"{suffix}_eq", **extras))

            # Bias solve for contour maps (only if no other sweep already applied bias)
            if vce_sweep is None and gummel_sweep is None:
                # Apply Vbe
                if abs(contour_vbe) > 1e-10:
                    sim.add_solve(Solve(
                        previous=True, v2=contour_vbe,
                        electrode=2, outfile="contour_vbe_set"
                    ))
                # Apply Vce in steps
                if abs(contour_vce) > 1e-10:
                    n_steps = max(1, int(abs(contour_vce) / 0.5))
                    v_step = contour_vce / n_steps
                    sim.add_solve(Solve(
                        project=True, v3=0.0,
                        vstep=v_step, nsteps=n_steps,
                        electrode=3, outfile="contour_bias"
                    ))

            # Bias dumps
            for qty in quantities:
                if qty == "doping":
                    continue  # doping doesn't change with bias
                if qty not in _qty_map:
                    continue
                kwarg, suffix = _qty_map[qty]
                sim.add_command(Plot3D(**{kwarg: True}, outfile=f"{suffix}_bias"))

    return sim


# Alias
bjt = create_bjt
