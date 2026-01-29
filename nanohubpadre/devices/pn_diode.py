"""
PN Junction Diode factory function.
"""

from typing import Optional, Tuple
from ..simulation import Simulation
from ..mesh import Mesh
from ..region import Region
from ..electrode import Electrode
from ..doping import Doping
from ..contact import Contact
from ..material import Material
from ..models import Models
from ..solver import System, Solve
from ..options import Options
from ..log import Log


def create_pn_diode(
    # Geometry parameters
    length: float = 1.0,
    width: float = 1.0,
    junction_position: float = 0.5,
    # Mesh parameters
    nx: int = 200,
    ny: int = 3,
    # Doping parameters
    p_doping: float = 1e17,
    n_doping: float = 1e17,
    # Physical models
    temperature: float = 300,
    srh: bool = True,
    conmob: bool = True,
    fldmob: bool = True,
    impact: bool = False,
    # Material parameters
    taun0: float = 1e-6,
    taup0: float = 1e-6,
    # Simulation options
    title: Optional[str] = None,
    postscript: bool = False,
    # Output logging options
    log_iv: bool = False,
    iv_file: str = "iv",
    log_bands_eq: bool = False,
    log_bands_bias: bool = False,
    # Voltage sweep options
    forward_sweep: Optional[Tuple[float, float, float]] = None,
    reverse_sweep: Optional[Tuple[float, float, float]] = None,
    sweep_electrode: int = 2,
) -> Simulation:
    """
    Create a PN junction diode simulation.

    Creates a 1D-like PN diode structure with configurable doping profiles,
    mesh refinement, and physical models.

    Parameters
    ----------
    length : float
        Total device length in microns (default: 1.0)
    width : float
        Device width in microns (default: 1.0)
    junction_position : float
        Position of the PN junction as fraction of length (default: 0.5)
    nx : int
        Number of mesh points in x direction (default: 200)
    ny : int
        Number of mesh points in y direction (default: 3)
    p_doping : float
        P-type doping concentration in cm^-3 (default: 1e17)
    n_doping : float
        N-type doping concentration in cm^-3 (default: 1e17)
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    srh : bool
        Enable Shockley-Read-Hall recombination (default: True)
    conmob : bool
        Enable concentration-dependent mobility (default: True)
    fldmob : bool
        Enable field-dependent mobility (default: True)
    impact : bool
        Enable impact ionization (default: False)
    taun0 : float
        Electron lifetime in seconds (default: 1e-6)
    taup0 : float
        Hole lifetime in seconds (default: 1e-6)
    title : str, optional
        Simulation title
    postscript : bool
        Enable PostScript output (default: False)
    log_iv : bool
        If True, add I-V logging (default: False)
    iv_file : str
        Filename for I-V log (default: "iv")
    log_bands_eq : bool
        If True, log band diagrams at equilibrium (default: False)
    log_bands_bias : bool
        If True, log band diagrams during voltage sweeps (default: False)
    forward_sweep : tuple (v_start, v_end, v_step), optional
        If provided, adds a forward bias voltage sweep.
        Example: (0.0, 0.8, 0.05) sweeps from 0V to 0.8V in 0.05V steps
    reverse_sweep : tuple (v_start, v_end, v_step), optional
        If provided, adds a reverse bias voltage sweep.
        Example: (0.0, -5.0, -0.5) sweeps from 0V to -5V in 0.5V steps
    sweep_electrode : int
        Electrode number to apply voltage sweeps (default: 2)

    Returns
    -------
    Simulation
        Configured PN diode simulation ready to run

    Example
    -------
    >>> # Basic diode - no solve commands, add your own
    >>> sim = create_pn_diode(length=2.0, p_doping=1e16, n_doping=1e18)
    >>> sim.add_solve(Solve(initial=True))
    >>> print(sim.generate_deck())
    >>>
    >>> # Complete simulation with sweeps and logging
    >>> sim = create_pn_diode(
    ...     log_iv=True,
    ...     log_bands_eq=True,
    ...     log_bands_bias=True,
    ...     forward_sweep=(0.0, 0.8, 0.05),
    ...     reverse_sweep=(0.0, -5.0, -0.5)
    ... )
    >>> result = sim.run()
    >>> sim.plot_band_diagram()  # Plots all logged band diagrams
    """
    sim = Simulation(title=title or "PN Junction Diode")

    # Options
    if postscript:
        sim.options = Options(postscript=True)

    # Mesh with refinement near junction
    sim.mesh = Mesh(nx=nx, ny=ny, width=width, outfile="mesh")
    junction_x = junction_position * length
    mid_point = nx // 2

    sim.mesh.add_x_mesh(1, 0, ratio=1)
    sim.mesh.add_x_mesh(mid_point, junction_x, ratio=0.8)
    sim.mesh.add_x_mesh(nx, length, ratio=1.05)
    sim.mesh.add_y_mesh(1, 0, ratio=1)
    sim.mesh.add_y_mesh(ny, width, ratio=1)

    # Silicon regions (split at junction for clarity)
    sim.add_region(Region(1, ix_low=1, ix_high=mid_point, iy_low=1, iy_high=ny, silicon=True))
    sim.add_region(Region(1, ix_low=mid_point, ix_high=nx, iy_low=1, iy_high=ny, silicon=True))

    # Electrodes at device ends
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=1, iy_low=1, iy_high=ny))
    sim.add_electrode(Electrode(2, ix_low=nx, ix_high=nx, iy_low=1, iy_high=ny))

    # Doping
    sim.add_doping(Doping(region=1, p_type=True, concentration=p_doping,
                          x_left=0, x_right=junction_x, y_top=0, y_bottom=width, uniform=True))
    sim.add_doping(Doping(region=1, n_type=True, concentration=n_doping,
                          x_left=junction_x, x_right=length, y_top=0, y_bottom=width, uniform=True))

    # Contacts - ohmic contacts for all electrodes
    sim.add_contact(Contact(all_contacts=True, neutral=True))

    # Material with lifetimes
    sim.add_material(Material(name="silicon", taun0=taun0, taup0=taup0,
                              trap_type="0", etrap=0))

    # Physical models
    sim.models = Models(srh=srh, conmob=conmob, fldmob=fldmob, impact=impact,
                        temperature=temperature)
    sim.system = System(electrons=True, holes=True, newton=True)

    # I-V logging
    if log_iv:
        sim.add_log(Log(ivfile=iv_file))

    # Line cut position for band diagrams (horizontal through middle of device)
    y_cut = width / 2.0

    # Only add solve commands if sweeps are specified
    if forward_sweep is not None or reverse_sweep is not None or log_bands_eq:
        # Always start with equilibrium solve
        sim.add_solve(Solve(initial=True, outfile="eq"))

        # Log equilibrium band diagram if requested
        if log_bands_eq:
            sim.log_band_diagram(
                outfile_prefix="eq",
                x_start=0.0, y_start=y_cut,
                x_end=length, y_end=y_cut
            )

        # Forward bias sweep
        if forward_sweep is not None:
            v_start, v_end, v_step = forward_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))
            sim.add_solve(Solve(
                project=True,
                v1=v_start if sweep_electrode == 1 else 0.0,
                v2=v_start if sweep_electrode == 2 else 0.0,
                vstep=v_step,
                nsteps=nsteps,
                electrode=sweep_electrode,
                outfile="fwd"
            ))
            if log_bands_bias:
                sim.log_band_diagram(
                    outfile_prefix="fwd",
                    x_start=0.0, y_start=y_cut,
                    x_end=length, y_end=y_cut,
                    include_qf=True
                )

        # Reverse bias sweep
        if reverse_sweep is not None:
            v_start, v_end, v_step = reverse_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))
            sim.add_solve(Solve(
                project=True,
                v1=v_start if sweep_electrode == 1 else 0.0,
                v2=v_start if sweep_electrode == 2 else 0.0,
                vstep=v_step,
                nsteps=nsteps,
                electrode=sweep_electrode,
                outfile="rev"
            ))
            if log_bands_bias:
                sim.log_band_diagram(
                    outfile_prefix="rev",
                    x_start=0.0, y_start=y_cut,
                    x_end=length, y_end=y_cut,
                    include_qf=True
                )

    return sim


# Alias
pn_diode = create_pn_diode
