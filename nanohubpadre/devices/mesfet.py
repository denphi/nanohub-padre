"""
MESFET factory function.
"""

from typing import Optional, Tuple
from ..simulation import Simulation
from ..mesh import Mesh
from ..region import Region
from ..electrode import Electrode
from ..doping import Doping
from ..contact import Contact
from ..models import Models
from ..solver import System, Solve
from ..log import Log


def create_mesfet(
    # Geometry parameters
    channel_length: float = 0.2,
    gate_length: float = 0.2,
    device_width: float = 0.6,
    channel_depth: float = 0.2,
    substrate_depth: float = 0.8,
    # Mesh parameters
    nx: int = 61,
    ny: int = 51,
    # Doping parameters
    channel_doping: float = 1e17,
    substrate_doping: float = 1e17,
    contact_doping: float = 1e20,
    device_type: str = "n",
    # Gate contact
    gate_workfunction: float = 4.87,
    # Physical models
    temperature: float = 300,
    bgn: bool = True,
    conmob: bool = True,
    fldmob: bool = True,
    # Simulation options
    title: Optional[str] = None,
    # Output logging options
    log_iv: bool = False,
    iv_file: str = "idvd",
    log_bands_eq: bool = False,
    # Voltage sweep options
    vgs: float = 0.0,
    vds_sweep: Optional[Tuple[float, float, float]] = None,
) -> Simulation:
    """
    Create a MESFET (Metal-Semiconductor FET) simulation.

    Creates a Schottky-gate FET structure with source, drain, and gate contacts.

    Parameters
    ----------
    channel_length : float
        Source-to-gate and gate-to-drain spacing in microns (default: 0.2)
    gate_length : float
        Gate length in microns (default: 0.2)
    device_width : float
        Total device width in microns (default: 0.6)
    channel_depth : float
        Channel depth in microns (default: 0.2)
    substrate_depth : float
        Substrate depth below channel in microns (default: 0.8)
    nx : int
        Mesh points in x direction (default: 61)
    ny : int
        Mesh points in y direction (default: 51)
    channel_doping : float
        Channel doping concentration in cm^-3 (default: 1e17)
    substrate_doping : float
        Substrate doping concentration in cm^-3 (default: 1e17)
    contact_doping : float
        Source/drain contact doping in cm^-3 (default: 1e20)
    device_type : str
        "n" for n-channel or "p" for p-channel (default: "n")
    gate_workfunction : float
        Gate metal workfunction in V (default: 4.87)
    temperature : float
        Simulation temperature in Kelvin (default: 300)
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
        Filename for I-V log (default: "idvd")
    log_bands_eq : bool
        If True, log band diagrams at equilibrium (default: False)
    vgs : float
        Gate-source voltage for output characteristic (default: 0.0)
    vds_sweep : tuple (v_start, v_end, v_step), optional
        Drain-source voltage sweep for output characteristic (Id vs Vds).
        Example: (0.0, 2.0, 0.1) sweeps Vds from 0V to 2V

    Returns
    -------
    Simulation
        Configured MESFET simulation

    Example
    -------
    >>> # Basic MESFET - add your own solve commands
    >>> sim = create_mesfet(channel_length=0.1, gate_workfunction=4.9)
    >>> sim.add_solve(Solve(initial=True))
    >>> print(sim.generate_deck())
    >>>
    >>> # Output characteristic (Id vs Vds)
    >>> sim = create_mesfet(
    ...     log_iv=True,
    ...     vgs=-0.4,
    ...     vds_sweep=(0.0, 2.0, 0.1)
    ... )
    >>> result = sim.run()
    """
    is_n_type = device_type.lower() == "n"
    sim = Simulation(title=title or f"{'N' if is_n_type else 'P'}-channel MESFET")

    total_depth = substrate_depth + channel_depth
    source_width = channel_length
    drain_width = channel_length
    gate_start = source_width + (device_width - source_width - drain_width - gate_length) / 2
    gate_end = gate_start + gate_length

    # Mesh
    sim.mesh = Mesh(nx=nx, ny=ny)
    sim.mesh.add_x_mesh(1, 0, ratio=1.1)
    sim.mesh.add_x_mesh(int(nx * source_width / device_width), source_width, ratio=0.8)
    sim.mesh.add_x_mesh(int(nx * gate_start / device_width), gate_start, ratio=0.8)
    sim.mesh.add_x_mesh(int(nx * gate_end / device_width), gate_end, ratio=0.8)
    sim.mesh.add_x_mesh(int(nx * (device_width - drain_width) / device_width), device_width - drain_width, ratio=0.8)
    sim.mesh.add_x_mesh(nx, device_width, ratio=1.1)
    sim.mesh.add_y_mesh(1, 0.0, ratio=1.1)
    sim.mesh.add_y_mesh(int(ny * substrate_depth / total_depth), substrate_depth, ratio=0.9)
    sim.mesh.add_y_mesh(ny, total_depth, ratio=0.8)

    ny_sub = int(ny * substrate_depth / total_depth)
    nx_src = int(nx * source_width / device_width)
    nx_gate_start = int(nx * gate_start / device_width)
    nx_gate_end = int(nx * gate_end / device_width)
    nx_drain_start = int(nx * (device_width - drain_width) / device_width)

    # Regions
    sim.add_region(Region(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=ny_sub, silicon=True))
    sim.add_region(Region(2, ix_low=1, ix_high=nx_src, iy_low=ny_sub, iy_high=ny, silicon=True))
    sim.add_region(Region(3, ix_low=nx_src, ix_high=nx_drain_start, iy_low=ny_sub, iy_high=ny, silicon=True))
    sim.add_region(Region(4, ix_low=nx_drain_start, ix_high=nx, iy_low=ny_sub, iy_high=ny, silicon=True))

    # Electrodes
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=nx_src, iy_low=ny, iy_high=ny))  # Source
    sim.add_electrode(Electrode(2, ix_low=nx_drain_start, ix_high=nx, iy_low=ny, iy_high=ny))  # Drain
    sim.add_electrode(Electrode(3, ix_low=nx_gate_start, ix_high=nx_gate_end, iy_low=ny, iy_high=ny))  # Gate

    # Doping
    sub_type = not is_n_type  # substrate opposite to channel
    sim.add_doping(Doping(region=1, p_type=sub_type, n_type=not sub_type,
                          uniform=True, concentration=substrate_doping))
    sim.add_doping(Doping(region=2, n_type=is_n_type, p_type=not is_n_type,
                          uniform=True, concentration=contact_doping))
    sim.add_doping(Doping(region=3, n_type=is_n_type, p_type=not is_n_type,
                          uniform=True, concentration=channel_doping))
    sim.add_doping(Doping(region=4, n_type=is_n_type, p_type=not is_n_type,
                          uniform=True, concentration=contact_doping))

    # Contacts
    sim.add_contact(Contact(all_contacts=True, neutral=True))
    sim.add_contact(Contact(number=3, workfunction=gate_workfunction))

    # Models
    sim.models = Models(temperature=temperature, bgn=bgn, conmob=conmob, fldmob=fldmob)
    sim.system = System(newton=True, carriers=1, electrons=is_n_type, holes=not is_n_type)

    # I-V logging
    if log_iv:
        sim.add_log(Log(ivfile=iv_file))

    # Only add solve commands if sweeps are specified
    if vds_sweep is not None or log_bands_eq:
        # Always start with equilibrium solve
        sim.add_solve(Solve(initial=True, outfile="eq"))

        # Log band diagram at equilibrium (horizontal cut through channel)
        if log_bands_eq:
            # Cut at surface (top of channel) where current flows
            y_channel = total_depth - channel_depth / 2
            sim.log_band_diagram(
                outfile_prefix="eq",
                x_start=0.0, x_end=device_width,
                y_start=y_channel, y_end=y_channel
            )

        # Output characteristic (Id vs Vds at fixed Vgs)
        if vds_sweep is not None:
            v_start, v_end, v_step = vds_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))

            # Set gate-source voltage first
            if abs(vgs) > 1e-10:
                sim.add_solve(Solve(project=True, v3=vgs, electrode=3, outfile="vgs_set"))

            # Sweep drain-source voltage (electrode 2 = drain)
            sim.add_solve(Solve(
                project=True,
                v2=v_start,
                vstep=v_step,
                nsteps=nsteps,
                electrode=2,
                outfile="idvd"
            ))

    return sim


# Alias
mesfet = create_mesfet
