"""
MOSFET factory function.
"""

from typing import Optional
from ..simulation import Simulation
from ..mesh import Mesh
from ..region import Region
from ..electrode import Electrode
from ..doping import Doping
from ..contact import Contact
from ..models import Models
from ..solver import System


def create_mosfet(
    # Geometry parameters
    channel_length: float = 0.025,
    gate_oxide_thickness: float = 0.012,
    junction_depth: float = 0.018,
    device_width: float = 0.125,
    device_depth: float = 0.068,
    # Mesh parameters
    nx: int = 51,
    ny: int = 51,
    # Doping parameters
    channel_doping: float = 1e19,
    substrate_doping: float = 5e16,
    source_drain_doping: float = 1e20,
    device_type: str = "nmos",
    # Physical models
    temperature: float = 300,
    bgn: bool = True,
    carriers: int = 1,
    # Simulation options
    title: Optional[str] = None,
) -> Simulation:
    """
    Create a MOSFET simulation.

    Creates an NMOS or PMOS transistor structure with source, drain,
    gate, and substrate contacts.

    Parameters
    ----------
    channel_length : float
        Gate/channel length in microns (default: 0.025)
    gate_oxide_thickness : float
        Gate oxide thickness in microns (default: 0.012)
    junction_depth : float
        Source/drain junction depth in microns (default: 0.018)
    device_width : float
        Total device width in microns (default: 0.125)
    device_depth : float
        Substrate depth in microns (default: 0.068)
    nx : int
        Mesh points in x direction (default: 51)
    ny : int
        Mesh points in y direction (default: 51)
    channel_doping : float
        Channel doping concentration in cm^-3 (default: 1e19)
    substrate_doping : float
        Substrate doping concentration in cm^-3 (default: 5e16)
    source_drain_doping : float
        Source/drain doping concentration in cm^-3 (default: 1e20)
    device_type : str
        "nmos" or "pmos" (default: "nmos")
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    bgn : bool
        Enable band-gap narrowing (default: True)
    carriers : int
        Number of carriers to solve (1 or 2, default: 1)
    title : str, optional
        Simulation title

    Returns
    -------
    Simulation
        Configured MOSFET simulation

    Example
    -------
    >>> sim = create_mosfet(channel_length=0.05, device_type="nmos")
    >>> sim.add_solve(Solve(initial=True))
    >>> # Transfer characteristic
    >>> sim.add_log(Log(ivfile="idvg"))
    >>> sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))
    >>> print(sim.generate_deck())
    """
    is_nmos = device_type.lower() == "nmos"
    sim = Simulation(title=title or f"{'NMOS' if is_nmos else 'PMOS'} MOSFET")

    # Calculate dimensions
    sd_width = (device_width - channel_length) / 2
    total_height = device_depth + junction_depth + gate_oxide_thickness

    # Mesh points distribution
    nx_sd = int(nx * sd_width / device_width)
    nx_ch = nx - 2 * nx_sd
    ny_sub = int(ny * device_depth / total_height)
    ny_junc = int(ny * junction_depth / total_height)
    ny_ox = ny - ny_sub - ny_junc

    sim.mesh = Mesh(nx=nx, ny=ny)

    # X mesh (source - channel - drain)
    sim.mesh.add_x_mesh(1, 0)
    sim.mesh.add_x_mesh(nx_sd, sd_width, ratio=0.8)
    sim.mesh.add_x_mesh(nx_sd + nx_ch // 2, sd_width + channel_length / 2, ratio=1.25)
    sim.mesh.add_x_mesh(nx_sd + nx_ch, sd_width + channel_length, ratio=0.8)
    sim.mesh.add_x_mesh(nx, device_width, ratio=1.25)

    # Y mesh (substrate - junction - oxide)
    sim.mesh.add_y_mesh(1, 0)
    sim.mesh.add_y_mesh(ny_sub, device_depth, ratio=0.8)
    sim.mesh.add_y_mesh(ny_sub + ny_junc, device_depth + junction_depth, ratio=1.25)
    sim.mesh.add_y_mesh(ny, total_height, ratio=1.25)

    # Regions
    # Substrate
    sim.add_region(Region(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=ny_sub, silicon=True))
    # Source region
    sim.add_region(Region(2, ix_low=1, ix_high=nx_sd, iy_low=ny_sub, iy_high=ny_sub + ny_junc, silicon=True))
    # Drain region
    sim.add_region(Region(3, ix_low=nx_sd + nx_ch, ix_high=nx, iy_low=ny_sub, iy_high=ny_sub + ny_junc, silicon=True))
    # Channel region
    sim.add_region(Region(4, ix_low=nx_sd, ix_high=nx_sd + nx_ch, iy_low=ny_sub, iy_high=ny_sub + ny_junc, silicon=True))
    # Gate oxide
    sim.add_region(Region(5, ix_low=nx_sd, ix_high=nx_sd + nx_ch, iy_low=ny_sub + ny_junc, iy_high=ny, oxide=True))
    # Filler oxides over source/drain
    sim.add_region(Region(6, ix_low=1, ix_high=nx_sd, iy_low=ny_sub + ny_junc, iy_high=ny, oxide=True))
    sim.add_region(Region(7, ix_low=nx_sd + nx_ch, ix_high=nx, iy_low=ny_sub + ny_junc, iy_high=ny, oxide=True))

    # Electrodes
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=nx_sd, iy_low=ny_sub + ny_junc, iy_high=ny_sub + ny_junc))  # Source
    sim.add_electrode(Electrode(2, ix_low=nx_sd + nx_ch, ix_high=nx, iy_low=ny_sub + ny_junc, iy_high=ny_sub + ny_junc))  # Drain
    sim.add_electrode(Electrode(3, ix_low=nx_sd, ix_high=nx_sd + nx_ch, iy_low=ny, iy_high=ny))  # Gate
    sim.add_electrode(Electrode(4, ix_low=1, ix_high=nx, iy_low=1, iy_high=1))  # Substrate

    # Doping (NMOS: n+ S/D, p-channel, p-sub; PMOS: opposite)
    if is_nmos:
        sim.add_doping(Doping(region=[2, 3], uniform=True, concentration=source_drain_doping, n_type=True))
        sim.add_doping(Doping(region=4, uniform=True, concentration=channel_doping, p_type=True))
        sim.add_doping(Doping(region=1, uniform=True, concentration=substrate_doping, p_type=True))
    else:
        sim.add_doping(Doping(region=[2, 3], uniform=True, concentration=source_drain_doping, p_type=True))
        sim.add_doping(Doping(region=4, uniform=True, concentration=channel_doping, n_type=True))
        sim.add_doping(Doping(region=1, uniform=True, concentration=substrate_doping, n_type=True))

    # Contacts
    sim.add_contact(Contact(number=3, n_polysilicon=is_nmos, p_polysilicon=not is_nmos))

    # Models
    sim.models = Models(temperature=temperature, bgn=bgn)
    if carriers == 1:
        sim.system = System(newton=True, carriers=1, electrons=is_nmos, holes=not is_nmos)
    else:
        sim.system = System(newton=True, carriers=2, electrons=True, holes=True)

    return sim


# Alias
mosfet = create_mosfet
