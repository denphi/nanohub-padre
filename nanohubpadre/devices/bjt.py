"""
Bipolar Junction Transistor (BJT) factory function.
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

    Returns
    -------
    Simulation
        Configured BJT simulation

    Example
    -------
    >>> sim = create_bjt(device_type="npn", base_width=0.3)
    >>> sim.add_solve(Solve(initial=True))
    >>> # Common-emitter output characteristic
    >>> sim.add_solve(Solve(v2=0.7))  # Forward bias base
    >>> sim.add_log(Log(ivfile="ic_vce"))
    >>> sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=30, electrode=3))
    >>> print(sim.generate_deck())
    """
    is_npn = device_type.lower() == "npn"
    sim = Simulation(title=title or f"{'NPN' if is_npn else 'PNP'} BJT")

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

    return sim


# Alias
bjt = create_bjt
