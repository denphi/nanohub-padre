"""
MOS Capacitor factory function.
"""

from typing import Optional
from ..simulation import Simulation
from ..mesh import Mesh
from ..region import Region
from ..electrode import Electrode
from ..doping import Doping
from ..contact import Contact
from ..material import Material
from ..models import Models
from ..solver import System


def create_mos_capacitor(
    # Geometry parameters
    oxide_thickness: float = 0.002,
    silicon_thickness: float = 0.03,
    device_width: float = 1.0,
    # Mesh parameters
    ny_oxide: int = 10,
    ny_silicon: int = 20,
    nx: int = 3,
    # Doping parameters
    substrate_doping: float = 1e18,
    substrate_type: str = "p",
    # Material parameters
    oxide_permittivity: float = 3.9,
    oxide_qf: float = 0,
    # Physical models
    temperature: float = 300,
    conmob: bool = True,
    fldmob: bool = True,
    # Gate contact
    gate_type: str = "n_poly",
    # Simulation options
    title: Optional[str] = None,
) -> Simulation:
    """
    Create a MOS capacitor simulation.

    Creates an oxide-semiconductor structure for C-V analysis.

    Parameters
    ----------
    oxide_thickness : float
        Gate oxide thickness in microns (default: 0.002 = 2nm)
    silicon_thickness : float
        Silicon substrate thickness in microns (default: 0.03)
    device_width : float
        Device width in microns (default: 1.0)
    ny_oxide : int
        Mesh points in oxide layer (default: 10)
    ny_silicon : int
        Mesh points in silicon (default: 20)
    nx : int
        Mesh points in x direction (default: 3)
    substrate_doping : float
        Substrate doping concentration in cm^-3 (default: 1e18)
    substrate_type : str
        Substrate doping type: "p" or "n" (default: "p")
    oxide_permittivity : float
        Relative permittivity of oxide (default: 3.9)
    oxide_qf : float
        Fixed oxide charge (default: 0)
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    conmob : bool
        Enable concentration-dependent mobility (default: True)
    fldmob : bool
        Enable field-dependent mobility (default: True)
    gate_type : str
        Gate contact type: "n_poly", "p_poly", or "metal" (default: "n_poly")
    title : str, optional
        Simulation title

    Returns
    -------
    Simulation
        Configured MOS capacitor simulation

    Example
    -------
    >>> sim = create_mos_capacitor(oxide_thickness=0.005, substrate_doping=1e17)
    >>> sim.add_solve(Solve(initial=True))
    >>> sim.add_solve(Solve(v1=0, vstep=-0.2, nsteps=10, electrode=1,
    ...                     ac_analysis=True, frequency=1e6))
    >>> print(sim.generate_deck())
    """
    sim = Simulation(title=title or "MOS Capacitor")

    total_ny = ny_oxide + ny_silicon
    total_thickness = oxide_thickness + silicon_thickness

    # Mesh
    sim.mesh = Mesh(nx=nx, ny=total_ny)
    sim.mesh.add_y_mesh(1, 0, ratio=1)
    sim.mesh.add_y_mesh(ny_oxide, oxide_thickness, ratio=0.8)
    sim.mesh.add_y_mesh(total_ny, total_thickness, ratio=1.25)
    sim.mesh.add_x_mesh(1, 0, ratio=1)
    sim.mesh.add_x_mesh(nx, device_width, ratio=1)

    # Regions
    sim.add_region(Region(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=ny_oxide,
                          material="sio2", insulator=True))
    sim.add_region(Region(2, ix_low=1, ix_high=nx, iy_low=ny_oxide, iy_high=total_ny,
                          material="silicon", semiconductor=True))

    # Electrodes
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=1))  # Gate
    sim.add_electrode(Electrode(2, ix_low=1, ix_high=nx, iy_low=total_ny, iy_high=total_ny))  # Back

    # Doping
    p_type = substrate_type.lower() == "p"
    sim.add_doping(Doping(region=2, p_type=p_type, n_type=not p_type,
                          concentration=substrate_doping, uniform=True))

    # Contacts
    sim.add_contact(Contact(all_contacts=True, neutral=True))
    if gate_type == "n_poly":
        sim.add_contact(Contact(number=1, n_polysilicon=True))
    elif gate_type == "p_poly":
        sim.add_contact(Contact(number=1, p_polysilicon=True))
    # else: use neutral (metal-like)

    # Materials
    sim.add_material(Material(name="silicon"))
    sim.add_material(Material(name="sio2", permittivity=oxide_permittivity, qf=oxide_qf))

    # Models
    sim.models = Models(temperature=temperature, conmob=conmob, fldmob=fldmob)
    sim.system = System(electrons=True, holes=True, newton=True)

    return sim


# Alias
mos_capacitor = create_mos_capacitor
