"""
MOS Capacitor factory function.
"""

from typing import Optional, Tuple
from ..simulation import Simulation
from ..mesh import Mesh
from ..region import Region
from ..electrode import Electrode
from ..doping import Doping
from ..contact import Contact
from ..interface import Interface, Surface
from ..material import Material
from ..models import Models
from ..solver import System, Solve
from ..log import Log


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
    oxide_qftrap: float = 0,
    # Carrier lifetimes
    taun0: float = 1e-6,
    taup0: float = 1e-6,
    # Physical models
    temperature: float = 300,
    conmob: bool = True,
    fldmob: bool = True,
    # Gate contact
    gate_type: str = "n_poly",
    gate_workfunction: Optional[float] = None,
    # Gate configuration
    gate_config: str = "single",
    back_oxide_thickness: float = 0.002,
    back_gate_type: str = "n_poly",
    back_gate_workfunction: Optional[float] = None,
    # Simulation options
    title: Optional[str] = None,
    # Output logging options
    log_cv: bool = False,
    cv_file: str = "cv_data",
    log_cv_lf: bool = False,
    cv_lf_file: str = "cv_lf_data",
    log_bands_eq: bool = False,
    log_bands_bias: bool = False,
    log_qf_eq: bool = False,
    log_qf_bias: bool = False,
    log_profiles_eq: bool = False,
    log_profiles_bias: bool = False,
    # Voltage sweep options
    vg_sweep: Optional[Tuple[float, float, float]] = None,
    ac_frequency: float = 1e6,
    ac_frequency_lf: float = 1.0,
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
        Fixed bulk charge density in oxide in cm^-3 (default: 0)
    oxide_qftrap : float
        Interface trap charge density at oxide-semiconductor interface in cm^-2
        (default: 0). Shifts the flat-band voltage.
    taun0 : float
        Electron minority carrier lifetime in seconds (default: 1e-6 = 1 µs).
        Smaller values improve low-frequency C-V accuracy.
    taup0 : float
        Hole minority carrier lifetime in seconds (default: 1e-6 = 1 µs).
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    conmob : bool
        Enable concentration-dependent mobility (default: True)
    fldmob : bool
        Enable field-dependent mobility (default: True)
    gate_type : str
        Top gate contact type: "n_poly", "p_poly", "aluminum", "tungsten",
        or "metal" for custom workfunction (default: "n_poly").
    gate_workfunction : float, optional
        Custom gate workfunction in eV. Used when gate_type="metal".
        Standard values: aluminum=4.17, tungsten≈4.55.
    gate_config : str
        Gate configuration: "single" (one top gate + ohmic back contact) or
        "double" (gate-oxide-Si-oxide-gate stack). (default: "single")
    back_oxide_thickness : float
        Bottom oxide thickness in microns (double-gate only, default: 0.002).
    back_gate_type : str
        Bottom gate type: "n_poly", "p_poly", "aluminum", "tungsten", or
        "metal" (double-gate only, default: "n_poly").
    back_gate_workfunction : float, optional
        Custom bottom gate workfunction in eV (double-gate only).
    title : str, optional
        Simulation title
    log_cv : bool
        If True, add high-frequency AC C-V logging (default: False)
    cv_file : str
        Filename for high-frequency C-V data (default: "cv_data")
    log_cv_lf : bool
        If True, add a second low-frequency AC C-V solve (default: False)
    cv_lf_file : str
        Filename for low-frequency C-V data (default: "cv_lf_data")
    log_bands_eq : bool
        If True, log band diagrams at equilibrium (default: False)
    log_bands_bias : bool
        If True, log band diagrams at each bias point during sweep (default: False)
    log_qf_eq : bool
        If True, also log quasi-Fermi levels (Efn, Efp) at equilibrium
        alongside the band diagram. Requires log_bands_eq=True or is added
        automatically when True (default: False)
    log_qf_bias : bool
        If True, also log quasi-Fermi levels at each bias point (default: False)
    log_profiles_eq : bool
        If True, log carrier densities, potential, and electric field at
        equilibrium (default: False)
    log_profiles_bias : bool
        If True, log carrier densities, potential, and electric field at
        the last bias point of the sweep (default: False)
    vg_sweep : tuple (v_start, v_end, v_step), optional
        Gate voltage sweep for C-V characteristic with AC analysis.
        Example: (-2.0, 2.0, 0.2) sweeps from -2V to 2V
    ac_frequency : float
        High-frequency AC analysis frequency in Hz (default: 1e6 = 1 MHz)
    ac_frequency_lf : float
        Low-frequency AC analysis frequency in Hz (default: 1.0 = 1 Hz).
        Only used when log_cv_lf=True.

    Returns
    -------
    Simulation
        Configured MOS capacitor simulation

    Example
    -------
    >>> # Basic MOS capacitor - add your own solve commands
    >>> sim = create_mos_capacitor(oxide_thickness=0.005, substrate_doping=1e17)
    >>> sim.add_solve(Solve(initial=True))
    >>> print(sim.generate_deck())
    >>>
    >>> # C-V characteristic with HF and LF curves + profiles
    >>> sim = create_mos_capacitor(
    ...     log_cv=True, log_cv_lf=True,
    ...     log_profiles_eq=True, log_profiles_bias=True,
    ...     vg_sweep=(-2.0, 2.0, 0.1),
    ... )
    >>> result = sim.run()
    >>>
    >>> # Double-gate with custom workfunction
    >>> sim = create_mos_capacitor(
    ...     gate_config="double",
    ...     gate_type="metal", gate_workfunction=4.5,
    ...     back_gate_type="metal", back_gate_workfunction=4.5,
    ... )
    """
    is_double = gate_config.lower() == "double"
    sim = Simulation(title=title or ("Double-Gate MOS Capacitor" if is_double else "MOS Capacitor"))
    sim._device_type = "mos_capacitor"
    sim._device_kwargs = dict(
        oxide_thickness=oxide_thickness, silicon_thickness=silicon_thickness,
        device_width=device_width, ny_oxide=ny_oxide, ny_silicon=ny_silicon,
        nx=nx, substrate_doping=substrate_doping, substrate_type=substrate_type,
        oxide_permittivity=oxide_permittivity, oxide_qf=oxide_qf,
        oxide_qftrap=oxide_qftrap, taun0=taun0, taup0=taup0,
        temperature=temperature, conmob=conmob, fldmob=fldmob,
        gate_type=gate_type, gate_workfunction=gate_workfunction,
        gate_config=gate_config,
        back_oxide_thickness=back_oxide_thickness, back_gate_type=back_gate_type,
        back_gate_workfunction=back_gate_workfunction,
        title=title, log_cv=log_cv, cv_file=cv_file,
        log_cv_lf=log_cv_lf, cv_lf_file=cv_lf_file,
        log_bands_eq=log_bands_eq, log_bands_bias=log_bands_bias,
        log_qf_eq=log_qf_eq, log_qf_bias=log_qf_bias,
        log_profiles_eq=log_profiles_eq, log_profiles_bias=log_profiles_bias,
        vg_sweep=vg_sweep, ac_frequency=ac_frequency,
        ac_frequency_lf=ac_frequency_lf,
    )

    # Transition zone near oxide-silicon interface (0.02 um thick) to avoid
    # obtuse mesh elements — matches Rappture reference deck pattern.
    ny_transition = max(2, ny_oxide // 5)

    if is_double:
        ny_back_oxide = ny_oxide  # same mesh density for back oxide
        total_ny = ny_oxide + ny_transition + ny_silicon + ny_back_oxide
        total_thickness = oxide_thickness + silicon_thickness + back_oxide_thickness
        ny_si_end = ny_oxide + ny_transition + ny_silicon
    else:
        total_ny = ny_oxide + ny_transition + ny_silicon
        total_thickness = oxide_thickness + silicon_thickness

    # Mesh — 5-point y specification mirroring Rappture reference deck:
    #   1) Start at 0
    #   2) Mid-oxide with ratio=1 (uniform oxide mesh)
    #   3) Oxide-silicon interface with ratio=0.8 (compress toward interface)
    #   4) Short transition zone past the interface with ratio=1 (decompress)
    #   5) Silicon bulk with ratio=1.05 (gently expanding mesh)
    ny_mid_oxide = max(1, ny_oxide // 2)
    transition_thickness = oxide_thickness + 0.02  # 20 nm past the interface

    sim.mesh = Mesh(nx=nx, ny=total_ny)
    sim.mesh.add_y_mesh(1, 0, ratio=1)
    sim.mesh.add_y_mesh(ny_mid_oxide, oxide_thickness / 2, ratio=1)
    sim.mesh.add_y_mesh(ny_oxide, oxide_thickness, ratio=0.8)
    sim.mesh.add_y_mesh(ny_oxide + ny_transition, transition_thickness, ratio=1)
    if is_double:
        sim.mesh.add_y_mesh(ny_si_end, oxide_thickness + silicon_thickness, ratio=1.05)
        sim.mesh.add_y_mesh(total_ny, total_thickness, ratio=1.05)
    else:
        sim.mesh.add_y_mesh(total_ny, total_thickness, ratio=1.05)
    sim.mesh.add_x_mesh(1, 0, ratio=1)
    sim.mesh.add_x_mesh(nx, device_width, ratio=1)

    # Regions
    sim.add_region(Region(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=ny_oxide,
                          material="sio2", insulator=True))
    if is_double:
        sim.add_region(Region(2, ix_low=1, ix_high=nx, iy_low=ny_oxide, iy_high=ny_si_end,
                              material="silicon", semiconductor=True))
        sim.add_region(Region(3, ix_low=1, ix_high=nx, iy_low=ny_si_end, iy_high=total_ny,
                              material="sio2", insulator=True))
    else:
        sim.add_region(Region(2, ix_low=1, ix_high=nx, iy_low=ny_oxide, iy_high=total_ny,
                              material="silicon", semiconductor=True))

    # Surface interface between oxide (region 1) and silicon (region 2) —
    # matches Rappture: "surface interface num=1 reg1=1 reg2=2"
    sim.add_surface(Surface(number=1, interface=True, reg1=1, reg2=2,
                            x_min=0, x_max=device_width,
                            y_min=oxide_thickness, y_max=oxide_thickness))

    # Electrodes: electrode 1 = top gate, electrode 2 = back contact / bottom gate
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=1))       # Top gate
    sim.add_electrode(Electrode(2, ix_low=1, ix_high=nx, iy_low=total_ny, iy_high=total_ny))  # Back

    # Doping
    p_type = substrate_type.lower() == "p"
    si_region = 2
    sim.add_doping(Doping(region=si_region, p_type=p_type, n_type=not p_type,
                          concentration=substrate_doping, uniform=True))

    # Contacts — set all to neutral first, then override gate(s)
    sim.add_contact(Contact(all_contacts=True, neutral=True))

    def _gate_contact(number: int, gtype: str, gwf: Optional[float]) -> Contact:
        """Build a Contact for a gate electrode."""
        if gtype == "n_poly":
            return Contact(number=number, n_polysilicon=True)
        elif gtype == "p_poly":
            return Contact(number=number, p_polysilicon=True)
        elif gtype == "aluminum":
            return Contact(number=number, aluminum=True)
        elif gtype == "tungsten":
            return Contact(number=number, tungsten=True)
        else:  # "metal" or anything else — use explicit workfunction if given
            if gwf is not None:
                return Contact(number=number, workfunction=gwf)
            return Contact(number=number, neutral=True)  # fallback: neutral metal

    sim.add_contact(_gate_contact(1, gate_type, gate_workfunction))
    if is_double:
        sim.add_contact(_gate_contact(2, back_gate_type, back_gate_workfunction))

    # Materials — include carrier lifetimes
    sim.add_material(Material(name="silicon", taun0=taun0, taup0=taup0))
    sim.add_material(Material(name="sio2", permittivity=oxide_permittivity, qf=oxide_qf))

    # Interface trap charge at oxide-semiconductor interface (interface 1)
    if oxide_qftrap != 0:
        sim.add_interface(Interface(number=1, qf=oxide_qftrap))

    # Models — srh matches Rappture reference deck
    sim.models = Models(temperature=temperature, srh=True, conmob=conmob, fldmob=fldmob)
    sim.system = System(electrons=True, holes=True, newton=True)

    # C-V logging (high frequency) — issued before the HF sweep
    if log_cv:
        sim.add_log(Log(acfile=cv_file))
    # Note: LF log is inserted later, after the HF sweep, when vg_sweep is built

    # Line cut for profiles: vertical through oxide and silicon at x = mid
    x_mid = device_width / 2

    needs_solve = (vg_sweep is not None or log_bands_eq or log_qf_eq
                   or log_profiles_eq or log_profiles_bias)

    if needs_solve:
        # Always start with equilibrium solve
        sim.add_solve(Solve(initial=True, outfile="eq"))

        # Equilibrium band diagram
        if log_bands_eq or log_qf_eq:
            sim.log_band_diagram(
                outfile_prefix="eq",
                x_start=x_mid, x_end=x_mid,
                y_start=0.0, y_end=total_thickness,
                include_qf=log_qf_eq,
            )

        # Equilibrium profiles: carriers, potential, E-field
        if log_profiles_eq:
            sim.log_carriers("eq", x_start=x_mid, x_end=x_mid,
                             y_start=0.0, y_end=total_thickness)
            sim.log_potential("pot_eq", x_start=x_mid, x_end=x_mid,
                              y_start=0.0, y_end=total_thickness)
            sim.log_efield("ef_eq", x_start=x_mid, x_end=x_mid,
                           y_start=0.0, y_end=total_thickness)

        # Gate voltage sweep — high-frequency C-V
        if vg_sweep is not None:
            v_start, v_end, v_step = vg_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))

            sim.add_solve(Solve(
                project=True,
                v1=v_start,
                vstep=v_step,
                nsteps=nsteps,
                electrode=1,
                ac_analysis=True,
                frequency=ac_frequency,
                outfile="cv",
                save=1 if (log_bands_bias or log_profiles_bias) else None,
            ))

            if log_bands_bias or log_qf_bias:
                sim.log_band_diagram(
                    outfile_prefix="bias",
                    x_start=x_mid, x_end=x_mid,
                    y_start=0.0, y_end=total_thickness,
                    include_qf=log_qf_bias,
                )

            if log_profiles_bias:
                sim.log_carriers("bias", x_start=x_mid, x_end=x_mid,
                                 y_start=0.0, y_end=total_thickness)
                sim.log_potential("pot_bias", x_start=x_mid, x_end=x_mid,
                                  y_start=0.0, y_end=total_thickness)
                sim.log_efield("ef_bias", x_start=x_mid, x_end=x_mid,
                               y_start=0.0, y_end=total_thickness)

            # Low-frequency C-V: issue a new log command to redirect output,
            # then repeat the sweep at the low frequency
            if log_cv_lf:
                sim.add_log(Log(acfile=cv_lf_file))
                sim.add_solve(Solve(
                    project=True,
                    v1=v_start,
                    vstep=v_step,
                    nsteps=nsteps,
                    electrode=1,
                    ac_analysis=True,
                    frequency=ac_frequency_lf,
                    outfile="cv_lf",
                ))

    return sim


# Alias
mos_capacitor = create_mos_capacitor
