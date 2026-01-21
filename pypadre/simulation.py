"""
Main Simulation class for PADRE.

Orchestrates all components to build and run PADRE simulations.
"""

import os
import subprocess
import tempfile
from typing import List, Optional, Union
from pathlib import Path

from .base import PadreCommand, Title, Comment
from .mesh import Mesh, XMesh, YMesh, ZMesh
from .region import Region
from .electrode import Electrode
from .doping import Doping
from .contact import Contact
from .material import Material, Alloy
from .models import Models
from .solver import Solve, Method, System, LinAlg
from .log import Log
from .interface import Interface, Surface
from .regrid import Regrid, Adapt
from .plotting import Plot1D, Plot2D, Contour, Vector
from .options import Options, Load
from .plot3d import Plot3D


class End(PadreCommand):
    """END command to terminate PADRE input deck."""
    command_name = "END"

    def to_padre(self) -> str:
        return "end"


class Simulation:
    """
    Main class to build and run PADRE simulations.

    A Simulation object collects all the components (mesh, regions, doping,
    electrodes, etc.) and generates a complete PADRE input deck.

    Parameters
    ----------
    title : str, optional
        Simulation title (max 60 characters)
    working_dir : str, optional
        Working directory for simulation files

    Example
    -------
    >>> sim = Simulation(title="Simple PN Diode")
    >>> sim.mesh = Mesh(nx=40, ny=17, outfile="mesh.pg")
    >>> sim.mesh.add_x_mesh(1, 0).add_x_mesh(40, 2)
    >>> sim.mesh.add_y_mesh(1, 0).add_y_mesh(17, 2, ratio=1.3)
    >>> sim.add_region(Region.silicon(1, ix_low=1, ix_high=40,
    ...                               iy_low=1, iy_high=17))
    >>> sim.add_doping(Doping.uniform_p(1e16))
    >>> sim.add_doping(Doping.gaussian_n(1e19, junction=0.5))
    >>> sim.add_electrode(Electrode(1, iy_low=1, iy_high=1,
    ...                             ix_low=1, ix_high=40))
    >>> sim.add_electrode(Electrode(2, iy_low=17, iy_high=17,
    ...                             ix_low=1, ix_high=40))
    >>> sim.add_contact(Contact.ohmic(all_contacts=True))
    >>> sim.models = Models.drift_diffusion(temperature=300, srh=True)
    >>> sim.system = System(carriers=2, newton=True)
    >>> sim.method = Method(trap=True)
    >>> sim.add_solve(Solve.equilibrium(outfile="sol0"))
    >>> sim.add_solve(Solve.bias_sweep(electrode=2, start=0, stop=1,
    ...                                step=0.1, outfile="sol_a"))
    >>> print(sim.generate_deck())
    """

    def __init__(self, title: Optional[str] = None,
                 working_dir: Optional[str] = None):
        self.title = title
        self.working_dir = working_dir or os.getcwd()

        # Options
        self._options: Optional[Options] = None

        # Core components
        self._mesh: Optional[Mesh] = None
        self._regions: List[Region] = []
        self._electrodes: List[Electrode] = []
        self._dopings: List[Doping] = []
        self._contacts: List[Contact] = []
        self._materials: List[Material] = []
        self._alloys: List[Alloy] = []
        self._interfaces: List[Interface] = []
        self._surfaces: List[Surface] = []

        # Models and solver configuration
        self._models: Optional[Models] = None
        self._system: Optional[System] = None
        self._method: Optional[Method] = None
        self._linalg: Optional[LinAlg] = None

        # Sequential commands (solve, log, load, plot mixed together)
        self._commands: List[PadreCommand] = []

        # Comments and raw commands
        self._preamble: List[PadreCommand] = []

        # Include end statement
        self._include_end: bool = True

    # Property accessors
    @property
    def mesh(self) -> Optional[Mesh]:
        return self._mesh

    @mesh.setter
    def mesh(self, value: Mesh):
        self._mesh = value

    @property
    def models(self) -> Optional[Models]:
        return self._models

    @models.setter
    def models(self, value: Models):
        self._models = value

    @property
    def system(self) -> Optional[System]:
        return self._system

    @system.setter
    def system(self, value: System):
        self._system = value

    @property
    def method(self) -> Optional[Method]:
        return self._method

    @method.setter
    def method(self, value: Method):
        self._method = value

    @property
    def linalg(self) -> Optional[LinAlg]:
        return self._linalg

    @linalg.setter
    def linalg(self, value: LinAlg):
        self._linalg = value

    @property
    def options(self) -> Optional[Options]:
        return self._options

    @options.setter
    def options(self, value: Options):
        self._options = value

    # Add methods
    def add_region(self, region: Region) -> "Simulation":
        """Add a region definition."""
        self._regions.append(region)
        return self

    def add_electrode(self, electrode: Electrode) -> "Simulation":
        """Add an electrode definition."""
        self._electrodes.append(electrode)
        return self

    def add_doping(self, doping: Doping) -> "Simulation":
        """Add a doping profile."""
        self._dopings.append(doping)
        return self

    def add_contact(self, contact: Contact) -> "Simulation":
        """Add a contact boundary condition."""
        self._contacts.append(contact)
        return self

    def add_material(self, material: Material) -> "Simulation":
        """Add a material definition."""
        self._materials.append(material)
        return self

    def add_alloy(self, alloy: Alloy) -> "Simulation":
        """Add an alloy definition."""
        self._alloys.append(alloy)
        return self

    def add_interface(self, interface: Interface) -> "Simulation":
        """Add an interface definition."""
        self._interfaces.append(interface)
        return self

    def add_surface(self, surface: Surface) -> "Simulation":
        """Add a surface definition."""
        self._surfaces.append(surface)
        return self

    def add_command(self, cmd: PadreCommand) -> "Simulation":
        """Add any command to the sequential command list."""
        self._commands.append(cmd)
        return self

    def add_solve(self, solve: Solve) -> "Simulation":
        """Add a solve step."""
        self._commands.append(solve)
        return self

    def add_log(self, log: Log) -> "Simulation":
        """Add a log command."""
        self._commands.append(log)
        return self

    def add_load(self, load: Load) -> "Simulation":
        """Add a load command."""
        self._commands.append(load)
        return self

    def add_regrid(self, regrid: Union[Regrid, Adapt]) -> "Simulation":
        """Add a regrid/adapt command."""
        self._commands.append(regrid)
        return self

    def add_plot(self, plot: PadreCommand) -> "Simulation":
        """Add a plotting command."""
        self._commands.append(plot)
        return self

    def add_comment(self, text: str) -> "Simulation":
        """Add a comment to the preamble."""
        self._preamble.append(Comment(text))
        return self

    def generate_deck(self) -> str:
        """
        Generate the complete PADRE input deck.

        Returns
        -------
        str
            Complete PADRE input deck as a string
        """
        lines = []

        # Title
        if self.title:
            lines.append(Title(self.title).to_padre())

        # Options
        if self._options:
            lines.append(self._options.to_padre())

        # Preamble comments
        for cmd in self._preamble:
            lines.append(cmd.to_padre())
        if self._preamble:
            lines.append("")

        # Mesh
        if self._mesh:
            lines.append(self._mesh.to_padre())

        # Regions
        for region in self._regions:
            lines.append(region.to_padre())

        # Electrodes
        for electrode in self._electrodes:
            lines.append(electrode.to_padre())

        # Doping
        for doping in self._dopings:
            lines.append(doping.to_padre())

        # Alloys (before materials)
        for alloy in self._alloys:
            lines.append(alloy.to_padre())

        # Materials
        for material in self._materials:
            lines.append(material.to_padre())

        # Interfaces
        for interface in self._interfaces:
            lines.append(interface.to_padre())

        # Surfaces
        for surface in self._surfaces:
            lines.append(surface.to_padre())

        # Contacts
        for contact in self._contacts:
            lines.append(contact.to_padre())

        # Models
        if self._models:
            lines.append(self._models.to_padre())

        # System
        if self._system:
            lines.append(self._system.to_padre())

        # Method
        if self._method:
            lines.append(self._method.to_padre())

        # Linear algebra
        if self._linalg:
            lines.append(self._linalg.to_padre())

        # Sequential commands (solve, log, load, plot)
        for cmd in self._commands:
            lines.append(cmd.to_padre())

        # End
        if self._include_end:
            lines.append("")
            lines.append("end")

        return "\n".join(lines)

    def write_deck(self, filename: str) -> str:
        """
        Write the input deck to a file.

        Parameters
        ----------
        filename : str
            Output filename (relative to working_dir or absolute)

        Returns
        -------
        str
            Full path to the written file
        """
        if not os.path.isabs(filename):
            filename = os.path.join(self.working_dir, filename)

        deck = self.generate_deck()
        with open(filename, 'w') as f:
            f.write(deck)

        return filename

    def run(self, padre_executable: str = "padre",
            input_file: Optional[str] = None,
            output_file: Optional[str] = None,
            capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Run the PADRE simulation.

        Parameters
        ----------
        padre_executable : str
            Path to PADRE executable (default: "padre")
        input_file : str, optional
            Input deck filename. If None, creates a temporary file.
        output_file : str, optional
            Output file for PADRE stdout
        capture_output : bool
            Whether to capture stdout/stderr

        Returns
        -------
        subprocess.CompletedProcess
            Result of the PADRE run
        """
        # Write deck to file
        if input_file is None:
            fd, input_file = tempfile.mkstemp(suffix=".inp", dir=self.working_dir)
            os.close(fd)

        self.write_deck(input_file)

        # Build command
        cmd = [padre_executable]

        # Run PADRE
        result = subprocess.run(
            cmd,
            stdin=open(input_file, 'r'),
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            cwd=self.working_dir,
            text=True
        )

        # Save output if requested
        if output_file and capture_output and result.stdout:
            output_path = os.path.join(self.working_dir, output_file)
            with open(output_path, 'w') as f:
                f.write(result.stdout)

        return result

    def __repr__(self) -> str:
        parts = []
        if self.title:
            parts.append(f"title='{self.title}'")
        parts.append(f"regions={len(self._regions)}")
        parts.append(f"electrodes={len(self._electrodes)}")
        parts.append(f"dopings={len(self._dopings)}")
        parts.append(f"solves={len(self._solves)}")
        return f"<Simulation {', '.join(parts)}>"


# Convenience function for quick simulations
def create_pn_diode(
    length: float = 2.0,
    nx: int = 40,
    ny: int = 17,
    substrate_doping: float = 1e16,
    junction_depth: float = 0.5,
    junction_doping: float = 1e19,
    temperature: float = 300,
) -> Simulation:
    """
    Create a simple 1D PN diode simulation.

    Parameters
    ----------
    length : float
        Device length in microns
    nx, ny : int
        Mesh nodes
    substrate_doping : float
        P-type substrate doping (/cm^3)
    junction_depth : float
        N+ junction depth (microns)
    junction_doping : float
        N+ peak doping (/cm^3)
    temperature : float
        Temperature (K)

    Returns
    -------
    Simulation
        Configured simulation object
    """
    sim = Simulation(title=f"PN Diode - {junction_depth}um junction")

    # Mesh
    sim.mesh = Mesh(nx=nx, ny=ny, outfile="mesh.pg")
    sim.mesh.add_x_mesh(1, 0)
    sim.mesh.add_x_mesh(nx, length)
    sim.mesh.add_y_mesh(1, 0)
    sim.mesh.add_y_mesh(ny, length, ratio=1.3)

    # Region
    sim.add_region(Region.silicon(1, ix_low=1, ix_high=nx,
                                  iy_low=1, iy_high=ny))

    # Doping
    sim.add_doping(Doping.uniform_p(substrate_doping))
    sim.add_doping(Doping.gaussian_n(junction_doping, junction=junction_depth))

    # Electrodes
    sim.add_electrode(Electrode(1, iy_low=1, iy_high=1, ix_low=1, ix_high=nx))
    sim.add_electrode(Electrode(2, iy_low=ny, iy_high=ny, ix_low=1, ix_high=nx))

    # Contact
    sim.add_contact(Contact.ohmic(all_contacts=True))

    # Models
    sim.models = Models.drift_diffusion(temperature=temperature, srh=True)
    sim.system = System(carriers=2, newton=True)
    sim.method = Method(trap=True)

    return sim
