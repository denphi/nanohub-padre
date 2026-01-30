"""
Main Simulation class for PADRE.

Orchestrates all components to build and run PADRE simulations.
"""

import os
import subprocess
import tempfile
from typing import List, Optional, Union, Any
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
from .parser import parse_padre_output, SimulationResult, parse_iv_file, IVData, parse_ac_file, ACData
from .solution import parse_solution_file, load_solution_series, SolutionData
from .outputs import OutputManager, OutputType, get_plot1d_variable, PlotData


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
        self.working_dir = (
            working_dir
            or os.environ.get("RESULTSDIR")
            or os.getcwd()
        )
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

        # Output manager for tracking and accessing results
        self._outputs: Optional[OutputManager] = None

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

    @property
    def outputs(self) -> OutputManager:
        """
        Access simulation outputs.

        The OutputManager provides easy access to all simulation outputs
        (plots, solutions, I-V data) without needing to specify file paths.

        Returns
        -------
        OutputManager
            Manager for accessing parsed output data

        Example
        -------
        >>> sim.run()
        >>> # List all outputs
        >>> print(sim.outputs.list())
        >>> # Get specific output
        >>> pot = sim.outputs.get("pot")
        >>> pot.plot()
        >>> # Get by variable type
        >>> sim.outputs.get_by_variable("potential")
        """
        if self._outputs is None:
            self._build_output_registry()
        return self._outputs

    def _build_output_registry(self) -> None:
        """Build the output registry from simulation commands."""
        self._outputs = OutputManager(working_dir=self.working_dir)

        # Register mesh output
        if self._mesh and hasattr(self._mesh, 'outfile') and self._mesh.outfile:
            self._outputs.register(
                self._mesh.outfile,
                OutputType.MESH,
                command=self._mesh
            )

        # Scan commands for outputs
        for cmd in self._commands:
            # Solve commands
            if isinstance(cmd, Solve):
                if hasattr(cmd, 'outfile') and cmd.outfile:
                    self._outputs.register(
                        cmd.outfile,
                        OutputType.SOLUTION,
                        command=cmd
                    )

            # Log commands
            elif isinstance(cmd, Log):
                if hasattr(cmd, 'ivfile') and cmd.ivfile:
                    self._outputs.register(
                        cmd.ivfile,
                        OutputType.IV_DATA,
                        command=cmd
                    )
                if hasattr(cmd, 'acfile') and cmd.acfile:
                    self._outputs.register(
                        cmd.acfile,
                        OutputType.AC_DATA,
                        command=cmd
                    )

            # Plot1D commands
            elif isinstance(cmd, Plot1D):
                if hasattr(cmd, 'outfile') and cmd.outfile:
                    variable = get_plot1d_variable(cmd)
                    self._outputs.register(
                        cmd.outfile,
                        OutputType.PLOT_1D,
                        command=cmd,
                        variable=variable
                    )

            # Plot2D commands
            elif isinstance(cmd, Plot2D):
                if hasattr(cmd, 'outfile') and cmd.outfile:
                    self._outputs.register(
                        cmd.outfile,
                        OutputType.PLOT_2D,
                        command=cmd
                    )

    def _load_outputs(self) -> None:
        """Load all output files after simulation run."""
        if self._outputs is None:
            self._build_output_registry()
        self._outputs.working_dir = self.working_dir
        self._outputs.load_all()

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

    # -----------------------------------------------------------------------
    # High-level output logging helpers
    # -----------------------------------------------------------------------

    def log_band_diagram(self, outfile_prefix: str = "band",
                         x_start: float = 0.0, y_start: float = 0.0,
                         x_end: Optional[float] = None, y_end: Optional[float] = None,
                         include_qf: bool = False) -> "Simulation":
        """
        Add Plot1D commands to log band diagram data (Ec, Ev, optionally Efn, Efp).

        Parameters
        ----------
        outfile_prefix : str
            Prefix for output files (default: "band").
            Creates files: {prefix}_ec, {prefix}_ev, and optionally {prefix}_efn, {prefix}_efp
        x_start, y_start : float
            Start point of the line cut (default: 0.0, 0.0)
        x_end, y_end : float, optional
            End point of the line cut. If None, uses mesh dimensions.
        include_qf : bool
            If True, also log quasi-Fermi levels (Efn, Efp)

        Returns
        -------
        Simulation
            Self for method chaining

        Example
        -------
        >>> sim = create_pn_diode()
        >>> sim.add_solve(Solve(initial=True))
        >>> sim.log_band_diagram("eq")  # Logs eq_ec, eq_ev
        >>> sim.add_solve(Solve(project=True, vstep=0.1, nsteps=5, electrode=1))
        >>> sim.log_band_diagram("fwd", include_qf=True)  # Logs fwd_ec, fwd_ev, fwd_efn, fwd_efp
        """
        # Default end point to mesh dimensions if not specified
        if x_end is None:
            x_end = x_start
        if y_end is None and self._mesh:
            # Try to get mesh extent from mesh definition
            y_end = 2.0  # Default assumption

        # Conduction band (Ec)
        self._commands.append(Plot1D(
            band_con=True,
            x_start=x_start, y_start=y_start,
            x_end=x_end, y_end=y_end,
            outfile=f"cb{outfile_prefix}",
            ascii=True
        ))

        # Valence band (Ev)
        self._commands.append(Plot1D(
            band_val=True,
            x_start=x_start, y_start=y_start,
            x_end=x_end, y_end=y_end,
            outfile=f"vb{outfile_prefix}",
            ascii=True
        ))

        # Quasi-Fermi levels if requested
        if include_qf:
            self._commands.append(Plot1D(
                qfn=True,
                x_start=x_start, y_start=y_start,
                x_end=x_end, y_end=y_end,
                outfile=f"qfn{outfile_prefix}",
                ascii=True
            ))
            self._commands.append(Plot1D(
                qfp=True,
                x_start=x_start, y_start=y_start,
                x_end=x_end, y_end=y_end,
                outfile=f"qfp{outfile_prefix}",
                ascii=True
            ))

        return self

    def log_carriers(self, outfile_prefix: str = "carrier",
                     x_start: float = 0.0, y_start: float = 0.0,
                     x_end: Optional[float] = None, y_end: Optional[float] = None) -> "Simulation":
        """
        Add Plot1D commands to log carrier concentrations (n, p).

        Parameters
        ----------
        outfile_prefix : str
            Prefix for output files (default: "carrier").
            Creates files: {prefix}_n, {prefix}_p
        x_start, y_start : float
            Start point of the line cut
        x_end, y_end : float, optional
            End point of the line cut

        Returns
        -------
        Simulation
            Self for method chaining
        """
        if x_end is None:
            x_end = x_start
        if y_end is None:
            y_end = 2.0

        self._commands.append(Plot1D(
            electrons=True,
            x_start=x_start, y_start=y_start,
            x_end=x_end, y_end=y_end,
            outfile=f"n{outfile_prefix}",
            ascii=True
        ))
        self._commands.append(Plot1D(
            holes=True,
            x_start=x_start, y_start=y_start,
            x_end=x_end, y_end=y_end,
            outfile=f"p{outfile_prefix}",
            ascii=True
        ))

        return self

    def log_potential(self, outfile: str = "potential",
                      x_start: float = 0.0, y_start: float = 0.0,
                      x_end: Optional[float] = None, y_end: Optional[float] = None) -> "Simulation":
        """
        Add Plot1D command to log electrostatic potential.

        Parameters
        ----------
        outfile : str
            Output filename (default: "potential")
        x_start, y_start : float
            Start point of the line cut
        x_end, y_end : float, optional
            End point of the line cut

        Returns
        -------
        Simulation
            Self for method chaining
        """
        if x_end is None:
            x_end = x_start
        if y_end is None:
            y_end = 2.0

        self._commands.append(Plot1D(
            potential=True,
            x_start=x_start, y_start=y_start,
            x_end=x_end, y_end=y_end,
            outfile=outfile,
            ascii=True
        ))

        return self

    def log_efield(self, outfile: str = "efield",
                   x_start: float = 0.0, y_start: float = 0.0,
                   x_end: Optional[float] = None, y_end: Optional[float] = None) -> "Simulation":
        """
        Add Plot1D command to log electric field.

        Parameters
        ----------
        outfile : str
            Output filename (default: "efield")
        x_start, y_start : float
            Start point of the line cut
        x_end, y_end : float, optional
            End point of the line cut

        Returns
        -------
        Simulation
            Self for method chaining
        """
        if x_end is None:
            x_end = x_start
        if y_end is None:
            y_end = 2.0

        self._commands.append(Plot1D(
            e_field=True,
            x_start=x_start, y_start=y_start,
            x_end=x_end, y_end=y_end,
            outfile=outfile,
            ascii=True
        ))

        return self

    # -----------------------------------------------------------------------
    # Voltage sweep helpers
    # -----------------------------------------------------------------------

    def add_voltage_sweep(self, electrode: int,
                          v_start: float = 0.0,
                          v_end: float = 1.0,
                          v_step: float = 0.1,
                          outfile: Optional[str] = None,
                          log_bands: bool = False,
                          log_carriers: bool = False,
                          band_prefix: str = "sweep",
                          carrier_prefix: str = "sweep",
                          x_start: float = 0.0, y_start: float = 0.0,
                          x_end: Optional[float] = None, y_end: Optional[float] = None) -> "Simulation":
        """
        Add a voltage sweep with optional band diagram and carrier logging at each step.

        Parameters
        ----------
        electrode : int
            Electrode number to sweep voltage on
        v_start : float
            Starting voltage (default: 0.0)
        v_end : float
            Ending voltage (default: 1.0)
        v_step : float
            Voltage step size (default: 0.1)
        outfile : str, optional
            Solution output file prefix
        log_bands : bool
            If True, log band diagrams at each bias point
        log_carriers : bool
            If True, log carrier concentrations at each bias point
        band_prefix : str
            Prefix for band diagram files (default: "sweep")
        carrier_prefix : str
            Prefix for carrier files (default: "sweep")
        x_start, y_start : float
            Start point for line cuts
        x_end, y_end : float, optional
            End point for line cuts

        Returns
        -------
        Simulation
            Self for method chaining

        Example
        -------
        >>> sim = create_pn_diode()
        >>> sim.add_solve(Solve(initial=True))
        >>> sim.add_voltage_sweep(
        ...     electrode=1,
        ...     v_start=0.0, v_end=0.8, v_step=0.1,
        ...     log_bands=True, band_prefix="fwd"
        ... )
        """
        nsteps = int(abs(v_end - v_start) / abs(v_step))

        # Add the solve command for the sweep
        self._commands.append(Solve(
            project=True,
            v1=v_start if electrode == 1 else 0.0,
            v2=v_start if electrode == 2 else 0.0,
            vstep=v_step,
            nsteps=nsteps,
            electrode=electrode,
            outfile=outfile
        ))

        # Add band logging if requested
        if log_bands:
            self.log_band_diagram(
                outfile_prefix=band_prefix,
                x_start=x_start, y_start=y_start,
                x_end=x_end, y_end=y_end,
                include_qf=True
            )

        # Add carrier logging if requested
        if log_carriers:
            self.log_carriers(
                outfile_prefix=carrier_prefix,
                x_start=x_start, y_start=y_start,
                x_end=x_end, y_end=y_end
            )

        return self

    def add_equilibrium_solve(self, outfile: str = "eq",
                              log_bands: bool = False,
                              log_carriers: bool = False,
                              x_start: float = 0.0, y_start: float = 0.0,
                              x_end: Optional[float] = None, y_end: Optional[float] = None) -> "Simulation":
        """
        Add equilibrium solve with optional output logging.

        Parameters
        ----------
        outfile : str
            Solution output file (default: "eq")
        log_bands : bool
            If True, log band diagrams at equilibrium
        log_carriers : bool
            If True, log carrier concentrations at equilibrium
        x_start, y_start : float
            Start point for line cuts
        x_end, y_end : float, optional
            End point for line cuts

        Returns
        -------
        Simulation
            Self for method chaining

        Example
        -------
        >>> sim = create_pn_diode()
        >>> sim.add_equilibrium_solve(log_bands=True)
        """
        self._commands.append(Solve(initial=True, outfile=outfile))

        if log_bands:
            self.log_band_diagram(
                outfile_prefix="eq",
                x_start=x_start, y_start=y_start,
                x_end=x_end, y_end=y_end
            )

        if log_carriers:
            self.log_carriers(
                outfile_prefix="eq",
                x_start=x_start, y_start=y_start,
                x_end=x_end, y_end=y_end
            )

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
            capture_output: bool = True,
            use_stdin: bool = False,
            verbose: bool = False,
            output_dir: Optional[str] = None,
            auto_output_dir: bool = True,
            force_rerun: bool = False) -> subprocess.CompletedProcess:
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
            Whether to capture stdout/stderr (ignored if verbose=True)
        use_stdin : bool
            If True, pass input via stdin (padre < file.inp).
            If False, pass input file as argument (padre file.inp).
        verbose : bool
            If True, stream output to console in real-time.
            Overrides capture_output.
        output_dir : str, optional
            Directory name to create for outputs. If provided, creates this
            subdirectory in working_dir and runs the simulation there.
            This prevents outputs from different runs from being mixed.
        auto_output_dir : bool
            If True (default), automatically creates a hash-based output directory.
            Equivalent to calling create_output_dir() before run().
            Ignored if output_dir is provided. Set to False to run in current working_dir.
        force_rerun : bool
            If True, deletes existing cached directory and forces a fresh simulation.
            Default is False (reuse cached results if available).

        Returns
        -------
        subprocess.CompletedProcess
            Result of the PADRE run

        Example
        -------
        >>> # Run with automatic output directory (default)
        >>> result = sim.run()
        >>> print(sim.working_dir)  # /path/to/simulation_20240115_143022
        >>>
        >>> # Run with specific output directory
        >>> result = sim.run(output_dir="forward_bias_sweep")
        >>> print(sim.working_dir)  # /path/to/forward_bias_sweep
        >>>
        >>> # Run in current working_dir without creating subdirectory
        >>> result = sim.run(auto_output_dir=False)
        """
        # Create output directory if requested
        if output_dir is not None:
            self.create_output_dir(name=output_dir, use_timestamp=False)
        elif auto_output_dir:
            self.create_output_dir()
            
        # Force rerun: delete cached directory if it exists
        if force_rerun and os.path.exists(self.working_dir):
            import shutil
            shutil.rmtree(self.working_dir)
            os.makedirs(self.working_dir, exist_ok=True)
            print(f"Forcing fresh simulation: deleted cached directory '{self.working_dir}'")
        # Write deck to file
        if input_file is None:
            # Use short filename to avoid PADRE's filename length limit
            fd, input_file = tempfile.mkstemp(suffix=".inp", prefix="p", dir=self.working_dir)
            os.close(fd)

        deck_path = self.write_deck(input_file)

        # PADRE has a filename length limit (~60 chars). Use basename if running in working_dir
        if not use_stdin:
            # Check if the path is too long for PADRE
            if len(deck_path) > 60:
                # Use just the filename if we're in the same directory
                deck_path = os.path.basename(deck_path)

        # Build command
        if use_stdin:
            cmd = [padre_executable]
        else:
            cmd = [padre_executable, deck_path]

        # Run PADRE
        if verbose:
            # Stream output in real-time
            output_lines = []
            if use_stdin:
                with open(deck_path, 'r') as infile:
                    process = subprocess.Popen(
                        cmd,
                        stdin=infile,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        cwd=self.working_dir,
                        text=True,
                        bufsize=1
                    )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=self.working_dir,
                    text=True,
                    bufsize=1
                )

            # Stream output line by line
            for line in process.stdout:
                print(line, end='', flush=True)
                output_lines.append(line)

            process.wait()
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=''.join(output_lines),
                stderr=''
            )
        elif use_stdin:
            with open(deck_path, 'r') as infile:
                result = subprocess.run(
                    cmd,
                    stdin=infile,
                    stdout=subprocess.PIPE if capture_output else None,
                    stderr=subprocess.PIPE if capture_output else None,
                    cwd=self.working_dir,
                    text=True
                )
        else:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                cwd=self.working_dir,
                text=True
            )

        # Save output if requested
        if output_file and (capture_output or verbose) and result.stdout:
            output_path = os.path.join(self.working_dir, output_file)
            with open(output_path, 'w') as f:
                f.write(result.stdout)

        # Automatically load outputs after successful run
        if result.returncode == 0:
            self._load_outputs()

        return result

    def parse_output(self, output: str) -> SimulationResult:
        """
        Parse PADRE simulation output.

        Parameters
        ----------
        output : str
            PADRE stdout output string (e.g., from result.stdout)

        Returns
        -------
        SimulationResult
            Parsed simulation results containing mesh statistics,
            bias points, I-V data, convergence info, and warnings.

        Example
        -------
        >>> result = sim.run(padre_executable="padre")
        >>> parsed = sim.parse_output(result.stdout)
        >>> print(parsed.summary())
        >>> vg, id = parsed.get_transfer_characteristic(gate_electrode=3, drain_electrode=2)
        """
        return parse_padre_output(output)

    def run_and_parse(self, padre_executable: str = "padre",
                      input_file: Optional[str] = None,
                      output_file: Optional[str] = None,
                      verbose: bool = False) -> SimulationResult:
        """
        Run the PADRE simulation and parse the output in one step.

        Parameters
        ----------
        padre_executable : str
            Path to PADRE executable (default: "padre")
        input_file : str, optional
            Input deck filename. If None, creates a temporary file.
        output_file : str, optional
            Output file for PADRE stdout
        verbose : bool
            If True, stream output to console in real-time.

        Returns
        -------
        SimulationResult
            Parsed simulation results

        Example
        -------
        >>> parsed = sim.run_and_parse(padre_executable="padre")
        >>> print(f"Converged: {parsed.all_converged}")
        >>> print(f"Bias points: {parsed.num_bias_points}")
        >>> voltages, currents = parsed.get_iv_data(electrode=2)
        """
        result = self.run(
            padre_executable=padre_executable,
            input_file=input_file,
            output_file=output_file,
            capture_output=True,
            verbose=verbose
        )
        return self.parse_output(result.stdout)

    def parse_iv_file(self, filename: str) -> IVData:
        """
        Parse a PADRE log file (ivfile) created by the Log command.

        Parameters
        ----------
        filename : str
            Path to the PADRE log file (relative to working_dir or absolute)

        Returns
        -------
        IVData
            Parsed I-V data with voltages and currents for each electrode.
            Use methods like get_transfer_characteristic() or get_iv_data()
            to extract specific data.

        Example
        -------
        >>> sim.add_log(Log(ivfile="idvg"))
        >>> sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))
        >>> result = sim.run(padre_executable="padre")
        >>> iv_data = sim.parse_iv_file("idvg")
        >>> vg, id = iv_data.get_transfer_characteristic(gate_electrode=3, drain_electrode=2)
        """
        if not os.path.isabs(filename):
            filename = os.path.join(self.working_dir, filename)
        return parse_iv_file(filename)

    def _get_log_files(self) -> list:
        """Get list of log files from the simulation configuration."""
        log_files = []
        for cmd in self._commands:
            if isinstance(cmd, Log) and cmd.ivfile:
                log_files.append(cmd.ivfile)
        return log_files

    def get_iv_data(self, filename: Optional[str] = None,
                    electrode: Optional[int] = None) -> Union[IVData, tuple]:
        """
        Get I-V data from the simulation log file.

        If no filename is provided, uses the first ivfile from the simulation
        configuration. If an electrode is specified, returns (voltages, currents)
        tuple directly.

        Parameters
        ----------
        filename : str, optional
            Log file name. If None, infers from simulation Log commands.
        electrode : int, optional
            If specified, returns (voltages, currents) tuple for this electrode.
            If None, returns the full IVData object.

        Returns
        -------
        IVData or tuple
            If electrode is None: IVData object with all parsed data
            If electrode is specified: (voltages, currents) tuple

        Raises
        ------
        ValueError
            If no filename provided and no Log command with ivfile found

        Example
        -------
        >>> sim.add_log(Log(ivfile="idvg"))
        >>> sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))
        >>> result = sim.run()
        >>> # Get full IVData object
        >>> iv_data = sim.get_iv_data()
        >>> # Or get specific electrode data directly
        >>> voltages, currents = sim.get_iv_data(electrode=2)
        """
        if filename is None:
            log_files = self._get_log_files()
            if not log_files:
                raise ValueError(
                    "No Log command with ivfile found in simulation. "
                    "Either add Log(ivfile='...') or specify filename parameter."
                )
            filename = log_files[-1]  # Use the last log file

        iv_data = self.parse_iv_file(filename)

        if electrode is not None:
            return iv_data.get_iv_data(electrode)
        return iv_data

    def get_transfer_characteristic(self, gate_electrode: int,
                                     drain_electrode: int,
                                     filename: Optional[str] = None) -> tuple:
        """
        Get transfer characteristic (Id vs Vg) from simulation log file.

        Parameters
        ----------
        gate_electrode : int
            Gate electrode number
        drain_electrode : int
            Drain electrode number
        filename : str, optional
            Log file name. If None, infers from simulation Log commands.

        Returns
        -------
        tuple
            (gate_voltages, drain_currents) lists

        Example
        -------
        >>> sim.add_log(Log(ivfile="idvg"))
        >>> sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))
        >>> result = sim.run()
        >>> vg, id = sim.get_transfer_characteristic(gate_electrode=3, drain_electrode=2)
        """
        iv_data = self.get_iv_data(filename=filename)
        return iv_data.get_transfer_characteristic(gate_electrode, drain_electrode)

    def get_output_characteristic(self, drain_electrode: int,
                                   filename: Optional[str] = None) -> tuple:
        """
        Get output characteristic (Id vs Vd) from simulation log file.

        Parameters
        ----------
        drain_electrode : int
            Drain electrode number
        filename : str, optional
            Log file name. If None, infers from simulation Log commands.

        Returns
        -------
        tuple
            (drain_voltages, drain_currents) lists

        Example
        -------
        >>> sim.add_log(Log(ivfile="idvd"))
        >>> sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))
        >>> result = sim.run()
        >>> vd, id = sim.get_output_characteristic(drain_electrode=2)
        """
        iv_data = self.get_iv_data(filename=filename)
        return iv_data.get_output_characteristic(drain_electrode)

    # -----------------------------------------------------------------------
    # Plotting methods
    # -----------------------------------------------------------------------

    def plot_iv(
        self,
        current_electrode: int,
        voltage_electrode: Optional[int] = None,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        log_scale: bool = False,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ):
        """
        Plot I-V data for a specific electrode.

        Parameters
        ----------
        current_electrode : int
            Electrode number for current (y-axis)
        voltage_electrode : int, optional
            Electrode number for voltage (x-axis). If None, auto-detects
            the swept electrode (electrode with largest voltage variation).
        filename : str, optional
            Log file name. If None, infers from simulation Log commands.
        title : str, optional
            Plot title
        log_scale : bool
            Use log scale for current
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object (matplotlib Axes or plotly Figure)

        Example
        -------
        >>> sim.add_log(Log(ivfile="idvg"))
        >>> sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))
        >>> result = sim.run()
        >>> # Plot drain current vs swept voltage (auto-detected)
        >>> sim.plot_iv(current_electrode=2)
        >>> # Explicitly specify gate voltage on x-axis
        >>> sim.plot_iv(current_electrode=2, voltage_electrode=3)
        """
        iv_data = self.get_iv_data(filename=filename)
        return iv_data.plot(
            current_electrode,
            voltage_electrode=voltage_electrode,
            title=title,
            log_scale=log_scale,
            backend=backend,
            show=show,
            **kwargs
        )

    def plot_transfer(
        self,
        gate_electrode: int,
        drain_electrode: int,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        log_scale: bool = True,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ):
        """
        Plot transfer characteristic (Id vs Vg).

        Parameters
        ----------
        gate_electrode : int
            Gate electrode number
        drain_electrode : int
            Drain electrode number
        filename : str, optional
            Log file name. If None, infers from simulation Log commands.
        title : str, optional
            Plot title
        log_scale : bool
            Use log scale for drain current (default True)
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object (matplotlib Axes or plotly Figure)

        Example
        -------
        >>> sim.add_log(Log(ivfile="idvg"))
        >>> sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))
        >>> result = sim.run()
        >>> sim.plot_transfer(gate_electrode=3, drain_electrode=2)
        """
        iv_data = self.get_iv_data(filename=filename)
        return iv_data.plot_transfer(
            gate_electrode,
            drain_electrode,
            title=title,
            log_scale=log_scale,
            backend=backend,
            show=show,
            **kwargs
        )

    def plot_output(
        self,
        drain_electrode: int,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        log_scale: bool = False,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ):
        """
        Plot output characteristic (Id vs Vd).

        Parameters
        ----------
        drain_electrode : int
            Drain electrode number
        filename : str, optional
            Log file name. If None, infers from simulation Log commands.
        title : str, optional
            Plot title
        log_scale : bool
            Use log scale for drain current (default False)
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object (matplotlib Axes or plotly Figure)

        Example
        -------
        >>> sim.add_log(Log(ivfile="idvd"))
        >>> sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))
        >>> result = sim.run()
        >>> sim.plot_output(drain_electrode=2)
        """
        iv_data = self.get_iv_data(filename=filename)
        return iv_data.plot_output(
            drain_electrode,
            title=title,
            log_scale=log_scale,
            backend=backend,
            show=show,
            **kwargs
        )

    def plot_all_electrodes(
        self,
        filename: Optional[str] = None,
        title: str = "I-V Characteristics - All Electrodes",
        log_scale: bool = False,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ):
        """
        Plot I-V data for all electrodes.

        Parameters
        ----------
        filename : str, optional
            Log file name. If None, infers from simulation Log commands.
        title : str
            Plot title
        log_scale : bool
            Use log scale for current
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object (matplotlib Axes or plotly Figure)

        Example
        -------
        >>> sim.add_log(Log(ivfile="idvg"))
        >>> sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))
        >>> result = sim.run()
        >>> sim.plot_all_electrodes(log_scale=True)
        """
        iv_data = self.get_iv_data(filename=filename)
        return iv_data.plot_all_electrodes(
            title=title,
            log_scale=log_scale,
            backend=backend,
            show=show,
            **kwargs
        )

    # -----------------------------------------------------------------------
    # Solution file methods
    # -----------------------------------------------------------------------

    def load_solution(self, filename: str) -> SolutionData:
        """
        Load a PADRE solution file.

        Parameters
        ----------
        filename : str
            Solution filename (relative to working_dir or absolute)

        Returns
        -------
        SolutionData
            Parsed solution with visualization methods

        Example
        -------
        >>> result = sim.run()
        >>> sol = sim.load_solution("pn_eq")
        >>> sol.plot_2d("potential")
        >>> sol.plot_line("electron", log_scale=True)
        """
        if not os.path.isabs(filename):
            filename = os.path.join(self.working_dir, filename)
        return parse_solution_file(filename)

    def load_solutions(self, pattern: str = "*") -> list:
        """
        Load multiple solution files matching a pattern.

        Parameters
        ----------
        pattern : str
            Glob pattern to match files (e.g., "pn_fwd_*")

        Returns
        -------
        List[SolutionData]
            List of parsed solutions in sorted order

        Example
        -------
        >>> result = sim.run()
        >>> solutions = sim.load_solutions("pn_fwd_*")
        >>> for sol in solutions:
        ...     print(sol.filename)
        """
        return load_solution_series(self.working_dir, pattern)

    def plot_solution(
        self,
        filename: str,
        variable: str = 'potential',
        plot_type: str = '2d',
        **kwargs
    ):
        """
        Plot a solution file.

        Parameters
        ----------
        filename : str
            Solution filename
        variable : str
            Variable to plot: 'potential', 'electron', 'hole', 'doping'
        plot_type : str
            '2d' for contour plot, 'line' for 1D cut, 'band' for band diagram
        **kwargs
            Additional arguments passed to the plot method

        Returns
        -------
        Any
            Plot object

        Example
        -------
        >>> sim.plot_solution("pn_eq", variable="potential", plot_type="2d")
        >>> sim.plot_solution("pn_eq", variable="electron", plot_type="line", log_scale=True)
        """
        sol = self.load_solution(filename)

        if plot_type == '2d':
            return sol.plot_2d(variable, **kwargs)
        elif plot_type == 'line':
            return sol.plot_line(variable, **kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Use '2d' or 'line'.")

    # -----------------------------------------------------------------------
    # High-level plotting (uses OutputManager or SolutionData)
    # -----------------------------------------------------------------------

    def plot_band_diagram(self, index: Optional[Union[int, List[int]]] = None,
                           suffix: str = "", **kwargs):
        """
        Plot energy band diagram from Plot1D outputs.

        Requires Plot1D commands for band_val and/or band_con in the simulation.

        Parameters
        ----------
        index : int or List[int], optional
            Index or list of indices of the band diagram sets to plot.
            If None, plots all sets on the same axes.
            Use outputs.get_band_diagram_sets() to see available sets.
        suffix : str
            Suffix to match Plot1D output names (e.g., "eq" for "vbeq", "cbeq").
            Ignored if index is provided.
        **kwargs
            Additional arguments (backend, show, title, etc.)

        Returns
        -------
        Any
            Plot object

        Example
        -------
        >>> # Plot all band diagrams
        >>> sim.plot_band_diagram()
        >>>
        >>> # Plot only equilibrium (first set)
        >>> sim.plot_band_diagram(index=0)
        >>>
        >>> # Plot equilibrium and biased
        >>> sim.plot_band_diagram(index=[0, 1])
        >>>
        >>> # Plot by suffix
        >>> sim.plot_band_diagram(suffix="iv")
        """
        return self.outputs.plot_band_diagram(index=index, suffix=suffix, **kwargs)

    def plot_carriers(self, suffix: str = "", log_scale: bool = True, **kwargs):
        """
        Plot carrier concentration profiles (electrons and holes) from Plot1D outputs.

        Requires Plot1D commands for electrons and/or holes in the simulation.

        Parameters
        ----------
        suffix : str
            Suffix to match Plot1D output names
        log_scale : bool
            Use logarithmic y-axis (default True)
        **kwargs
            Additional arguments

        Returns
        -------
        Any
            Plot object
        """
        return self.outputs.plot_carriers(suffix=suffix, log_scale=log_scale, **kwargs)

    def plot_currents(self, suffix: str = "", **kwargs):
        """
        Plot current density profiles.

        Parameters
        ----------
        suffix : str
            Suffix to match Plot1D output names
        **kwargs
            Additional arguments

        Returns
        -------
        Any
            Plot object
        """
        return self.outputs.plot_currents(suffix=suffix, **kwargs)

    def plot_variable(self, variable: str, suffix: str = "", **kwargs):
        """
        Plot a single variable from Plot1D outputs.

        Requires a Plot1D command for the specified variable in the simulation.

        Parameters
        ----------
        variable : str
            Variable to plot: 'potential', 'electrons', 'holes', 'doping',
            'band_val', 'band_con', 'qfn', 'qfp', 'e_field', etc.
        suffix : str
            Suffix to match Plot1D output names
        **kwargs
            Additional arguments

        Returns
        -------
        Any
            Plot object
        """
        data_list = self.outputs.get_by_variable(variable)
        if data_list:
            # Filter by suffix if provided
            for data in data_list:
                if not suffix or suffix in data.name:
                    return data.plot(**kwargs)

        raise ValueError(
            f"No data available for variable '{variable}'. "
            "Check that the simulation has the appropriate Plot1D commands."
        )

    # -----------------------------------------------------------------------
    # Output management
    # -----------------------------------------------------------------------

    def list_outputs(self, include_existing: bool = False) -> dict:
        """
        List all output files that will be or have been generated by this simulation.

        Analyzes the simulation configuration to find all output files specified
        in Mesh, Solve, Log, Plot1D, Plot2D, Contour, Vector, and Plot3D commands.

        Parameters
        ----------
        include_existing : bool
            If True, also checks which files already exist in working_dir

        Returns
        -------
        dict
            Dictionary with keys:
            - 'mesh': List of mesh output files (.pg)
            - 'solution': List of solution output files (from Solve outfile)
            - 'log': List of log/IV files (from Log ivfile)
            - 'plot': List of plot output files
            - 'all': Combined list of all output files
            If include_existing=True, also includes:
            - 'existing': List of files that already exist
            - 'missing': List of files that don't exist yet

        Example
        -------
        >>> sim = create_pn_diode()
        >>> sim.add_solve(Solve.equilibrium(outfile="sol0"))
        >>> sim.add_solve(Solve.bias_sweep(electrode=2, outfile="sol_fwd"))
        >>> sim.add_log(Log(ivfile="diode_iv"))
        >>> outputs = sim.list_outputs()
        >>> print(outputs['solution'])  # ['sol0', 'sol_fwd']
        >>> print(outputs['log'])       # ['diode_iv']
        """
        outputs = {
            'mesh': [],
            'solution': [],
            'log': [],
            'plot': [],
            'all': []
        }

        # Mesh output file
        if self._mesh and hasattr(self._mesh, 'outfile') and self._mesh.outfile:
            outputs['mesh'].append(self._mesh.outfile)

        # Scan all commands for output files
        for cmd in self._commands:
            # Solve commands - outfile parameter
            if isinstance(cmd, Solve):
                if hasattr(cmd, 'outfile') and cmd.outfile:
                    outputs['solution'].append(cmd.outfile)

            # Log commands - ivfile parameter
            elif isinstance(cmd, Log):
                if hasattr(cmd, 'ivfile') and cmd.ivfile:
                    outputs['log'].append(cmd.ivfile)

            # Plot commands - outfile parameter
            elif isinstance(cmd, (Plot1D, Plot2D, Contour, Vector, Plot3D)):
                if hasattr(cmd, 'outfile') and cmd.outfile:
                    outputs['plot'].append(cmd.outfile)

        # Combine all outputs
        outputs['all'] = (
            outputs['mesh'] +
            outputs['solution'] +
            outputs['log'] +
            outputs['plot']
        )

        # Check existing files if requested
        if include_existing:
            outputs['existing'] = []
            outputs['missing'] = []
            for filename in outputs['all']:
                filepath = os.path.join(self.working_dir, filename)
                if os.path.exists(filepath):
                    outputs['existing'].append(filename)
                else:
                    outputs['missing'].append(filename)

        return outputs

    def create_output_dir(self, name: Optional[str] = None,
                          use_timestamp: bool = True) -> str:
        """
        Create a new output directory for this simulation run.

        Creates a subdirectory in working_dir and updates working_dir to point
        to the new directory. This prevents output files from different runs
        from being mixed together.

        Parameters
        ----------
        name : str, optional
            Base name for the output directory. If None, uses simulation title
            or 'run'.
        use_timestamp : bool
            If True (default), uses content hash for directory name to enable
            caching. If False, uses just the name without hash.

        Returns
        -------
        str
            Path to the created output directory

        Example
        -------
        >>> sim = create_pn_diode()
        >>> # Create timestamped output directory
        >>> output_dir = sim.create_output_dir()
        >>> print(output_dir)  # /path/to/working_dir/pn_diode_20240115_143022
        >>>
        >>> # Create named directory without timestamp
        >>> output_dir = sim.create_output_dir(name="forward_bias", use_timestamp=False)
        >>> print(output_dir)  # /path/to/working_dir/forward_bias
        >>>
        >>> # Now run() will save outputs to this directory
        >>> result = sim.run()
        """
        import hashlib

        # Determine base name
        if name is None:
            if self.title:
                # Clean title for use as directory name
                name = self.title.lower().replace(' ', '_')
                name = ''.join(c for c in name if c.isalnum() or c == '_')
            else:
                name = 'run'

        # Add content hash if requested (default)
        if use_timestamp:
            # Generate hash from deck content
            deck_content = self.generate_deck()
            content_hash = hashlib.md5(deck_content.encode()).hexdigest()
            dir_name = f"{name}_{content_hash}"
        else:
            dir_name = name

        # Create the directory
        output_dir = os.path.join(self.working_dir, dir_name)
        
        # Check if directory already exists (cached results)
        if os.path.exists(output_dir):
            import warnings
            warnings.warn(
                f"Output directory '{output_dir}' already exists. "
                f"Reusing existing simulation results. "
                f"If you want to re-run, add force_rerun=True to run()",
                UserWarning
            )
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Update working_dir to point to new directory
        self.working_dir = output_dir

        return output_dir

    def scan_output_dir(self, directory: Optional[str] = None) -> dict:
        """
        Scan a directory for PADRE output files.

        Useful for finding all outputs from a previous simulation run.

        Parameters
        ----------
        directory : str, optional
            Directory to scan. If None, uses working_dir.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'mesh': List of mesh files (.pg)
            - 'solution': List of potential solution files (no extension)
            - 'log': List of potential log/IV files
            - 'input': List of input deck files (.inp)
            - 'output': List of output files (.out)
            - 'all': List of all files found

        Example
        -------
        >>> # After running a simulation
        >>> files = sim.scan_output_dir()
        >>> print(files['solution'])  # ['pn_eq', 'pn_fwd_a', ...]
        >>> print(files['mesh'])      # ['pn_mesh.pg']
        """
        if directory is None:
            directory = self.working_dir

        results = {
            'mesh': [],
            'solution': [],
            'log': [],
            'input': [],
            'output': [],
            'all': []
        }

        if not os.path.isdir(directory):
            return results

        # Get all files in directory
        all_files = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                all_files.append(item)

        results['all'] = sorted(all_files)

        # Categorize files
        for filename in all_files:
            # Mesh files
            if filename.endswith('.pg'):
                results['mesh'].append(filename)

            # Input deck files
            elif filename.endswith('.inp'):
                results['input'].append(filename)

            # Output files
            elif filename.endswith('.out'):
                results['output'].append(filename)

            # Log/IV files - typically no extension, check for PADRE format
            elif not '.' in filename:
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r') as f:
                        first_line = f.readline()
                        # PADRE IV files start with "# PADRE"
                        if 'PADRE' in first_line:
                            results['log'].append(filename)
                        # Solution files have numeric data
                        elif first_line.strip() and any(c.isdigit() for c in first_line):
                            results['solution'].append(filename)
                except (IOError, UnicodeDecodeError):
                    pass

        return results

    # -----------------------------------------------------------------------
    # Plotting helpers
    # -----------------------------------------------------------------------

    def plot_transfer(self, gate_electrode: int, drain_electrode: int, **kwargs):
        """Plot transfer characteristic from simulation results."""
        if self._outputs is None:
            self._load_outputs()
        
        # Try to find an I-V output
        iv_data = self.outputs.get_iv_data()
        if not iv_data:
            raise ValueError("No I-V data found in simulation outputs.")
        
        # Use the first available I-V dataset
        # In most cases there's only one relevant for transfer char
        first_iv = list(iv_data.values())[0]
        return first_iv.plot_transfer(gate_electrode, drain_electrode, **kwargs)

    def plot_output(self, drain_electrode: int, **kwargs):
        """Plot output characteristic from simulation results."""
        if self._outputs is None:
            self._load_outputs()
            
        iv_data = self.outputs.get_iv_data()
        if not iv_data:
            raise ValueError("No I-V data found in simulation outputs.")
            
        first_iv = list(iv_data.values())[0]
        return first_iv.plot_output(drain_electrode, **kwargs)

    def plot_gummel(self, base_electrode: int = 2, collector_electrode: int = 3, **kwargs):
        """Plot Gummel characteristic from simulation results."""
        if self._outputs is None:
            self._load_outputs()
            
        iv_data = self.outputs.get_iv_data()
        if not iv_data:
            raise ValueError("No I-V data found in simulation outputs.")
            
        first_iv = list(iv_data.values())[0]
        return first_iv.plot_gummel(base_electrode, collector_electrode, **kwargs)

    def get_cv_data(self) -> Optional[ACData]:
        """
        Get C-V data from simulation outputs.
        
        Returns
        -------
        ACData or None
            Parsed AC analysis data
        """
        if self._outputs is None:
            self._load_outputs()
            
        # Try to find AC data
        # Note: OutputManager needs to support AC_DATA retrieval
        if hasattr(self.outputs, 'get_ac_data'):
            ac_data_map = self.outputs.get_ac_data()
            if ac_data_map:
                return list(ac_data_map.values())[0]
        
        # Fallback: check raw files if output manager support is incomplete
        for cmd in self._commands:
            if isinstance(cmd, Log) and hasattr(cmd, 'acfile') and cmd.acfile:
                path = os.path.join(self.working_dir, cmd.acfile)
                if os.path.exists(path):
                    return parse_ac_file(path)
                    
        return None

    def __repr__(self) -> str:
        parts = []
        if self.title:
            parts.append(f"title='{self.title}'")
        parts.append(f"regions={len(self._regions)}")
        parts.append(f"electrodes={len(self._electrodes)}")
        parts.append(f"dopings={len(self._dopings)}")
        parts.append(f"commands={len(self._commands)}")
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
