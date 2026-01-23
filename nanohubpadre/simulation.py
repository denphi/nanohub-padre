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
from .parser import parse_padre_output, SimulationResult, parse_iv_file, IVData
from .solution import parse_solution_file, load_solution_series, SolutionData


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
            capture_output: bool = True,
            use_stdin: bool = False,
            verbose: bool = False) -> subprocess.CompletedProcess:
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

        Returns
        -------
        subprocess.CompletedProcess
            Result of the PADRE run
        """
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
        >>> sol.plot_band_diagram()
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
        >>> sim.plot_solution("pn_fwd_a", plot_type="band")
        """
        sol = self.load_solution(filename)

        if plot_type == '2d':
            return sol.plot_2d(variable, **kwargs)
        elif plot_type == 'line':
            return sol.plot_line(variable, **kwargs)
        elif plot_type == 'band':
            return sol.plot_band_diagram(**kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Use '2d', 'line', or 'band'.")

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
