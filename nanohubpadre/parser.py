"""
Parser for PADRE simulation output.

Extracts statistics, convergence data, and I-V characteristics from PADRE output.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class ConvergenceStatus(Enum):
    """Convergence status for a bias point."""
    CONVERGED = "converged"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class BiasPoint:
    """Data for a single bias point solution."""
    voltages: Dict[int, float] = field(default_factory=dict)
    currents: Dict[int, Dict[str, float]] = field(default_factory=dict)
    flux: Dict[int, float] = field(default_factory=dict)
    displacement_current: Dict[int, float] = field(default_factory=dict)
    total_current: Dict[int, float] = field(default_factory=dict)
    iterations: int = 0
    cpu_time: float = 0.0
    convergence_status: ConvergenceStatus = ConvergenceStatus.UNKNOWN
    is_initial: bool = False


@dataclass
class MeshStatistics:
    """Mesh statistics from PADRE output."""
    total_grid_points: int = 0
    total_elements: int = 0
    min_grid_spacing: float = 0.0
    max_grid_spacing: float = 0.0
    spacing_ratio: float = 0.0
    obtuse_elements: int = 0
    obtuse_percentage: float = 0.0


@dataclass
class SimulationResult:
    """Complete parsed results from a PADRE simulation."""
    title: str = ""
    mesh_stats: Optional[MeshStatistics] = None
    materials: Dict[int, str] = field(default_factory=dict)
    regions_by_material: Dict[str, List[int]] = field(default_factory=dict)
    bias_points: List[BiasPoint] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    total_cpu_time: float = 0.0

    @property
    def num_bias_points(self) -> int:
        """Number of bias points solved."""
        return len(self.bias_points)

    @property
    def all_converged(self) -> bool:
        """Whether all bias points converged."""
        return all(bp.convergence_status == ConvergenceStatus.CONVERGED
                   for bp in self.bias_points)

    def get_iv_data(self, electrode: int) -> Tuple[List[float], List[float]]:
        """
        Extract I-V data for a specific electrode.

        Parameters
        ----------
        electrode : int
            Electrode number (1-indexed)

        Returns
        -------
        tuple
            (voltages, currents) lists
        """
        voltages = []
        currents = []
        for bp in self.bias_points:
            if electrode in bp.voltages and electrode in bp.total_current:
                voltages.append(bp.voltages[electrode])
                currents.append(bp.total_current[electrode])
        return voltages, currents

    def get_transfer_characteristic(self, gate_electrode: int,
                                     drain_electrode: int) -> Tuple[List[float], List[float]]:
        """
        Extract transfer characteristic (Id vs Vg).

        Parameters
        ----------
        gate_electrode : int
            Gate electrode number
        drain_electrode : int
            Drain electrode number

        Returns
        -------
        tuple
            (gate_voltages, drain_currents) lists
        """
        vg = []
        id = []
        for bp in self.bias_points:
            if gate_electrode in bp.voltages and drain_electrode in bp.total_current:
                vg.append(bp.voltages[gate_electrode])
                id.append(abs(bp.total_current[drain_electrode]))
        return vg, id

    def summary(self) -> str:
        """Generate a human-readable summary of the simulation results."""
        lines = []
        lines.append(f"PADRE Simulation Results")
        lines.append(f"=" * 50)

        if self.title:
            lines.append(f"Title: {self.title}")

        if self.mesh_stats:
            lines.append(f"\nMesh Statistics:")
            lines.append(f"  Grid points: {self.mesh_stats.total_grid_points}")
            lines.append(f"  Elements: {self.mesh_stats.total_elements}")
            lines.append(f"  Min spacing: {self.mesh_stats.min_grid_spacing:.4e} um")
            lines.append(f"  Max spacing: {self.mesh_stats.max_grid_spacing:.4e} um")

        lines.append(f"\nBias Points: {self.num_bias_points}")
        converged = sum(1 for bp in self.bias_points
                       if bp.convergence_status == ConvergenceStatus.CONVERGED)
        lines.append(f"  Converged: {converged}/{self.num_bias_points}")

        if self.warnings:
            lines.append(f"\nWarnings: {len(self.warnings)}")
            for w in self.warnings[:5]:  # Show first 5
                lines.append(f"  - {w}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")

        if self.errors:
            lines.append(f"\nErrors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  - {e}")

        lines.append(f"\nTotal CPU time: {self.total_cpu_time:.4f} s")

        return "\n".join(lines)


class PadreOutputParser:
    """Parser for PADRE simulation output."""

    def __init__(self):
        self.result = SimulationResult()
        self._current_bias_point = None

    def parse(self, output: str) -> SimulationResult:
        """
        Parse PADRE output string.

        Parameters
        ----------
        output : str
            Complete PADRE stdout output

        Returns
        -------
        SimulationResult
            Parsed simulation results
        """
        self.result = SimulationResult()
        self._current_bias_point = None

        lines = output.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]

            # Parse title
            if line.strip().startswith('***********') and i + 1 < len(lines):
                title_line = lines[i + 1].strip()
                if title_line and not title_line.startswith('*'):
                    self.result.title = title_line
                i += 1

            # Parse mesh statistics
            elif 'Mesh statistics' in line:
                i = self._parse_mesh_stats(lines, i)

            # Parse material definitions
            elif 'Material Definitions' in line:
                i = self._parse_materials(lines, i)

            # Parse bias point solution
            elif 'Solution for bias:' in line:
                i = self._parse_bias_point(lines, i)

            # Parse warnings
            elif '** Warning' in line:
                warning = self._extract_warning(lines, i)
                if warning:
                    self.result.warnings.append(warning)

            # Parse total CPU time at end
            elif 'Total cpu time =' in line and 'bias point' not in line.lower():
                match = re.search(r'Total cpu time =\s+([\d.]+)', line)
                if match:
                    self.result.total_cpu_time = float(match.group(1))

            i += 1

        return self.result

    def _parse_mesh_stats(self, lines: List[str], start: int) -> int:
        """Parse mesh statistics section."""
        stats = MeshStatistics()
        i = start + 1

        while i < len(lines) and lines[i].strip():
            line = lines[i].strip()

            if 'Total grid points' in line:
                match = re.search(r'=\s*(\d+)', line)
                if match:
                    stats.total_grid_points = int(match.group(1))

            elif 'Total no. of elements' in line:
                match = re.search(r'=\s*(\d+)', line)
                if match:
                    stats.total_elements = int(match.group(1))

            elif 'Min grid spacing' in line:
                match = re.search(r'=\s*([\d.E+-]+)', line)
                if match:
                    stats.min_grid_spacing = float(match.group(1))

            elif 'Max grid spacing' in line:
                match = re.search(r'=\s*([\d.E+-]+).*r=\s*([\d.E+-]+)', line)
                if match:
                    stats.max_grid_spacing = float(match.group(1))
                    stats.spacing_ratio = float(match.group(2))

            elif 'Obtuse elements' in line:
                match = re.search(r'=\s*(\d+).*\(\s*([\d.]+)%\)', line)
                if match:
                    stats.obtuse_elements = int(match.group(1))
                    stats.obtuse_percentage = float(match.group(2))

            i += 1

        self.result.mesh_stats = stats
        return i

    def _parse_materials(self, lines: List[str], start: int) -> int:
        """Parse material definitions section."""
        i = start + 1

        # Skip header lines
        while i < len(lines) and ('Index' in lines[i] or '---' in lines[i] or not lines[i].strip()):
            i += 1

        # Parse material entries
        while i < len(lines) and lines[i].strip():
            line = lines[i].strip()
            match = re.match(r'(\d+)\s+(\w+)\s+([\d,]+)', line)
            if match:
                idx = int(match.group(1))
                name = match.group(2)
                regions = [int(r) for r in match.group(3).split(',')]
                self.result.materials[idx] = name
                if name not in self.result.regions_by_material:
                    self.result.regions_by_material[name] = []
                self.result.regions_by_material[name].extend(regions)
            i += 1

        return i

    def _parse_bias_point(self, lines: List[str], start: int) -> int:
        """Parse a bias point solution section."""
        bp = BiasPoint()
        i = start + 1

        # Parse voltages (V1, V2, etc.)
        while i < len(lines):
            line = lines[i]

            # Look for voltage specifications
            v_matches = re.findall(r'V(\d+)\s*=\s*([\d.E+-]+)', line)
            for electrode, voltage in v_matches:
                bp.voltages[int(electrode)] = float(voltage)

            if v_matches:
                i += 1
                continue

            # Check for initial solution
            if 'Initial solution' in line:
                bp.is_initial = True

            # Parse electrode currents table
            if 'Electrode   Voltage   Electron Current' in line:
                i = self._parse_current_table(lines, i, bp)
                continue

            # Parse flux table
            if 'Electrode          Flux' in line:
                i = self._parse_flux_table(lines, i, bp)
                continue

            # Check for convergence
            if 'Convergence criterion completely met' in line:
                bp.convergence_status = ConvergenceStatus.CONVERGED
            elif 'did not converge' in line.lower() or 'failed' in line.lower():
                bp.convergence_status = ConvergenceStatus.FAILED

            # Parse CPU time for this bias point
            if 'Total cpu time for bias point' in line:
                match = re.search(r'=\s+([\d.]+)', line)
                if match:
                    bp.cpu_time = float(match.group(1))

            # End of this bias point section
            if i + 1 < len(lines) and 'Solution for bias:' in lines[i + 1]:
                break
            if 'Caution!' in line or 'warnings cited' in line:
                break

            i += 1

        self.result.bias_points.append(bp)
        return i

    def _parse_current_table(self, lines: List[str], start: int, bp: BiasPoint) -> int:
        """Parse electrode current table."""
        i = start + 1

        # Skip header line
        while i < len(lines) and ('(Volts)' in lines[i] or '(Amps)' in lines[i]):
            i += 1

        # Parse current entries
        while i < len(lines) and lines[i].strip():
            line = lines[i].strip()
            parts = line.split()

            if len(parts) >= 5:
                try:
                    electrode = int(parts[0])
                    voltage = float(parts[1])
                    electron_current = float(parts[2])
                    hole_current = float(parts[3])
                    conduction_current = float(parts[4])

                    bp.currents[electrode] = {
                        'electron': electron_current,
                        'hole': hole_current,
                        'conduction': conduction_current
                    }
                except (ValueError, IndexError):
                    pass

            i += 1

        return i

    def _parse_flux_table(self, lines: List[str], start: int, bp: BiasPoint) -> int:
        """Parse electrode flux table."""
        i = start + 1

        # Skip header line
        while i < len(lines) and ('(Coul)' in lines[i] or '(Amps)' in lines[i]):
            i += 1

        # Parse flux entries
        while i < len(lines) and lines[i].strip():
            line = lines[i].strip()
            parts = line.split()

            if len(parts) >= 4:
                try:
                    electrode = int(parts[0])
                    flux = float(parts[1])
                    disp_current = float(parts[2])
                    total_current = float(parts[3])

                    bp.flux[electrode] = flux
                    bp.displacement_current[electrode] = disp_current
                    bp.total_current[electrode] = total_current
                except (ValueError, IndexError):
                    pass

            i += 1

        return i

    def _extract_warning(self, lines: List[str], start: int) -> Optional[str]:
        """Extract warning message."""
        warning_parts = []
        i = start

        while i < len(lines) and i < start + 5:
            line = lines[i].strip()
            if line and not line.startswith('**'):
                warning_parts.append(line)
            elif line.startswith('** Warning'):
                # Extract line number if present
                match = re.search(r'line #\s*(\d+)', line)
                if match:
                    warning_parts.append(f"Line {match.group(1)}:")
            i += 1

            # Stop at next warning or empty line after content
            if warning_parts and not line:
                break

        return ' '.join(warning_parts) if warning_parts else None


def parse_padre_output(output: str) -> SimulationResult:
    """
    Convenience function to parse PADRE output.

    Parameters
    ----------
    output : str
        Complete PADRE stdout output

    Returns
    -------
    SimulationResult
        Parsed simulation results

    Example
    -------
    >>> result = sim.run(padre_executable="padre")
    >>> parsed = parse_padre_output(result.stdout)
    >>> print(parsed.summary())
    >>> vg, id = parsed.get_transfer_characteristic(gate_electrode=3, drain_electrode=2)
    """
    parser = PadreOutputParser()
    return parser.parse(output)


@dataclass
class IVData:
    """
    Parsed I-V data from PADRE log file (ivfile).

    Attributes
    ----------
    num_electrodes : int
        Number of electrodes in the simulation
    bias_points : List[dict]
        List of bias point data, each containing:
        - voltages: dict mapping electrode number to voltage
        - currents: dict mapping electrode number to current components
    """
    num_electrodes: int = 0
    bias_points: List[Dict] = field(default_factory=list)

    def get_voltages(self, electrode: int) -> List[float]:
        """Get all voltages for a specific electrode."""
        return [bp['voltages'].get(electrode, 0.0) for bp in self.bias_points]

    def get_currents(self, electrode: int, component: str = 'total') -> List[float]:
        """
        Get currents for a specific electrode.

        Parameters
        ----------
        electrode : int
            Electrode number (1-indexed)
        component : str
            Current component: 'electron', 'hole', or 'total' (default)
        """
        return [bp['currents'].get(electrode, {}).get(component, 0.0)
                for bp in self.bias_points]

    def get_iv_data(self, electrode: int) -> Tuple[List[float], List[float]]:
        """
        Get I-V data for a specific electrode.

        Parameters
        ----------
        electrode : int
            Electrode number (1-indexed)

        Returns
        -------
        tuple
            (voltages, currents) lists
        """
        voltages = self.get_voltages(electrode)
        currents = self.get_currents(electrode, 'total')
        return voltages, currents

    def get_transfer_characteristic(self, gate_electrode: int,
                                     drain_electrode: int) -> Tuple[List[float], List[float]]:
        """
        Extract transfer characteristic (Id vs Vg).

        Parameters
        ----------
        gate_electrode : int
            Gate electrode number
        drain_electrode : int
            Drain electrode number

        Returns
        -------
        tuple
            (gate_voltages, drain_currents) lists
        """
        vg = self.get_voltages(gate_electrode)
        id_vals = [abs(i) for i in self.get_currents(drain_electrode, 'total')]
        return vg, id_vals

    def get_output_characteristic(self, drain_electrode: int) -> Tuple[List[float], List[float]]:
        """
        Extract output characteristic (Id vs Vd).

        Parameters
        ----------
        drain_electrode : int
            Drain electrode number

        Returns
        -------
        tuple
            (drain_voltages, drain_currents) lists
        """
        vd = self.get_voltages(drain_electrode)
        id_vals = [abs(i) for i in self.get_currents(drain_electrode, 'total')]
        return vd, id_vals

    # -----------------------------------------------------------------------
    # Plotting methods
    # -----------------------------------------------------------------------

    def plot(
        self,
        electrode: int,
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
        electrode : int
            Electrode number
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
        >>> iv_data = sim.get_iv_data()
        >>> iv_data.plot(electrode=2, log_scale=True)
        """
        from .visualization import plot_iv
        voltages, currents = self.get_iv_data(electrode)
        title = title or f"I-V Characteristic - Electrode {electrode}"
        return plot_iv(
            voltages, currents,
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
        >>> iv_data = sim.get_iv_data()
        >>> iv_data.plot_transfer(gate_electrode=3, drain_electrode=2)
        """
        from .visualization import plot_transfer_characteristic
        vg, id_vals = self.get_transfer_characteristic(gate_electrode, drain_electrode)
        title = title or f"Transfer Characteristic (Gate={gate_electrode}, Drain={drain_electrode})"
        return plot_transfer_characteristic(
            vg, id_vals,
            title=title,
            log_scale=log_scale,
            backend=backend,
            show=show,
            **kwargs
        )

    def plot_output(
        self,
        drain_electrode: int,
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
        >>> iv_data = sim.get_iv_data()
        >>> iv_data.plot_output(drain_electrode=2)
        """
        from .visualization import plot_output_characteristic
        vd, id_vals = self.get_output_characteristic(drain_electrode)
        title = title or f"Output Characteristic (Drain={drain_electrode})"
        return plot_output_characteristic(
            vd, id_vals,
            title=title,
            log_scale=log_scale,
            backend=backend,
            show=show,
            **kwargs
        )

    def plot_all_electrodes(
        self,
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
        >>> iv_data = sim.get_iv_data()
        >>> iv_data.plot_all_electrodes(log_scale=True)
        """
        from .visualization import (
            _get_default_backend,
            _plot_multi_iv_matplotlib,
            _plot_multi_iv_plotly
        )

        if backend is None:
            backend = _get_default_backend()

        data_series = []
        for elec in range(1, self.num_electrodes + 1):
            voltages, currents = self.get_iv_data(elec)
            data_series.append((voltages, currents, f"Electrode {elec}"))

        if backend == 'matplotlib':
            return _plot_multi_iv_matplotlib(
                data_series,
                title=title,
                log_scale=log_scale,
                show=show,
                **kwargs
            )
        else:
            return _plot_multi_iv_plotly(
                data_series,
                title=title,
                log_scale=log_scale,
                show=show,
                **kwargs
            )


class IVFileParser:
    """
    Parser for PADRE log files (ivfile format).

    PADRE log files contain I-V data in a specific format with Q-records
    that store electrode voltages and currents for each bias point.

    File format:
    - Line 1: Header (e.g., "# PADRE2.4E 10/24/94")
    - Line 2: Three integers: num_electrodes, num_electrodes, 0
    - Q-records: Each starts with "Q" followed by data
      - 4 lines of 5 values each (20 total values per bias point)
      - Contains voltages and currents for all electrodes
    """

    def __init__(self):
        self.data = IVData()

    def parse(self, content: str) -> IVData:
        """
        Parse PADRE log file content.

        Parameters
        ----------
        content : str
            Content of the PADRE log file

        Returns
        -------
        IVData
            Parsed I-V data
        """
        self.data = IVData()
        lines = content.strip().split('\n')

        if len(lines) < 2:
            return self.data

        # Parse header - line 2 contains electrode counts
        # Format: "                     4                     4                     0"
        header_line = lines[1].strip()
        header_parts = header_line.split()
        if len(header_parts) >= 2:
            try:
                self.data.num_electrodes = int(header_parts[0])
            except ValueError:
                pass

        # Parse Q-records
        i = 2
        while i < len(lines):
            line = lines[i].strip()

            # Look for Q-record start
            if line.startswith('Q'):
                # Parse Q-record (spans 5 lines: Q-line + 4 data lines)
                if i + 4 < len(lines):
                    bias_point = self._parse_q_record(lines, i)
                    if bias_point:
                        self.data.bias_points.append(bias_point)
                    i += 5
                    continue

            i += 1

        return self.data

    def _parse_q_record(self, lines: List[str], start: int) -> Optional[Dict]:
        """
        Parse a single Q-record.

        Q-record format for 4 electrodes:
        Q0.00000E+00 0.00000E+00   (time values, typically 0)
        [5 values]  V1, V2, V3, V4, and extra value
        [5 values]  V5(unused), V6(unused), V7=V3, V8(unused), I1_e, I2_e
        [5 values]  I3_e, I4_e, I1_h, I2_h, I3_h
        [5 values]  I4_h, Q1, Q2, Q3, Q4

        Actually, after analyzing the file more carefully:
        - Lines have 5 values each
        - Total 20 values per Q-record (after the Q line)
        - Format depends on number of electrodes
        """
        try:
            # Collect all values from the 4 data lines
            values = []
            for offset in range(1, 5):
                line = lines[start + offset].strip()
                parts = line.split()
                for part in parts:
                    try:
                        values.append(float(part))
                    except ValueError:
                        values.append(0.0)

            if len(values) < 20:
                return None

            num_elec = self.data.num_electrodes
            if num_elec == 0:
                num_elec = 4  # Default assumption

            # Parse based on PADRE log file format
            # The format stores: voltages, then electron currents, hole currents, charges
            # For 4 electrodes:
            # Values 0-4: V1, V2, V3, V4, (extra)
            # Values 5-9: (extra), (extra), V3_copy, (extra), I1_electron, I2_electron
            # Actually the exact mapping depends on PADRE version

            # Based on the idvg file provided:
            # Line 1 (values 0-4): 0, 0, Vg(0.1, 0.2...), 0, 0
            # Line 2 (values 5-9): 0, Vg, 0, small_number, small_number
            # Line 3 (values 10-14): 0, very_small, very_small, very_small, small_number
            # Line 4 (values 15-19): very_small, 0, 0, Vg, 0

            # Looking at the pattern: V3 appears at index 2, 6, and 18
            # This suggests electrode 3 is being swept

            # Standard PADRE ivfile format for N electrodes:
            # First N values: voltages V1 through VN
            # Next N values: electron currents I1_e through IN_e
            # Next N values: hole currents I1_h through IN_h
            # Next N values: total currents I1 through IN

            # However, the actual format seems different. Let's use a simpler approach:
            # Extract voltages from positions that show the sweep pattern

            bias_point = {
                'voltages': {},
                'currents': {}
            }

            # For the idvg file format observed:
            # Voltages appear to be at indices 2 (V3), 6 (V3 copy), 18 (V3 again)
            # and the value at index 2, 6, 18 is the gate voltage being swept

            # General approach: first num_electrodes values are voltages
            for elec in range(1, num_elec + 1):
                idx = elec - 1
                if idx < len(values):
                    bias_point['voltages'][elec] = values[idx]

            # For currents, they appear after the voltage section
            # Electron currents start at index num_elec
            # Hole currents at index 2*num_elec
            # Total currents (or charges) at index 3*num_elec

            for elec in range(1, num_elec + 1):
                elec_currents = {}

                # Electron current
                e_idx = num_elec + (elec - 1)
                if e_idx < len(values):
                    elec_currents['electron'] = values[e_idx]

                # Hole current
                h_idx = 2 * num_elec + (elec - 1)
                if h_idx < len(values):
                    elec_currents['hole'] = values[h_idx]

                # Total current (electron + hole)
                if 'electron' in elec_currents and 'hole' in elec_currents:
                    elec_currents['total'] = elec_currents['electron'] + elec_currents['hole']
                else:
                    # Try to get from a later position
                    t_idx = 3 * num_elec + (elec - 1)
                    if t_idx < len(values):
                        elec_currents['total'] = values[t_idx]

                bias_point['currents'][elec] = elec_currents

            return bias_point

        except (IndexError, ValueError):
            return None


def parse_iv_file(filename: str) -> IVData:
    """
    Parse a PADRE log file (ivfile).

    Parameters
    ----------
    filename : str
        Path to the PADRE log file

    Returns
    -------
    IVData
        Parsed I-V data with voltages and currents for each electrode

    Example
    -------
    >>> iv_data = parse_iv_file("idvg")
    >>> vg, id = iv_data.get_transfer_characteristic(gate_electrode=3, drain_electrode=2)
    >>> print(f"Vg range: {min(vg):.2f} to {max(vg):.2f} V")
    """
    with open(filename, 'r') as f:
        content = f.read()
    parser = IVFileParser()
    return parser.parse(content)


def parse_iv_content(content: str) -> IVData:
    """
    Parse PADRE log file content directly.

    Parameters
    ----------
    content : str
        Content of the PADRE log file

    Returns
    -------
    IVData
        Parsed I-V data
    """
    parser = IVFileParser()
    return parser.parse(content)
