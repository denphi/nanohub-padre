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
