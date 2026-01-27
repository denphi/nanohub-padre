"""
Parser and visualization for PADRE solution files.

PADRE saves solution data in binary-like format containing:
- Mesh information (node coordinates)
- Device variables at each node (potential, electron/hole concentrations, etc.)

This module provides tools to read, analyze, and visualize these solution files.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np


@dataclass
class MeshData:
    """
    Mesh information from PADRE output.

    Attributes
    ----------
    nx : int
        Number of nodes in x direction
    ny : int
        Number of nodes in y direction
    x : np.ndarray
        X coordinates of mesh nodes (flat array, in microns)
    y : np.ndarray
        Y coordinates of mesh nodes (flat array, in microns)
    x_unique : np.ndarray
        Unique X coordinates (1D, in microns)
    y_unique : np.ndarray
        Unique Y coordinates (1D, in microns)
    """
    nx: int = 0
    ny: int = 0
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    x_unique: np.ndarray = field(default_factory=lambda: np.array([]))
    y_unique: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def num_nodes(self) -> int:
        """Total number of mesh nodes."""
        return self.nx * self.ny

    def get_y_positions(self) -> np.ndarray:
        """Get unique Y positions in microns."""
        if len(self.y_unique) > 0:
            return self.y_unique
        elif len(self.y) > 0:
            return np.unique(self.y)
        else:
            # Generate default Y mesh for 2 micron device
            return np.linspace(0, 2, self.ny)

    def get_x_positions(self) -> np.ndarray:
        """Get unique X positions in microns."""
        if len(self.x_unique) > 0:
            return self.x_unique
        elif len(self.x) > 0:
            return np.unique(self.x)
        else:
            # Generate default X mesh for 2 micron device
            return np.linspace(0, 2, self.nx)


@dataclass
class SolutionData:
    """
    Solution data from a PADRE output file.

    Contains the device state at a specific bias point including
    electrostatic potential, carrier concentrations, and other variables.

    Attributes
    ----------
    mesh : MeshData
        Mesh information
    potential : np.ndarray
        Electrostatic potential (V) at each node
    electron_conc : np.ndarray
        Electron concentration (/cm³) at each node
    hole_conc : np.ndarray
        Hole concentration (/cm³) at each node
    bias_voltages : Dict[int, float]
        Applied voltages at each electrode
    filename : str
        Source filename
    """
    mesh: MeshData = field(default_factory=MeshData)
    potential: np.ndarray = field(default_factory=lambda: np.array([]))
    electron_conc: np.ndarray = field(default_factory=lambda: np.array([]))
    hole_conc: np.ndarray = field(default_factory=lambda: np.array([]))
    doping: np.ndarray = field(default_factory=lambda: np.array([]))
    electric_field_x: np.ndarray = field(default_factory=lambda: np.array([]))
    electric_field_y: np.ndarray = field(default_factory=lambda: np.array([]))
    bias_voltages: Dict[int, float] = field(default_factory=dict)
    filename: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if the solution contains valid data."""
        return (
            len(self.potential) > 0 and
            len(self.electron_conc) > 0 and
            self.mesh.nx > 0 and
            self.mesh.ny > 0
        )

    def summary(self) -> str:
        """Get a summary of the solution data."""
        lines = [f"Solution: {self.filename}"]
        lines.append(f"  Mesh: {self.mesh.nx} x {self.mesh.ny} = {self.mesh.num_nodes} nodes")
        lines.append(f"  Potential: {len(self.potential)} values")
        lines.append(f"  Electrons: {len(self.electron_conc)} values")
        lines.append(f"  Holes: {len(self.hole_conc)} values")
        if self.is_valid:
            lines.append(f"  Potential range: {self.potential.min():.4f} to {self.potential.max():.4f} V")
            lines.append(f"  Electron range: {self.electron_conc.min():.2e} to {self.electron_conc.max():.2e} /cm³")
        else:
            lines.append("  WARNING: Solution data appears incomplete or invalid")
        return "\n".join(lines)

    def get_2d_data(self, variable: str) -> np.ndarray:
        """
        Get a variable reshaped as 2D array matching the mesh.

        Parameters
        ----------
        variable : str
            Variable name: 'potential', 'electron', 'hole', 'doping',
            'net_doping', 'electric_field_x', 'electric_field_y'

        Returns
        -------
        np.ndarray
            2D array of shape (ny, nx)
        """
        var_map = {
            'potential': self.potential,
            'electron': self.electron_conc,
            'hole': self.hole_conc,
            'doping': self.doping,
            'electric_field_x': self.electric_field_x,
            'electric_field_y': self.electric_field_y,
        }

        if variable not in var_map:
            raise ValueError(f"Unknown variable: {variable}. "
                           f"Available: {list(var_map.keys())}")

        data = var_map[variable]
        if len(data) == 0:
            raise ValueError(f"No data for variable: {variable}")

        # Verify mesh dimensions are set
        if self.mesh.nx == 0 or self.mesh.ny == 0:
            raise ValueError(
                f"Mesh dimensions not set (nx={self.mesh.nx}, ny={self.mesh.ny}). "
                "The solution file may not have been parsed correctly."
            )

        # Verify data length matches mesh
        expected_nodes = self.mesh.nx * self.mesh.ny
        if len(data) != expected_nodes:
            raise ValueError(
                f"Data length ({len(data)}) doesn't match mesh "
                f"({self.mesh.nx}x{self.mesh.ny}={expected_nodes} nodes). "
                "The solution file may be corrupted or incomplete."
            )

        return data.reshape(self.mesh.ny, self.mesh.nx)

    def get_line_cut(self, variable: str, direction: str = 'y',
                     position: Optional[float] = None,
                     index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a 1D line cut from the 2D data.

        Parameters
        ----------
        variable : str
            Variable to extract
        direction : str
            Direction of the cut: 'x' (horizontal) or 'y' (vertical/depth)
        position : float, optional
            Position of the cut in microns. If None, uses middle or first.
        index : int, optional
            Direct index of the cut. Overrides position if provided.

        Returns
        -------
        tuple
            (coordinates_in_microns, values) along the cut
        """
        data_2d = self.get_2d_data(variable)
        x = self.mesh.get_x_positions()
        y = self.mesh.get_y_positions()

        if direction == 'y':
            # Vertical cut (along y/depth at fixed x)
            if index is None:
                if position is not None:
                    # Find closest x index
                    index = np.argmin(np.abs(x - position))
                else:
                    index = 0  # Default to first x position (surface)
            return y, data_2d[:, index]
        else:
            # Horizontal cut (along x at fixed y/depth)
            if index is None:
                if position is not None:
                    # Find closest y index
                    index = np.argmin(np.abs(y - position))
                else:
                    index = self.mesh.ny // 2  # Middle depth
            return x, data_2d[index, :]

    def get_1d_profile(self, variable: str = 'potential') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a 1D profile for devices uniform in X (like PN diodes).

        For 1D devices, this identifies unique device states and orders them
        by position (from surface to substrate) based on electron concentration
        gradient.

        Parameters
        ----------
        variable : str
            Variable to extract: 'potential', 'electron', 'hole'

        Returns
        -------
        tuple
            (y_positions, values) arrays sorted from surface (y=0) to substrate
        """
        if len(self.potential) == 0 or len(self.electron_conc) == 0:
            return np.array([]), np.array([])

        ny = self.mesh.ny
        y_positions = self.mesh.get_y_positions()

        # Find unique device states based on (potential, n, p) combinations
        # Round to reduce numerical noise
        states = {}
        for i, (psi, n, p) in enumerate(zip(self.potential, self.electron_conc, self.hole_conc)):
            key = (round(psi, 5), round(np.log10(max(n, 1)), 1), round(np.log10(max(p, 1)), 1))
            if key not in states:
                states[key] = {'psi': psi, 'n': n, 'p': p, 'count': 0}
            states[key]['count'] += 1

        # Sort states by electron concentration (high n = surface, low n = substrate)
        sorted_states = sorted(states.values(), key=lambda s: -s['n'])

        # Take up to ny states (one per Y level)
        profile_psi = []
        profile_n = []
        profile_p = []
        for state in sorted_states[:ny]:
            profile_psi.append(state['psi'])
            profile_n.append(state['n'])
            profile_p.append(state['p'])

        # Pad if needed
        while len(profile_psi) < ny:
            profile_psi.append(profile_psi[-1] if profile_psi else 0)
            profile_n.append(profile_n[-1] if profile_n else 0)
            profile_p.append(profile_p[-1] if profile_p else 0)

        # Select the requested variable
        var_map = {
            'potential': profile_psi,
            'electron': profile_n,
            'hole': profile_p,
        }

        if variable not in var_map:
            raise ValueError(f"Unknown variable: {variable}")

        return y_positions, np.array(var_map[variable])

    def to_csv(self, variable: str = 'potential',
               use_1d_profile: bool = True) -> str:
        """
        Export 1D profile data as CSV string.

        Parameters
        ----------
        variable : str
            Variable to export: 'potential', 'electron', 'hole'
        use_1d_profile : bool
            If True (default), uses get_1d_profile for physically-ordered output.
            If False, uses raw get_line_cut along Y.

        Returns
        -------
        str
            CSV formatted string with Position (um), Variable (unit)
        """
        if use_1d_profile:
            coords, values = self.get_1d_profile(variable)
        else:
            coords, values = self.get_line_cut(variable, 'y', index=0)

        labels = {
            'potential': ('Electrostatic Potential', 'V'),
            'electron': ('Electron Concentration', '/cm³'),
            'hole': ('Hole Concentration', '/cm³'),
            'doping': ('Net Doping', '/cm³'),
        }
        label, unit = labels.get(variable, (variable, ''))

        lines = [f"Position (um), {label} ({unit})"]
        for pos, val in zip(coords, values):
            lines.append(f"{pos},      {val}")

        return '\n'.join(lines)

    # -------------------------------------------------------------------
    # Plotting methods
    # -------------------------------------------------------------------

    def plot_2d(
        self,
        variable: str = 'potential',
        title: Optional[str] = None,
        cmap: str = 'viridis',
        log_scale: bool = False,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> Any:
        """
        Plot 2D contour of a variable.

        Parameters
        ----------
        variable : str
            Variable to plot: 'potential', 'electron', 'hole', 'doping'
        title : str, optional
            Plot title
        cmap : str
            Colormap name
        log_scale : bool
            Use logarithmic scale for color
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional arguments for the plotting function

        Returns
        -------
        Any
            Plot object
        """
        data = self.get_2d_data(variable)

        # Set up coordinates using actual mesh data
        x = self.mesh.get_x_positions()
        y = self.mesh.get_y_positions()

        # Variable labels
        labels = {
            'potential': ('Electrostatic Potential', 'V'),
            'electron': ('Electron Concentration', '/cm³'),
            'hole': ('Hole Concentration', '/cm³'),
            'doping': ('Net Doping', '/cm³'),
        }
        label, unit = labels.get(variable, (variable, ''))

        if title is None:
            title = f"{label}"
            if self.filename:
                title += f" - {os.path.basename(self.filename)}"

        if backend is None:
            backend = _get_default_backend()

        if log_scale and variable in ['electron', 'hole']:
            data = np.abs(data)
            data = np.where(data > 0, data, 1e-10)

        if backend == 'matplotlib':
            return self._plot_2d_matplotlib(
                x, y, data, title, cmap, log_scale, label, unit, show, **kwargs
            )
        else:
            return self._plot_2d_plotly(
                x, y, data, title, cmap, log_scale, label, unit, show, **kwargs
            )

    def _plot_2d_matplotlib(
        self, x, y, data, title, cmap, log_scale, label, unit, show, **kwargs
    ) -> Any:
        """Plot 2D using matplotlib."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))

        if log_scale:
            norm = LogNorm(vmin=data[data > 0].min(), vmax=data.max())
            im = ax.pcolormesh(x, y, data, cmap=cmap, norm=norm, shading='auto')
        else:
            im = ax.pcolormesh(x, y, data, cmap=cmap, shading='auto')

        cbar = plt.colorbar(im, ax=ax, label=f'{label} ({unit})')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(title)
        ax.set_aspect('equal')

        if show:
            plt.show()

        return ax

    def _plot_2d_plotly(
        self, x, y, data, title, cmap, log_scale, label, unit, show, **kwargs
    ) -> Any:
        """Plot 2D using plotly."""
        import plotly.graph_objects as go

        if log_scale:
            data = np.log10(np.abs(data) + 1e-30)
            label = f'log₁₀({label})'

        fig = go.Figure(data=go.Heatmap(
            x=x,
            y=y,
            z=data,
            colorscale=cmap,
            colorbar=dict(title=f'{label} ({unit})')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='X (μm)',
            yaxis_title='Y (μm)',
            width=kwargs.get('width', 700),
            height=kwargs.get('height', 600),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        if show:
            fig.show()

        return fig

    def plot_line(
        self,
        variable: str = 'potential',
        direction: str = 'y',
        position: Optional[float] = None,
        use_1d_profile: bool = True,
        title: Optional[str] = None,
        log_scale: bool = False,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> Any:
        """
        Plot 1D line cut of a variable.

        Parameters
        ----------
        variable : str
            Variable to plot: 'potential', 'electron', 'hole'
        direction : str
            Direction of cut: 'x' or 'y' (ignored if use_1d_profile=True)
        position : float, optional
            Position of the cut in microns (ignored if use_1d_profile=True)
        use_1d_profile : bool
            If True (default), uses get_1d_profile for physically-ordered data.
            Set to False for raw data extraction.
        title : str, optional
            Plot title
        log_scale : bool
            Use logarithmic scale
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional arguments

        Returns
        -------
        Any
            Plot object
        """
        if use_1d_profile and variable in ['potential', 'electron', 'hole']:
            coords, values = self.get_1d_profile(variable)
            direction = 'y'  # 1D profile is always along Y
            # Fallback to line cut if 1d profile is empty
            if len(coords) == 0 or len(values) == 0:
                coords, values = self.get_line_cut(variable, direction, position)
        else:
            coords, values = self.get_line_cut(variable, direction, position)

        # Check if we have valid data
        if len(coords) == 0 or len(values) == 0:
            raise ValueError(
                f"No {variable} data available for plotting. "
                "Ensure the solution file was parsed correctly."
            )

        labels = {
            'potential': ('Electrostatic Potential', 'V'),
            'electron': ('Electron Concentration', '/cm³'),
            'hole': ('Hole Concentration', '/cm³'),
            'doping': ('Net Doping', '/cm³'),
        }
        label, unit = labels.get(variable, (variable, ''))

        if title is None:
            title = f"{label} - {direction.upper()} cut"

        if backend is None:
            backend = _get_default_backend()

        if backend == 'matplotlib':
            return self._plot_line_matplotlib(
                coords, values, direction, title, label, unit, log_scale, show, **kwargs
            )
        else:
            return self._plot_line_plotly(
                coords, values, direction, title, label, unit, log_scale, show, **kwargs
            )

    def _plot_line_matplotlib(
        self, coords, values, direction, title, label, unit, log_scale, show, **kwargs
    ) -> Any:
        """Plot line cut using matplotlib."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))

        if log_scale:
            ax.semilogy(coords, np.abs(values), 'b-', linewidth=2)
        else:
            ax.plot(coords, values, 'b-', linewidth=2)

        xlabel = 'Y (μm)' if direction == 'y' else 'X (μm)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f'{label} ({unit})')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return ax

    def _plot_line_plotly(
        self, coords, values, direction, title, label, unit, log_scale, show, **kwargs
    ) -> Any:
        """Plot line cut using plotly."""
        import plotly.graph_objects as go

        fig = go.Figure()

        plot_values = np.abs(values) if log_scale else values

        fig.add_trace(go.Scatter(
            x=coords,
            y=plot_values,
            mode='lines',
            line=dict(width=2)
        ))

        xlabel = 'Y (μm)' if direction == 'y' else 'X (μm)'
        yaxis_type = 'log' if log_scale else 'linear'

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=f'{label} ({unit})',
            yaxis_type=yaxis_type,
            width=kwargs.get('width', 700),
            height=kwargs.get('height', 500),
            template='plotly_white'
        )

        if show:
            fig.show()

        return fig


class SolutionFileParser:
    """
    Parser for PADRE solution files.

    PADRE solution files are ASCII text files with the following structure:
    - Header section (first ~16 lines): ASCII codes and mesh dimensions
    - Line with mesh info: nx ny num_nodes num_edges 2
    - Line with electrode info: electrode_num voltage temperature
    - Line with 0.0 values
    - Node data: 2 lines per node
      - Line 1: potential, electron_conc, hole_conc
      - Line 2: 3 additional values (Jacobian-related)
    - Trailing data: electrode currents and other info
    """

    def __init__(self):
        self.data = SolutionData()

    def parse(self, filename: str) -> SolutionData:
        """
        Parse a PADRE solution file.

        Parameters
        ----------
        filename : str
            Path to the solution file

        Returns
        -------
        SolutionData
            Parsed solution data
        """
        self.data = SolutionData()
        self.data.filename = filename

        with open(filename, 'r') as f:
            lines = f.readlines()

        # Solution file format:
        # Lines 0-6: ASCII header codes
        # Line 7+: "35  0.0" or similar
        # Line 15: "680 1248 2" - num_nodes, num_edges, 2
        # Line 16: "1  0.0  300.0" - electrode info
        # Line 17: "0.0  0.0" - zeros
        # Line 18+: node data (2 lines per node)

        # Find the line with num_nodes
        num_nodes = 0
        header_end = 0
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 1:
                try:
                    v0 = int(float(parts[0]))
                    # Look for num_nodes (should be > 100 for typical meshes)
                    if v0 > 100 and v0 < 100000:
                        # This is likely the num_nodes line
                        num_nodes = v0
                        header_end = i + 1
                        break
                except (ValueError, OverflowError):
                    continue

        if num_nodes == 0:
            return self.data

        # Infer mesh dimensions - common cases
        # Look for factors of num_nodes that make sense
        for ny in range(2, 100):
            if num_nodes % ny == 0:
                nx = num_nodes // ny
                if 2 <= nx <= 200 and 2 <= ny <= 200:
                    # Prefer wider than tall (nx > ny for typical devices)
                    if nx >= ny:
                        self.data.mesh.nx = nx
                        self.data.mesh.ny = ny
                        break

        if self.data.mesh.nx == 0:
            # Fallback
            self.data.mesh.nx = num_nodes
            self.data.mesh.ny = 1

        # Skip electrode info lines to find data start
        data_start = header_end
        for i in range(header_end, min(header_end + 5, len(lines))):
            parts = lines[i].split()
            if len(parts) >= 2:
                try:
                    v0 = float(parts[0])
                    v1 = float(parts[1])
                    # Skip electrode info line (e.g., "1  0.0  300.0")
                    if abs(v0) < 10 and abs(v0 - int(v0)) < 0.01:
                        data_start = i + 1
                        continue
                    # Skip zeros line
                    if all(abs(float(p)) < 1e-10 for p in parts):
                        data_start = i + 1
                        continue
                    # Found data line
                    break
                except ValueError:
                    data_start = i + 1
                    continue

        # Parse node data: potential, n, p on data lines, skip Jacobian lines
        # PADRE solution format per node:
        #   Line 1: potential, electron_conc, hole_conc (n, p are large: 1e6 to 1e19)
        #   Line 2: three small values ~0.025 (mesh spacing related, skip these)
        potentials = []
        electrons = []
        holes = []

        node_count = 0
        i = data_start
        while i < len(lines) and node_count < num_nodes:
            parts = lines[i].split()
            if len(parts) >= 3:
                try:
                    psi = float(parts[0])
                    n = float(parts[1])
                    p = float(parts[2])

                    # Valid node data line criteria:
                    # - Potential is reasonable (< 50V)
                    # - n and p are carrier concentrations (typically > 1e5 /cm³)
                    # - Not a Jacobian line (those have small repeated values < 1)
                    is_jacobian_line = (abs(psi) < 1 and abs(n) < 1 and abs(p) < 1)
                    is_valid_node = (abs(psi) < 50 and n > 1e3 and p > 1)

                    if is_valid_node and not is_jacobian_line:
                        potentials.append(psi)
                        electrons.append(n)
                        holes.append(p)
                        node_count += 1
                        # Skip the next line (Jacobian data)
                        i += 2
                        continue
                except ValueError:
                    pass
            i += 1

        if len(potentials) == num_nodes:
            self.data.potential = np.array(potentials)
            self.data.electron_conc = np.array(electrons)
            self.data.hole_conc = np.array(holes)

        return self.data

    def parse_mesh_file(self, filename: str,
                        x_extent: float = 2.0,
                        y_extent: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse a PADRE mesh file (.pg) to extract mesh dimensions and coordinates.

        Parameters
        ----------
        filename : str
            Path to the mesh file
        x_extent : float
            Device width in microns (default: 2.0)
        y_extent : float
            Device depth in microns (default: 2.0)

        Returns
        -------
        tuple
            (x_coords, y_coords) arrays in microns
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Find mesh dimensions: line with nx ny num_nodes (where nx*ny == num_nodes)
        nx, ny = 0, 0
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    v1 = int(float(parts[0]))
                    v2 = int(float(parts[1]))
                    v3 = int(float(parts[2]))
                    if v1 > 1 and v2 > 1 and v1 * v2 == v3:
                        nx, ny = v1, v2
                        break
                except (ValueError, OverflowError):
                    continue

        if nx == 0:
            return np.array([]), np.array([])

        # Update mesh dimensions
        self.data.mesh.nx = nx
        self.data.mesh.ny = ny

        # Parse mesh file to extract actual coordinate values
        # Data starts after line with just "1" (region marker)
        data_start = 0
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) == 1 and parts[0] == '1':
                data_start = i + 1
                break

        # Collect all numeric values from data section
        all_values = []
        for line in lines[data_start:]:
            for part in line.split():
                try:
                    all_values.append(float(part))
                except ValueError:
                    continue

        # Extract coordinate-like values (in cm, convert to µm)
        # Coordinates are small positive values (0 to ~0.0002 cm = 0 to 2 µm)
        coord_candidates = sorted(set(round(v * 1e4, 6) for v in all_values
                                       if 0 <= v < 0.00025))

        # Strategy: X coordinates are uniform, Y may not be
        # First, identify exact uniform X values in candidates
        dx = x_extent / (nx - 1) if nx > 1 else x_extent
        x_uniform = set(round(i * dx, 6) for i in range(nx))

        # Candidates matching X exactly (within floating point tolerance)
        x_matched = set()
        for c in coord_candidates:
            for x in x_uniform:
                if abs(c - x) < 1e-5:  # Very tight tolerance for exact match
                    x_matched.add(x)
                    break

        # Remaining candidates are Y coordinates
        y_from_file = []
        for c in coord_candidates:
            is_x_exact = any(abs(c - x) < 1e-5 for x in x_uniform)
            if not is_x_exact and c <= y_extent:
                y_from_file.append(c)

        # Use uniform X (we know it's uniform from PADRE spec)
        x_vals = np.linspace(0, x_extent, nx)

        # For Y, use extracted values plus boundaries
        y_from_file = sorted(set(y_from_file))
        if len(y_from_file) >= ny - 3:
            # Add boundary values
            y_vals = list(y_from_file)
            if 0.0 not in y_vals:
                y_vals.insert(0, 0.0)
            if y_extent not in y_vals:
                y_vals.append(y_extent)
            y_vals = sorted(y_vals)
            # If we have exactly ny values, use them
            if len(y_vals) == ny:
                y_vals = np.array(y_vals)
            else:
                # Fall back to uniform
                y_vals = np.linspace(0, y_extent, ny)
        else:
            y_vals = np.linspace(0, y_extent, ny)

        self.data.mesh.x_unique = x_vals
        self.data.mesh.y_unique = y_vals

        # Generate full coordinate arrays
        # PADRE data is stored in column-major order (y varies fastest within each x)
        num_nodes = nx * ny
        x_coords = np.zeros(num_nodes)
        y_coords = np.zeros(num_nodes)
        for ix in range(nx):
            for iy in range(ny):
                idx = ix * ny + iy
                x_i = x_vals[ix] if ix < len(x_vals) else x_extent * ix / (nx - 1)
                y_i = y_vals[iy] if iy < len(y_vals) else y_extent * iy / (ny - 1)
                x_coords[idx] = x_i
                y_coords[idx] = y_i

        return x_coords, y_coords


def _get_default_backend() -> str:
    """Get the default plotting backend."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return 'matplotlib'
    except ImportError:
        pass
    try:
        import plotly.graph_objects as go  # noqa: F401
        return 'plotly'
    except ImportError:
        pass
    raise ImportError(
        "No plotting backend available. Install matplotlib or plotly."
    )


def parse_solution_file(filename: str, mesh_file: Optional[str] = None) -> SolutionData:
    """
    Parse a PADRE solution file.

    Parameters
    ----------
    filename : str
        Path to the solution file (e.g., 'pn_eq', 'pn_fwd_a')
    mesh_file : str, optional
        Path to the mesh file (.pg). If not provided, looks for common mesh
        file names in the same directory.

    Returns
    -------
    SolutionData
        Parsed solution with methods for visualization

    Example
    -------
    >>> sol = parse_solution_file('outputs/pn_eq')
    >>> sol.plot_2d('potential')
    >>> sol.plot_line('electron', direction='y', log_scale=True)
    >>> sol.plot_band_diagram()
    >>> print(sol.to_csv('potential'))  # Export to CSV format
    """
    parser = SolutionFileParser()
    solution = parser.parse(filename)

    # Try to load mesh coordinates from mesh file
    if mesh_file is None:
        # Look for mesh file in the same directory
        directory = os.path.dirname(filename)
        if not directory:
            directory = '.'

        # Common mesh file patterns
        mesh_patterns = ['*_mesh.pg', '*.pg', 'mesh.pg']
        import glob
        for pattern in mesh_patterns:
            matches = glob.glob(os.path.join(directory, pattern))
            if matches:
                mesh_file = matches[0]
                break

    if mesh_file and os.path.exists(mesh_file):
        x_coords, y_coords = parser.parse_mesh_file(mesh_file)
        if len(x_coords) == solution.mesh.num_nodes:
            solution.mesh.x = x_coords
            solution.mesh.y = y_coords
            # Extract unique coordinates
            solution.mesh.x_unique = np.unique(x_coords)
            solution.mesh.y_unique = np.unique(y_coords)

    return solution


def load_solution_series(
    directory: str,
    pattern: str = "pn_fwd_*"
) -> List[SolutionData]:
    """
    Load a series of solution files.

    Parameters
    ----------
    directory : str
        Directory containing solution files
    pattern : str
        Glob pattern to match files

    Returns
    -------
    List[SolutionData]
        List of parsed solutions in sorted order

    Example
    -------
    >>> solutions = load_solution_series('outputs/', 'pn_fwd_*')
    >>> for sol in solutions:
    ...     sol.plot_2d('potential', show=False)
    """
    import glob

    files = sorted(glob.glob(os.path.join(directory, pattern)))
    solutions = []

    for f in files:
        try:
            sol = parse_solution_file(f)
            solutions.append(sol)
        except Exception:
            pass

    return solutions
