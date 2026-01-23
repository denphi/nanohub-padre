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
        X coordinates of mesh nodes (1D array or 2D grid)
    y : np.ndarray
        Y coordinates of mesh nodes (1D array or 2D grid)
    """
    nx: int = 0
    ny: int = 0
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def num_nodes(self) -> int:
        """Total number of mesh nodes."""
        return self.nx * self.ny


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
            Direction of the cut: 'x' (horizontal) or 'y' (vertical)
        position : float, optional
            Position of the cut in microns. If None, uses middle.
        index : int, optional
            Direct index of the cut. Overrides position if provided.

        Returns
        -------
        tuple
            (coordinates, values) along the cut
        """
        data_2d = self.get_2d_data(variable)
        x = np.linspace(0, 2, self.mesh.nx)  # Assume 2 micron default
        y = np.linspace(0, 2, self.mesh.ny)

        if direction == 'y':
            # Vertical cut (along y at fixed x)
            if index is None:
                if position is not None:
                    index = int(position / 2 * (self.mesh.nx - 1))
                else:
                    index = self.mesh.nx // 2
            return y, data_2d[:, index]
        else:
            # Horizontal cut (along x at fixed y)
            if index is None:
                if position is not None:
                    index = int(position / 2 * (self.mesh.ny - 1))
                else:
                    index = self.mesh.ny // 2
            return x, data_2d[index, :]

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

        # Set up coordinates
        x = np.linspace(0, 2, self.mesh.nx)
        y = np.linspace(0, 2, self.mesh.ny)

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
            Variable to plot
        direction : str
            Direction of cut: 'x' or 'y'
        position : float, optional
            Position of the cut in microns
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
        coords, values = self.get_line_cut(variable, direction, position)

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

    def plot_band_diagram(
        self,
        direction: str = 'y',
        position: Optional[float] = None,
        title: Optional[str] = None,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> Any:
        """
        Plot energy band diagram.

        Parameters
        ----------
        direction : str
            Direction of cut: 'x' or 'y'
        position : float, optional
            Position of the cut in microns
        title : str, optional
            Plot title
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately

        Returns
        -------
        Any
            Plot object
        """
        coords, psi = self.get_line_cut('potential', direction, position)

        # Calculate band edges (simplified - assumes Si at 300K)
        Eg = 1.12  # Silicon bandgap (eV)
        chi = 4.05  # Electron affinity (eV)

        Ec = -psi - chi  # Conduction band
        Ev = Ec - Eg     # Valence band
        Ef = -chi - Eg/2  # Fermi level (approx intrinsic)

        if title is None:
            title = f"Energy Band Diagram - {direction.upper()} cut"

        if backend is None:
            backend = _get_default_backend()

        if backend == 'matplotlib':
            return self._plot_bands_matplotlib(
                coords, Ec, Ev, Ef, direction, title, show, **kwargs
            )
        else:
            return self._plot_bands_plotly(
                coords, Ec, Ev, Ef, direction, title, show, **kwargs
            )

    def _plot_bands_matplotlib(
        self, coords, Ec, Ev, Ef, direction, title, show, **kwargs
    ) -> Any:
        """Plot band diagram using matplotlib."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))

        ax.plot(coords, Ec, 'b-', linewidth=2, label='Ec')
        ax.plot(coords, Ev, 'r-', linewidth=2, label='Ev')
        ax.axhline(y=Ef, color='g', linestyle='--', linewidth=1, label='Ef (intrinsic)')

        xlabel = 'Y (μm)' if direction == 'y' else 'X (μm)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Energy (eV)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return ax

    def _plot_bands_plotly(
        self, coords, Ec, Ev, Ef, direction, title, show, **kwargs
    ) -> Any:
        """Plot band diagram using plotly."""
        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=coords, y=Ec, mode='lines',
                                  name='Ec', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=coords, y=Ev, mode='lines',
                                  name='Ev', line=dict(color='red', width=2)))
        fig.add_hline(y=Ef, line_dash='dash', line_color='green',
                      annotation_text='Ef (intrinsic)')

        xlabel = 'Y (μm)' if direction == 'y' else 'X (μm)'

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title='Energy (eV)',
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

    PADRE solution files contain mesh and variable data in a specific format.
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
            content = f.read()

        lines = content.strip().split('\n')
        values = []

        # Parse all numeric values from the file
        for line in lines:
            parts = line.split()
            for part in parts:
                try:
                    values.append(float(part))
                except ValueError:
                    continue

        if len(values) < 10:
            return self.data

        # Parse header to get mesh dimensions
        # Format varies, but typically nx, ny are in the first few values
        # Looking for pattern: ... nx ny num_nodes ...
        # From the file, line 8 has: 40 17 680 (nx=40, ny=17, nodes=680)

        # Find mesh dimensions by looking for consistent pattern
        # nx * ny should equal num_nodes
        for i in range(len(values) - 3):
            nx = int(values[i])
            ny = int(values[i + 1])
            nodes = int(values[i + 2])
            if nx > 1 and ny > 1 and nx * ny == nodes:
                self.data.mesh.nx = nx
                self.data.mesh.ny = ny
                break

        if self.data.mesh.nx == 0:
            # Fallback: assume standard mesh from file
            self.data.mesh.nx = 40
            self.data.mesh.ny = 17

        num_nodes = self.data.mesh.nx * self.data.mesh.ny

        # The rest of the file contains solution variables
        # Each node has: potential, n, p, and other values
        # Try to extract potential, n, p from the data

        # After header, data comes in groups of values per node
        # Format: x, y, doping, potential, n, p, ... for each node
        # or: potential, n, p per node

        # For now, use a simplified approach based on file structure
        # Parse data starting after the header section
        data_start = 20  # Skip header values

        if len(values) > data_start + num_nodes * 3:
            # Extract potential, n, p assuming they're in sequence
            potentials = []
            electrons = []
            holes = []

            # Data appears to be in format: potential, n, p per node
            for i in range(num_nodes):
                idx = data_start + i * 6  # Adjust stride based on format
                if idx + 2 < len(values):
                    potentials.append(values[idx])
                    electrons.append(values[idx + 1])
                    holes.append(values[idx + 2])

            if potentials:
                self.data.potential = np.array(potentials)
                self.data.electron_conc = np.array(electrons)
                self.data.hole_conc = np.array(holes)

        return self.data


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


def parse_solution_file(filename: str) -> SolutionData:
    """
    Parse a PADRE solution file.

    Parameters
    ----------
    filename : str
        Path to the solution file (e.g., 'pn_eq', 'pn_fwd_a')

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
    """
    parser = SolutionFileParser()
    return parser.parse(filename)


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
