"""
Output management for PADRE simulations.

Automatically tracks, parses, and provides access to simulation outputs
based on the commands in the simulation configuration.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum


class OutputType(Enum):
    """Types of PADRE output data."""
    MESH = "mesh"
    SOLUTION = "solution"
    IV_DATA = "iv_data"
    AC_DATA = "ac_data"
    PLOT_1D = "plot_1d"
    PLOT_2D = "plot_2d"


# Variable groups for composite visualizations
VARIABLE_GROUPS = {
    'band_diagram': ['band_val', 'band_con', 'qfn', 'qfp'],
    'carriers': ['electrons', 'holes'],
    'currents': ['j_electr', 'j_hole', 'j_total'],
    'quasi_fermi': ['qfn', 'qfp'],
    'bands': ['band_val', 'band_con'],
}


@dataclass
class PlotData:
    """
    Parsed 1D plot data from PADRE ASCII output.

    Attributes
    ----------
    name : str
        Output file name (without path)
    variable : str
        Variable being plotted (e.g., 'potential', 'electrons', 'doping')
    x : np.ndarray
        X coordinates (position in microns)
    y : np.ndarray
        Y values (the plotted quantity)
    x_label : str
        X axis label
    y_label : str
        Y axis label with units
    logarithmic : bool
        Whether data is logarithmic
    absolute : bool
        Whether absolute values were taken
    line_spec : dict
        Line specification (x_start, y_start, x_end, y_end)
    """
    name: str = ""
    variable: str = ""
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    x_label: str = "Position (μm)"
    y_label: str = ""
    logarithmic: bool = False
    absolute: bool = False
    line_spec: dict = field(default_factory=dict)

    def plot(self, backend: Optional[str] = None, show: bool = True,
             title: Optional[str] = None, **kwargs) -> Any:
        """
        Plot the data.

        Parameters
        ----------
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        title : str, optional
            Plot title (default: uses variable name)
        **kwargs
            Additional plot arguments

        Returns
        -------
        Any
            Plot object
        """
        if backend is None:
            backend = _get_default_backend()

        if title is None:
            title = f"{self.variable} vs Position"

        if backend == 'matplotlib':
            return self._plot_matplotlib(title, show, **kwargs)
        else:
            return self._plot_plotly(title, show, **kwargs)

    def _plot_matplotlib(self, title: str, show: bool, **kwargs) -> Any:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))

        ax.plot(self.x, self.y, 'b-', linewidth=2)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label if self.y_label else self.variable)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if self.logarithmic:
            ax.set_yscale('log')

        if show:
            plt.show()

        return ax

    def _plot_plotly(self, title: str, show: bool, **kwargs) -> Any:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.x, y=self.y, mode='lines',
                                  line=dict(color='blue', width=2)))

        yaxis_type = 'log' if self.logarithmic else 'linear'

        fig.update_layout(
            title=title,
            xaxis_title=self.x_label,
            yaxis_title=self.y_label if self.y_label else self.variable,
            yaxis_type=yaxis_type,
            width=kwargs.get('width', 700),
            height=kwargs.get('height', 500),
            template='plotly_white'
        )

        if show:
            fig.show()

        return fig


@dataclass
class OutputEntry:
    """
    A single output entry tracking a command and its output.

    Attributes
    ----------
    name : str
        Output name (outfile parameter)
    output_type : OutputType
        Type of output
    command : Any
        The command that generates this output
    variable : str
        Variable being output (for plots)
    data : Any
        Parsed data (populated after running)
    file_path : str
        Full path to output file (populated after running)
    """
    name: str
    output_type: OutputType
    command: Any = None
    variable: str = ""
    data: Any = None
    file_path: str = ""


class OutputManager:
    """
    Manages simulation outputs - tracks expected outputs and provides access to data.

    The OutputManager analyzes the simulation commands to determine what outputs
    will be generated, then after running provides methods to access the parsed
    data directly without needing to specify file paths.

    Example
    -------
    >>> sim = Simulation(title="PN Diode")
    >>> # ... configure simulation ...
    >>> sim.add_command(Plot1D(potential=True, outfile="pot", ...))
    >>> sim.add_command(Plot1D(electrons=True, outfile="ele", ...))
    >>> sim.add_log(Log(ivfile="iv"))
    >>>
    >>> # Run simulation
    >>> result = sim.run()
    >>>
    >>> # Access outputs directly
    >>> sim.outputs.list()  # Shows all outputs
    >>> pot_data = sim.outputs.get("pot")  # Get parsed data
    >>> pot_data.plot()  # Plot directly
    >>>
    >>> # Or access by variable type
    >>> sim.outputs.get_by_variable("potential")
    """

    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir
        self._entries: Dict[str, OutputEntry] = {}
        self._by_type: Dict[OutputType, List[str]] = {t: [] for t in OutputType}
        self._by_variable: Dict[str, List[str]] = {}

    def register(self, name: str, output_type: OutputType,
                 command: Any = None, variable: str = "") -> None:
        """Register an expected output."""
        entry = OutputEntry(
            name=name,
            output_type=output_type,
            command=command,
            variable=variable
        )
        self._entries[name] = entry
        self._by_type[output_type].append(name)

        if variable:
            if variable not in self._by_variable:
                self._by_variable[variable] = []
            self._by_variable[variable].append(name)

    def list(self, output_type: Optional[OutputType] = None) -> List[str]:
        """
        List all registered output names.

        Parameters
        ----------
        output_type : OutputType, optional
            Filter by type

        Returns
        -------
        List[str]
            List of output names
        """
        if output_type:
            return list(self._by_type.get(output_type, []))
        return list(self._entries.keys())

    def list_by_variable(self) -> Dict[str, List[str]]:
        """
        List outputs grouped by variable.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping variable names to output names
        """
        return dict(self._by_variable)

    def get(self, name: str) -> Optional[Any]:
        """
        Get parsed data for an output.

        Parameters
        ----------
        name : str
            Output name

        Returns
        -------
        Any
            Parsed data (PlotData, IVData, SolutionData, etc.)
            Returns None if not found or not yet loaded.
        """
        entry = self._entries.get(name)
        if entry:
            return entry.data
        return None

    def get_entry(self, name: str) -> Optional[OutputEntry]:
        """Get the full OutputEntry for an output."""
        return self._entries.get(name)

    def get_by_variable(self, variable: str) -> List[Any]:
        """
        Get all outputs for a specific variable.

        Parameters
        ----------
        variable : str
            Variable name (e.g., 'potential', 'electrons')

        Returns
        -------
        List[Any]
            List of parsed data objects
        """
        names = self._by_variable.get(variable, [])
        return [self._entries[n].data for n in names if self._entries[n].data is not None]

    def get_plots(self) -> Dict[str, PlotData]:
        """Get all 1D plot outputs as a dictionary."""
        result = {}
        for name in self._by_type[OutputType.PLOT_1D]:
            entry = self._entries[name]
            if entry.data is not None:
                result[name] = entry.data
        return result

    def get_solutions(self) -> Dict[str, Any]:
        """Get all solution outputs as a dictionary."""
        result = {}
        for name in self._by_type[OutputType.SOLUTION]:
            entry = self._entries[name]
            if entry.data is not None:
                result[name] = entry.data
        return result

    def get_iv_data(self) -> Dict[str, Any]:
        """Get all I-V data outputs as a dictionary."""
        result = {}
        for name in self._by_type[OutputType.IV_DATA]:
            entry = self._entries[name]
            if entry.data is not None:
                result[name] = entry.data
        return result

    def get_ac_data(self) -> Dict[str, Any]:
        """Get all AC data outputs as a dictionary."""
        result = {}
        for name in self._by_type.get(OutputType.AC_DATA, []):
            entry = self._entries[name]
            if entry.data is not None:
                result[name] = entry.data
        return result

    def load_all(self) -> None:
        """Load/parse all output files."""
        for name, entry in self._entries.items():
            self._load_entry(entry)

    def _load_entry(self, entry: OutputEntry) -> None:
        """Load data for a single entry."""
        file_path = os.path.join(self.working_dir, entry.name)
        entry.file_path = file_path

        if not os.path.exists(file_path):
            return

        if entry.output_type == OutputType.PLOT_1D:
            entry.data = self._parse_plot1d(file_path, entry)
        elif entry.output_type == OutputType.IV_DATA:
            entry.data = self._parse_iv_file(file_path)
        elif entry.output_type == OutputType.AC_DATA:
            entry.data = self._parse_ac_file(file_path)
        elif entry.output_type == OutputType.SOLUTION:
            entry.data = self._parse_solution(file_path)

    def _parse_plot1d(self, file_path: str, entry: OutputEntry) -> PlotData:
        """Parse a PADRE ASCII plot file."""
        x_vals = []
        y_vals = []

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Skip header lines and parse data
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('$'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    continue

        # Determine variable info from command
        variable = entry.variable
        logarithmic = False
        absolute = False
        line_spec = {}

        if entry.command:
            cmd = entry.command
            logarithmic = getattr(cmd, 'logarithm', False)
            absolute = getattr(cmd, 'absolute', False)
            line_spec = {
                'x_start': getattr(cmd, 'x_start', None),
                'y_start': getattr(cmd, 'y_start', None),
                'x_end': getattr(cmd, 'x_end', None),
                'y_end': getattr(cmd, 'y_end', None),
            }

        # Variable labels
        labels = {
            'potential': ('Electrostatic Potential', 'V'),
            'electrons': ('Electron Concentration', '/cm³'),
            'holes': ('Hole Concentration', '/cm³'),
            'doping': ('Doping Concentration', '/cm³'),
            'band_val': ('Valence Band Edge', 'eV'),
            'band_con': ('Conduction Band Edge', 'eV'),
            'qfn': ('Electron Quasi-Fermi Level', 'eV'),
            'qfp': ('Hole Quasi-Fermi Level', 'eV'),
            'e_field': ('Electric Field', 'V/cm'),
            'recomb': ('Recombination Rate', '/cm³/s'),
            'net_charge': ('Net Charge', '/cm³'),
            'j_electr': ('Electron Current Density', 'A/cm²'),
            'j_hole': ('Hole Current Density', 'A/cm²'),
            'j_total': ('Total Current Density', 'A/cm²'),
        }

        y_label = labels.get(variable, (variable, ''))[0]
        if labels.get(variable):
            y_label = f"{labels[variable][0]} ({labels[variable][1]})"

        return PlotData(
            name=entry.name,
            variable=variable,
            x=np.array(x_vals),
            y=np.array(y_vals),
            x_label="Position (μm)",
            y_label=y_label,
            logarithmic=logarithmic,
            absolute=absolute,
            line_spec=line_spec
        )

    def _parse_iv_file(self, file_path: str) -> Any:
        """Parse a PADRE I-V log file."""
        # Import here to avoid circular imports
        from .parser import parse_iv_file
        try:
            return parse_iv_file(file_path)
        except Exception:
            return None

    def _parse_ac_file(self, file_path: str) -> Any:
        """Parse a PADRE AC log file."""
        from .parser import parse_ac_file
        try:
            return parse_ac_file(file_path)
        except Exception:
            return None

    def _parse_solution(self, file_path: str) -> Any:
        """Parse a PADRE solution file."""
        from .solution import parse_solution_file
        try:
            return parse_solution_file(file_path)
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Composite plot methods
    # -----------------------------------------------------------------------

    def get_group(self, group_name: str) -> Dict[str, PlotData]:
        """
        Get all outputs belonging to a variable group.

        Parameters
        ----------
        group_name : str
            Group name: 'band_diagram', 'carriers', 'currents', 'quasi_fermi', 'bands'

        Returns
        -------
        Dict[str, PlotData]
            Dictionary mapping variable names to PlotData objects
        """
        if group_name not in VARIABLE_GROUPS:
            raise ValueError(f"Unknown group: {group_name}. "
                           f"Available: {list(VARIABLE_GROUPS.keys())}")

        result = {}
        for var in VARIABLE_GROUPS[group_name]:
            data_list = self.get_by_variable(var)
            if data_list:
                # Take the first (or most recent) output for this variable
                result[var] = data_list[-1]
        return result

    def get_band_diagram_sets(self) -> List[Dict[str, Any]]:
        """
        Get all band diagram data sets grouped by their naming pattern.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries, each containing:
            - 'name': identifier for this set (e.g., 'eq', 'iv')
            - 'band_val': PlotData for valence band (or None)
            - 'band_con': PlotData for conduction band (or None)
            - 'qfn': PlotData for electron quasi-Fermi level (or None)
            - 'qfp': PlotData for hole quasi-Fermi level (or None)
        """
        # Group band outputs by their base name pattern
        # e.g., vband/cband -> 'band', vbiv/cbiv -> 'iv'
        band_sets = {}

        for name, entry in self._entries.items():
            if entry.data is None:
                continue
            if entry.variable not in ['band_val', 'band_con', 'qfn', 'qfp']:
                continue

            # Extract the suffix/identifier from the name
            # Common patterns: vband/cband, vbiv/cbiv, vbeq/cbeq
            if entry.variable == 'band_val':
                # Remove common prefixes for valence band
                for prefix in ['vband', 'vb', 'ev', 'valence']:
                    if name.lower().startswith(prefix):
                        suffix = name[len(prefix):] or 'default'
                        break
                else:
                    suffix = name
            elif entry.variable == 'band_con':
                for prefix in ['cband', 'cb', 'ec', 'conduction']:
                    if name.lower().startswith(prefix):
                        suffix = name[len(prefix):] or 'default'
                        break
                else:
                    suffix = name
            elif entry.variable == 'qfn':
                for prefix in ['qfn', 'efn']:
                    if name.lower().startswith(prefix):
                        suffix = name[len(prefix):] or 'default'
                        break
                else:
                    suffix = name
            elif entry.variable == 'qfp':
                for prefix in ['qfp', 'efp']:
                    if name.lower().startswith(prefix):
                        suffix = name[len(prefix):] or 'default'
                        break
                else:
                    suffix = name

            if suffix not in band_sets:
                band_sets[suffix] = {
                    'name': suffix,
                    'band_val': None,
                    'band_con': None,
                    'qfn': None,
                    'qfp': None
                }

            band_sets[suffix][entry.variable] = entry.data

        # Convert to list and sort by name
        result = list(band_sets.values())
        # Sort with 'default' first, then alphabetically
        result.sort(key=lambda x: (x['name'] != 'default', x['name']))
        return result

    def plot_band_diagram(self, index: Optional[Union[int, List[int]]] = None,
                          suffix: str = "", title: Optional[str] = None,
                          backend: Optional[str] = None, show: bool = True,
                          **kwargs) -> Any:
        """
        Plot energy band diagram combining band_val and band_con outputs.

        Parameters
        ----------
        index : int or List[int], optional
            Index or list of indices of the band diagram sets to plot.
            If None, plots all sets on the same axes.
            Use get_band_diagram_sets() to see available sets.
        suffix : str
            Suffix to match output names (e.g., "iv" to match "vbiv", "cbiv").
            Ignored if index is provided.
        title : str, optional
            Plot title
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional plot arguments

        Returns
        -------
        Any
            Plot object

        Example
        -------
        >>> # Plot all band diagrams on the same axes
        >>> sim.outputs.plot_band_diagram()
        >>>
        >>> # Plot only the first set (e.g., equilibrium)
        >>> sim.outputs.plot_band_diagram(index=0)
        >>>
        >>> # Plot specific sets (e.g., equilibrium and under bias)
        >>> sim.outputs.plot_band_diagram(index=[0, 1])
        >>>
        >>> # Plot by suffix
        >>> sim.outputs.plot_band_diagram(suffix="iv")
        """
        all_band_sets = self.get_band_diagram_sets()

        if not all_band_sets:
            raise ValueError("No band data found. Need outputs with band_val or band_con.")

        # Select by index/indices if provided
        if index is not None:
            # Convert single int to list
            indices = [index] if isinstance(index, int) else list(index)

            # Validate indices
            for idx in indices:
                if idx < 0 or idx >= len(all_band_sets):
                    raise ValueError(
                        f"Index {idx} out of range. Available: 0-{len(all_band_sets)-1}"
                    )

            band_sets = [all_band_sets[i] for i in indices]
        # Filter by suffix if provided (and index not specified)
        elif suffix:
            band_sets = [s for s in all_band_sets
                         if suffix in s['name'] or s['name'] == suffix]
            if not band_sets:
                raise ValueError(f"No band data found matching suffix '{suffix}'.")
        else:
            band_sets = all_band_sets

        if backend is None:
            backend = _get_default_backend()

        if title is None:
            if len(band_sets) == 1:
                title = f"Energy Band Diagram ({band_sets[0]['name']})"
            else:
                title = "Energy Band Diagram"

        if backend == 'matplotlib':
            return self._plot_bands_matplotlib_multi(band_sets, title, show, **kwargs)
        else:
            return self._plot_bands_plotly_multi(band_sets, title, show, **kwargs)

    def _plot_bands_matplotlib_multi(self, band_sets: List[Dict], title: str,
                                       show: bool, **kwargs):
        """Plot multiple band diagram sets using matplotlib."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))

        # Color palette for multiple sets
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']

        for i, band_set in enumerate(band_sets):
            color = colors[i % len(colors)]
            label_suffix = f" ({band_set['name']})" if len(band_sets) > 1 else ""

            cband = band_set['band_con']
            vband = band_set['band_val']
            qfn = band_set['qfn']
            qfp = band_set['qfp']

            if cband is not None:
                ax.plot(cband.x, cband.y, color=color, linestyle='-',
                        linewidth=2, label=f'Ec{label_suffix}')
            if vband is not None:
                ax.plot(vband.x, vband.y, color=color, linestyle='-',
                        linewidth=2, label=f'Ev{label_suffix}',
                        alpha=0.7 if cband is not None else 1.0)
            if qfn is not None:
                ax.plot(qfn.x, qfn.y, color=color, linestyle='--',
                        linewidth=1, label=f'Efn{label_suffix}')
            if qfp is not None:
                ax.plot(qfp.x, qfp.y, color=color, linestyle=':',
                        linewidth=1, label=f'Efp{label_suffix}')

        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return ax

    def _plot_bands_plotly_multi(self, band_sets: List[Dict], title: str,
                                  show: bool, **kwargs):
        """Plot multiple band diagram sets using plotly."""
        import plotly.graph_objects as go

        fig = go.Figure()

        # Color palette for multiple sets
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']

        for i, band_set in enumerate(band_sets):
            color = colors[i % len(colors)]
            label_suffix = f" ({band_set['name']})" if len(band_sets) > 1 else ""

            cband = band_set['band_con']
            vband = band_set['band_val']
            qfn = band_set['qfn']
            qfp = band_set['qfp']

            if cband is not None:
                fig.add_trace(go.Scatter(
                    x=cband.x, y=cband.y, mode='lines',
                    name=f'Ec{label_suffix}',
                    line=dict(color=color, width=2)
                ))
            if vband is not None:
                fig.add_trace(go.Scatter(
                    x=vband.x, y=vband.y, mode='lines',
                    name=f'Ev{label_suffix}',
                    line=dict(color=color, width=2),
                    opacity=0.7 if cband is not None else 1.0
                ))
            if qfn is not None:
                fig.add_trace(go.Scatter(
                    x=qfn.x, y=qfn.y, mode='lines',
                    name=f'Efn{label_suffix}',
                    line=dict(color=color, width=1, dash='dash')
                ))
            if qfp is not None:
                fig.add_trace(go.Scatter(
                    x=qfp.x, y=qfp.y, mode='lines',
                    name=f'Efp{label_suffix}',
                    line=dict(color=color, width=1, dash='dot')
                ))

        fig.update_layout(
            title=title,
            xaxis_title='Position (μm)',
            yaxis_title='Energy (eV)',
            width=kwargs.get('width', 700),
            height=kwargs.get('height', 500),
            template='plotly_white'
        )

        if show:
            fig.show()

        return fig

    def plot_carriers(self, suffix: str = "", title: Optional[str] = None,
                      log_scale: bool = True, backend: Optional[str] = None,
                      show: bool = True, **kwargs) -> Any:
        """
        Plot electron and hole concentration profiles.

        Parameters
        ----------
        suffix : str
            Suffix to match output names
        title : str, optional
            Plot title
        log_scale : bool
            Use logarithmic y-axis (default True)
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately

        Returns
        -------
        Any
            Plot object
        """
        ele_data = None
        hole_data = None

        for name, entry in self._entries.items():
            if entry.data is None:
                continue
            if suffix and not name.endswith(suffix) and suffix not in name:
                continue

            if entry.variable == 'electrons':
                ele_data = entry.data
            elif entry.variable == 'holes':
                hole_data = entry.data

        if ele_data is None and hole_data is None:
            raise ValueError("No carrier data found. Need outputs with electrons or holes.")

        if backend is None:
            backend = _get_default_backend()

        if title is None:
            title = "Carrier Concentration"
            if suffix:
                title += f" ({suffix})"

        if backend == 'matplotlib':
            return self._plot_carriers_matplotlib(
                ele_data, hole_data, title, log_scale, show, **kwargs
            )
        else:
            return self._plot_carriers_plotly(
                ele_data, hole_data, title, log_scale, show, **kwargs
            )

    def _plot_carriers_matplotlib(self, ele, hole, title, log_scale, show, **kwargs):
        """Plot carrier profiles using matplotlib."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))

        if ele is not None:
            ax.plot(ele.x, ele.y, 'b-', linewidth=2, label='Electrons')
        if hole is not None:
            ax.plot(hole.x, hole.y, 'r-', linewidth=2, label='Holes')

        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Concentration (/cm³)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale('log')

        if show:
            plt.show()

        return ax

    def _plot_carriers_plotly(self, ele, hole, title, log_scale, show, **kwargs):
        """Plot carrier profiles using plotly."""
        import plotly.graph_objects as go

        fig = go.Figure()

        if ele is not None:
            fig.add_trace(go.Scatter(x=ele.x, y=ele.y, mode='lines',
                                      name='Electrons', line=dict(color='blue', width=2)))
        if hole is not None:
            fig.add_trace(go.Scatter(x=hole.x, y=hole.y, mode='lines',
                                      name='Holes', line=dict(color='red', width=2)))

        yaxis_type = 'log' if log_scale else 'linear'

        fig.update_layout(
            title=title,
            xaxis_title='Position (μm)',
            yaxis_title='Concentration (/cm³)',
            yaxis_type=yaxis_type,
            width=kwargs.get('width', 700),
            height=kwargs.get('height', 500),
            template='plotly_white'
        )

        if show:
            fig.show()

        return fig

    def plot_currents(self, suffix: str = "", title: Optional[str] = None,
                      backend: Optional[str] = None, show: bool = True,
                      **kwargs) -> Any:
        """
        Plot current density profiles (electron, hole, total).

        Parameters
        ----------
        suffix : str
            Suffix to match output names
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
        j_ele = None
        j_hole = None
        j_total = None

        for name, entry in self._entries.items():
            if entry.data is None:
                continue
            if suffix and not name.endswith(suffix) and suffix not in name:
                continue

            if entry.variable == 'j_electr':
                j_ele = entry.data
            elif entry.variable == 'j_hole':
                j_hole = entry.data
            elif entry.variable == 'j_total':
                j_total = entry.data

        if j_ele is None and j_hole is None and j_total is None:
            raise ValueError("No current data found.")

        if backend is None:
            backend = _get_default_backend()

        if title is None:
            title = "Current Density"
            if suffix:
                title += f" ({suffix})"

        if backend == 'matplotlib':
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))

            if j_ele is not None:
                ax.plot(j_ele.x, j_ele.y, 'b-', linewidth=2, label='Jn')
            if j_hole is not None:
                ax.plot(j_hole.x, j_hole.y, 'r-', linewidth=2, label='Jp')
            if j_total is not None:
                ax.plot(j_total.x, j_total.y, 'k-', linewidth=2, label='Jtotal')

            ax.set_xlabel('Position (μm)')
            ax.set_ylabel('Current Density (A/cm²)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if show:
                plt.show()
            return ax
        else:
            import plotly.graph_objects as go
            fig = go.Figure()

            if j_ele is not None:
                fig.add_trace(go.Scatter(x=j_ele.x, y=j_ele.y, mode='lines',
                                          name='Jn', line=dict(color='blue', width=2)))
            if j_hole is not None:
                fig.add_trace(go.Scatter(x=j_hole.x, y=j_hole.y, mode='lines',
                                          name='Jp', line=dict(color='red', width=2)))
            if j_total is not None:
                fig.add_trace(go.Scatter(x=j_total.x, y=j_total.y, mode='lines',
                                          name='Jtotal', line=dict(color='black', width=2)))

            fig.update_layout(
                title=title,
                xaxis_title='Position (μm)',
                yaxis_title='Current Density (A/cm²)',
                width=kwargs.get('width', 700),
                height=kwargs.get('height', 500),
                template='plotly_white'
            )

            if show:
                fig.show()
            return fig

    def plot_multiple(self, variables: List[str], suffix: str = "",
                      title: Optional[str] = None, log_scale: bool = False,
                      backend: Optional[str] = None, show: bool = True,
                      **kwargs) -> Any:
        """
        Plot multiple variables on the same axes.

        Parameters
        ----------
        variables : List[str]
            List of variable names to plot
        suffix : str
            Suffix to match output names
        title : str, optional
            Plot title
        log_scale : bool
            Use logarithmic y-axis
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately

        Returns
        -------
        Any
            Plot object

        Example
        -------
        >>> sim.outputs.plot_multiple(['potential', 'e_field'])
        >>> sim.outputs.plot_multiple(['electrons', 'holes', 'doping'], log_scale=True)
        """
        data_dict = {}
        for var in variables:
            for name, entry in self._entries.items():
                if entry.data is None:
                    continue
                if suffix and not name.endswith(suffix) and suffix not in name:
                    continue
                if entry.variable == var:
                    data_dict[var] = entry.data
                    break

        if not data_dict:
            raise ValueError(f"No data found for variables: {variables}")

        if backend is None:
            backend = _get_default_backend()

        if title is None:
            title = " vs ".join(variables)

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        if backend == 'matplotlib':
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))

            for i, (var, data) in enumerate(data_dict.items()):
                color = colors[i % len(colors)]
                ax.plot(data.x, data.y, color=color, linewidth=2, label=var)

            ax.set_xlabel('Position (μm)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if log_scale:
                ax.set_yscale('log')

            if show:
                plt.show()
            return ax
        else:
            import plotly.graph_objects as go
            fig = go.Figure()

            for i, (var, data) in enumerate(data_dict.items()):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(x=data.x, y=data.y, mode='lines',
                                          name=var, line=dict(color=color, width=2)))

            yaxis_type = 'log' if log_scale else 'linear'
            fig.update_layout(
                title=title,
                xaxis_title='Position (μm)',
                yaxis_type=yaxis_type,
                width=kwargs.get('width', 700),
                height=kwargs.get('height', 500),
                template='plotly_white'
            )

            if show:
                fig.show()
            return fig

    def summary(self) -> str:
        """
        Get a summary of all outputs.

        Returns
        -------
        str
            Formatted summary string
        """
        lines = ["Simulation Outputs:"]
        lines.append("-" * 40)

        for output_type in OutputType:
            names = self._by_type[output_type]
            if names:
                lines.append(f"\n{output_type.value.upper()}:")
                for name in names:
                    entry = self._entries[name]
                    status = "loaded" if entry.data is not None else "not loaded"
                    var_info = f" ({entry.variable})" if entry.variable else ""
                    lines.append(f"  - {name}{var_info} [{status}]")

        # Show available groups
        available_groups = []
        for group, vars in VARIABLE_GROUPS.items():
            has_data = any(self.get_by_variable(v) for v in vars)
            if has_data:
                available_groups.append(group)

        if available_groups:
            lines.append(f"\nAvailable composite plots: {', '.join(available_groups)}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        total = len(self._entries)
        loaded = sum(1 for e in self._entries.values() if e.data is not None)
        return f"<OutputManager: {total} outputs, {loaded} loaded>"


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
    raise ImportError("No plotting backend available. Install matplotlib or plotly.")


def get_plot1d_variable(cmd) -> str:
    """
    Determine which variable a Plot1D command is plotting.

    Parameters
    ----------
    cmd : Plot1D
        The Plot1D command object

    Returns
    -------
    str
        Variable name (e.g., 'potential', 'electrons')
    """
    variable_map = {
        'potential': 'potential',
        'qfn': 'qfn',
        'qfp': 'qfp',
        'doping': 'doping',
        'electrons': 'electrons',
        'holes': 'holes',
        'band_val': 'band_val',
        'band_con': 'band_con',
        'e_field': 'e_field',
        'recomb': 'recomb',
        'net_charge': 'net_charge',
        'net_carrier': 'net_carrier',
        'j_electr': 'j_electr',
        'j_hole': 'j_hole',
        'j_total': 'j_total',
        'j_conduc': 'j_conduc',
        'v_electr': 'v_electr',
        'v_hole': 'v_hole',
        'n_temp': 'n_temp',
        'p_temp': 'p_temp',
        'ion_imp': 'ion_imp',
        'j_displa': 'j_displa',
    }

    for attr, var_name in variable_map.items():
        if getattr(cmd, attr, False):
            return var_name

    return "unknown"
