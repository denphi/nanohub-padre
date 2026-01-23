"""
Visualization module for PADRE simulation results.

Provides plotting functions for I-V characteristics using matplotlib and plotly
as rendering backends. Supports transfer characteristics (Id-Vg), output
characteristics (Id-Vd), and general I-V curves.

Example
-------
>>> from nanohubpadre import create_mosfet, Solve, Log
>>> sim = create_mosfet()
>>> sim.add_log(Log(ivfile="idvg"))
>>> sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))
>>> result = sim.run()
>>> # Plot using matplotlib
>>> sim.plot_transfer(gate_electrode=3, drain_electrode=2)
>>> # Plot using plotly
>>> sim.plot_transfer(gate_electrode=3, drain_electrode=2, backend="plotly")
"""

from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .parser import IVData

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _check_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return True
    except ImportError:
        return False


def _check_plotly() -> bool:
    """Check if plotly is available."""
    try:
        import plotly.graph_objects as go  # noqa: F401
        return True
    except ImportError:
        return False


def get_available_backends() -> List[str]:
    """
    Get list of available plotting backends.

    Returns
    -------
    List[str]
        List of available backend names ('matplotlib', 'plotly')
    """
    backends = []
    if _check_matplotlib():
        backends.append('matplotlib')
    if _check_plotly():
        backends.append('plotly')
    return backends


def _get_default_backend() -> str:
    """Get the default plotting backend."""
    if _check_matplotlib():
        return 'matplotlib'
    elif _check_plotly():
        return 'plotly'
    else:
        raise ImportError(
            "No plotting backend available. Install matplotlib or plotly:\n"
            "  pip install matplotlib\n"
            "  pip install plotly"
        )


# ---------------------------------------------------------------------------
# Matplotlib plotting functions
# ---------------------------------------------------------------------------

def _plot_iv_matplotlib(
    voltages: List[float],
    currents: List[float],
    xlabel: str = "Voltage (V)",
    ylabel: str = "Current (A)",
    title: str = "I-V Characteristic",
    log_scale: bool = False,
    abs_current: bool = True,
    marker: str = 'o-',
    color: Optional[str] = None,
    label: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (8, 6),
    grid: bool = True,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot I-V data using matplotlib.

    Parameters
    ----------
    voltages : List[float]
        Voltage values
    currents : List[float]
        Current values
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current (y-axis)
    abs_current : bool
        Plot absolute value of current
    marker : str
        Matplotlib marker style
    color : str, optional
        Line color
    label : str, optional
        Legend label
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
    figsize : tuple
        Figure size (width, height) in inches
    grid : bool
        Show grid
    show : bool
        Call plt.show() after plotting

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Process currents
    y_data = [abs(i) for i in currents] if abs_current else currents

    # Plot
    plot_kwargs = {'label': label}
    if color:
        plot_kwargs['color'] = color
    plot_kwargs.update(kwargs)

    if log_scale:
        ax.semilogy(voltages, y_data, marker, **plot_kwargs)
    else:
        ax.plot(voltages, y_data, marker, **plot_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if grid:
        ax.grid(True, alpha=0.3)

    if label:
        ax.legend()

    if show:
        plt.show()

    return ax


def _plot_multi_iv_matplotlib(
    data_series: List[Tuple[List[float], List[float], str]],
    xlabel: str = "Voltage (V)",
    ylabel: str = "Current (A)",
    title: str = "I-V Characteristics",
    log_scale: bool = False,
    abs_current: bool = True,
    figsize: Tuple[float, float] = (8, 6),
    grid: bool = True,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot multiple I-V curves using matplotlib.

    Parameters
    ----------
    data_series : List[Tuple[List[float], List[float], str]]
        List of (voltages, currents, label) tuples
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current
    abs_current : bool
        Plot absolute value of current
    figsize : tuple
        Figure size
    grid : bool
        Show grid
    show : bool
        Call plt.show()

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for voltages, currents, label in data_series:
        y_data = [abs(i) for i in currents] if abs_current else currents

        if log_scale:
            ax.semilogy(voltages, y_data, 'o-', label=label, **kwargs)
        else:
            ax.plot(voltages, y_data, 'o-', label=label, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if grid:
        ax.grid(True, alpha=0.3)

    ax.legend()

    if show:
        plt.show()

    return ax


# ---------------------------------------------------------------------------
# Plotly plotting functions
# ---------------------------------------------------------------------------

def _plot_iv_plotly(
    voltages: List[float],
    currents: List[float],
    xlabel: str = "Voltage (V)",
    ylabel: str = "Current (A)",
    title: str = "I-V Characteristic",
    log_scale: bool = False,
    abs_current: bool = True,
    color: Optional[str] = None,
    label: Optional[str] = None,
    fig: Optional[Any] = None,
    width: int = 800,
    height: int = 600,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot I-V data using plotly.

    Parameters
    ----------
    voltages : List[float]
        Voltage values
    currents : List[float]
        Current values
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current
    abs_current : bool
        Plot absolute value of current
    color : str, optional
        Line color
    label : str, optional
        Legend label
    fig : plotly.graph_objects.Figure, optional
        Existing figure to add trace to
    width : int
        Figure width in pixels
    height : int
        Figure height in pixels
    show : bool
        Call fig.show()

    Returns
    -------
    plotly.graph_objects.Figure
        The figure object
    """
    import plotly.graph_objects as go

    # Process currents
    y_data = [abs(i) for i in currents] if abs_current else currents

    # Create or update figure
    if fig is None:
        fig = go.Figure()

    # Add trace
    trace_kwargs = {
        'x': voltages,
        'y': y_data,
        'mode': 'lines+markers',
        'name': label or 'I-V',
    }
    if color:
        trace_kwargs['line'] = {'color': color}
    trace_kwargs.update(kwargs)

    fig.add_trace(go.Scatter(**trace_kwargs))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        template='plotly_white',
    )

    if log_scale:
        fig.update_yaxes(type='log')

    if show:
        fig.show()

    return fig


def _plot_multi_iv_plotly(
    data_series: List[Tuple[List[float], List[float], str]],
    xlabel: str = "Voltage (V)",
    ylabel: str = "Current (A)",
    title: str = "I-V Characteristics",
    log_scale: bool = False,
    abs_current: bool = True,
    width: int = 800,
    height: int = 600,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot multiple I-V curves using plotly.

    Parameters
    ----------
    data_series : List[Tuple[List[float], List[float], str]]
        List of (voltages, currents, label) tuples
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current
    abs_current : bool
        Plot absolute value of current
    width : int
        Figure width
    height : int
        Figure height
    show : bool
        Call fig.show()

    Returns
    -------
    plotly.graph_objects.Figure
        The figure object
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    for voltages, currents, label in data_series:
        y_data = [abs(i) for i in currents] if abs_current else currents

        fig.add_trace(go.Scatter(
            x=voltages,
            y=y_data,
            mode='lines+markers',
            name=label,
            **kwargs
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        template='plotly_white',
    )

    if log_scale:
        fig.update_yaxes(type='log')

    if show:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# Public API - Generic plotting functions
# ---------------------------------------------------------------------------

def plot_iv(
    voltages: List[float],
    currents: List[float],
    xlabel: str = "Voltage (V)",
    ylabel: str = "Current (A)",
    title: str = "I-V Characteristic",
    log_scale: bool = False,
    abs_current: bool = True,
    backend: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot I-V characteristic curve.

    Parameters
    ----------
    voltages : List[float]
        Voltage values
    currents : List[float]
        Current values
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current (y-axis)
    abs_current : bool
        Plot absolute value of current (default True)
    backend : str, optional
        Plotting backend: 'matplotlib' or 'plotly'. If None, uses first available.
    show : bool
        Display the plot immediately
    **kwargs
        Additional arguments passed to the backend plotting function

    Returns
    -------
    Any
        matplotlib.axes.Axes or plotly.graph_objects.Figure depending on backend

    Example
    -------
    >>> from nanohubpadre.visualization import plot_iv
    >>> voltages = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    >>> currents = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    >>> plot_iv(voltages, currents, log_scale=True)
    """
    if backend is None:
        backend = _get_default_backend()

    if backend == 'matplotlib':
        return _plot_iv_matplotlib(
            voltages, currents, xlabel, ylabel, title,
            log_scale, abs_current, show=show, **kwargs
        )
    elif backend == 'plotly':
        return _plot_iv_plotly(
            voltages, currents, xlabel, ylabel, title,
            log_scale, abs_current, show=show, **kwargs
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'matplotlib' or 'plotly'.")


def plot_transfer_characteristic(
    gate_voltages: List[float],
    drain_currents: List[float],
    title: str = "Transfer Characteristic (Id-Vg)",
    log_scale: bool = True,
    backend: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot MOSFET transfer characteristic (Id vs Vg).

    Parameters
    ----------
    gate_voltages : List[float]
        Gate voltage values (Vg)
    drain_currents : List[float]
        Drain current values (Id)
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for drain current (default True for subthreshold)
    backend : str, optional
        Plotting backend: 'matplotlib' or 'plotly'
    show : bool
        Display the plot immediately
    **kwargs
        Additional arguments passed to the backend

    Returns
    -------
    Any
        Plot object (axes or figure)

    Example
    -------
    >>> vg, id = sim.get_transfer_characteristic(gate_electrode=3, drain_electrode=2)
    >>> plot_transfer_characteristic(vg, id)
    """
    return plot_iv(
        gate_voltages, drain_currents,
        xlabel="Gate Voltage Vg (V)",
        ylabel="Drain Current |Id| (A)",
        title=title,
        log_scale=log_scale,
        abs_current=True,
        backend=backend,
        show=show,
        **kwargs
    )


def plot_output_characteristic(
    drain_voltages: List[float],
    drain_currents: List[float],
    title: str = "Output Characteristic (Id-Vd)",
    log_scale: bool = False,
    backend: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot MOSFET output characteristic (Id vs Vd).

    Parameters
    ----------
    drain_voltages : List[float]
        Drain voltage values (Vd)
    drain_currents : List[float]
        Drain current values (Id)
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for drain current (default False for output)
    backend : str, optional
        Plotting backend: 'matplotlib' or 'plotly'
    show : bool
        Display the plot immediately
    **kwargs
        Additional arguments passed to the backend

    Returns
    -------
    Any
        Plot object (axes or figure)

    Example
    -------
    >>> vd, id = sim.get_output_characteristic(drain_electrode=2)
    >>> plot_output_characteristic(vd, id)
    """
    return plot_iv(
        drain_voltages, drain_currents,
        xlabel="Drain Voltage Vd (V)",
        ylabel="Drain Current |Id| (A)",
        title=title,
        log_scale=log_scale,
        abs_current=True,
        backend=backend,
        show=show,
        **kwargs
    )


def plot_diode_iv(
    voltages: List[float],
    currents: List[float],
    title: str = "Diode I-V Characteristic",
    log_scale: bool = True,
    backend: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot diode I-V characteristic.

    Parameters
    ----------
    voltages : List[float]
        Applied voltage values
    currents : List[float]
        Current values
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current (default True)
    backend : str, optional
        Plotting backend: 'matplotlib' or 'plotly'
    show : bool
        Display the plot immediately
    **kwargs
        Additional arguments passed to the backend

    Returns
    -------
    Any
        Plot object (axes or figure)
    """
    return plot_iv(
        voltages, currents,
        xlabel="Voltage (V)",
        ylabel="|Current| (A)",
        title=title,
        log_scale=log_scale,
        abs_current=True,
        backend=backend,
        show=show,
        **kwargs
    )


# ---------------------------------------------------------------------------
# IVData plotting mixin
# ---------------------------------------------------------------------------

class IVDataPlotMixin:
    """
    Mixin class to add plotting methods to IVData.

    This mixin provides convenient plotting methods that can be added to the
    IVData class for direct visualization of simulation results.
    """

    def plot(
        self,
        electrode: int,
        title: Optional[str] = None,
        log_scale: bool = False,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> Any:
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
            Plot object
        """
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
    ) -> Any:
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
            Plot object
        """
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
    ) -> Any:
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
            Plot object
        """
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
    ) -> Any:
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
            Plot object
        """
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
