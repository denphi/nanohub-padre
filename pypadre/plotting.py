"""
Plotting commands for PADRE simulations.

Includes PLOT.1D, PLOT.2D, CONTOUR, and VECTOR commands.
"""

from typing import Optional, Union, List
from .base import PadreCommand


class Plot2D(PadreCommand):
    """
    Set up 2D plot area and optionally plot grid/boundaries.

    Parameters
    ----------
    Area definition:
    x_min, x_max : float
        X bounds (microns)
    y_min, y_max : float
        Y bounds (microns)
    z_pos : float
        Z position for xy-plane slice

    What to plot:
    grid : bool
        Plot the mesh grid
    boundary : bool
        Plot region boundaries
    junction : bool
        Plot doping junctions
    depl_edg : bool
        Plot depletion edges
    crosses : bool
        Mark grid points with crosses

    Control:
    no_fill : bool
        Draw to scale (don't fill screen)
    no_clear : bool
        Don't clear screen before plot
    no_tic : bool
        No tic marks
    title : bool
        Show title
    labels : bool
        Show color labels
    flip_x : bool
        Mirror plot about y-axis
    tilt : bool
        3D tilted view
    a_elevation : float
        Elevation angle for tilt
    a_azimuth : float
        Azimuth angle for tilt
    pause : bool
        Pause after plot

    outfile : str
        Output plot file

    Example
    -------
    >>> # Plot grid to scale
    >>> p = Plot2D(grid=True, no_fill=True)
    >>>
    >>> # Plot boundaries and junctions in region
    >>> p = Plot2D(x_min=0, x_max=5, y_min=0, y_max=10,
    ...            junction=True, boundary=True)
    """

    command_name = "PLOT.2D"

    def __init__(
        self,
        # Area
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        z_pos: Optional[float] = None,
        top: bool = False,
        bottom: bool = False,
        left: bool = False,
        right: bool = False,
        # What to plot
        grid: bool = False,
        mesh: bool = False,
        obtuse: bool = False,
        crosses: bool = False,
        boundary: bool = False,
        interface: int = 1,
        depl_edg: bool = False,
        junction: bool = False,
        # Control
        no_tic: bool = False,
        no_top: bool = False,
        no_fill: bool = False,
        no_clear: bool = False,
        no_end: bool = False,
        no_diag: bool = False,
        labels: bool = False,
        title: bool = False,
        flip_x: bool = False,
        tilt: bool = False,
        a_elevation: float = 30,
        a_azimuth: float = -30,
        pause: bool = False,
        spline: bool = False,
        nspline: int = 100,
        # Line types
        l_elect: Optional[int] = None,
        l_deple: Optional[int] = None,
        l_junct: Optional[int] = None,
        l_bound: Optional[int] = None,
        l_grid: Optional[int] = None,
        color: bool = False,
        grey: bool = False,
        # Output
        outfile: Optional[str] = None,
        geomfile: Optional[str] = None,
        criter: Optional[float] = None,
    ):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_pos = z_pos
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

        self.grid = grid or mesh
        self.obtuse = obtuse
        self.crosses = crosses
        self.boundary = boundary
        self.interface = interface
        self.depl_edg = depl_edg
        self.junction = junction

        self.no_tic = no_tic
        self.no_top = no_top
        self.no_fill = no_fill
        self.no_clear = no_clear
        self.no_end = no_end
        self.no_diag = no_diag
        self.labels = labels
        self.title = title
        self.flip_x = flip_x
        self.tilt = tilt
        self.a_elevation = a_elevation
        self.a_azimuth = a_azimuth
        self.pause = pause
        self.spline = spline
        self.nspline = nspline

        self.l_elect = l_elect
        self.l_deple = l_deple
        self.l_junct = l_junct
        self.l_bound = l_bound
        self.l_grid = l_grid
        self.color = color
        self.grey = grey

        self.outfile = outfile
        self.geomfile = geomfile
        self.criter = criter

    def to_padre(self) -> str:
        params = {}
        flags = []

        # Area
        if self.x_min is not None:
            params["X.MIN"] = self.x_min
        if self.x_max is not None:
            params["X.MAX"] = self.x_max
        if self.y_min is not None:
            params["Y.MIN"] = self.y_min
        if self.y_max is not None:
            params["Y.MAX"] = self.y_max
        if self.z_pos is not None:
            params["Z.POS"] = self.z_pos
        if self.top:
            flags.append("TOP")
        if self.bottom:
            flags.append("BOTTOM")
        if self.left:
            flags.append("LEFT")
        if self.right:
            flags.append("RIGHT")

        # What to plot
        if self.grid:
            flags.append("GRID")
        if self.obtuse:
            flags.append("OBTUSE")
        if self.crosses:
            flags.append("CROSSES")
        if self.boundary:
            flags.append("BOUND")
        if self.interface != 1:
            params["INTERFACE"] = self.interface
        if self.depl_edg:
            flags.append("DEPL.EDG")
        if self.junction:
            flags.append("JUNC")

        # Control
        if self.no_tic:
            flags.append("NO.TIC")
        if self.no_top:
            flags.append("NO.TOP")
        if self.no_fill:
            flags.append("NO.FILL")
        if self.no_clear:
            flags.append("NO.CLEAR")
        if self.no_end:
            flags.append("NO.END")
        if self.no_diag:
            flags.append("NO.DIAG")
        if self.labels:
            flags.append("LABELS")
        if self.title:
            flags.append("TITLE")
        if self.flip_x:
            flags.append("FLIP.X")
        if self.tilt:
            flags.append("TILT")
            params["A.ELEVATION"] = self.a_elevation
            params["A.AZIMUTH"] = self.a_azimuth
        if self.pause:
            flags.append("PAUSE")
        if self.spline:
            flags.append("SPLINE")
            params["NSPLINE"] = self.nspline

        # Line types
        if self.l_elect is not None:
            params["L.ELECT"] = self.l_elect
        if self.l_deple is not None:
            params["L.DEPLE"] = self.l_deple
        if self.l_junct is not None:
            params["L.JUNCT"] = self.l_junct
        if self.l_bound is not None:
            params["L.BOUND"] = self.l_bound
        if self.l_grid is not None:
            params["L.GRID"] = self.l_grid
        if self.color:
            flags.append("COLOR")
        if self.grey:
            flags.append("GREY")

        # Output
        if self.outfile:
            params["OUTF"] = self.outfile
        if self.geomfile:
            params["GEOMF"] = self.geomfile
        if self.criter is not None:
            params["CRITER"] = self.criter

        return self._build_command(params, flags)


class Contour(PadreCommand):
    """
    Plot contours of a quantity (requires preceding PLOT.2D).

    Parameters
    ----------
    Quantity (one of):
    potential, qfn, qfp : bool
        Potentials
    electrons, holes : bool
        Carrier concentrations
    doping : bool
        Doping concentration
    e_field : bool
        Electric field magnitude
    j_electr, j_hole, j_total : bool
        Current densities
    recomb : bool
        Recombination rate
    flowlines : bool
        Current flowlines

    Range:
    min_value, max_value : float
        Contour bounds
    del_value : float
        Contour spacing
    ncontours : int
        Number of contours

    Control:
    logarithm : bool
        Logarithmic scale
    absolute : bool
        Absolute value
    x_compon, y_compon : bool
        Vector components
    line_type : int
        Line style
    color : bool
        Color fill
    grey : bool
        Grey scale fill
    pause : bool
        Pause after plot

    Example
    -------
    >>> # Potential contours
    >>> c = Contour(potential=True, min_value=-1, max_value=3, del_value=0.25)
    >>>
    >>> # Log doping contours
    >>> c = Contour(doping=True, logarithm=True, absolute=True,
    ...             min_value=10, max_value=20, del_value=1)
    """

    command_name = "CONTOUR"

    def __init__(
        self,
        # Quantity
        potential: bool = False,
        qfn: bool = False,
        qfp: bool = False,
        n_temp: bool = False,
        p_temp: bool = False,
        band_val: bool = False,
        band_cond: bool = False,
        doping: bool = False,
        ion_imp: bool = False,
        electrons: bool = False,
        holes: bool = False,
        net_charge: bool = False,
        net_carrier: bool = False,
        j_conduc: bool = False,
        j_electr: bool = False,
        v_electr: bool = False,
        j_hole: bool = False,
        v_hole: bool = False,
        j_displa: bool = False,
        j_total: bool = False,
        e_field: bool = False,
        recomb: bool = False,
        flowlines: bool = False,
        # Range
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        del_value: Optional[float] = None,
        ncontours: Optional[int] = None,
        constrain: bool = True,
        # Control
        line_type: int = 1,
        absolute: bool = False,
        logarithm: bool = False,
        x_compon: bool = False,
        y_compon: bool = False,
        mix_mater: bool = False,
        pause: bool = False,
        color: bool = False,
        grey: bool = False,
    ):
        super().__init__()
        self.potential = potential
        self.qfn = qfn
        self.qfp = qfp
        self.n_temp = n_temp
        self.p_temp = p_temp
        self.band_val = band_val
        self.band_cond = band_cond
        self.doping = doping
        self.ion_imp = ion_imp
        self.electrons = electrons
        self.holes = holes
        self.net_charge = net_charge
        self.net_carrier = net_carrier
        self.j_conduc = j_conduc
        self.j_electr = j_electr
        self.v_electr = v_electr
        self.j_hole = j_hole
        self.v_hole = v_hole
        self.j_displa = j_displa
        self.j_total = j_total
        self.e_field = e_field
        self.recomb = recomb
        self.flowlines = flowlines

        self.min_value = min_value
        self.max_value = max_value
        self.del_value = del_value
        self.ncontours = ncontours
        self.constrain = constrain

        self.line_type = line_type
        self.absolute = absolute
        self.logarithm = logarithm
        self.x_compon = x_compon
        self.y_compon = y_compon
        self.mix_mater = mix_mater
        self.pause = pause
        self.color = color
        self.grey = grey

    def to_padre(self) -> str:
        params = {}
        flags = []

        # Quantity flags
        if self.potential:
            flags.append("POTEN")
        if self.qfn:
            flags.append("QFN")
        if self.qfp:
            flags.append("QFP")
        if self.n_temp:
            flags.append("N.TEMP")
        if self.p_temp:
            flags.append("P.TEMP")
        if self.band_val:
            flags.append("BAND.VAL")
        if self.band_cond:
            flags.append("BAND.COND")
        if self.doping:
            flags.append("DOPING")
        if self.ion_imp:
            flags.append("ION.IMP")
        if self.electrons:
            flags.append("ELECT")
        if self.holes:
            flags.append("HOLES")
        if self.net_charge:
            flags.append("NET.CH")
        if self.net_carrier:
            flags.append("NET.CA")
        if self.j_conduc:
            flags.append("J.CONDUC")
        if self.j_electr:
            flags.append("J.ELECTR")
        if self.v_electr:
            flags.append("V.ELECTR")
        if self.j_hole:
            flags.append("J.HOLE")
        if self.v_hole:
            flags.append("V.HOLE")
        if self.j_displa:
            flags.append("J.DISPLA")
        if self.j_total:
            flags.append("J.TOTAL")
        if self.e_field:
            flags.append("E.FIELD")
        if self.recomb:
            flags.append("RECOMB")
        if self.flowlines:
            flags.append("FLOW")

        # Range
        if self.min_value is not None:
            params["MIN"] = self.min_value
        if self.max_value is not None:
            params["MAX"] = self.max_value
        if self.del_value is not None:
            params["DEL"] = self.del_value
        if self.ncontours is not None:
            params["NCONT"] = self.ncontours
        if not self.constrain:
            params["CONSTRAIN"] = False

        # Control
        if self.line_type != 1:
            params["LINE.TYPE"] = self.line_type
        if self.absolute:
            flags.append("ABS")
        if self.logarithm:
            flags.append("LOG")
        if self.x_compon:
            flags.append("X.COMPON")
        if self.y_compon:
            flags.append("Y.COMPON")
        if self.mix_mater:
            flags.append("MIX.MATER")
        if self.pause:
            flags.append("PAUSE")
        if self.color:
            flags.append("COLOR")
        if self.grey:
            flags.append("GREY")

        return self._build_command(params, flags)


class Plot1D(PadreCommand):
    """
    1D line plot through device or I-V curve plot.

    Parameters
    ----------
    Mode A - Line through device:
    x_start, y_start : float
        Start point (A)
    x_end, y_end : float
        End point (B)
    z_pos : float
        Z position

    Quantity (one of):
    potential, qfn, qfp, doping, electrons, holes, etc.

    Mode B - I-V plot:
    x_axis : str
        X axis quantity (e.g., "V1", "V2", "I1")
    y_axis : str
        Y axis quantity
    infile : str
        Log file to plot from
    frequency : float
        Frequency for AC plots

    Axes control:
    min_value, max_value : float
        Y axis bounds
    x_min, x_max : float
        X axis bounds
    x_scale, y_scale : float
        Scale factors
    x_label, y_label : str
        Axis labels

    Plot control:
    logarithm : bool
        Log Y scale
    x_log : bool
        Log X scale
    absolute : bool
        Absolute value
    points : bool
        Mark data points
    no_line : bool
        Don't connect points
    line_type : int
        Line style
    no_clear : bool
        Don't clear screen
    unchanged : bool
        Keep same axes
    pause : bool
        Pause after plot

    outfile : str
        Output file

    Example
    -------
    >>> # Plot potential along a line
    >>> p = Plot1D(potential=True, x_start=0, y_start=0,
    ...            x_end=5, y_end=0)
    >>>
    >>> # I-V curve
    >>> p = Plot1D(x_axis="V2", y_axis="I1")
    """

    command_name = "PLOT.1D"

    def __init__(
        self,
        # Segment
        x_start: Optional[float] = None,
        y_start: Optional[float] = None,
        x_end: Optional[float] = None,
        y_end: Optional[float] = None,
        z_pos: Optional[float] = None,
        coordinate: Optional[str] = None,
        # Quantity for Mode A
        potential: bool = False,
        qfn: bool = False,
        qfp: bool = False,
        n_temp: bool = False,
        p_temp: bool = False,
        doping: bool = False,
        ion_imp: bool = False,
        electrons: bool = False,
        holes: bool = False,
        net_charge: bool = False,
        net_carrier: bool = False,
        j_conduc: bool = False,
        j_electr: bool = False,
        v_electr: bool = False,
        j_hole: bool = False,
        v_hole: bool = False,
        j_displa: bool = False,
        j_total: bool = False,
        e_field: bool = False,
        recomb: bool = False,
        band_val: bool = False,
        band_con: bool = False,
        # Mode B - I-V
        x_axis: Optional[str] = None,
        y_axis: Optional[str] = None,
        frequency: Optional[float] = None,
        infile: Optional[str] = None,
        # Axes
        right_axis: bool = False,
        short_axis: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        unscale: bool = False,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        x_mark: Optional[float] = None,
        y_mark: Optional[float] = None,
        title: bool = True,
        # Plot control
        no_clear: bool = False,
        no_axis: bool = False,
        unchanged: bool = False,
        no_end: bool = False,
        no_order: bool = False,
        order_y: bool = False,
        unique: float = 1e-6,
        points: bool = False,
        no_line: bool = False,
        pause: bool = False,
        line_type: int = 1,
        # Data control
        absolute: bool = False,
        logarithm: bool = False,
        x_log: bool = False,
        decibels: bool = False,
        integral: bool = False,
        negative: bool = False,
        inverse: bool = False,
        d_order: float = 0,
        x_component: bool = False,
        y_component: bool = False,
        spline: bool = False,
        nspline: int = 100,
        # Output
        outfile: Optional[str] = None,
        ascii: bool = False,
    ):
        super().__init__()
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.z_pos = z_pos
        self.coordinate = coordinate

        # Quantities
        self.potential = potential
        self.qfn = qfn
        self.qfp = qfp
        self.n_temp = n_temp
        self.p_temp = p_temp
        self.doping = doping
        self.ion_imp = ion_imp
        self.electrons = electrons
        self.holes = holes
        self.net_charge = net_charge
        self.net_carrier = net_carrier
        self.j_conduc = j_conduc
        self.j_electr = j_electr
        self.v_electr = v_electr
        self.j_hole = j_hole
        self.v_hole = v_hole
        self.j_displa = j_displa
        self.j_total = j_total
        self.e_field = e_field
        self.recomb = recomb
        self.band_val = band_val
        self.band_con = band_con

        # Mode B
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.frequency = frequency
        self.infile = infile

        # Axes
        self.right_axis = right_axis
        self.short_axis = short_axis
        self.min_value = min_value
        self.max_value = max_value
        self.x_min = x_min
        self.x_max = x_max
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.unscale = unscale
        self.x_label = x_label
        self.y_label = y_label
        self.x_mark = x_mark
        self.y_mark = y_mark
        self.title = title

        # Plot control
        self.no_clear = no_clear
        self.no_axis = no_axis
        self.unchanged = unchanged
        self.no_end = no_end
        self.no_order = no_order
        self.order_y = order_y
        self.unique = unique
        self.points = points
        self.no_line = no_line
        self.pause = pause
        self.line_type = line_type

        # Data control
        self.absolute = absolute
        self.logarithm = logarithm
        self.x_log = x_log
        self.decibels = decibels
        self.integral = integral
        self.negative = negative
        self.inverse = inverse
        self.d_order = d_order
        self.x_component = x_component
        self.y_component = y_component
        self.spline = spline
        self.nspline = nspline

        # Output
        self.outfile = outfile
        self.ascii = ascii

    def to_padre(self) -> str:
        params = {}
        flags = []

        # Segment
        if self.x_start is not None:
            params["A.X"] = self.x_start
        if self.y_start is not None:
            params["A.Y"] = self.y_start
        if self.x_end is not None:
            params["B.X"] = self.x_end
        if self.y_end is not None:
            params["B.Y"] = self.y_end
        if self.z_pos is not None:
            params["Z.POS"] = self.z_pos
        if self.coordinate:
            params["COORD"] = self.coordinate

        # Quantities
        if self.potential:
            flags.append("POT")
        if self.qfn:
            flags.append("QFN")
        if self.qfp:
            flags.append("QFP")
        if self.n_temp:
            flags.append("N.TEMP")
        if self.p_temp:
            flags.append("P.TEMP")
        if self.doping:
            flags.append("DOP")
        if self.ion_imp:
            flags.append("ION.IMP")
        if self.electrons:
            flags.append("ELE")
        if self.holes:
            flags.append("HOLE")
        if self.net_charge:
            flags.append("NET.CHARGE")
        if self.net_carrier:
            flags.append("NET.CA")
        if self.j_conduc:
            flags.append("J.CONDUC")
        if self.j_electr:
            flags.append("J.ELECTR")
        if self.v_electr:
            flags.append("V.ELECTR")
        if self.j_hole:
            flags.append("J.HOLE")
        if self.v_hole:
            flags.append("V.HOLE")
        if self.j_displa:
            flags.append("J.DISPLA")
        if self.j_total:
            flags.append("J.TOTAL")
        if self.e_field:
            flags.append("E.FIELD")
        if self.recomb:
            flags.append("RECOMB")
        if self.band_val:
            flags.append("BAND.VAL")
        if self.band_con:
            flags.append("BAND.CON")

        # Mode B
        if self.x_axis:
            params["X.AXIS"] = self.x_axis
        if self.y_axis:
            params["Y.AXIS"] = self.y_axis
        if self.frequency is not None:
            params["FREQ"] = self.frequency
        if self.infile:
            params["INF"] = self.infile

        # Axes
        if self.right_axis:
            flags.append("RIGHT.AXIS")
        if self.short_axis:
            flags.append("SHORT.AXIS")
        if self.min_value is not None:
            params["MIN"] = self.min_value
        if self.max_value is not None:
            params["MAX"] = self.max_value
        if self.x_min is not None:
            params["X.MIN"] = self.x_min
        if self.x_max is not None:
            params["X.MAX"] = self.x_max
        if self.x_scale != 1.0:
            params["X.SCALE"] = self.x_scale
        if self.y_scale != 1.0:
            params["Y.SCALE"] = self.y_scale
        if self.unscale:
            flags.append("UNSCALE")
        if self.x_label:
            params["X.LABEL"] = self.x_label
        if self.y_label:
            params["Y.LABEL"] = self.y_label
        if self.x_mark is not None:
            params["X.MARK"] = self.x_mark
        if self.y_mark is not None:
            params["Y.MARK"] = self.y_mark
        if not self.title:
            params["TITLE"] = False

        # Plot control
        if self.no_clear:
            flags.append("NO.CLEAR")
        if self.no_axis:
            flags.append("NO.AXIS")
        if self.unchanged:
            flags.append("UNCH")
        if self.no_end:
            flags.append("NO.END")
        if self.no_order:
            flags.append("NO.ORDER")
        if self.order_y:
            flags.append("ORDER.Y")
        if self.unique != 1e-6:
            params["UNIQUE"] = self.unique
        if self.points:
            flags.append("POINTS")
        if self.no_line:
            flags.append("NO.LINE")
        if self.pause:
            flags.append("PAUSE")
        if self.line_type != 1:
            params["LINE"] = self.line_type

        # Data control
        if self.absolute:
            flags.append("ABS")
        if self.logarithm:
            flags.append("LOG")
        if self.x_log:
            flags.append("X.LOG")
        if self.decibels:
            flags.append("DECIBELS")
        if self.integral:
            flags.append("INTEGRAL")
        if self.negative:
            flags.append("NEGATIVE")
        if self.inverse:
            flags.append("INVERSE")
        if self.d_order != 0:
            params["D.ORDER"] = self.d_order
        if self.x_component:
            flags.append("X.COMP")
        if self.y_component:
            flags.append("Y.COMP")
        if self.spline:
            flags.append("SPLINE")
            params["NSPL"] = self.nspline

        # Output
        if self.outfile:
            params["OUTF"] = self.outfile
        if self.ascii:
            flags.append("ASCII")

        return self._build_command(params, flags)


class Vector(PadreCommand):
    """
    Plot vector quantities (requires preceding PLOT.2D).

    Parameters
    ----------
    Quantity (one of):
    j_conduc, j_electr, j_hole, j_total, j_displa : bool
        Current densities
    v_electr, v_hole : bool
        Velocities
    e_field : bool
        Electric field

    Control:
    logarithm : bool
        Logarithmic scaling
    minimum, maximum : float
        Magnitude bounds
    scale : float
        Scale factor
    clipfact : float
        Threshold for not plotting
    line_type : int
        Line style

    Example
    -------
    >>> # Plot electron and hole currents
    >>> v1 = Vector(j_electr=True, line_type=2)
    >>> v2 = Vector(j_hole=True, line_type=3)
    """

    command_name = "VECTOR"

    def __init__(
        self,
        j_conduc: bool = False,
        j_electr: bool = False,
        v_electr: bool = False,
        j_hole: bool = False,
        v_hole: bool = False,
        j_displa: bool = False,
        j_total: bool = False,
        e_field: bool = False,
        logarithm: bool = False,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        scale: float = 1.0,
        clipfact: float = 0.1,
        line_type: int = 1,
    ):
        super().__init__()
        self.j_conduc = j_conduc
        self.j_electr = j_electr
        self.v_electr = v_electr
        self.j_hole = j_hole
        self.v_hole = v_hole
        self.j_displa = j_displa
        self.j_total = j_total
        self.e_field = e_field
        self.logarithm = logarithm
        self.minimum = minimum
        self.maximum = maximum
        self.scale = scale
        self.clipfact = clipfact
        self.line_type = line_type

    def to_padre(self) -> str:
        params = {}
        flags = []

        if self.j_conduc:
            flags.append("J.CONDUC")
        if self.j_electr:
            flags.append("J.ELECTR")
        if self.v_electr:
            flags.append("V.ELECTR")
        if self.j_hole:
            flags.append("J.HOLE")
        if self.v_hole:
            flags.append("V.HOLE")
        if self.j_displa:
            flags.append("J.DISPLA")
        if self.j_total:
            flags.append("J.TOTAL")
        if self.e_field:
            flags.append("E.FIELD")

        if self.logarithm:
            flags.append("LOG")
        if self.minimum is not None:
            params["MIN"] = self.minimum
        if self.maximum is not None:
            params["MAX"] = self.maximum
        if self.scale != 1.0:
            params["SCALE"] = self.scale
        if self.clipfact != 0.1:
            params["CLIPFACT"] = self.clipfact
        if self.line_type != 1:
            params["LINE"] = self.line_type

        return self._build_command(params, flags)
