"""
Solver configuration for PADRE simulations.

Includes SOLVE, METHOD, SYSTEM, and LINALG commands.
"""

from typing import Optional, Union, List
from .base import PadreCommand


class System(PadreCommand):
    """
    Define which PDEs to solve and their coupling.

    Parameters
    ----------
    carriers : int
        Number of carriers (0, 1, or 2)
    electrons : bool
        Solve electron continuity equation
    holes : bool
        Solve hole continuity equation
    n_temperature : bool
        Solve electron energy balance
    p_temperature : bool
        Solve hole energy balance

    Coupling:
    newton : bool
        Fully coupled Newton iteration
    gummel : bool
        Decoupled Gummel iteration
    coupling : list of str
        Custom coupling specification

    print_info : bool
        Print memory allocation info

    Example
    -------
    >>> # Two-carrier drift-diffusion
    >>> sys = System(carriers=2, newton=True)
    >>>
    >>> # Energy balance with custom coupling
    >>> sys = System(electrons=True, n_temperature=True,
    ...              coupling=["12", "4"])
    """

    command_name = "SYSTEM"

    def __init__(
        self,
        carriers: int = 0,
        electrons: bool = False,
        holes: bool = False,
        n_temperature: bool = False,
        p_temperature: bool = False,
        newton: bool = False,
        gummel: bool = False,
        coupling: Optional[List[str]] = None,
        print_info: bool = False,
        symmetric: bool = True,
    ):
        super().__init__()
        self.carriers = carriers
        self.electrons = electrons
        self.holes = holes
        self.n_temperature = n_temperature
        self.p_temperature = p_temperature
        self.newton = newton
        self.gummel = gummel
        self.coupling = coupling
        self.print_info = print_info
        self.symmetric = symmetric

    def to_padre(self) -> str:
        params = {}
        flags = []

        if self.carriers > 0:
            params["CARR"] = self.carriers
        if self.electrons:
            flags.append("ELECTRONS")
        if self.holes:
            flags.append("HOLES")
        if self.n_temperature:
            flags.append("N.TEMPERATURE")
        if self.p_temperature:
            flags.append("P.TEMPERATURE")

        if self.newton:
            flags.append("NEWTON")
        elif self.gummel:
            flags.append("GUMMEL")
        elif self.coupling:
            params["COUPLING"] = ",".join(self.coupling)

        if self.print_info:
            flags.append("PRINT")
        if not self.symmetric:
            params["SYMMETRIC"] = False

        return self._build_command(params, flags)


class LinAlg(PadreCommand):
    """
    Configure linear algebra solver.

    Parameters
    ----------
    dir_def : bool
        Use direct solver defaults
    iter_def : bool
        Use iterative solver defaults
    method : str
        Method specification
    precondition : str
        Preconditioner specification
    acceleration : str
        Acceleration technique

    itmax : int
        Maximum iterations
    lin_tol : float
        Linear convergence tolerance
    lin_atol : float
        Absolute tolerance

    Example
    -------
    >>> # Direct solver
    >>> la = LinAlg(dir_def=True)
    >>>
    >>> # Iterative solver
    >>> la = LinAlg(iter_def=True, itmax=500)
    """

    command_name = "LINALG"

    def __init__(
        self,
        dir_def: bool = False,
        iter_def: bool = False,
        method: Optional[str] = None,
        precondition: Optional[str] = None,
        acceleration: Optional[str] = None,
        s_method: Optional[str] = None,
        s_precondition: Optional[str] = None,
        s_acceleration: Optional[str] = None,
        itmax: Optional[int] = None,
        nrmax: Optional[int] = None,
        lin_tol: Optional[float] = None,
        lin_atol: Optional[float] = None,
        nlin_tol: Optional[float] = None,
        maxfill: Optional[int] = None,
        k: Optional[int] = None,
        restart: Optional[float] = None,
        linscal: bool = True,
        ptscl: bool = True,
        scale: bool = False,
    ):
        super().__init__()
        self.dir_def = dir_def
        self.iter_def = iter_def
        self.method = method
        self.precondition = precondition
        self.acceleration = acceleration
        self.s_method = s_method
        self.s_precondition = s_precondition
        self.s_acceleration = s_acceleration
        self.itmax = itmax
        self.nrmax = nrmax
        self.lin_tol = lin_tol
        self.lin_atol = lin_atol
        self.nlin_tol = nlin_tol
        self.maxfill = maxfill
        self.k = k
        self.restart = restart
        self.linscal = linscal
        self.ptscl = ptscl
        self.scale = scale

    def to_padre(self) -> str:
        params = {}
        flags = []

        if self.dir_def:
            flags.append("DIR.DEF")
        if self.iter_def:
            flags.append("ITER.DEF")

        if self.method:
            params["METHOD"] = self.method
        if self.precondition:
            params["PRECONDITION"] = self.precondition
        if self.acceleration:
            params["ACCELERATION"] = self.acceleration
        if self.s_method:
            params["S.METHOD"] = self.s_method
        if self.s_precondition:
            params["S.PRECONDITION"] = self.s_precondition
        if self.s_acceleration:
            params["S.ACCELERATION"] = self.s_acceleration

        if self.itmax is not None:
            params["ITMAX"] = self.itmax
        if self.nrmax is not None:
            params["NRMAX"] = self.nrmax
        if self.lin_tol is not None:
            params["LIN.TOL"] = self.lin_tol
        if self.lin_atol is not None:
            params["LIN.ATOL"] = self.lin_atol
        if self.nlin_tol is not None:
            params["NLIN.TOL"] = self.nlin_tol
        if self.maxfill is not None:
            params["MAXFILL"] = self.maxfill
        if self.k is not None:
            params["K"] = self.k
        if self.restart is not None:
            params["RESTART"] = self.restart

        if not self.linscal:
            params["LINSCAL"] = False
        if not self.ptscl:
            params["PTSCL"] = False
        if self.scale:
            flags.append("SCALE")

        return self._build_command(params, flags)


class Method(PadreCommand):
    """
    Configure nonlinear iteration and numerical methods.

    Parameters
    ----------
    Convergence control:
    itlimit : int
        Maximum inner iterations (default 20)
    outloops : int
        Maximum outer iterations (default 20)
    gloops : int
        Number of Gummel smoothing loops
    x_toler : float or list
        Update norm tolerance
    rhs_toler : float or list
        Residual norm tolerance
    xnorm : bool
        Use update norm (default True)
    rhsnorm : bool
        Use residual norm
    l2norm : bool
        Use L2 norm for residual

    Pseudo-continuation (trap):
    trap : bool
        Enable bias stepping on convergence failure
    a_trap : float
        Bias reduction factor (default 0.5)
    n_trap : int
        Newton iterations before trap check
    dv_trap : float
        Minimum voltage step
    di_trap : float
        Minimum current step
    stop : bool
        Stop on convergence failure

    Damping:
    damped : str
        Damping mode ("single", "all", "none")
    dvlimit : float
        Maximum potential update

    Time stepping:
    second_order : bool
        Use 2nd order TR-BDF2 (default True)
    tauto : bool
        Automatic time step (default True)
    tolr_time : float
        Relative truncation error tolerance
    tola_time : float
        Absolute truncation error tolerance
    dt_min : float
        Minimum time step

    Newton-Richardson:
    autonr : bool
        Automatic Newton-Richardson

    print_iter : bool
        Print terminal values after each iteration

    Example
    -------
    >>> # Standard method with trap
    >>> m = Method(trap=True, a_trap=0.5, itlimit=30)
    >>>
    >>> # Transient with auto timestep
    >>> m = Method(second_order=True, tauto=True, tolr_time=1e-3)
    """

    command_name = "METHOD"

    def __init__(
        self,
        # Convergence
        itlimit: Optional[int] = None,
        outloops: Optional[int] = None,
        min_inner: Optional[int] = None,
        min_outer: Optional[int] = None,
        gloops: Optional[int] = None,
        x_toler: Optional[Union[float, List[float]]] = None,
        rhs_toler: Optional[Union[float, List[float]]] = None,
        xnorm: bool = True,
        rhsnorm: bool = False,
        l2norm: bool = True,
        # Trap
        trap: bool = True,
        dgmin: Optional[float] = None,
        a_trap: Optional[float] = None,
        n_trap: Optional[int] = None,
        m_trap: Optional[int] = None,
        i_trap: Optional[int] = None,
        maxneg: Optional[int] = None,
        dv_trap: Optional[float] = None,
        di_trap: Optional[float] = None,
        out_trap: bool = True,
        ign_inner: bool = False,
        stop: bool = True,
        # Damping
        damped: str = "single",
        itdamp: Optional[int] = None,
        delta: Optional[float] = None,
        damploop: Optional[int] = None,
        dfactor: Optional[float] = None,
        dpower: Optional[float] = None,
        dvlimit: Optional[float] = None,
        truncate: bool = False,
        vmargin: Optional[float] = None,
        # Time stepping
        second_order: bool = True,
        tr_print: bool = False,
        tauto: bool = True,
        tolr_time: Optional[float] = None,
        tola_time: Optional[float] = None,
        l2tnorm: bool = True,
        dt_min: Optional[float] = None,
        t_lima: Optional[float] = None,
        t_limb: Optional[float] = None,
        extrapolate: bool = False,
        # Newton-Richardson
        autonr: bool = True,
        nrcriterion: Optional[float] = None,
        nrloop: Optional[int] = None,
        # AC
        ac_method: Optional[str] = None,
        # Misc
        print_iter: bool = False,
        err_estimate: bool = False,
        lin_proj: bool = False,
        lin_fail: bool = False,
    ):
        super().__init__()
        # Convergence
        self.itlimit = itlimit
        self.outloops = outloops
        self.min_inner = min_inner
        self.min_outer = min_outer
        self.gloops = gloops
        self.x_toler = x_toler
        self.rhs_toler = rhs_toler
        self.xnorm = xnorm
        self.rhsnorm = rhsnorm
        self.l2norm = l2norm

        # Trap
        self.trap = trap
        self.dgmin = dgmin
        self.a_trap = a_trap
        self.n_trap = n_trap
        self.m_trap = m_trap
        self.i_trap = i_trap
        self.maxneg = maxneg
        self.dv_trap = dv_trap
        self.di_trap = di_trap
        self.out_trap = out_trap
        self.ign_inner = ign_inner
        self.stop = stop

        # Damping
        self.damped = damped
        self.itdamp = itdamp
        self.delta = delta
        self.damploop = damploop
        self.dfactor = dfactor
        self.dpower = dpower
        self.dvlimit = dvlimit
        self.truncate = truncate
        self.vmargin = vmargin

        # Time stepping
        self.second_order = second_order
        self.tr_print = tr_print
        self.tauto = tauto
        self.tolr_time = tolr_time
        self.tola_time = tola_time
        self.l2tnorm = l2tnorm
        self.dt_min = dt_min
        self.t_lima = t_lima
        self.t_limb = t_limb
        self.extrapolate = extrapolate

        # Newton-Richardson
        self.autonr = autonr
        self.nrcriterion = nrcriterion
        self.nrloop = nrloop

        # AC
        self.ac_method = ac_method

        # Misc
        self.print_iter = print_iter
        self.err_estimate = err_estimate
        self.lin_proj = lin_proj
        self.lin_fail = lin_fail

    def _format_vector(self, value: Union[float, List[float]]) -> str:
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return str(value)

    def to_padre(self) -> str:
        params = {}
        flags = []

        # Convergence
        if self.itlimit is not None:
            params["ITLIMIT"] = self.itlimit
        if self.outloops is not None:
            params["OUTLOOPS"] = self.outloops
        if self.min_inner is not None:
            params["MIN.INNER"] = self.min_inner
        if self.min_outer is not None:
            params["MIN.OUTER"] = self.min_outer
        if self.gloops is not None:
            params["GLOOPS"] = self.gloops
        if self.x_toler is not None:
            params["X.TOL"] = self._format_vector(self.x_toler)
        if self.rhs_toler is not None:
            params["RHS.TOL"] = self._format_vector(self.rhs_toler)
        if not self.xnorm:
            params["XNORM"] = False
        if self.rhsnorm:
            flags.append("RHSNORM")
        if not self.l2norm:
            params["L2NORM"] = False

        # Trap
        if self.trap:
            flags.append("TRAP")
        if self.dgmin is not None:
            params["DGMIN"] = self.dgmin
        if self.a_trap is not None:
            params["A.TRAP"] = self.a_trap
        if self.n_trap is not None:
            params["N.TRAP"] = self.n_trap
        if self.m_trap is not None:
            params["M.TRAP"] = self.m_trap
        if self.i_trap is not None:
            params["I.TRAP"] = self.i_trap
        if self.maxneg is not None:
            params["MAXNEG"] = self.maxneg
        if self.dv_trap is not None:
            params["DV.TRAP"] = self.dv_trap
        if self.di_trap is not None:
            params["DI.TRAP"] = self.di_trap
        if not self.out_trap:
            params["OUT.TRAP"] = False
        if self.ign_inner:
            flags.append("IGN.INNER")
        if not self.stop:
            params["STOP"] = False

        # Damping
        if self.damped != "single":
            params["DAMPED"] = self.damped
        if self.itdamp is not None:
            params["ITDAMP"] = self.itdamp
        if self.delta is not None:
            params["DELTA"] = self.delta
        if self.damploop is not None:
            params["DAMPLOOP"] = self.damploop
        if self.dfactor is not None:
            params["DFACTOR"] = self.dfactor
        if self.dpower is not None:
            params["DPOWER"] = self.dpower
        if self.dvlimit is not None:
            params["DVLIMIT"] = self.dvlimit
        if self.truncate:
            flags.append("TRUNCATE")
        if self.vmargin is not None:
            params["VMARGIN"] = self.vmargin

        # Time stepping
        if not self.second_order:
            params["2NDORDER"] = False
        if self.tr_print:
            flags.append("TR.PRINT")
        if not self.tauto:
            params["TAUTO"] = False
        if self.tolr_time is not None:
            params["TOLR.TIME"] = self.tolr_time
        if self.tola_time is not None:
            params["TOLA.TIME"] = self.tola_time
        if not self.l2tnorm:
            params["L2TNORM"] = False
        if self.dt_min is not None:
            params["DT.MIN"] = self.dt_min
        if self.t_lima is not None:
            params["T.LIMA"] = self.t_lima
        if self.t_limb is not None:
            params["T.LIMB"] = self.t_limb
        if self.extrapolate:
            flags.append("EXTRAPOLATE")

        # Newton-Richardson
        if not self.autonr:
            params["AUTONR"] = False
        if self.nrcriterion is not None:
            params["NRCRITERION"] = self.nrcriterion
        if self.nrloop is not None:
            params["NRLOOP"] = self.nrloop

        # AC
        if self.ac_method:
            params["AC.METHOD"] = self.ac_method

        # Misc
        if self.print_iter:
            flags.append("PRINT")
        if self.err_estimate:
            flags.append("ERR.ESTIMATE")
        if self.lin_proj:
            flags.append("LIN.PROJ")
        if self.lin_fail:
            flags.append("LIN.FAIL")

        return self._build_command(params, flags)


class Solve(PadreCommand):
    """
    Solve for one or more bias points.

    Parameters
    ----------
    Initial guess:
    initial : bool
        Compute equilibrium solution (first solve)
    previous : bool
        Use previous solution as guess
    project : bool
        Project from two previous solutions
    euler : bool
        Euler projection guess
    local : bool
        Local quasi-Fermi guess (good for reverse bias)

    Bias specification:
    v1-v0 : float
        Voltage on electrode 1-10
    i1-i0 : float
        Current on electrode 1-10 (for current BC)
    vstep : float
        Voltage step for stepping
    istep : float
        Current step for stepping
    nsteps : int
        Number of steps
    electrode : int or list
        Electrode(s) to step
    multiply : bool
        Multiply by step instead of add

    Generation:
    generation : float
        Blanket generation rate (/s-cm^3)
    dose_rad : float
        Radiation dose (rad)
    absorption : float
        Absorption coefficient (/um)

    Transient:
    tstep : float
        Time step (seconds)
    tstop : float
        Stop time (seconds)
    tdelta : float
        Time interval to simulate
    ramptime : float
        Ramp duration for bias changes
    endramp : float
        End time of ramp
    seu : bool
        Single event upset mode
    dt_seu : float
        SEU pulse width

    AC analysis:
    ac_analysis : bool
        Perform AC analysis
    frequency : float
        AC frequency (Hz)
    fstep : float
        Frequency step
    nfsteps : int
        Number of frequency steps
    mult_freq : bool
        Multiply frequency by step
    vss : float
        Small-signal voltage (default 0.1*kT/q)
    terminal : int
        Terminal for AC excitation

    Output:
    outfile : str
        Solution output file
    currents : bool
        Save current data
    ascii : bool
        ASCII output format
    save : int
        Save frequency (every n points)

    Example
    -------
    >>> # Initial equilibrium
    >>> s = Solve(initial=True, outfile="sol0")
    >>>
    >>> # Voltage sweep
    >>> s = Solve(v1=0.5, vstep=0.1, nsteps=10, electrode=1,
    ...           outfile="sol_a")
    >>>
    >>> # Transient simulation
    >>> s = Solve(v1=2, tstep=1e-12, tstop=1e-9, ramptime=10e-9,
    ...           outfile="trans")
    """

    command_name = "SOLVE"

    def __init__(
        self,
        # Initial guess
        initial: bool = False,
        previous: bool = False,
        project: bool = False,
        euler: bool = False,
        local: bool = False,
        # Bias
        v1: Optional[float] = None,
        v2: Optional[float] = None,
        v3: Optional[float] = None,
        v4: Optional[float] = None,
        v5: Optional[float] = None,
        v6: Optional[float] = None,
        v7: Optional[float] = None,
        v8: Optional[float] = None,
        v9: Optional[float] = None,
        v0: Optional[float] = None,
        i1: Optional[float] = None,
        i2: Optional[float] = None,
        i3: Optional[float] = None,
        i4: Optional[float] = None,
        i5: Optional[float] = None,
        i6: Optional[float] = None,
        i7: Optional[float] = None,
        i8: Optional[float] = None,
        i9: Optional[float] = None,
        i0: Optional[float] = None,
        vstep: Optional[float] = None,
        istep: Optional[float] = None,
        nsteps: Optional[int] = None,
        electrode: Optional[Union[int, List[int]]] = None,
        multiply: bool = False,
        n_bias: Optional[float] = None,
        p_bias: Optional[float] = None,
        # Generation
        generation: Optional[float] = None,
        dose_rad: Optional[float] = None,
        absorption: Optional[float] = None,
        dir_gen: str = "y",
        pk_gen: Optional[float] = None,
        reg_gen: Optional[List[int]] = None,
        # Transient
        tstep: Optional[float] = None,
        tstop: Optional[float] = None,
        tdelta: Optional[float] = None,
        tend_refine: bool = False,
        ramptime: Optional[float] = None,
        endramp: Optional[float] = None,
        seu: bool = False,
        dt_seu: Optional[float] = None,
        g_tau: Optional[float] = None,
        # AC
        ac_analysis: bool = False,
        frequency: Optional[float] = None,
        fstep: Optional[float] = None,
        mult_freq: bool = False,
        nfsteps: Optional[int] = None,
        vss: Optional[float] = None,
        terminal: Optional[int] = None,
        s_omega: Optional[float] = None,
        max_inner: Optional[int] = None,
        tolerance: Optional[float] = None,
        # Noise
        noise_anal: bool = False,
        i_noise: bool = False,
        e_noise: Optional[int] = None,
        # Output
        outfile: Optional[str] = None,
        currents: bool = True,
        no_append: bool = False,
        ascii: bool = True,
        save: Optional[int] = None,
        t_save: Optional[List[float]] = None,
        nfile: Optional[str] = None,
        mostrans: Optional[int] = None,
    ):
        super().__init__()
        # Initial guess
        self.initial = initial
        self.previous = previous
        self.project = project
        self.euler = euler
        self.local = local

        # Bias
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v4
        self.v5 = v5
        self.v6 = v6
        self.v7 = v7
        self.v8 = v8
        self.v9 = v9
        self.v0 = v0
        self.i1 = i1
        self.i2 = i2
        self.i3 = i3
        self.i4 = i4
        self.i5 = i5
        self.i6 = i6
        self.i7 = i7
        self.i8 = i8
        self.i9 = i9
        self.i0 = i0
        self.vstep = vstep
        self.istep = istep
        self.nsteps = nsteps
        self.electrode = electrode
        self.multiply = multiply
        self.n_bias = n_bias
        self.p_bias = p_bias

        # Generation
        self.generation = generation
        self.dose_rad = dose_rad
        self.absorption = absorption
        self.dir_gen = dir_gen
        self.pk_gen = pk_gen
        self.reg_gen = reg_gen

        # Transient
        self.tstep = tstep
        self.tstop = tstop
        self.tdelta = tdelta
        self.tend_refine = tend_refine
        self.ramptime = ramptime
        self.endramp = endramp
        self.seu = seu
        self.dt_seu = dt_seu
        self.g_tau = g_tau

        # AC
        self.ac_analysis = ac_analysis
        self.frequency = frequency
        self.fstep = fstep
        self.mult_freq = mult_freq
        self.nfsteps = nfsteps
        self.vss = vss
        self.terminal = terminal
        self.s_omega = s_omega
        self.max_inner = max_inner
        self.tolerance = tolerance

        # Noise
        self.noise_anal = noise_anal
        self.i_noise = i_noise
        self.e_noise = e_noise

        # Output
        self.outfile = outfile
        self.currents = currents
        self.no_append = no_append
        self.ascii = ascii
        self.save = save
        self.t_save = t_save
        self.nfile = nfile
        self.mostrans = mostrans

    def _format_electrode(self, electrode: Union[int, List[int]]) -> str:
        if isinstance(electrode, list):
            return "".join(str(e) for e in electrode)
        return str(electrode)

    def to_padre(self) -> str:
        params = {}
        flags = []

        # Initial guess
        if self.initial:
            flags.append("INIT")
        if self.previous:
            flags.append("PREV")
        if self.project:
            flags.append("PROJ")
        if self.euler:
            flags.append("EULER")
        if self.local:
            flags.append("LOCAL")

        # Bias
        for i in range(1, 10):
            v = getattr(self, f"v{i}")
            if v is not None:
                params[f"V{i}"] = v
        if self.v0 is not None:
            params["V0"] = self.v0

        for i in range(1, 10):
            curr = getattr(self, f"i{i}")
            if curr is not None:
                params[f"I{i}"] = curr
        if self.i0 is not None:
            params["I0"] = self.i0

        if self.vstep is not None:
            params["VSTEP"] = self.vstep
        if self.istep is not None:
            params["ISTEP"] = self.istep
        if self.nsteps is not None:
            params["NSTEPS"] = self.nsteps
        if self.electrode is not None:
            params["ELECT"] = self._format_electrode(self.electrode)
        if self.multiply:
            flags.append("MULTIPLY")
        if self.n_bias is not None:
            params["N.BIAS"] = self.n_bias
        if self.p_bias is not None:
            params["P.BIAS"] = self.p_bias

        # Generation
        if self.generation is not None:
            params["GEN"] = self.generation
        if self.dose_rad is not None:
            params["DOSE.RAD"] = self.dose_rad
        if self.absorption is not None:
            params["ABSORP"] = self.absorption
        if self.dir_gen != "y":
            params["DIR.GEN"] = self.dir_gen
        if self.pk_gen is not None:
            params["PK.GEN"] = self.pk_gen
        if self.reg_gen:
            params["REG.GEN"] = ",".join(str(r) for r in self.reg_gen)

        # Transient
        if self.tstep is not None:
            params["TSTEP"] = self.tstep
        if self.tstop is not None:
            params["TSTOP"] = self.tstop
        if self.tdelta is not None:
            params["TDELTA"] = self.tdelta
        if self.tend_refine:
            flags.append("TEND.REFINE")
        if self.ramptime is not None:
            params["RAMPTIME"] = self.ramptime
        if self.endramp is not None:
            params["ENDRAMP"] = self.endramp
        if self.seu:
            flags.append("SEU")
        if self.dt_seu is not None:
            params["DT.SEU"] = self.dt_seu
        if self.g_tau is not None:
            params["G.TAU"] = self.g_tau

        # AC
        if self.ac_analysis:
            flags.append("AC.ANALYSIS")
        if self.frequency is not None:
            params["FREQ"] = self.frequency
        if self.fstep is not None:
            params["FSTEP"] = self.fstep
        if self.mult_freq:
            flags.append("MULT.FREQ")
        if self.nfsteps is not None:
            params["NFSTEPS"] = self.nfsteps
        if self.vss is not None:
            params["VSS"] = self.vss
        if self.terminal is not None:
            params["TERMINAL"] = self.terminal
        if self.s_omega is not None:
            params["S.OMEGA"] = self.s_omega
        if self.max_inner is not None:
            params["MAX.INNER"] = self.max_inner
        if self.tolerance is not None:
            params["TOLERANCE"] = self.tolerance

        # Noise
        if self.noise_anal:
            flags.append("NOISE.ANAL")
        if self.i_noise:
            flags.append("I.NOISE")
        if self.e_noise is not None:
            params["E.NOISE"] = self.e_noise

        # Output
        if self.outfile:
            params["OUTF"] = self.outfile
        if not self.currents:
            params["CURRENTS"] = False
        if self.no_append:
            flags.append("NO.APPEND")
        if not self.ascii:
            params["ASCII"] = False
        if self.save is not None:
            params["SAVE"] = self.save
        if self.t_save:
            params["T.SAVE"] = ",".join(str(t) for t in self.t_save)
        if self.nfile:
            params["NFILE"] = self.nfile
        if self.mostrans is not None:
            params["MOSTRANS"] = self.mostrans

        return self._build_command(params, flags)

    @classmethod
    def equilibrium(cls, outfile: Optional[str] = None) -> "Solve":
        """Create initial equilibrium solve."""
        return cls(initial=True, outfile=outfile)

    @classmethod
    def bias_sweep(cls, electrode: int, start: float, stop: float,
                   step: float, outfile: Optional[str] = None,
                   project: bool = True, **kwargs) -> "Solve":
        """Create a bias sweep solve."""
        nsteps = int((stop - start) / step)
        biases = {f"v{electrode}": start}
        return cls(vstep=step, nsteps=nsteps, electrode=electrode,
                   project=project, outfile=outfile, **biases, **kwargs)
