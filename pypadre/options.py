"""
Options configuration for PADRE simulations.

Sets global run options including plotting devices.
"""

from typing import Optional
from .base import PadreCommand


class Options(PadreCommand):
    """
    Configure global run options.

    Parameters
    ----------
    Plot devices (one of):
    hp2648 : bool
        HP2648 graphics terminal
    hp2623 : bool
        HP2623 graphics terminal
    tek4107 : bool
        Tektronix 4107 color terminal
    tek4014 : bool
        Tektronix 4014 terminal
    vt240 : bool
        DEC VT240 terminal
    pic : bool
        PIC language output
    splot : bool
        SPLOT format for 1D plots
    postscript : bool
        PostScript output

    Screen size:
    x_screen : float
        Screen width (inches, default 10)
    y_screen : float
        Screen height (inches, default 5)
    x_offset : float
        X offset from bottom-left (inches)
    y_offset : float
        Y offset from bottom-left (inches)

    Debug:
    news : bool
        Print version news
    g_debug : bool
        Print general debug info
    n_debug : bool
        Print numerical debug info
    cpustat : bool
        Print CPU profile
    cpufile : str
        CPU profile output file
    max_cpu : float
        Maximum CPU time (seconds)

    mode : str
        Compatibility mode ("2.1" or "2.3")

    Example
    -------
    >>> # PostScript output
    >>> opt = Options(postscript=True)
    >>>
    >>> # Custom screen size
    >>> opt = Options(tek4107=True, x_screen=6, y_screen=5)
    """

    command_name = "OPTIONS"

    def __init__(
        self,
        # Plot devices
        hp2648: bool = False,
        hp2623: bool = False,
        tek4107: bool = False,
        tek4014: bool = False,
        vt240: bool = False,
        pic: bool = False,
        splot: bool = False,
        postscript: bool = False,
        # Screen size
        x_screen: Optional[float] = None,
        y_screen: Optional[float] = None,
        x_offset: Optional[float] = None,
        y_offset: Optional[float] = None,
        # Debug
        news: bool = False,
        g_debug: bool = False,
        n_debug: bool = False,
        ns_debug: bool = False,
        cpustat: bool = False,
        cpufile: Optional[str] = None,
        max_cpu: Optional[float] = None,
        mode: Optional[str] = None,
        # Size
        size_buf: Optional[float] = None,
        wee_size: bool = False,
        med_size: bool = False,
        big_size: bool = False,
        wow_size: bool = False,
        crush: bool = False,
    ):
        super().__init__()
        self.hp2648 = hp2648
        self.hp2623 = hp2623
        self.tek4107 = tek4107
        self.tek4014 = tek4014
        self.vt240 = vt240
        self.pic = pic
        self.splot = splot
        self.postscript = postscript
        self.x_screen = x_screen
        self.y_screen = y_screen
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.news = news
        self.g_debug = g_debug
        self.n_debug = n_debug
        self.ns_debug = ns_debug
        self.cpustat = cpustat
        self.cpufile = cpufile
        self.max_cpu = max_cpu
        self.mode = mode
        self.size_buf = size_buf
        self.wee_size = wee_size
        self.med_size = med_size
        self.big_size = big_size
        self.wow_size = wow_size
        self.crush = crush

    def to_padre(self) -> str:
        params = {}
        flags = []

        # Plot devices
        if self.hp2648:
            flags.append("HP2648")
        if self.hp2623:
            flags.append("HP2623")
        if self.tek4107:
            flags.append("TEK4107")
        if self.tek4014:
            flags.append("4014")
        if self.vt240:
            flags.append("VT240")
        if self.pic:
            flags.append("PIC")
        if self.splot:
            flags.append("SPLOT")
        if self.postscript:
            flags.append("PO")

        # Screen size
        if self.x_screen is not None:
            params["X.S"] = self.x_screen
        if self.y_screen is not None:
            params["Y.S"] = self.y_screen
        if self.x_offset is not None:
            params["X.OFF"] = self.x_offset
        if self.y_offset is not None:
            params["Y.OFF"] = self.y_offset

        # Debug
        if self.news:
            flags.append("NEWS")
        if self.g_debug:
            flags.append("DEBUG")
        if self.n_debug:
            flags.append("N.DEBUG")
        if self.ns_debug:
            flags.append("NS.DEBUG")
        if self.cpustat:
            flags.append("CPUSTAT")
        if self.cpufile:
            params["CPUFILE"] = self.cpufile
        if self.max_cpu is not None:
            params["MAX.CPU"] = self.max_cpu
        if self.mode:
            params["MODE"] = self.mode

        # Size
        if self.size_buf is not None:
            params["SIZE.BUF"] = self.size_buf
        if self.wee_size:
            flags.append("WEE.SIZE")
        if self.med_size:
            flags.append("MED.SIZE")
        if self.big_size:
            flags.append("BIG.SIZE")
        if self.wow_size:
            flags.append("WOW.SIZE")
        if self.crush:
            flags.append("CRUSH")

        return self._build_command(params, flags)


class Load(PadreCommand):
    """
    Load a previously saved solution.

    Parameters
    ----------
    infile : str
        Solution file to load
    ascii : bool
        File is ASCII format

    Example
    -------
    >>> load = Load(infile="initsol")
    """

    command_name = "LOAD"

    def __init__(self, infile: str, ascii: bool = True):
        super().__init__()
        self.infile = infile
        self.ascii = ascii

    def to_padre(self) -> str:
        params = {"INF": self.infile}
        flags = []
        if not self.ascii:
            params["ASCII"] = False
        return self._build_command(params, flags)
