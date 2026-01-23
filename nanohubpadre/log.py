"""
Logging configuration for PADRE simulations.

Controls I-V and AC data logging.
"""

from typing import Optional
from .base import PadreCommand


class Log(PadreCommand):
    """
    Configure I-V and AC data logging.

    Parameters
    ----------
    ivfile : str, optional
        Output file for I-V data
    acfile : str, optional
        Output file for AC data
    last : bool
        Only log last bias point
    no_trap : bool
        Don't log intermediate trap points
    off : bool
        Turn off logging

    Example
    -------
    >>> # Log IV and AC data
    >>> log = Log(ivfile="iv_data", acfile="ac_data")
    >>>
    >>> # Turn off logging
    >>> log = Log(off=True)
    """

    command_name = "LOG"

    def __init__(
        self,
        ivfile: Optional[str] = None,
        acfile: Optional[str] = None,
        last: bool = False,
        no_trap: bool = False,
        off: bool = False,
    ):
        super().__init__()
        self.ivfile = ivfile
        self.acfile = acfile
        self.last = last
        self.no_trap = no_trap
        self.off = off

    def to_padre(self) -> str:
        params = {}
        flags = []

        if self.ivfile:
            params["OUTF"] = self.ivfile
        if self.acfile:
            params["ACF"] = self.acfile
        if self.last:
            flags.append("LAST")
        if self.no_trap:
            flags.append("NO.TRAP")
        if self.off:
            flags.append("OFF")

        return self._build_command(params, flags)
