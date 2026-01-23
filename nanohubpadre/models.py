"""
Physical models configuration for PADRE simulations.

Controls recombination, mobility, and other physical models.
"""

from typing import Optional, Union, List
from .base import PadreCommand


class Models(PadreCommand):
    """
    Configure physical models for the simulation.

    Parameters
    ----------
    temperature : float
        Ambient temperature in Kelvin (default 300K)

    Recombination models:
    srh : bool
        Shockley-Read-Hall recombination
    auger : bool
        Auger recombination
    direct : bool
        Direct radiative recombination
    deeptrap : bool
        Deep-level traps
    conlife : bool
        Concentration-dependent lifetimes
    impact : bool
        Impact ionization generation
    tunneling : bool
        Band-to-band tunneling
    laser : bool
        Stimulated emission

    Band structure:
    bgn : bool
        Band-gap narrowing
    statistics : str
        Carrier statistics ("boltzmann", "fermi", "2dgas")
    incomplete : bool
        Incomplete ionization

    Mobility models:
    conmob : bool
        Concentration-dependent mobility (ionized impurity scattering)
    ccmob : bool
        Carrier-carrier scattering
    fldmob : bool
        Field-dependent mobility (velocity saturation)
    gatmob : bool
        Gate-field dependent mobility
    flddif : bool
        Field-dependent diffusivity
    neutral : bool
        Neutral impurity scattering

    Region selection:
    e_region : int or list
        Regions for velocity saturation model
    g_region : int or list
        Regions for gate-field model
    d_region : int or list
        Regions for diffusivity model

    Driving force specification:
    e_drive : str
        Parallel field drive term
    g_drive : str
        Gate field drive term
    d_drive : str
        Diffusivity drive term
    i_drive : str
        Impact ionization drive term

    print_models : bool
        Print model status and coefficients

    Example
    -------
    >>> # Basic drift-diffusion with SRH
    >>> models = Models(temperature=300, srh=True, conmob=True, fldmob=True)
    >>>
    >>> # Energy balance simulation
    >>> models = Models(srh=True, auger=True, bgn=True,
    ...                 statistics="fermi", fldmob=True)
    """

    command_name = "MODELS"

    # Valid drive term options
    DRIVE_OPTIONS = ["eoj", "eoqf", "ex", "ey", "emag", "qf", "qfb"]
    G_DRIVE_OPTIONS = ["exj", "exqf", "ex", "ey", "emag"]

    def __init__(
        self,
        temperature: float = 300.0,
        # Recombination
        srh: bool = False,
        auger: bool = False,
        direct: bool = False,
        deeptrap: bool = False,
        conlife: bool = False,
        impact: bool = False,
        tunneling: bool = False,
        laser: bool = False,
        # Band structure
        bgn: bool = False,
        statistics: str = "boltzmann",
        incomplete: bool = False,
        # Mobility
        conmob: bool = False,
        ccmob: bool = False,
        fldmob: bool = False,
        gatmob: bool = False,
        flddif: bool = False,
        neutral: bool = False,
        # Region selection
        e_region: Optional[Union[int, List[int]]] = None,
        g_region: Optional[Union[int, List[int]]] = None,
        d_region: Optional[Union[int, List[int]]] = None,
        # Driving forces
        e_drive: Optional[str] = None,
        g_drive: Optional[str] = None,
        d_drive: Optional[str] = None,
        i_drive: Optional[str] = None,
        i_current: Optional[str] = None,
        # Energy transport
        jtherm: bool = True,
        et_form: bool = False,
        # Carrier forms
        c1_sign: Optional[int] = None,
        c2_sign: Optional[int] = None,
        c1_type: Optional[int] = None,
        c2_type: Optional[int] = None,
        # Noise
        gen_nois: bool = False,
        diff_noi: bool = False,
        one_over_f_n: bool = False,
        # Misc
        print_models: bool = False,
    ):
        super().__init__()
        self.temperature = temperature

        # Recombination
        self.srh = srh
        self.auger = auger
        self.direct = direct
        self.deeptrap = deeptrap
        self.conlife = conlife
        self.impact = impact
        self.tunneling = tunneling
        self.laser = laser

        # Band structure
        self.bgn = bgn
        self.statistics = statistics
        self.incomplete = incomplete

        # Mobility
        self.conmob = conmob
        self.ccmob = ccmob
        self.fldmob = fldmob
        self.gatmob = gatmob
        self.flddif = flddif
        self.neutral = neutral

        # Regions
        self.e_region = e_region
        self.g_region = g_region
        self.d_region = d_region

        # Drives
        self.e_drive = e_drive
        self.g_drive = g_drive
        self.d_drive = d_drive
        self.i_drive = i_drive
        self.i_current = i_current

        # Energy transport
        self.jtherm = jtherm
        self.et_form = et_form

        # Carrier forms
        self.c1_sign = c1_sign
        self.c2_sign = c2_sign
        self.c1_type = c1_type
        self.c2_type = c2_type

        # Noise
        self.gen_nois = gen_nois
        self.diff_noi = diff_noi
        self.one_over_f_n = one_over_f_n

        # Misc
        self.print_models = print_models

    def _format_region(self, region: Union[int, List[int]]) -> str:
        """Format region specification as concatenated integers."""
        if isinstance(region, list):
            return "".join(str(r) for r in region)
        return str(region)

    def to_padre(self) -> str:
        params = {"TEMP": self.temperature}
        flags = []

        # Recombination
        if self.srh:
            flags.append("SRH")
        if self.auger:
            flags.append("AUGER")
        if self.direct:
            flags.append("DIRECT")
        if self.deeptrap:
            flags.append("DEEPTRAP")
        if self.conlife:
            flags.append("CONLIFE")
        if self.impact:
            flags.append("IMPACT")
        if self.tunneling:
            flags.append("TUNNELING")
        if self.laser:
            flags.append("LASER")

        # Band structure
        if self.bgn:
            flags.append("BGN")
        if self.statistics != "boltzmann":
            params["STATISTICS"] = self.statistics
        if self.incomplete:
            flags.append("INCOMPLETE")

        # Mobility
        if self.conmob:
            flags.append("CONMOB")
        if self.ccmob:
            flags.append("CCMOB")
        if self.fldmob:
            flags.append("FLDMOB")
        if self.gatmob:
            flags.append("GATMOB")
        if self.flddif:
            flags.append("FLDDIF")
        if self.neutral:
            flags.append("NEUTRAL")

        # Regions
        if self.e_region is not None:
            params["E.REGION"] = self._format_region(self.e_region)
        if self.g_region is not None:
            params["G.REGION"] = self._format_region(self.g_region)
        if self.d_region is not None:
            params["D.REGION"] = self._format_region(self.d_region)

        # Drives
        if self.e_drive:
            params["E.DRIVE"] = self.e_drive
        if self.g_drive:
            params["G.DRIVE"] = self.g_drive
        if self.d_drive:
            params["D.DRIVE"] = self.d_drive
        if self.i_drive:
            params["I.DRIVE"] = self.i_drive
        if self.i_current:
            params["I.CURRENT"] = self.i_current

        # Energy transport
        if not self.jtherm:
            params["JTHERM"] = False
        if self.et_form:
            flags.append("ET.FORM")

        # Carrier forms
        if self.c1_sign is not None:
            params["C1.SIGN"] = self.c1_sign
        if self.c2_sign is not None:
            params["C2.SIGN"] = self.c2_sign
        if self.c1_type is not None:
            params["C1.TYPE"] = self.c1_type
        if self.c2_type is not None:
            params["C2.TYPE"] = self.c2_type

        # Noise
        if self.gen_nois:
            flags.append("GEN.NOIS")
        if self.diff_noi:
            flags.append("DIFF.NOI")
        if self.one_over_f_n:
            flags.append("1OVERF.N")

        # Misc
        if self.print_models:
            flags.append("PRINT")

        return self._build_command(params, flags)

    @classmethod
    def drift_diffusion(cls, temperature: float = 300,
                        srh: bool = True, auger: bool = False,
                        conmob: bool = True, fldmob: bool = True,
                        **kwargs) -> "Models":
        """Configure standard drift-diffusion simulation."""
        return cls(temperature=temperature, srh=srh, auger=auger,
                   conmob=conmob, fldmob=fldmob, **kwargs)

    @classmethod
    def energy_balance(cls, temperature: float = 300,
                       srh: bool = True, auger: bool = True,
                       bgn: bool = True, **kwargs) -> "Models":
        """Configure energy balance simulation."""
        return cls(temperature=temperature, srh=srh, auger=auger,
                   bgn=bgn, statistics="fermi", **kwargs)
