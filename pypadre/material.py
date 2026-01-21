"""
Material property definitions for PADRE simulations.

Defines physical parameters for semiconductor and insulator materials.
"""

from typing import Optional, Union, List
from .base import PadreCommand


class Alloy(PadreCommand):
    """
    Define an alloy material from predefined materials.

    The alloy can then be used with MATERIAL lines to create
    new material systems with interpolated properties.

    Parameters
    ----------
    name : str
        Name of the alloy to create
    material1 : str
        First material name
    x1 : float
        Compositional fraction for material1 (0-1)
    material2 : str
        Second material name
    x2 : float
        Compositional fraction for material2
    material3 : str, optional
        Third material name
    x3 : float, optional
        Compositional fraction for material3
    material4 : str, optional
        Fourth material name
    x4 : float, optional
        Compositional fraction for material4

    Example
    -------
    >>> # SiGe alloy
    >>> alloy = Alloy(name="sige",
    ...               material1="silicon", x1=0,
    ...               material2="germanium", x2=1)
    """

    command_name = "ALLOY"

    def __init__(
        self,
        name: str,
        material1: str,
        x1: float,
        material2: str,
        x2: float,
        material3: Optional[str] = None,
        x3: Optional[float] = None,
        material4: Optional[str] = None,
        x4: Optional[float] = None,
    ):
        super().__init__()
        self.name = name
        self.material1 = material1
        self.x1 = x1
        self.material2 = material2
        self.x2 = x2
        self.material3 = material3
        self.x3 = x3
        self.material4 = material4
        self.x4 = x4

    def to_padre(self) -> str:
        params = {
            "NAME": self.name,
            "M1ALLOY": self.material1,
            "X1ALLOY": self.x1,
            "M2ALLOY": self.material2,
            "X2ALLOY": self.x2,
        }

        if self.material3 and self.x3 is not None:
            params["M3ALLOY"] = self.material3
            params["X3ALLOY"] = self.x3

        if self.material4 and self.x4 is not None:
            params["M4ALLOY"] = self.material4
            params["X4ALLOY"] = self.x4

        return self._build_command(params)


class Material(PadreCommand):
    """
    Define material properties for a region.

    Many parameters have defaults for standard materials (silicon, gaas, etc.).

    Parameters
    ----------
    name : str
        Material name (new or predefined)
    default : str, optional
        Copy parameters from this material
    alloy : str, optional
        Alloy name to interpolate from
    composition : float, optional
        Compositional fraction for alloy (0-1)

    Band structure:
    eg300 : float
        Energy gap at 300K (eV)
    eg_alpha : float
        Eg temperature coefficient alpha
    eg_beta : float
        Eg temperature coefficient beta
    affinity : float
        Electron affinity (eV)
    ec_off : float
        Conduction band offset (eV)

    Constants:
    permittivity : float
        Dielectric permittivity (F/cm)
    nc300 : float
        Conduction band density at 300K (/cm^3)
    nv300 : float
        Valence band density at 300K (/cm^3)
    mun : float
        Electron mobility at 300K (cm^2/V-s)
    mup : float
        Hole mobility at 300K (cm^2/V-s)
    vsatn : float
        Electron saturation velocity at 300K (cm/s)
    vsatp : float
        Hole saturation velocity at 300K (cm/s)

    Lifetimes:
    taun0 : float or list
        Electron lifetime(s) (s)
    taup0 : float or list
        Hole lifetime(s) (s)
    ntaun : float
        Electron lifetime concentration parameter (/cm^3)
    ntaup : float
        Hole lifetime concentration parameter (/cm^3)

    Auger:
    augn : float
        Electron Auger coefficient (cm^6/s)
    augp : float
        Hole Auger coefficient (cm^6/s)

    Models:
    in_model : str
        Ionized impurity scattering model
    en_model : str
        Velocity saturation model (electrons)
    ep_model : str
        Velocity saturation model (holes)
    gn_model : str
        Gate-field mobility model (electrons)
    gp_model : str
        Gate-field mobility model (holes)
    bgnn_model : str
        Band-gap narrowing model (electrons)
    bgnp_model : str
        Band-gap narrowing model (holes)

    n_type, p_type : bool
        Set separate values for n-type/p-type material

    Example
    -------
    >>> # Define silicon with custom lifetimes
    >>> mat = Material(name="silicon", taun0=1e-6, taup0=1e-6)
    >>>
    >>> # SiGe alloy material
    >>> mat = Material(name="sige30", alloy="sige", composition=0.3,
    ...                eg300=0.830)
    """

    command_name = "MATERIAL"

    def __init__(
        self,
        name: str,
        default: Optional[str] = None,
        alloy: Optional[str] = None,
        composition: Optional[float] = None,
        # Type
        n_type: bool = False,
        p_type: bool = False,
        no_charge: bool = False,
        # Band structure
        eg300: Optional[float] = None,
        eg_alpha: Optional[float] = None,
        eg_beta: Optional[float] = None,
        affinity: Optional[float] = None,
        decdev: Optional[float] = None,
        decdeg: Optional[float] = None,
        ec_off: Optional[float] = None,
        refoff: Optional[str] = None,
        # Constants
        permittivity: Optional[float] = None,
        qf: Optional[float] = None,
        nc300: Optional[float] = None,
        nv300: Optional[float] = None,
        gcb: Optional[float] = None,
        gvb: Optional[float] = None,
        edb: Optional[float] = None,
        eab: Optional[float] = None,
        arichn: Optional[float] = None,
        arichp: Optional[float] = None,
        # Mobility
        mun: Optional[float] = None,
        mup: Optional[float] = None,
        vsatn: Optional[float] = None,
        vsatp: Optional[float] = None,
        tauwn: Optional[float] = None,
        tauwp: Optional[float] = None,
        # Lifetimes
        taun0: Optional[Union[float, List[float]]] = None,
        taup0: Optional[Union[float, List[float]]] = None,
        taur0: Optional[float] = None,
        ntaun: Optional[float] = None,
        ntaup: Optional[float] = None,
        ntaur: Optional[float] = None,
        b0dir: Optional[float] = None,
        # Auger
        augn: Optional[float] = None,
        augp: Optional[float] = None,
        # Traps
        trap_type: Optional[str] = None,
        etrap: Optional[Union[float, List[float]]] = None,
        ntrap: Optional[Union[float, List[float]]] = None,
        # Generation
        gen_con: Optional[float] = None,
        # Models
        in_model: Optional[str] = None,
        ip_model: Optional[str] = None,
        en_model: Optional[str] = None,
        ep_model: Optional[str] = None,
        gn_model: Optional[str] = None,
        gp_model: Optional[str] = None,
        cn_model: Optional[str] = None,
        cp_model: Optional[str] = None,
        dn_model: Optional[str] = None,
        dp_model: Optional[str] = None,
        wn_model: Optional[str] = None,
        wp_model: Optional[str] = None,
        bgnn_model: Optional[str] = None,
        bgnp_model: Optional[str] = None,
        # Model coefficients
        ln_mu: Optional[List[float]] = None,
        lp_mu: Optional[List[float]] = None,
        iin_mu: Optional[List[float]] = None,
        iip_mu: Optional[List[float]] = None,
        en_mu: Optional[List[float]] = None,
        ep_mu: Optional[List[float]] = None,
        gn_mu: Optional[List[float]] = None,
        gp_mu: Optional[List[float]] = None,
        # Impact ionization
        en_ion: Optional[List[float]] = None,
        ep_ion: Optional[List[float]] = None,
        an_ion: Optional[List[float]] = None,
        ap_ion: Optional[List[float]] = None,
        bn_ion: Optional[List[float]] = None,
        bp_ion: Optional[List[float]] = None,
        # BGN coefficients
        n_bgn: Optional[List[float]] = None,
        p_bgn: Optional[List[float]] = None,
    ):
        super().__init__()
        self.name = name
        self.default = default
        self.alloy = alloy
        self.composition = composition
        self.n_type = n_type
        self.p_type = p_type
        self.no_charge = no_charge

        # Band structure
        self.eg300 = eg300
        self.eg_alpha = eg_alpha
        self.eg_beta = eg_beta
        self.affinity = affinity
        self.decdev = decdev
        self.decdeg = decdeg
        self.ec_off = ec_off
        self.refoff = refoff

        # Constants
        self.permittivity = permittivity
        self.qf = qf
        self.nc300 = nc300
        self.nv300 = nv300
        self.gcb = gcb
        self.gvb = gvb
        self.edb = edb
        self.eab = eab
        self.arichn = arichn
        self.arichp = arichp

        # Mobility
        self.mun = mun
        self.mup = mup
        self.vsatn = vsatn
        self.vsatp = vsatp
        self.tauwn = tauwn
        self.tauwp = tauwp

        # Lifetimes
        self.taun0 = taun0
        self.taup0 = taup0
        self.taur0 = taur0
        self.ntaun = ntaun
        self.ntaup = ntaup
        self.ntaur = ntaur
        self.b0dir = b0dir

        # Auger
        self.augn = augn
        self.augp = augp

        # Traps
        self.trap_type = trap_type
        self.etrap = etrap
        self.ntrap = ntrap

        # Generation
        self.gen_con = gen_con

        # Models
        self.in_model = in_model
        self.ip_model = ip_model
        self.en_model = en_model
        self.ep_model = ep_model
        self.gn_model = gn_model
        self.gp_model = gp_model
        self.cn_model = cn_model
        self.cp_model = cp_model
        self.dn_model = dn_model
        self.dp_model = dp_model
        self.wn_model = wn_model
        self.wp_model = wp_model
        self.bgnn_model = bgnn_model
        self.bgnp_model = bgnp_model

        # Model coefficients
        self.ln_mu = ln_mu
        self.lp_mu = lp_mu
        self.iin_mu = iin_mu
        self.iip_mu = iip_mu
        self.en_mu = en_mu
        self.ep_mu = ep_mu
        self.gn_mu = gn_mu
        self.gp_mu = gp_mu

        # Impact ionization
        self.en_ion = en_ion
        self.ep_ion = ep_ion
        self.an_ion = an_ion
        self.ap_ion = ap_ion
        self.bn_ion = bn_ion
        self.bp_ion = bp_ion

        # BGN
        self.n_bgn = n_bgn
        self.p_bgn = p_bgn

    def _format_vector(self, value: Union[float, List[float]]) -> str:
        """Format a scalar or vector value."""
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return str(value)

    def to_padre(self) -> str:
        params = {"NAME": self.name}
        flags = []

        if self.default:
            params["DEF"] = self.default
        if self.alloy:
            params["ALLOY"] = self.alloy
        if self.composition is not None:
            params["COMP"] = self.composition
        if self.n_type:
            flags.append("N.TYPE")
        if self.p_type:
            flags.append("P.TYPE")
        if self.no_charge:
            flags.append("NO.CHARGE")

        # Band structure
        if self.eg300 is not None:
            params["EG300"] = self.eg300
        if self.eg_alpha is not None:
            params["EGALPHA"] = self.eg_alpha
        if self.eg_beta is not None:
            params["EGBETA"] = self.eg_beta
        if self.affinity is not None:
            params["AFFINITY"] = self.affinity
        if self.decdev is not None:
            params["DECDEV"] = self.decdev
        if self.decdeg is not None:
            params["DECDEG"] = self.decdeg
        if self.ec_off is not None:
            params["EC.OFF"] = self.ec_off
        if self.refoff:
            params["REFOFF"] = self.refoff

        # Constants
        if self.permittivity is not None:
            params["PERMITTIVITY"] = self.permittivity
        if self.qf is not None:
            params["QF"] = self.qf
        if self.nc300 is not None:
            params["NC300"] = self.nc300
        if self.nv300 is not None:
            params["NV300"] = self.nv300
        if self.gcb is not None:
            params["GCB"] = self.gcb
        if self.gvb is not None:
            params["GVB"] = self.gvb
        if self.edb is not None:
            params["EDB"] = self.edb
        if self.eab is not None:
            params["EAB"] = self.eab
        if self.arichn is not None:
            params["ARICHN"] = self.arichn
        if self.arichp is not None:
            params["ARICHP"] = self.arichp

        # Mobility
        if self.mun is not None:
            params["MUN"] = self.mun
        if self.mup is not None:
            params["MUP"] = self.mup
        if self.vsatn is not None:
            params["VSATN"] = self.vsatn
        if self.vsatp is not None:
            params["VSATP"] = self.vsatp
        if self.tauwn is not None:
            params["TAUWN"] = self.tauwn
        if self.tauwp is not None:
            params["TAUWP"] = self.tauwp

        # Lifetimes
        if self.taun0 is not None:
            params["TAUN0"] = self._format_vector(self.taun0)
        if self.taup0 is not None:
            params["TAUP0"] = self._format_vector(self.taup0)
        if self.taur0 is not None:
            params["TAUR0"] = self.taur0
        if self.ntaun is not None:
            params["NTAUN"] = self.ntaun
        if self.ntaup is not None:
            params["NTAUP"] = self.ntaup
        if self.ntaur is not None:
            params["NTAUR"] = self.ntaur
        if self.b0dir is not None:
            params["B0DIR"] = self.b0dir

        # Auger
        if self.augn is not None:
            params["AUGN"] = self.augn
        if self.augp is not None:
            params["AUGP"] = self.augp

        # Traps
        if self.trap_type:
            params["TRAP.TYP"] = self.trap_type
        if self.etrap is not None:
            params["ETRAP"] = self._format_vector(self.etrap)
        if self.ntrap is not None:
            params["NTRAP"] = self._format_vector(self.ntrap)

        # Generation
        if self.gen_con is not None:
            params["GEN.CON"] = self.gen_con

        # Models
        if self.in_model:
            params["IN.MOD"] = self.in_model
        if self.ip_model:
            params["IP.MOD"] = self.ip_model
        if self.en_model:
            params["EN.MOD"] = self.en_model
        if self.ep_model:
            params["EP.MOD"] = self.ep_model
        if self.gn_model:
            params["GN.MOD"] = self.gn_model
        if self.gp_model:
            params["GP.MOD"] = self.gp_model
        if self.cn_model:
            params["CN.MOD"] = self.cn_model
        if self.cp_model:
            params["CP.MOD"] = self.cp_model
        if self.dn_model:
            params["DN.MOD"] = self.dn_model
        if self.dp_model:
            params["DP.MOD"] = self.dp_model
        if self.wn_model:
            params["WN.MOD"] = self.wn_model
        if self.wp_model:
            params["WP.MOD"] = self.wp_model
        if self.bgnn_model:
            params["BGNN.MOD"] = self.bgnn_model
        if self.bgnp_model:
            params["BGNP.MOD"] = self.bgnp_model

        # Model coefficients
        if self.ln_mu:
            params["LN.MU"] = self._format_vector(self.ln_mu)
        if self.lp_mu:
            params["LP.MU"] = self._format_vector(self.lp_mu)
        if self.iin_mu:
            params["IIN.MU"] = self._format_vector(self.iin_mu)
        if self.iip_mu:
            params["IIP.MU"] = self._format_vector(self.iip_mu)
        if self.en_mu:
            params["EN.MU"] = self._format_vector(self.en_mu)
        if self.ep_mu:
            params["EP.MU"] = self._format_vector(self.ep_mu)
        if self.gn_mu:
            params["GN.MU"] = self._format_vector(self.gn_mu)
        if self.gp_mu:
            params["GP.MU"] = self._format_vector(self.gp_mu)

        # Impact ionization
        if self.en_ion:
            params["EN.ION"] = self._format_vector(self.en_ion)
        if self.ep_ion:
            params["EP.ION"] = self._format_vector(self.ep_ion)
        if self.an_ion:
            params["AN.ION"] = self._format_vector(self.an_ion)
        if self.ap_ion:
            params["AP.ION"] = self._format_vector(self.ap_ion)
        if self.bn_ion:
            params["BN.ION"] = self._format_vector(self.bn_ion)
        if self.bp_ion:
            params["BP.ION"] = self._format_vector(self.bp_ion)

        # BGN
        if self.n_bgn:
            params["N.BGN"] = self._format_vector(self.n_bgn)
        if self.p_bgn:
            params["P.BGN"] = self._format_vector(self.p_bgn)

        return self._build_command(params, flags)

    @classmethod
    def silicon(cls, name: str = "silicon", **kwargs) -> "Material":
        """Create silicon material with optional customizations."""
        return cls(name=name, default="silicon", **kwargs)

    @classmethod
    def gaas(cls, name: str = "gaas", **kwargs) -> "Material":
        """Create GaAs material with optional customizations."""
        return cls(name=name, default="gaas", **kwargs)
