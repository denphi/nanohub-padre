"""
Doping profile definitions for PADRE simulations.

Supports uniform, Gaussian, linear, and file-based doping profiles.
"""

from typing import List, Optional, Union
from .base import PadreCommand


class Doping(PadreCommand):
    """
    Define doping profiles in the device.

    Supports multiple profile types: uniform, Gaussian, linear,
    SUPREM-III, 1D tables, and 2D profiles.

    Parameters
    ----------
    Profile type (one of):
    uniform : bool
        Uniform doping concentration
    gaussian : bool
        Gaussian profile
    linear : bool
        Linear profile
    suprem3 : bool
        Read from SUPREM-III output
    new_suprem3 : bool
        Read from newer SUPREM-III "export" format
    table_1d : bool
        Read from 1D depth vs concentration table
    profile_2d : bool
        Load 2D profile from BIPAD format file
    bison : bool
        Load 2D profile from BISON format
    reload : bool
        Reload from previous PADRE doping file

    Dopant type (one of):
    n_type : bool
        N-type dopant
    p_type : bool
        P-type dopant
    donor : bool
        Donor impurity (same as n_type)
    acceptor : bool
        Acceptor impurity (same as p_type)

    Profile parameters:
    concentration : float
        Peak concentration (atoms/cm^3)
    dose : float
        Total dose
    characteristic : float
        Characteristic length (microns)
    junction : float
        Junction depth (microns)
    peak : float
        Peak position along profile direction

    Location bounds:
    x_left, x_right : float
        X bounds (microns)
    y_top, y_bottom : float
        Y bounds (microns)
    z_front, z_back : float
        Z bounds (microns)
    x_origin, y_origin : float
        Origin for 2D profiles

    File input:
    infile : str
        Input file name
    outfile : str
        Output doping file

    region : list of int, optional
        Region number(s) to dope

    direction : str
        Profile direction ("x" or "y", default "y")
    ratio_lateral : float
        Lateral to principal characteristic length ratio
    lat_char : float
        Lateral characteristic length
    erfc_lateral : bool
        Use error function for lateral profile

    Example
    -------
    >>> # Uniform substrate doping
    >>> d1 = Doping(uniform=True, concentration=1e16, p_type=True)
    >>>
    >>> # Gaussian N+ source/drain
    >>> d2 = Doping(gaussian=True, concentration=9e19, n_type=True,
    ...             x_right=4, junction=1.3, ratio_lateral=0.6,
    ...             erfc_lateral=True)
    """

    command_name = "DOP"

    def __init__(
        self,
        # Profile type
        uniform: bool = False,
        gaussian: bool = False,
        linear: bool = False,
        suprem3: bool = False,
        new_suprem3: bool = False,
        table_1d: bool = False,
        profile_2d: bool = False,
        bison: bool = False,
        reload: bool = False,
        # Dopant type
        n_type: bool = False,
        p_type: bool = False,
        donor: bool = False,
        acceptor: bool = False,
        # Specific dopant species
        boron: bool = False,
        phosphorus: bool = False,
        arsenic: bool = False,
        antimony: bool = False,
        # Profile parameters
        concentration: Optional[float] = None,
        dose: Optional[float] = None,
        characteristic: Optional[float] = None,
        junction: Optional[float] = None,
        peak: Optional[float] = None,
        # Location
        x_left: Optional[float] = None,
        x_right: Optional[float] = None,
        y_top: Optional[float] = None,
        y_bottom: Optional[float] = None,
        z_front: Optional[float] = None,
        z_back: Optional[float] = None,
        x_origin: Optional[float] = None,
        y_origin: Optional[float] = None,
        # Region
        region: Optional[Union[int, List[int]]] = None,
        # Files
        infile: Optional[str] = None,
        outfile: Optional[str] = None,
        ascii_in: bool = True,
        ascii_out: bool = True,
        # Lateral spread
        direction: str = "y",
        ratio_lateral: Optional[float] = None,
        lat_char: Optional[float] = None,
        z_char: Optional[float] = None,
        erfc_lateral: bool = False,
        start: Optional[float] = None,
        # Misc
        initialize: bool = False,
        compensate: bool = False,
        negate: bool = False,
        maxdop: Optional[float] = None,
    ):
        super().__init__()
        # Profile type
        self.uniform = uniform
        self.gaussian = gaussian
        self.linear = linear
        self.suprem3 = suprem3
        self.new_suprem3 = new_suprem3
        self.table_1d = table_1d
        self.profile_2d = profile_2d
        self.bison = bison
        self.reload = reload

        # Dopant type
        self.n_type = n_type or donor
        self.p_type = p_type or acceptor
        self.boron = boron
        self.phosphorus = phosphorus
        self.arsenic = arsenic
        self.antimony = antimony

        # Profile parameters
        self.concentration = concentration
        self.dose = dose
        self.characteristic = characteristic
        self.junction = junction
        self.peak = peak

        # Location
        self.x_left = x_left
        self.x_right = x_right
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.z_front = z_front
        self.z_back = z_back
        self.x_origin = x_origin
        self.y_origin = y_origin

        # Region
        self.region = region

        # Files
        self.infile = infile
        self.outfile = outfile
        self.ascii_in = ascii_in
        self.ascii_out = ascii_out

        # Lateral spread
        self.direction = direction
        self.ratio_lateral = ratio_lateral
        self.lat_char = lat_char
        self.z_char = z_char
        self.erfc_lateral = erfc_lateral
        self.start = start

        # Misc
        self.initialize = initialize
        self.compensate = compensate
        self.negate = negate
        self.maxdop = maxdop

    def to_padre(self) -> str:
        params = {}
        flags = []

        # Profile type
        if self.uniform:
            flags.append("UNIFORM")
        if self.gaussian:
            flags.append("GAUSSIAN")
        if self.linear:
            flags.append("LINEAR")
        if self.suprem3:
            flags.append("SUPREM3")
        if self.new_suprem3:
            flags.append("NEW.SUPREM3")
        if self.table_1d:
            flags.append("TABLE.1D")
        if self.profile_2d:
            flags.append("2D.PROFILE")
        if self.bison:
            flags.append("BISON")
        if self.reload:
            flags.append("RELOAD")

        # Dopant type
        if self.n_type:
            flags.append("N.TYPE")
        if self.p_type:
            flags.append("P.TYPE")
        if self.boron:
            flags.append("BORON")
        if self.phosphorus:
            flags.append("PHOSPHORUS")
        if self.arsenic:
            flags.append("ARSENIC")
        if self.antimony:
            flags.append("ANTIMONY")

        # Profile parameters
        if self.concentration is not None:
            params["CONC"] = self.concentration
        if self.dose is not None:
            params["DOSE"] = self.dose
        if self.characteristic is not None:
            params["CHAR"] = self.characteristic
        if self.junction is not None:
            params["JUNC"] = self.junction
        if self.peak is not None:
            params["PEAK"] = self.peak

        # Region (before location, matches original format)
        if self.region is not None:
            if isinstance(self.region, list):
                params["REG"] = ",".join(str(r) for r in self.region)
            else:
                params["REG"] = self.region

        # Location
        if self.x_left is not None:
            params["X.L"] = self.x_left
        if self.x_right is not None:
            params["X.R"] = self.x_right
        if self.y_top is not None:
            params["Y.TOP"] = self.y_top
        if self.y_bottom is not None:
            params["Y.BOT"] = self.y_bottom
        if self.z_front is not None:
            params["Z.FRONT"] = self.z_front
        if self.z_back is not None:
            params["Z.BACK"] = self.z_back
        if self.x_origin is not None:
            params["X.ORIG"] = self.x_origin
        if self.y_origin is not None:
            params["Y.ORIG"] = self.y_origin

        # Files
        if self.infile:
            params["INF"] = self.infile
        if self.outfile:
            params["OUTF"] = self.outfile
        if not self.ascii_in:
            params["ASCII.IN"] = False
        if not self.ascii_out:
            params["ASCII.OUT"] = False

        # Lateral spread
        if self.direction != "y":
            params["DIR"] = self.direction
        if self.ratio_lateral is not None:
            params["R.LAT"] = self.ratio_lateral
        if self.lat_char is not None:
            params["LAT.CHAR"] = self.lat_char
        if self.z_char is not None:
            params["Z.CHAR"] = self.z_char
        if self.erfc_lateral:
            flags.append("ERFC.LAT")
        if self.start is not None:
            params["START"] = self.start

        # Misc
        if self.initialize:
            flags.append("INIT")
        if self.compensate:
            flags.append("COMPENSATE")
        if self.negate:
            flags.append("NEGATE")
        if self.maxdop is not None:
            params["MAXDOP"] = self.maxdop

        return self._build_command(params, flags)

    @classmethod
    def uniform_n(cls, concentration: float, **kwargs) -> "Doping":
        """Create uniform N-type doping."""
        return cls(uniform=True, n_type=True, concentration=concentration, **kwargs)

    @classmethod
    def uniform_p(cls, concentration: float, **kwargs) -> "Doping":
        """Create uniform P-type doping."""
        return cls(uniform=True, p_type=True, concentration=concentration, **kwargs)

    @classmethod
    def gaussian_n(cls, concentration: float, junction: float,
                   peak: float = 0, **kwargs) -> "Doping":
        """Create Gaussian N-type doping profile."""
        return cls(gaussian=True, n_type=True, concentration=concentration,
                   junction=junction, peak=peak, **kwargs)

    @classmethod
    def gaussian_p(cls, concentration: float, junction: float,
                   peak: float = 0, **kwargs) -> "Doping":
        """Create Gaussian P-type doping profile."""
        return cls(gaussian=True, p_type=True, concentration=concentration,
                   junction=junction, peak=peak, **kwargs)

    @classmethod
    def from_suprem(cls, infile: str, dopant_type: str = "n",
                    **kwargs) -> "Doping":
        """Load doping from SUPREM-III file."""
        return cls(suprem3=True, infile=infile,
                   n_type=(dopant_type.lower() == "n"),
                   p_type=(dopant_type.lower() == "p"),
                   **kwargs)
