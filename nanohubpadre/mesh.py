"""
Mesh-related classes for PADRE simulations.

Provides classes for defining rectangular and triangular meshes,
including X.MESH, Y.MESH, Z.MESH, and MESH commands.
"""

from typing import Any, Dict, List, Optional, Union
from .base import PadreCommand


class XMesh(PadreCommand):
    """
    Define X grid line locations in a rectangular mesh.

    Parameters
    ----------
    node : int
        Node number (line number in mesh, 1-300)
    location : float
        X coordinate in microns
    ratio : float, optional
        Grid spacing ratio for interpolation (0.667-1.5 recommended)
    density : float, optional
        Exact grid density/spacing in microns (alternative to ratio)

    Example
    -------
    >>> x = XMesh(node=1, location=0.0)
    >>> x = XMesh(node=30, location=2.0, ratio=1.2)
    """

    command_name = "X.M"

    def __init__(self, node: int, location: float,
                 ratio: Optional[float] = None,
                 density: Optional[float] = None):
        super().__init__()
        self.node = node
        self.location = location
        self.ratio = ratio
        self.density = density

    def to_padre(self) -> str:
        params = {
            "N": self.node,
            "L": self.location,
        }
        if self.ratio is not None:
            params["R"] = self.ratio
        if self.density is not None:
            params["H"] = self.density
        return self._build_command(params)


class YMesh(PadreCommand):
    """
    Define Y grid line locations in a rectangular mesh.

    Parameters
    ----------
    node : int
        Node number (line number in mesh, 1-300)
    location : float
        Y coordinate in microns
    ratio : float, optional
        Grid spacing ratio for interpolation
    density : float, optional
        Exact grid density/spacing in microns

    Example
    -------
    >>> y = YMesh(node=1, location=-0.04)  # oxide surface
    >>> y = YMesh(node=20, location=1.0, ratio=1.4)
    """

    command_name = "Y.M"

    def __init__(self, node: int, location: float,
                 ratio: Optional[float] = None,
                 density: Optional[float] = None):
        super().__init__()
        self.node = node
        self.location = location
        self.ratio = ratio
        self.density = density

    def to_padre(self) -> str:
        params = {
            "N": self.node,
            "L": self.location,
        }
        if self.ratio is not None:
            params["R"] = self.ratio
        if self.density is not None:
            params["H"] = self.density
        return self._build_command(params)


class ZMesh(PadreCommand):
    """
    Define Z grid plane locations for 3D meshes.

    Parameters
    ----------
    node : int
        Node number (plane number in mesh)
    location : float
        Z coordinate in microns
    ratio : float, optional
        Grid spacing ratio for interpolation
    density : float, optional
        Exact grid density/spacing in microns
    """

    command_name = "Z.M"

    def __init__(self, node: int, location: float,
                 ratio: Optional[float] = None,
                 density: Optional[float] = None):
        super().__init__()
        self.node = node
        self.location = location
        self.ratio = ratio
        self.density = density

    def to_padre(self) -> str:
        params = {
            "N": self.node,
            "L": self.location,
        }
        if self.ratio is not None:
            params["R"] = self.ratio
        if self.density is not None:
            params["H"] = self.density
        return self._build_command(params)


class Mesh(PadreCommand):
    """
    Define mesh configuration for PADRE simulation.

    The Mesh class handles rectangular mesh generation, loading previous meshes,
    or reading triangular meshes from external files.

    Parameters
    ----------
    rectangular : bool, optional
        Create a rectangular mesh using X.MESH/Y.MESH lines
    nx : int, optional
        Number of nodes in x-direction (implies rectangular=True)
    ny : int, optional
        Number of nodes in y-direction
    nz : int, optional
        Number of grid planes in z-direction (default=1 for 2D)
    width : float, optional
        Width in z-dimension for 2D/3D simulations (microns)
    infile : str, optional
        Input mesh file (from previous PADRE run or tri generator)
    outfile : str, optional
        Output mesh file name
    previous : bool, optional
        Load a previously generated PADRE mesh
    tri : bool, optional
        Load mesh from tri mesh generator
    cylindrical : bool, optional
        Use cylindrical symmetry about x=xmin
    ascii_in : bool, optional
        Input file is ASCII format (default True)
    ascii_out : bool, optional
        Output file in ASCII format (default True)
    smooth_key : int, optional
        Mesh smoothing options (see PADRE manual)
    it_smooth : int, optional
        Maximum smoothing iterations
    hetero : bool, optional
        Force heterostructure data format
    condense : str, optional
        Region condensation ("all", "ins", "semi", "none")

    Example
    -------
    >>> # Rectangular mesh
    >>> mesh = Mesh(nx=40, ny=17, outfile="mesh1.pg")
    >>>
    >>> # Load previous mesh
    >>> mesh = Mesh(infile="mesh1.pg")
    >>>
    >>> # 3D mesh
    >>> mesh = Mesh(infile="mesh2d", nz=10, width=5, outfile="mesh3d")
    """

    command_name = "MESH"

    def __init__(
        self,
        rectangular: Optional[bool] = None,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        nz: Optional[int] = None,
        width: Optional[float] = None,
        infile: Optional[str] = None,
        outfile: Optional[str] = None,
        previous: Optional[bool] = None,
        tri: Optional[bool] = None,
        cylindrical: bool = False,
        ascii_in: bool = True,
        ascii_out: bool = True,
        smooth_key: Optional[int] = None,
        it_smooth: Optional[int] = None,
        hetero: bool = False,
        condense: Optional[str] = None,
        flip_y: Optional[bool] = None,
        scale: Optional[int] = None,
        diag_flip: bool = False,
        pack_reg: bool = False,
        unpack_reg: bool = False,
    ):
        super().__init__()
        self.rectangular = rectangular
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.width = width
        self.infile = infile
        self.outfile = outfile
        self.previous = previous
        self.tri = tri
        self.cylindrical = cylindrical
        self.ascii_in = ascii_in
        self.ascii_out = ascii_out
        self.smooth_key = smooth_key
        self.it_smooth = it_smooth
        self.hetero = hetero
        self.condense = condense
        self.flip_y = flip_y
        self.scale = scale
        self.diag_flip = diag_flip
        self.pack_reg = pack_reg
        self.unpack_reg = unpack_reg

        # Mesh lines stored in order of addition
        self._mesh_lines: List[PadreCommand] = []

    def add_x_mesh(self, node: int, location: float,
                   ratio: Optional[float] = None,
                   density: Optional[float] = None) -> "Mesh":
        """Add an X.MESH line."""
        self._mesh_lines.append(XMesh(node, location, ratio, density))
        return self

    def add_y_mesh(self, node: int, location: float,
                   ratio: Optional[float] = None,
                   density: Optional[float] = None) -> "Mesh":
        """Add a Y.MESH line."""
        self._mesh_lines.append(YMesh(node, location, ratio, density))
        return self

    def add_z_mesh(self, node: int, location: float,
                   ratio: Optional[float] = None,
                   density: Optional[float] = None) -> "Mesh":
        """Add a Z.MESH line."""
        self._mesh_lines.append(ZMesh(node, location, ratio, density))
        return self

    def to_padre(self) -> str:
        lines = []

        # Build main MESH command
        params = {}
        flags = []

        # Type flags
        if self.rectangular or (self.nx is not None and self.ny is not None):
            flags.append("RECT")
        if self.previous:
            flags.append("PREV")
        if self.tri:
            flags.append("TRI")

        # Specs
        if self.nx is not None:
            params["NX"] = self.nx
        if self.ny is not None:
            params["NY"] = self.ny
        if self.nz is not None:
            params["NZ"] = self.nz
        if self.width is not None:
            params["WIDTH"] = self.width
        if self.infile:
            params["INF"] = self.infile
        if self.outfile:
            params["OUTF"] = self.outfile
        if self.cylindrical:
            flags.append("CYLINDRICAL")
        if not self.ascii_in:
            params["ASCII.IN"] = False
        if not self.ascii_out:
            params["ASCII.OUT"] = False

        # Adjustments
        if self.smooth_key is not None:
            params["SMOOTH.K"] = self.smooth_key
        if self.it_smooth is not None:
            params["IT.SMOOTH"] = self.it_smooth
        if self.hetero:
            flags.append("HETERO")
        if self.condense:
            params["CONDENSE"] = self.condense
        if self.flip_y is not None:
            params["FLIP.Y"] = self.flip_y
        if self.scale is not None:
            params["SCALE"] = self.scale
        if self.diag_flip:
            flags.append("DIAG.FLI")
        if self.pack_reg:
            flags.append("PACK.REG")
        if self.unpack_reg:
            flags.append("UNPACK.REG")

        lines.append(self._build_command(params, flags))

        # Add mesh lines in order they were added
        for mesh_line in self._mesh_lines:
            lines.append(mesh_line.to_padre())

        return "\n".join(lines)

    @classmethod
    def rectangular_grid(cls, x_nodes: List[tuple], y_nodes: List[tuple],
                         outfile: Optional[str] = None) -> "Mesh":
        """
        Create a rectangular mesh from node specifications.

        Parameters
        ----------
        x_nodes : list of tuples
            List of (node, location) or (node, location, ratio) tuples
        y_nodes : list of tuples
            List of (node, location) or (node, location, ratio) tuples
        outfile : str, optional
            Output file name

        Example
        -------
        >>> mesh = Mesh.rectangular_grid(
        ...     x_nodes=[(1, 0), (30, 2)],
        ...     y_nodes=[(1, -0.04), (5, 0), (20, 1, 1.4)],
        ...     outfile="mesh.pg"
        ... )
        """
        mesh = cls(
            nx=max(n[0] for n in x_nodes),
            ny=max(n[0] for n in y_nodes),
            outfile=outfile
        )

        for node_spec in x_nodes:
            node, loc = node_spec[0], node_spec[1]
            ratio = node_spec[2] if len(node_spec) > 2 else None
            mesh.add_x_mesh(node, loc, ratio=ratio)

        for node_spec in y_nodes:
            node, loc = node_spec[0], node_spec[1]
            ratio = node_spec[2] if len(node_spec) > 2 else None
            mesh.add_y_mesh(node, loc, ratio=ratio)

        return mesh
