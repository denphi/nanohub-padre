"""
PyPADRE - Python library for PADRE semiconductor device simulator

This library provides a Pythonic interface to create and run PADRE simulations
for semiconductor device modeling.
"""

from .simulation import Simulation
from .mesh import Mesh, XMesh, YMesh, ZMesh
from .region import Region
from .electrode import Electrode
from .doping import Doping
from .contact import Contact
from .material import Material, Alloy
from .models import Models
from .solver import Solve, Method, System, LinAlg
from .log import Log
from .interface import Interface
from .regrid import Regrid, Adapt
from .plotting import Plot1D, Plot2D, Contour, Vector
from .options import Options, Load
from .plot3d import Plot3D
from .base import Comment, Title

__version__ = "0.1.0"
__all__ = [
    "Simulation",
    "Mesh", "XMesh", "YMesh", "ZMesh",
    "Region",
    "Electrode",
    "Doping",
    "Contact",
    "Material", "Alloy",
    "Models",
    "Solve", "Method", "System", "LinAlg",
    "Log",
    "Interface",
    "Regrid", "Adapt",
    "Plot1D", "Plot2D", "Contour", "Vector",
    "Options", "Load",
    "Plot3D",
    "Comment", "Title",
]
