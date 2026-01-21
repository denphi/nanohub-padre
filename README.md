# PyPADRE

A Python library for creating and running PADRE semiconductor device simulations.

## Overview

PyPADRE provides a Pythonic interface to generate PADRE input decks, making it easier to set up complex device simulations programmatically. PADRE (Physics-based Accurate Device Resolution and Evaluation) is a 2D/3D device simulator that solves the drift-diffusion equations for semiconductor devices.

## Features

- **Pythonic Interface**: Define meshes, regions, doping profiles, and solver settings using Python objects
- **Complete PADRE Support**: Covers mesh generation, material properties, physical models, and solve commands
- **Validation**: Built-in parameter validation and helpful error messages
- **Examples**: Ready-to-run examples for common device structures

## Installation

```bash
pip install pypadre
```

Or install from source:

```bash
git clone https://github.com/yourusername/padre.git
cd padre
pip install -e .
```

## Quick Start

```python
from pypadre import (
    Simulation, Mesh, Region, Electrode, Doping,
    Contact, Models, System, Solve
)

# Create simulation
sim = Simulation(title="Simple PN Diode")

# Define mesh
sim.mesh = Mesh(nx=100, ny=3)
sim.mesh.add_x_mesh(1, 0)
sim.mesh.add_x_mesh(100, 1.0)
sim.mesh.add_y_mesh(1, 0)
sim.mesh.add_y_mesh(3, 1)

# Define silicon region
sim.add_region(Region(1, ix_low=1, ix_high=100, iy_low=1, iy_high=3, silicon=True))

# Define electrodes
sim.add_electrode(Electrode(1, ix_low=1, ix_high=1, iy_low=1, iy_high=3))
sim.add_electrode(Electrode(2, ix_low=100, ix_high=100, iy_low=1, iy_high=3))

# Define doping
sim.add_doping(Doping(p_type=True, concentration=1e17, uniform=True, x_right=0.5))
sim.add_doping(Doping(n_type=True, concentration=1e17, uniform=True, x_left=0.5))

# Set contacts
sim.add_contact(Contact(all_contacts=True, neutral=True))

# Configure models
sim.models = Models(temperature=300, srh=True, conmob=True, fldmob=True)
sim.system = System(electrons=True, holes=True, newton=True)

# Solve
sim.add_solve(Solve(initial=True))

# Generate and print the input deck
print(sim.generate_deck())
```

## Examples

The `examples/` directory contains Python equivalents of common PADRE simulations:

- **pndiode.py**: PN junction diode I-V characterization
- **moscap.py**: MOS capacitor C-V analysis
- **mosfet_equivalent.py**: NMOS transistor transfer and output characteristics
- **mesfet.py**: Metal-Semiconductor FET simulation
- **single_mosgap.py**: Simple oxide-silicon structure

Run an example:

```bash
PYTHONPATH=. python3 examples/pndiode.py > pndiode.inp
padre < pndiode.inp > pndiode.out
```

## Supported Commands

PyPADRE supports all major PADRE commands:

| Category | Commands |
|----------|----------|
| Mesh | MESH, X.MESH, Y.MESH, Z.MESH |
| Structure | REGION, ELECTRODE |
| Doping | DOPING (uniform, Gaussian, ERFC, file) |
| Boundaries | CONTACT, INTERFACE, SURFACE |
| Materials | MATERIAL, ALLOY |
| Models | MODELS |
| Solver | SYSTEM, METHOD, LINALG, SOLVE |
| Output | LOG, PLOT.1D, PLOT.2D, PLOT.3D, CONTOUR, VECTOR |
| Control | OPTIONS, LOAD, REGRID, ADAPT |

## Documentation

Full documentation is available at [https://pypadre.readthedocs.io/](https://pypadre.readthedocs.io/)

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
