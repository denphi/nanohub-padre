.. PyPADRE documentation master file

Welcome to PyPADRE's documentation!
===================================

**PyPADRE** is a Python library for creating and running PADRE semiconductor device simulations.
It provides a Pythonic interface to generate PADRE input decks, making it easier to set up
complex device simulations programmatically.

PADRE (Physics-based Accurate Device Resolution and Evaluation) is a 2D/3D device simulator
that solves the drift-diffusion equations for semiconductor devices.

Features
--------

* **Pythonic Interface**: Define meshes, regions, doping profiles, and solver settings using Python objects
* **Complete PADRE Support**: Covers mesh generation, material properties, physical models, and solve commands
* **Validation**: Built-in parameter validation and helpful error messages
* **Examples**: Ready-to-run examples for common device structures (PN diode, MOS capacitor, MOSFET, MESFET)

Quick Start
-----------

.. code-block:: python

   from pypadre import Simulation, Mesh, Region, Electrode, Doping, Models, System, Solve

   # Create a simple PN diode simulation
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

   # Configure models
   sim.models = Models(temperature=300, srh=True, conmob=True, fldmob=True)
   sim.system = System(electrons=True, holes=True, newton=True)

   # Add solve commands
   sim.add_solve(Solve(initial=True))

   # Generate and print the input deck
   print(sim.generate_deck())

Installation
------------

.. code-block:: bash

   pip install pypadre

Or install from source:

.. code-block:: bash

   git clone https://github.com/yourusername/padre.git
   cd padre
   pip install -e .

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide
   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
