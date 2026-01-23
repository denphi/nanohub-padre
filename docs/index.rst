.. nanohub-padre documentation master file

Welcome to nanohub-padre's documentation!
=========================================

**nanohub-padre** is a Python library for creating and running PADRE semiconductor device simulations.
It provides a Pythonic interface to generate PADRE input decks, making it easier to set up
complex device simulations programmatically.

PADRE (Physics-based Accurate Device Resolution and Evaluation) is a 2D/3D device simulator
that solves the drift-diffusion equations for semiconductor devices.

Features
--------

* **Pythonic Interface**: Define meshes, regions, doping profiles, and solver settings using Python objects
* **Device Factory Functions**: Pre-built functions to create common devices (PN diode, MOSFET, BJT, solar cell, etc.)
* **Complete PADRE Support**: Covers mesh generation, material properties, physical models, and solve commands
* **Validation**: Built-in parameter validation and helpful error messages
* **Examples**: Ready-to-run examples for common device structures

Quick Start
-----------

**Using Device Factory Functions (Recommended)**

.. code-block:: python

   from nanohubpadre import create_mosfet, Solve, Log

   # Create an NMOS transistor with one line
   sim = create_mosfet(
       channel_length=0.05,
       device_type="nmos",
       temperature=300
   )

   # Add solve commands
   sim.add_solve(Solve(initial=True))
   sim.add_log(Log(ivfile="idvg"))
   sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))

   # Generate the input deck
   print(sim.generate_deck())

**Building from Scratch**

.. code-block:: python

   from nanohubpadre import Simulation, Mesh, Region, Electrode, Doping, Models, System, Solve

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

   pip install nanohub-padre

Or install from source:

.. code-block:: bash

   git clone https://github.com/nanohub/nanohub-padre.git
   cd nanohub-padre
   pip install -e .

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide
   devices
   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
