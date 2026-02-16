Getting Started
===============

This guide will help you get started with nanohub-padre for semiconductor device simulation.

Prerequisites
-------------

* Python 3.7 or higher
* PADRE simulator installed and accessible in your PATH (for running simulations)

Installation
------------

From PyPI
~~~~~~~~~

.. code-block:: bash

   pip install nanohubpadre

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/yourusername/padre.git
   cd padre
   pip install -e .

Basic Concepts
--------------

nanohub-padre mirrors the structure of PADRE input decks. A typical simulation consists of:

1. **Mesh**: Define the computational grid
2. **Regions**: Assign materials to mesh areas
3. **Electrodes**: Define contact points
4. **Doping**: Specify dopant profiles
5. **Contacts**: Set boundary conditions for electrodes
6. **Materials**: Customize material properties
7. **Models**: Select physical models (mobility, recombination, etc.)
8. **System/Method**: Configure the solver
9. **Solve**: Run the simulation

Your First Simulation
---------------------

Let's create a simple 1D PN diode:

.. code-block:: python

   from nanohubpadre import (
       Simulation, Mesh, Region, Electrode, Doping,
       Contact, Models, System, Solve, Log
   )

   # Create simulation with title
   sim = Simulation(title="My First PN Diode")

   # Define a rectangular mesh (100 nodes in x, 3 in y)
   sim.mesh = Mesh(nx=100, ny=3, outfile="mesh.pg")
   sim.mesh.add_x_mesh(1, 0.0)        # Start at x=0
   sim.mesh.add_x_mesh(50, 0.5)       # Junction at x=0.5
   sim.mesh.add_x_mesh(100, 1.0)      # End at x=1.0
   sim.mesh.add_y_mesh(1, 0.0)
   sim.mesh.add_y_mesh(3, 0.1)

   # Define a silicon region covering the entire mesh
   sim.add_region(Region(
       number=1,
       ix_low=1, ix_high=100,
       iy_low=1, iy_high=3,
       silicon=True
   ))

   # Add electrodes at the ends
   sim.add_electrode(Electrode(1, ix_low=1, ix_high=1, iy_low=1, iy_high=3))
   sim.add_electrode(Electrode(2, ix_low=100, ix_high=100, iy_low=1, iy_high=3))

   # Define doping: p-type on left, n-type on right
   sim.add_doping(Doping(
       p_type=True,
       concentration=1e17,
       uniform=True,
       x_left=0.0,
       x_right=0.5
   ))
   sim.add_doping(Doping(
       n_type=True,
       concentration=1e17,
       uniform=True,
       x_left=0.5,
       x_right=1.0
   ))

   # Set ohmic contacts for all electrodes
   sim.add_contact(Contact(all_contacts=True, neutral=True))

   # Configure physical models
   sim.models = Models(
       temperature=300,
       srh=True,        # SRH recombination
       conmob=True,     # Concentration-dependent mobility
       fldmob=True      # Field-dependent mobility
   )

   # Configure solver
   sim.system = System(electrons=True, holes=True, newton=True)

   # Solve for equilibrium
   sim.add_solve(Solve(initial=True, outfile="equilibrium"))

   # Voltage sweep
   sim.add_log(Log(ivfile="iv_data"))
   sim.add_solve(Solve(
       v2=0,
       vstep=0.1,
       nsteps=10,
       electrode=2
   ))

   # Generate the input deck
   deck = sim.generate_deck()
   print(deck)

   # Write to file
   sim.write_deck("pn_diode.inp")

Running the Simulation
----------------------

Use ``sim.run()`` to execute PADRE directly from Python. Always raise on failure
so errors surface as exceptions rather than silent output:

.. code-block:: python

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

After a successful run, all outputs are automatically parsed and accessible:

.. code-block:: python

   # Plot I-V characteristic (current_electrode defaults to 1)
   sim.plot_iv(title="PN Diode Forward I-V", log_scale=True)

   # Plot band diagram
   sim.plot_band_diagram(title="PN Diode Band Diagram")

   # Access raw data
   iv = sim.outputs.get_iv_data()
   vg, i = iv.get_iv_data(electrode=2)

Using Factory Functions (Recommended)
--------------------------------------

For most devices, the factory functions handle all setup automatically:

.. code-block:: python

   from nanohubpadre import create_mos_capacitor

   sim = create_mos_capacitor(
       oxide_thickness=0.005,
       substrate_doping=1e17,
       substrate_type="p",
       gate_type="n_poly",
       log_cv=True,
       log_bands_eq=True,
       vg_sweep=(-2.0, 2.0, 0.05),
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_band_diagram(suffix="eq")
   sim.plot_cv()

Next Steps
----------

* Explore the :doc:`user_guide` for detailed information on each component
* See the :doc:`devices` page for all factory functions and their parameters
* Check out the :doc:`examples` for complete simulation setups
* Refer to the :doc:`api` for full API documentation
