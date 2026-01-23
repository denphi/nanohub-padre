User Guide
==========

This guide provides detailed information on using nanohub-padre components.

Mesh Definition
---------------

The mesh defines the computational grid for your device simulation.

Rectangular Mesh
~~~~~~~~~~~~~~~~

The most common mesh type is rectangular:

.. code-block:: python

   from nanohubpadre import Mesh

   # Create a 2D rectangular mesh
   mesh = Mesh(nx=100, ny=50, outfile="mesh.pg")

   # Define X grid lines
   mesh.add_x_mesh(node=1, location=0.0)
   mesh.add_x_mesh(node=50, location=0.5, ratio=0.8)  # Fine mesh near center
   mesh.add_x_mesh(node=100, location=1.0, ratio=1.2)

   # Define Y grid lines
   mesh.add_y_mesh(node=1, location=0.0)
   mesh.add_y_mesh(node=25, location=0.1, ratio=0.7)  # Fine mesh near surface
   mesh.add_y_mesh(node=50, location=1.0, ratio=1.3)

The ``ratio`` parameter controls mesh grading:

* ``ratio < 1``: Mesh becomes finer toward this node
* ``ratio = 1``: Uniform mesh spacing
* ``ratio > 1``: Mesh becomes coarser toward this node

3D Mesh
~~~~~~~

For 3D simulations, add Z mesh lines:

.. code-block:: python

   mesh = Mesh(nx=50, ny=50, nz=10, width=1.0)
   # ... add x and y mesh lines ...
   mesh.add_z_mesh(node=1, location=0.0)
   mesh.add_z_mesh(node=10, location=1.0)

Loading Existing Mesh
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load from previous PADRE run
   mesh = Mesh(infile="previous_mesh.pg", previous=True)

   # Load from triangular mesh generator
   mesh = Mesh(infile="tri_mesh.dat", tri=True)

Regions
-------

Regions define material assignments for mesh areas.

Basic Region Definition
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nanohubpadre import Region

   # Silicon semiconductor region by index bounds
   silicon = Region(
       number=1,
       ix_low=1, ix_high=100,
       iy_low=5, iy_high=50,
       material="silicon",
       semiconductor=True
   )

   # Gate oxide insulator
   oxide = Region(
       number=2,
       ix_low=20, ix_high=80,
       iy_low=1, iy_high=5,
       material="sio2",
       insulator=True
   )

Using Coordinate Bounds
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define region by physical coordinates (in microns)
   region = Region(
       number=1,
       x_min=0.0, x_max=2.0,
       y_min=-0.01, y_max=1.0,
       material="silicon",
       semiconductor=True
   )

Shorthand Methods
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Quick silicon region
   sim.add_region(Region(1, ix_low=1, ix_high=100, iy_low=1, iy_high=50, silicon=True))

   # Quick oxide region
   sim.add_region(Region(2, ix_low=1, ix_high=100, iy_low=50, iy_high=55, oxide=True))

Electrodes
----------

Electrodes define electrical contacts on the device.

.. code-block:: python

   from nanohubpadre import Electrode

   # Source contact (by index)
   source = Electrode(
       number=1,
       ix_low=1, ix_high=20,
       iy_low=5, iy_high=5
   )

   # Gate contact (by coordinates)
   gate = Electrode(
       number=2,
       x_min=0.3, x_max=0.7,
       y_min=-0.01, y_max=-0.01
   )

   # Drain contact
   drain = Electrode(
       number=3,
       ix_low=80, ix_high=100,
       iy_low=5, iy_high=5
   )

   # Substrate contact (back)
   substrate = Electrode(
       number=4,
       ix_low=1, ix_high=100,
       iy_low=50, iy_high=50
   )

Doping Profiles
---------------

nanohub-padre supports various doping profile types.

Uniform Doping
~~~~~~~~~~~~~~

.. code-block:: python

   from nanohubpadre import Doping

   # Uniform p-type substrate
   sim.add_doping(Doping(
       p_type=True,
       concentration=1e17,
       uniform=True,
       region=1
   ))

   # Uniform n-type in specific area
   sim.add_doping(Doping(
       n_type=True,
       concentration=1e20,
       uniform=True,
       x_left=0.0, x_right=0.3,
       y_top=0.0, y_bottom=0.2
   ))

Gaussian Profile
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Gaussian n+ junction
   sim.add_doping(Doping(
       gaussian=True,
       n_type=True,
       concentration=5e19,
       junction=0.2,      # Junction depth (microns)
       peak=0.0,          # Peak position
       characteristic=0.05  # Characteristic length
   ))

   # Lateral Gaussian with ERFC
   sim.add_doping(Doping(
       gaussian=True,
       n_type=True,
       concentration=1e19,
       junction=0.15,
       x_right=0.5,           # Right boundary
       ratio_lateral=0.7,     # Lateral/vertical ratio
       erfc_lateral=True      # Use ERFC for lateral
   ))

ERFC Profile
~~~~~~~~~~~~

.. code-block:: python

   sim.add_doping(Doping(
       erfc=True,
       n_type=True,
       concentration=1e18,
       junction=0.3
   ))

Doping from File
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load 1D profile
   sim.add_doping(Doping(infile="profile.dat"))

   # Load 2D doping map
   sim.add_doping(Doping(infile="doping2d.dat", ascii=True))

Contacts
--------

Contacts specify boundary conditions for electrodes.

Ohmic Contacts
~~~~~~~~~~~~~~

.. code-block:: python

   from nanohubpadre import Contact

   # Ohmic contact for all electrodes
   sim.add_contact(Contact(all_contacts=True, neutral=True))

   # Ohmic contact for specific electrode
   sim.add_contact(Contact(number=1, neutral=True))

Schottky Contacts
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Schottky contact with work function
   sim.add_contact(Contact(
       number=2,
       workfunction=4.87  # Work function in eV
   ))

   # Aluminum Schottky contact
   sim.add_contact(Contact(number=2, aluminum=True))

Gate Contacts
~~~~~~~~~~~~~

.. code-block:: python

   # N+ polysilicon gate
   sim.add_contact(Contact(number=1, n_polysilicon=True))

   # P+ polysilicon gate
   sim.add_contact(Contact(number=1, p_polysilicon=True))

   # Tungsten gate
   sim.add_contact(Contact(number=1, tungsten=True))

Surface Recombination
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   sim.add_contact(Contact(
       number=2,
       surf_rec=True,
       vsurfn=1e4,  # Electron surface recombination velocity
       vsurfp=1e4   # Hole surface recombination velocity
   ))

Materials
---------

Customize material properties beyond defaults.

.. code-block:: python

   from nanohubpadre import Material

   # Silicon with custom lifetimes
   sim.add_material(Material(
       name="silicon",
       taun0=1e-6,  # Electron lifetime (s)
       taup0=1e-6   # Hole lifetime (s)
   ))

   # Gate oxide with fixed charge
   sim.add_material(Material(
       name="sio2",
       permittivity=3.9,
       qf=1e10  # Fixed charge (/cm^2)
   ))

   # Custom material with modified band structure
   sim.add_material(Material(
       name="custom_si",
       default="silicon",
       eg300=1.12,       # Bandgap at 300K (eV)
       affinity=4.05,    # Electron affinity (eV)
       mun=1400,         # Electron mobility (cm^2/V-s)
       mup=450           # Hole mobility (cm^2/V-s)
   ))

Physical Models
---------------

Configure physical models for the simulation.

.. code-block:: python

   from nanohubpadre import Models

   # Basic drift-diffusion
   sim.models = Models(
       temperature=300,
       srh=True,      # SRH recombination
       conmob=True,   # Concentration-dependent mobility
       fldmob=True    # Field-dependent mobility
   )

   # Advanced models
   sim.models = Models(
       temperature=300,
       srh=True,
       auger=True,        # Auger recombination
       direct=True,       # Radiative recombination
       bgn=True,          # Band-gap narrowing
       impact=True,       # Impact ionization
       conmob=True,
       fldmob=True,
       gatmob=True,       # Gate-field mobility degradation
       e_region=[1, 2],   # Regions for velocity saturation
       g_region=[1]       # Regions for gate mobility
   )

System and Method
-----------------

Configure the solver.

.. code-block:: python

   from nanohubpadre import System, Method

   # Two-carrier Newton solver
   sim.system = System(
       carriers=2,
       electrons=True,
       holes=True,
       newton=True
   )

   # Single carrier (electrons only)
   sim.system = System(
       carriers=1,
       electrons=True,
       newton=True
   )

   # Solver method options
   sim.method = Method(
       trap=True,       # Enable trap convergence aid
       a_trap=0.5,      # Trap factor
       itlimit=50       # Maximum iterations
   )

Solve Commands
--------------

Control the simulation flow.

Initial Solution
~~~~~~~~~~~~~~~~

.. code-block:: python

   from nanohubpadre import Solve

   # Solve for equilibrium
   sim.add_solve(Solve(initial=True, outfile="equilibrium"))

Voltage Sweep
~~~~~~~~~~~~~

.. code-block:: python

   # Linear voltage sweep on electrode 1
   sim.add_solve(Solve(
       v1=0,
       vstep=0.1,
       nsteps=20,
       electrode=1
   ))

   # Multiple electrode voltages
   sim.add_solve(Solve(
       v1=1.0,    # Gate
       v2=0.0,    # Source
       v3=0.5,    # Drain
       v4=0.0     # Substrate
   ))

AC Analysis
~~~~~~~~~~~

.. code-block:: python

   # Small-signal AC analysis
   sim.add_solve(Solve(
       v1=-2.0,
       vstep=0.1,
       nsteps=20,
       electrode=1,
       ac_analysis=True,
       frequency=1e6
   ))

Continuation Methods
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Previous solution as initial guess
   sim.add_solve(Solve(previous=True))

   # Projected continuation
   sim.add_solve(Solve(project=True, vstep=0.05, nsteps=10, electrode=1))

Logging and Output
------------------

.. code-block:: python

   from nanohubpadre import Log

   # Log I-V data
   sim.add_log(Log(ivfile="iv_data"))

   # Log AC data
   sim.add_log(Log(acfile="ac_data"))

   # Turn off logging
   sim.add_log(Log(off=True))

Loading Solutions
-----------------

.. code-block:: python

   from nanohubpadre import Load

   # Load a previous solution
   sim.add_load(Load(infile="equilibrium"))

Plotting
--------

1D Plots
~~~~~~~~

.. code-block:: python

   from nanohubpadre import Plot1D

   # Plot potential along a line
   sim.add_command(Plot1D(
       potential=True,
       x_start=0, x_end=1.0,
       y_start=0.5, y_end=0.5,
       ascii=True,
       outfile="potential"
   ))

   # Plot carrier concentrations
   sim.add_command(Plot1D(
       electrons=True,
       holes=True,
       logarithm=True,
       x_start=0, x_end=1.0,
       y_start=0.05, y_end=0.05,
       ascii=True,
       outfile="carriers"
   ))

3D Plots
~~~~~~~~

.. code-block:: python

   from nanohubpadre import Plot3D

   # Save 3D data for visualization
   sim.add_command(Plot3D(
       potential=True,
       electrons=True,
       outfile="solution3d"
   ))

   # Doping profile
   sim.add_command(Plot3D(
       doping=True,
       outfile="doping3d"
   ))

Complete Workflow Example
-------------------------

.. code-block:: python

   from nanohubpadre import (
       Simulation, Mesh, Region, Electrode, Doping,
       Contact, Material, Models, System, Method,
       Solve, Log, Load, Plot1D, Plot3D
   )

   # Create simulation
   sim = Simulation(title="Complete MOSFET Example")

   # Mesh
   sim.mesh = Mesh(nx=60, ny=40, outfile="mesh")
   sim.mesh.add_x_mesh(1, 0.0)
   sim.mesh.add_x_mesh(30, 0.5, ratio=0.9)
   sim.mesh.add_x_mesh(60, 1.0)
   sim.mesh.add_y_mesh(1, -0.01)  # Oxide
   sim.mesh.add_y_mesh(5, 0.0)    # Interface
   sim.mesh.add_y_mesh(40, 1.0, ratio=1.2)

   # Regions
   sim.add_region(Region(1, ix_low=1, ix_high=60, iy_low=1, iy_high=5,
                        material="sio2", insulator=True))
   sim.add_region(Region(2, ix_low=1, ix_high=60, iy_low=5, iy_high=40,
                        material="silicon", semiconductor=True))

   # Electrodes
   sim.add_electrode(Electrode(1, x_min=0.3, x_max=0.7, iy_low=1, iy_high=1))  # Gate
   sim.add_electrode(Electrode(2, ix_low=1, ix_high=10, iy_low=5, iy_high=5))  # Source
   sim.add_electrode(Electrode(3, ix_low=50, ix_high=60, iy_low=5, iy_high=5)) # Drain
   sim.add_electrode(Electrode(4, ix_low=1, ix_high=60, iy_low=40, iy_high=40)) # Sub

   # Doping
   sim.add_doping(Doping(p_type=True, concentration=1e17, uniform=True, region=2))
   sim.add_doping(Doping(gaussian=True, n_type=True, concentration=1e20,
                        junction=0.1, x_right=0.2, region=2))
   sim.add_doping(Doping(gaussian=True, n_type=True, concentration=1e20,
                        junction=0.1, x_left=0.8, region=2))

   # Contacts
   sim.add_contact(Contact(number=1, n_polysilicon=True))
   sim.add_contact(Contact(number=2, neutral=True))
   sim.add_contact(Contact(number=3, neutral=True))
   sim.add_contact(Contact(number=4, neutral=True))

   # Materials
   sim.add_material(Material(name="silicon", taun0=1e-6, taup0=1e-6))
   sim.add_material(Material(name="sio2", permittivity=3.9, qf=1e10))

   # Models
   sim.models = Models(temperature=300, srh=True, conmob=True, fldmob=True, gatmob=True)
   sim.system = System(carriers=2, newton=True)
   sim.method = Method(trap=True)

   # Solve equilibrium
   sim.add_solve(Solve(initial=True, outfile="eq"))
   sim.add_command(Plot3D(potential=True, electrons=True, outfile="eq_3d"))

   # Gate sweep (Id-Vg)
   sim.add_solve(Solve(v3=0.05))  # Small Vds
   sim.add_log(Log(ivfile="idvg"))
   sim.add_solve(Solve(v1=0, vstep=0.1, nsteps=15, electrode=1))
   sim.add_log(Log(off=True))

   # Load and do drain sweep (Id-Vd)
   sim.add_load(Load(infile="eq"))
   sim.add_solve(Solve(v1=1.0))  # Set Vgs
   sim.add_log(Log(ivfile="idvd"))
   sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=20, electrode=3))

   # Generate deck
   print(sim.generate_deck())
