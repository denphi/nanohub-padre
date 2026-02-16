User Guide
==========

This guide covers all components of nanohub-padre in detail.

Running Simulations
-------------------

The recommended workflow is to call ``sim.run()`` after building the simulation.
On failure it returns a non-zero ``returncode`` — always check and raise so
errors surface as Jupyter cell exceptions rather than silent output.

.. code-block:: python

   from nanohubpadre import create_pn_diode

   sim = create_pn_diode(p_doping=1e17, n_doping=1e17, log_iv=True,
                         forward_sweep=(0.0, 1.0, 0.05))

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   # Access outputs directly
   sim.plot_iv(title="PN Diode I-V")
   sim.plot_band_diagram(title="PN Diode Band Diagram")

Visualization Methods
---------------------

After a successful ``sim.run()`` all output data is accessible through
``sim.outputs`` or via convenience methods on the ``Simulation`` object.

Band Diagrams
~~~~~~~~~~~~~

.. code-block:: python

   # Plot equilibrium band diagram (Ec, Ev, Efn, Efp)
   sim.plot_band_diagram(title="Equilibrium")

   # Plot a specific suffix (e.g. after a bias sweep)
   sim.plot_band_diagram(suffix="eq")
   sim.plot_band_diagram(suffix="bias")

   # Choose backend
   sim.plot_band_diagram(backend="matplotlib")
   sim.plot_band_diagram(backend="plotly")

Carrier Concentrations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   sim.plot_carriers(suffix="eq", log_scale=True,
                     title="Carrier Concentrations at Equilibrium")

C-V Characteristics
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Normalized C/Cox (default)
   sim.plot_cv(title="MOS Capacitor C-V")

   # Raw capacitance in Farads
   sim.plot_cv(normalize=False)

Electrostatics
~~~~~~~~~~~~~~

.. code-block:: python

   # Side-by-side: potential and electric field
   sim.plot_electrostatics(suffix="eq", title="Electrostatics at Equilibrium")

I-V Characteristics
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Default: current_electrode=1
   sim.plot_iv(title="Forward I-V")

   # Specify electrode explicitly
   sim.plot_iv(current_electrode=2, log_scale=True)

Transfer / Output Characteristics (FET/BJT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   sim.plot_transfer(gate_electrode=3, drain_electrode=2)
   sim.plot_output(drain_electrode=2)
   sim.plot_gummel(base_electrode=2, collector_electrode=3)

2D Contour Maps
~~~~~~~~~~~~~~~

.. code-block:: python

   sim.plot_contour("pot_bias", title="Potential", colorscale="RdBu_r")
   sim.plot_contour("el_bias", title="Electrons", log_scale=True)

Accessing Raw Output Data
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get a named output (returns PlotData with .x and .y arrays)
   pot = sim.outputs.get("pot_eq")
   print(pot.x, pot.y)

   # Get all AC (C-V) data
   ac_map = sim.outputs.get_ac_data()  # dict name -> ACData

   # Get by variable type
   band_data = sim.outputs.get_by_variable("band_con")

Mesh Definition
---------------

The mesh defines the computational grid for your device simulation.

Rectangular Mesh
~~~~~~~~~~~~~~~~

.. code-block:: python

   from nanohubpadre import Mesh

   mesh = Mesh(nx=100, ny=50)

   # X grid lines — first node has no ratio
   mesh.add_x_mesh(1, 0.0)
   mesh.add_x_mesh(50, 0.5, ratio=0.8)
   mesh.add_x_mesh(100, 1.0, ratio=1.2)

   # Y grid lines
   mesh.add_y_mesh(1, 0.0)
   mesh.add_y_mesh(25, 0.1, ratio=0.7)
   mesh.add_y_mesh(50, 1.0, ratio=1.3)

The ``ratio`` parameter controls mesh grading:

* ``ratio < 1``: Mesh becomes finer toward this node
* ``ratio = 1``: Uniform spacing
* ``ratio > 1``: Mesh becomes coarser toward this node
* Omit ``ratio`` (or set to ``None``) on the first node — PADRE ignores it there

Loading Existing Mesh
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   mesh = Mesh(infile="previous_mesh.pg", previous=True)

Regions
-------

.. code-block:: python

   from nanohubpadre import Region

   silicon = Region(1, ix_low=1, ix_high=100, iy_low=5, iy_high=50,
                    material="silicon", semiconductor=True)

   oxide = Region(2, ix_low=20, ix_high=80, iy_low=1, iy_high=5,
                  material="sio2", insulator=True)

The material type keyword (``semiconductor=True`` / ``insulator=True``) is
appended **after** the ``name=`` parameter in the generated PADRE deck, which
is required by PADRE's parser.

Electrodes
----------

.. code-block:: python

   from nanohubpadre import Electrode

   source    = Electrode(1, ix_low=1,   ix_high=20,  iy_low=5, iy_high=5)
   gate      = Electrode(2, x_min=0.3,  x_max=0.7,   y_min=-0.01, y_max=-0.01)
   drain     = Electrode(3, ix_low=80,  ix_high=100, iy_low=5, iy_high=5)
   substrate = Electrode(4, ix_low=1,   ix_high=100, iy_low=50, iy_high=50)

Doping Profiles
---------------

Uniform Doping
~~~~~~~~~~~~~~

.. code-block:: python

   from nanohubpadre import Doping

   sim.add_doping(Doping(p_type=True, concentration=1e17, uniform=True, region=1))
   sim.add_doping(Doping(n_type=True, concentration=1e20, uniform=True,
                         x_left=0.0, x_right=0.3))

Gaussian Profile
~~~~~~~~~~~~~~~~

.. code-block:: python

   sim.add_doping(Doping(gaussian=True, n_type=True, concentration=5e19,
                         junction=0.2, peak=0.0, characteristic=0.05))

Contacts
--------

.. code-block:: python

   from nanohubpadre import Contact

   sim.add_contact(Contact(all_contacts=True, neutral=True))
   sim.add_contact(Contact(number=1, n_polysilicon=True))
   sim.add_contact(Contact(number=2, workfunction=4.87))

Card ordering in the generated deck: doping → contact → material → interface → models.
This matches the Rappture reference decks.

Materials
---------

.. code-block:: python

   from nanohubpadre import Material

   sim.add_material(Material(name="silicon", taun0=1e-6, taup0=1e-6))

   # Gate oxide — use permi= for permittivity (PADRE keyword)
   sim.add_material(Material(name="sio2", permittivity=3.9, qf=0))

The ``permittivity`` parameter is output as ``permi=`` in the deck, which is
the keyword PADRE accepts.

Interfaces
----------

Interface cards are always emitted for oxide/semiconductor boundaries, even
when ``qf=0``.

.. code-block:: python

   from nanohubpadre import Interface

   sim.add_interface(Interface(number=1, qf=0))

Physical Models
---------------

.. code-block:: python

   from nanohubpadre import Models

   sim.models = Models(
       temperature=300,
       srh=True,
       conmob=True,
       fldmob=True,
       print_models=True,   # Emit 'print' flag in PADRE deck
   )

Solve Commands
--------------

.. code-block:: python

   from nanohubpadre import Solve

   sim.add_solve(Solve(initial=True, outfile="eq"))

   # Voltage sweep
   sim.add_solve(Solve(v1=0, vstep=0.1, nsteps=20, electrode=1))

   # AC analysis (C-V)
   sim.add_solve(Solve(v1=-2.0, vstep=0.1, nsteps=20, electrode=1,
                       ac_analysis=True, frequency=1e6))

   # Continuation — needs one prior solution
   sim.add_solve(Solve(previous=True))

   # Projection — needs two prior solutions
   sim.add_solve(Solve(project=True, vstep=0.05, nsteps=10, electrode=1))

Logging and Output
------------------

.. code-block:: python

   from nanohubpadre import Log

   sim.add_log(Log(ivfile="iv_data"))    # I-V log
   sim.add_log(Log(acfile="cv_data"))    # AC/C-V log
   sim.add_log(Log(off=True))            # Stop logging

1D Profile Logging
~~~~~~~~~~~~~~~~~~

Use the helper methods on ``Simulation`` to add ``Plot1D`` commands cleanly:

.. code-block:: python

   x_mid = device_width / 2

   sim.log_band_diagram("eq", x_start=x_mid, x_end=x_mid,
                         y_start=0.0, y_end=total_thickness, include_qf=True)

   sim.log_carriers("carriers_eq", x_start=x_mid, x_end=x_mid,
                    y_start=0.0, y_end=total_thickness)

   sim.log_potential("pot_eq", x_start=x_mid, x_end=x_mid,
                     y_start=0.0, y_end=total_thickness)

   sim.log_efield("ef_eq", x_start=x_mid, x_end=x_mid,
                  y_start=0.0, y_end=total_thickness)

Complete Workflow Example
-------------------------

.. code-block:: python

   from nanohubpadre import (
       Simulation, Mesh, Region, Electrode, Doping,
       Contact, Material, Interface, Models, System,
       Solve, Log, Plot3D
   )

   sim = Simulation(title="MOSFET Example")

   sim.mesh = Mesh(nx=60, ny=40)
   sim.mesh.add_x_mesh(1, 0.0)
   sim.mesh.add_x_mesh(30, 0.5, ratio=0.9)
   sim.mesh.add_x_mesh(60, 1.0)
   sim.mesh.add_y_mesh(1, -0.01)
   sim.mesh.add_y_mesh(5, 0.0)
   sim.mesh.add_y_mesh(40, 1.0, ratio=1.2)

   sim.add_region(Region(1, ix_low=1, ix_high=60, iy_low=1, iy_high=5,
                         material="sio2", insulator=True))
   sim.add_region(Region(2, ix_low=1, ix_high=60, iy_low=5, iy_high=40,
                         material="silicon", semiconductor=True))

   sim.add_electrode(Electrode(1, x_min=0.3, x_max=0.7, iy_low=1, iy_high=1))
   sim.add_electrode(Electrode(2, ix_low=1,  ix_high=10, iy_low=5, iy_high=5))
   sim.add_electrode(Electrode(3, ix_low=50, ix_high=60, iy_low=5, iy_high=5))
   sim.add_electrode(Electrode(4, ix_low=1,  ix_high=60, iy_low=40, iy_high=40))

   sim.add_doping(Doping(p_type=True, concentration=1e17, uniform=True, region=2))
   sim.add_doping(Doping(gaussian=True, n_type=True, concentration=1e20,
                         junction=0.1, x_right=0.2, region=2))
   sim.add_doping(Doping(gaussian=True, n_type=True, concentration=1e20,
                         junction=0.1, x_left=0.8, region=2))

   # Card order: doping → contact → material → interface → models
   sim.add_contact(Contact(number=1, n_polysilicon=True))
   sim.add_contact(Contact(number=2, neutral=True))
   sim.add_contact(Contact(number=3, neutral=True))
   sim.add_contact(Contact(number=4, neutral=True))

   sim.add_material(Material(name="silicon", taun0=1e-6, taup0=1e-6))
   sim.add_material(Material(name="sio2", permittivity=3.9, qf=0))

   sim.add_interface(Interface(number=1, qf=0))

   sim.models = Models(temperature=300, srh=True, conmob=True, fldmob=True,
                       print_models=True)
   sim.system = System(electrons=True, holes=True, newton=True)

   sim.add_solve(Solve(initial=True, outfile="eq"))
   sim.add_solve(Solve(v3=0.05))
   sim.add_log(Log(ivfile="idvg"))
   sim.add_solve(Solve(v1=0, vstep=0.1, nsteps=15, electrode=1))
   sim.add_log(Log(off=True))

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_transfer(gate_electrode=1, drain_electrode=3,
                     title="NMOS Transfer Characteristic")
