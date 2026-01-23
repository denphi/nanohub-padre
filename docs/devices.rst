Device Factory Functions
========================

nanohub-padre provides convenient factory functions for creating common semiconductor devices.
These functions simplify the process of setting up device structures by providing
sensible defaults while allowing full customization.

Overview
--------

The ``nanohubpadre.devices`` module contains factory functions for:

* **PN Diode** - Basic PN junction diode
* **MOS Capacitor** - Metal-oxide-semiconductor capacitor for C-V analysis
* **MOSFET** - NMOS and PMOS transistors
* **MESFET** - Metal-semiconductor FET with Schottky gate
* **BJT** - NPN and PNP bipolar junction transistors
* **Schottky Diode** - Metal-semiconductor junction diode
* **Solar Cell** - PN junction photovoltaic device

All factory functions return a ``Simulation`` object that can be further customized
before generating the input deck.

Quick Example
-------------

.. code-block:: python

   from nanohubpadre import create_mosfet, Solve, Log

   # Create an NMOS transistor with one line
   sim = create_mosfet(channel_length=0.05, device_type="nmos")

   # Add solve commands
   sim.add_solve(Solve(initial=True))
   sim.add_log(Log(ivfile="idvg"))
   sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))

   # Generate the input deck
   print(sim.generate_deck())

PN Diode
--------

.. autofunction:: nanohubpadre.devices.pn_diode.create_pn_diode

**Example:**

.. code-block:: python

   from nanohubpadre import create_pn_diode, Solve, Log

   sim = create_pn_diode(
       length=1.0,
       junction_position=0.5,
       p_doping=1e17,
       n_doping=1e17,
       temperature=300
   )

   sim.add_solve(Solve(initial=True))
   sim.add_log(Log(ivfile="iv"))
   sim.add_solve(Solve(project=True, vstep=0.05, nsteps=20, electrode=1))

MOS Capacitor
-------------

.. autofunction:: nanohubpadre.devices.mos_capacitor.create_mos_capacitor

**Example:**

.. code-block:: python

   from nanohubpadre import create_mos_capacitor, Solve, Log

   sim = create_mos_capacitor(
       oxide_thickness=0.002,      # 2nm oxide
       silicon_thickness=0.03,
       substrate_doping=1e18,
       substrate_type="p",
       gate_type="n_poly"
   )

   sim.add_solve(Solve(initial=True))
   sim.add_log(Log(acfile="cv"))
   sim.add_solve(Solve(
       v1=-2.0, vstep=0.2, nsteps=20, electrode=1,
       ac_analysis=True, frequency=1e6
   ))

MOSFET
------

.. autofunction:: nanohubpadre.devices.mosfet.create_mosfet

**Example:**

.. code-block:: python

   from nanohubpadre import create_mosfet, Solve, Log, Load

   # Create NMOS transistor
   sim = create_mosfet(
       channel_length=0.025,
       source_drain_doping=1e20,
       channel_doping=1e19,
       device_type="nmos"
   )

   # Equilibrium
   sim.add_solve(Solve(initial=True, outfile="init"))

   # Transfer characteristic
   sim.add_solve(Solve(v2=0.05))
   sim.add_log(Log(ivfile="idvg"))
   sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))

   # Output characteristic
   sim.add_load(Load(infile="init"))
   sim.add_solve(Solve(v3=1.0))
   sim.add_log(Log(ivfile="idvd"))
   sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))

MESFET
------

.. autofunction:: nanohubpadre.devices.mesfet.create_mesfet

**Example:**

.. code-block:: python

   from nanohubpadre import create_mesfet, Solve, Log

   sim = create_mesfet(
       channel_length=0.2,
       gate_length=0.2,
       channel_doping=1e17,
       gate_workfunction=4.87,
       device_type="n"
   )

   sim.add_solve(Solve(initial=True))
   sim.add_solve(Solve(v3=0, vstep=-0.1, nsteps=5, electrode=3))
   sim.add_log(Log(ivfile="idvd"))
   sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))

Bipolar Junction Transistor (BJT)
---------------------------------

.. autofunction:: nanohubpadre.devices.bjt.create_bjt

**Example:**

.. code-block:: python

   from nanohubpadre import create_bjt, Solve, Log

   sim = create_bjt(
       emitter_width=1.0,
       base_width=0.3,
       collector_width=2.0,
       emitter_doping=1e20,
       base_doping=1e17,
       collector_doping=1e16,
       device_type="npn"
   )

   sim.add_solve(Solve(initial=True))
   sim.add_solve(Solve(v2=0.7))  # Forward bias base
   sim.add_log(Log(ivfile="ic_vce"))
   sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=30, electrode=3))

Schottky Diode
--------------

.. autofunction:: nanohubpadre.devices.schottky_diode.create_schottky_diode

**Example:**

.. code-block:: python

   from nanohubpadre import create_schottky_diode, Solve, Log

   sim = create_schottky_diode(
       length=2.0,
       doping=1e16,
       doping_type="n",
       workfunction=4.8,
       barrier_lowering=True
   )

   sim.add_solve(Solve(initial=True))
   sim.add_log(Log(ivfile="iv"))
   sim.add_solve(Solve(project=True, vstep=0.05, nsteps=20, electrode=1))

Solar Cell
----------

.. autofunction:: nanohubpadre.devices.solar_cell.create_solar_cell

**Example:**

.. code-block:: python

   from nanohubpadre import create_solar_cell, Solve, Log

   sim = create_solar_cell(
       emitter_depth=0.5,
       base_thickness=200.0,
       emitter_doping=1e19,
       base_doping=1e16,
       device_type="n_on_p",
       front_surface_velocity=1e4,
       back_surface_velocity=1e7
   )

   sim.add_solve(Solve(initial=True))
   sim.add_log(Log(ivfile="iv_dark"))
   sim.add_solve(Solve(v1=0, vstep=0.05, nsteps=15, electrode=1))

Aliases
-------

For convenience, shorter aliases are available for all factory functions:

.. code-block:: python

   from nanohubpadre import (
       pn_diode,        # alias for create_pn_diode
       mos_capacitor,   # alias for create_mos_capacitor
       mosfet,          # alias for create_mosfet
       mesfet,          # alias for create_mesfet
       bjt,             # alias for create_bjt
       schottky_diode,  # alias for create_schottky_diode
       solar_cell       # alias for create_solar_cell
   )

   # These are equivalent:
   sim1 = create_mosfet(device_type="nmos")
   sim2 = mosfet(device_type="nmos")

Customizing Generated Devices
-----------------------------

The returned ``Simulation`` object can be further customized:

.. code-block:: python

   from nanohubpadre import create_pn_diode, Material, Models

   # Create base device
   sim = create_pn_diode()

   # Add custom material properties
   sim.add_material(Material(
       name="silicon",
       taun0=1e-7,
       taup0=1e-7
   ))

   # Override models
   sim.models = Models(
       temperature=350,
       srh=True,
       auger=True,
       bgn=True,
       conmob=True,
       fldmob=True
   )

   # Add additional regions, electrodes, etc.
   # ...

   print(sim.generate_deck())
