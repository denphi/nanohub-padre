Examples
========

This section contains complete example simulations for common semiconductor devices.

Using Device Factory Functions
------------------------------

The easiest way to create device simulations is using the factory functions.
See the :doc:`devices` documentation for details on all available factories.

.. code-block:: python

   from nanohubpadre import create_pn_diode, create_mosfet, create_bjt, Solve, Log

   # Create devices with one line each
   diode = create_pn_diode(p_doping=1e17, n_doping=1e17)
   nmos = create_mosfet(channel_length=0.05, device_type="nmos")
   npn = create_bjt(base_width=0.3, device_type="npn")

   # Add solve commands and generate deck
   diode.add_solve(Solve(initial=True))
   print(diode.generate_deck())

Device Factory Examples
~~~~~~~~~~~~~~~~~~~~~~~

Complete examples using device factory functions are in ``examples/devices/``:

* ``pn_diode_example.py`` - PN diode I-V characteristics
* ``mos_capacitor_example.py`` - MOS capacitor C-V analysis
* ``mosfet_example.py`` - NMOS transfer and output characteristics
* ``mesfet_example.py`` - MESFET output characteristics
* ``bjt_example.py`` - NPN common-emitter characteristics
* ``schottky_diode_example.py`` - Schottky diode forward/reverse I-V
* ``solar_cell_example.py`` - Solar cell dark I-V

Run any example:

.. code-block:: bash

   cd padre
   PYTHONPATH=. python3 examples/devices/mosfet_example.py > mosfet.inp
   padre < mosfet.inp > mosfet.out

Manual Device Construction Examples
-----------------------------------

The examples below show how to build devices manually, giving you full control
over the structure. These are useful when you need custom device geometries.

PN Diode
--------

A simple PN junction diode for I-V characterization.

.. literalinclude:: ../examples/pndiode.py
   :language: python
   :linenos:
   :caption: examples/pndiode.py

The generated input deck:

.. code-block:: text

   TITLE  pn diode (setup)
   options po
   mesh rect nx=200 ny=3 width=1 outf=mesh
   x.m n=1 l=0 r=1
   x.m n=100 l=0.5 r=0.8
   x.m n=200 l=1.0 r=1.05
   y.m n=1 l=0 r=1
   y.m n=3 l=1 r=1
   region silicon num=1 ix.l=1 ix.h=100 iy.l=1 iy.h=3
   region silicon num=1 ix.l=100 ix.h=200 iy.l=1 iy.h=3
   elec num=1 ix.l=1 ix.h=1 iy.l=1 iy.h=3
   elec num=2 ix.l=200 ix.h=200 iy.l=1 iy.h=3
   ...
   solve init
   ...
   solve proj vstep=0.03 nsteps=20 elect=1
   end

MOS Capacitor
-------------

A MOS capacitor with oxide-silicon-oxide structure for C-V analysis.

.. literalinclude:: ../examples/moscap.py
   :language: python
   :linenos:
   :caption: examples/moscap.py

Key features:

* Three-region structure (oxide-silicon-oxide)
* AC analysis for capacitance extraction
* N+ polysilicon gate contacts
* P-type substrate doping

MOSFET
------

An NMOS transistor with source, drain, gate, and substrate contacts.

.. literalinclude:: ../examples/mosfet_equivalent.py
   :language: python
   :linenos:
   :caption: examples/mosfet_equivalent.py

Key features:

* Seven-region structure (substrate, source, drain, channel, gate oxide, fillers)
* Four electrodes (source, drain, gate, substrate)
* High source/drain doping (N+ 1e20)
* P-type channel and substrate
* Transfer characteristic (Id-Vg) and output characteristic (Id-Vd) sweeps
* Save/load solution capability for multiple bias conditions

MESFET
------

A Metal-Semiconductor FET with Schottky gate.

.. literalinclude:: ../examples/mesfet.py
   :language: python
   :linenos:
   :caption: examples/mesfet.py

Key features:

* Four silicon regions with different doping
* Schottky gate contact with specified work function
* Neutral ohmic contacts for source and drain
* Single carrier (electron) simulation
* Band-gap narrowing model

Single MOS Gap
--------------

A simple oxide-silicon structure for MOS physics study.

.. literalinclude:: ../examples/single_mosgap.py
   :language: python
   :linenos:
   :caption: examples/single_mosgap.py

Key features:

* Two-region structure (oxide on silicon)
* Gate sweep with AC analysis
* Comprehensive 1D plots of band structure, carriers, and fields
* SRH recombination model

Running the Examples
--------------------

To run any example:

.. code-block:: bash

   # Navigate to the padre directory
   cd padre

   # Run the example and save the output deck
   PYTHONPATH=. python3 examples/pndiode.py > pndiode.inp

   # Run PADRE (if installed)
   padre < pndiode.inp > pndiode.out

Or run interactively:

.. code-block:: python

   from examples.pndiode import create_pndiode_simulation

   sim = create_pndiode_simulation()

   # Print the deck
   print(sim.generate_deck())

   # Write to file
   sim.write_deck("pndiode.inp")

   # Run PADRE (if installed)
   result = sim.run(padre_executable="padre")
   print(result.stdout)

Customizing Examples
--------------------

The example functions can be modified for different device parameters:

.. code-block:: python

   from examples.pndiode import create_pndiode_simulation

   # Create base simulation
   sim = create_pndiode_simulation()

   # Modify doping levels
   sim._dopings.clear()
   from nanohubpadre import Doping
   sim.add_doping(Doping(region=1, p_type=True, concentration=5e16, uniform=True,
                        x_left=0, x_right=0.5, y_top=0, y_bottom=1))
   sim.add_doping(Doping(region=1, n_type=True, concentration=1e18, uniform=True,
                        x_left=0.5, x_right=1.0, y_top=0, y_bottom=1))

   # Change temperature
   from nanohubpadre import Models
   sim.models = Models(srh=True, conmob=True, fldmob=True, temperature=350)

   print(sim.generate_deck())

Creating New Device Structures
------------------------------

Use the examples as templates for your own devices:

.. code-block:: python

   from nanohubpadre import (
       Simulation, Mesh, Region, Electrode, Doping,
       Contact, Material, Models, System, Solve
   )

   def create_my_device():
       sim = Simulation(title="My Custom Device")

       # Define your mesh...
       sim.mesh = Mesh(nx=..., ny=...)

       # Add regions...
       sim.add_region(Region(...))

       # Continue with electrodes, doping, etc.

       return sim

   if __name__ == "__main__":
       sim = create_my_device()
       print(sim.generate_deck())
