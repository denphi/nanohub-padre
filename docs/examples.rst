Examples
========

All examples are available as Jupyter notebooks in ``examples/notebooks/``.
Each notebook can be run on nanoHUB or locally with PADRE installed.

Jupyter Notebooks
-----------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - #
     - Notebook
     - Description
   * - 00
     - ``00_Introduction.ipynb``
     - Overview and setup
   * - 01
     - ``01_Library_Overview.ipynb``
     - Component-by-component tour of the API
   * - 02
     - ``02_PN_Diode.ipynb``
     - I-V characteristics, band diagrams, ideality factor
   * - 03
     - ``03_Schottky_Diode.ipynb``
     - Schottky barrier, forward/reverse I-V, workfunction effects
   * - 04
     - ``04_MOSFET.ipynb``
     - Transfer and output characteristics, threshold voltage
   * - 05
     - ``05_BJT.ipynb``
     - Gummel plot, β extraction, common-emitter output curves
   * - 06
     - ``06_MOS_Capacitor.ipynb``
     - C-V analysis, band bending, single- and double-gate configurations
   * - 07
     - ``07_MESFET.ipynb``
     - Output and transfer characteristics, doping and workfunction effects

Using Device Factory Functions
-------------------------------

The simplest workflow uses factory functions that handle mesh, regions,
doping, contacts, materials, and solve commands automatically:

.. code-block:: python

   from nanohubpadre import create_mos_capacitor

   sim = create_mos_capacitor(
       oxide_thickness=0.1,
       silicon_thickness=5.0,
       substrate_doping=1e15,
       substrate_type="p",
       gate_type="n_poly",
       log_cv=True,
       log_cv_lf=True,
       log_bands_eq=True,
       log_profiles_eq=True,
       vg_sweep=(-3.0, 5.0, 0.08),
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   # All visualization through sim methods — no manual parsing
   sim.plot_band_diagram(suffix="eq")
   sim.plot_carriers(suffix="eq", log_scale=True)
   sim.plot_electrostatics(suffix="eq")
   sim.plot_cv()

MOS Capacitor (Rappture Reference)
-----------------------------------

The following reproduces the nanoHUB MOSCap Rappture tool defaults exactly.

.. code-block:: python

   from nanohubpadre import create_mos_capacitor

   sim = create_mos_capacitor(
       oxide_thickness=0.1,         # 100 nm (Rappture default)
       ny_oxide=100,
       silicon_thickness=5.0,
       ny_silicon=200,
       substrate_doping=1e15,       # Na = 1e15 /cm³
       substrate_type="p",
       gate_type="n_poly",          # n+ poly silicon gate
       oxide_permittivity=3.9,
       taun0=1e-9,                  # 1 ns carrier lifetimes
       taup0=1e-9,
       temperature=300,
       log_cv=True,
       cv_file="cv_hf",
       ac_frequency=1e6,            # HF: 1 MHz
       log_cv_lf=True,
       cv_lf_file="cv_lf",
       ac_frequency_lf=1.0,         # LF: 1 Hz
       log_bands_eq=True,
       log_qf_eq=True,
       log_profiles_eq=True,
       vg_sweep=(-3.0, 5.0, 0.08),  # -3V to +5V, 100 steps
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_band_diagram(suffix="eq", title="Band Diagram at Equilibrium (Vg=0)")
   sim.plot_carriers(suffix="eq", log_scale=True)
   sim.plot_electrostatics(suffix="eq")
   sim.plot_cv(title="MOS Capacitor C-V (HF 1 MHz + LF 1 Hz)")

Double-Gate MOS Capacitor
--------------------------

.. code-block:: python

   from nanohubpadre import create_mos_capacitor

   sim = create_mos_capacitor(
       oxide_thickness=0.005,
       silicon_thickness=0.02,
       substrate_doping=1e17,
       substrate_type="p",
       gate_type="n_poly",
       gate_config="double",
       back_oxide_thickness=0.005,
       back_gate_type="n_poly",
       log_cv=True,
       log_bands_eq=True,
       vg_sweep=(-2.0, 2.0, 0.1),
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_band_diagram(title="Double-Gate — Equilibrium")
   sim.plot_cv(title="Double-Gate C-V")

PN Diode
--------

.. code-block:: python

   from nanohubpadre import create_pn_diode

   sim = create_pn_diode(
       length=2.0,
       p_doping=1e17,
       n_doping=1e17,
       temperature=300,
       log_iv=True,
       forward_sweep=(0.0, 1.0, 0.05),
       log_bands_eq=True,
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_iv(title="PN Diode I-V", log_scale=True)
   sim.plot_band_diagram(title="PN Diode Band Diagram")

MOSFET
------

.. code-block:: python

   from nanohubpadre import create_mosfet

   sim = create_mosfet(channel_length=0.025, device_type="nmos")

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_transfer(gate_electrode=3, drain_electrode=2,
                     title="NMOS Transfer Characteristic")

Low-Level (Manual) Example
---------------------------

For full control over the device structure, build the simulation from scratch:

.. literalinclude:: ../examples/moscap.py
   :language: python
   :linenos:
   :caption: examples/moscap.py — Manual MOS capacitor

Running the Examples
--------------------

To run a Python example script:

.. code-block:: bash

   cd padre
   PYTHONPATH=. python3 examples/moscap.py

To run a notebook on nanoHUB, upload it and select the PADRE kernel.
To run locally, ensure PADRE is in your ``PATH`` and start Jupyter:

.. code-block:: bash

   jupyter notebook examples/notebooks/06_MOS_Capacitor.ipynb
