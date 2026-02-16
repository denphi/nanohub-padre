Device Factory Functions
========================

nanohub-padre provides factory functions for creating common semiconductor devices.
Each function returns a fully configured ``Simulation`` object that can be run
immediately or customized further.

All devices support direct ``sim.run()`` execution and built-in visualization
methods (``plot_band_diagram()``, ``plot_iv()``, ``plot_cv()``, etc.).

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Device
   * - ``create_pn_diode``
     - PN junction diode
   * - ``create_mos_capacitor``
     - MOS capacitor with C-V analysis (single- or double-gate)
   * - ``create_mosfet``
     - NMOS / PMOS transistor
   * - ``create_mesfet``
     - Metal-semiconductor FET (Schottky gate)
   * - ``create_bjt``
     - NPN / PNP bipolar junction transistor
   * - ``create_schottky_diode``
     - Schottky (metal-semiconductor) diode
   * - ``create_solar_cell``
     - PN junction solar cell

Error Handling
--------------

Always raise on simulation failure — do not silently ignore a non-zero return code:

.. code-block:: python

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

PN Diode
--------

.. autofunction:: nanohubpadre.devices.pn_diode.create_pn_diode

**Example:**

.. code-block:: python

   from nanohubpadre import create_pn_diode

   sim = create_pn_diode(
       length=2.0,
       p_doping=1e17,
       n_doping=1e17,
       temperature=300,
       log_iv=True,
       forward_sweep=(0.0, 1.0, 0.05),
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_iv(title="PN Diode Forward I-V")
   sim.plot_band_diagram(title="PN Diode Band Diagram")

MOS Capacitor
-------------

.. autofunction:: nanohubpadre.devices.mos_capacitor.create_mos_capacitor

The ``create_mos_capacitor`` function supports:

* **Single-gate** (default, ``gate_config="single"``) and **double-gate** (``gate_config="double"``) configurations
* Built-in HF and LF C-V sweeps (``log_cv=True``, ``log_cv_lf=True``, ``vg_sweep=(v_start, v_end, v_step)``)
* Equilibrium and bias band diagram logging (``log_bands_eq=True``, ``log_bands_bias=True``)
* Equilibrium carrier, potential, and E-field profiles (``log_profiles_eq=True``)

**Single-gate example (Rappture defaults):**

.. code-block:: python

   from nanohubpadre import create_mos_capacitor

   sim = create_mos_capacitor(
       oxide_thickness=0.1,        # 100 nm
       silicon_thickness=5.0,
       substrate_doping=1e15,
       substrate_type="p",
       gate_type="n_poly",
       oxide_permittivity=3.9,
       taun0=1e-9,
       taup0=1e-9,
       temperature=300,
       log_cv=True,
       cv_file="cv_hf",
       ac_frequency=1e6,
       log_cv_lf=True,
       cv_lf_file="cv_lf",
       log_bands_eq=True,
       log_qf_eq=True,
       log_profiles_eq=True,
       vg_sweep=(-3.0, 5.0, 0.08),
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_band_diagram(suffix="eq", title="Band Diagram at Equilibrium")
   sim.plot_carriers(suffix="eq", log_scale=True)
   sim.plot_electrostatics(suffix="eq")
   sim.plot_cv(title="C-V Characteristics (HF + LF)")

**Double-gate example:**

.. code-block:: python

   sim_dg = create_mos_capacitor(
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

   result = sim_dg.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim_dg.plot_band_diagram(title="Double-Gate MOS Cap — Equilibrium")
   sim_dg.plot_cv(title="Double-Gate C-V")

MOSFET
------

.. autofunction:: nanohubpadre.devices.mosfet.create_mosfet

**Example:**

.. code-block:: python

   from nanohubpadre import create_mosfet

   sim = create_mosfet(
       channel_length=0.025,
       device_type="nmos",
       temperature=300,
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_transfer(gate_electrode=3, drain_electrode=2,
                     title="NMOS Transfer Characteristic")
   sim.plot_output(drain_electrode=2, title="NMOS Output Characteristic")

MESFET
------

.. autofunction:: nanohubpadre.devices.mesfet.create_mesfet

**Example:**

.. code-block:: python

   from nanohubpadre import create_mesfet

   sim = create_mesfet(
       channel_length=0.2,
       gate_length=0.2,
       channel_doping=1e17,
       gate_workfunction=4.87,
       device_type="n",
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_transfer(gate_electrode=3, drain_electrode=2)
   sim.plot_output(drain_electrode=2)
   sim.plot_band_diagram()

Bipolar Junction Transistor (BJT)
---------------------------------

.. autofunction:: nanohubpadre.devices.bjt.create_bjt

**Example:**

.. code-block:: python

   from nanohubpadre import create_bjt

   sim = create_bjt(
       emitter_width=1.0,
       base_width=0.3,
       collector_width=2.0,
       emitter_doping=1e20,
       base_doping=1e17,
       collector_doping=1e16,
       device_type="npn",
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_band_diagram(title="NPN BJT Band Diagram")
   sim.plot_gummel(base_electrode=2, collector_electrode=3)

Schottky Diode
--------------

.. autofunction:: nanohubpadre.devices.schottky_diode.create_schottky_diode

**Example:**

.. code-block:: python

   from nanohubpadre import create_schottky_diode

   sim = create_schottky_diode(
       length=2.0,
       doping=1e16,
       doping_type="n",
       workfunction=4.8,
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_iv(log_scale=True, title="Schottky Diode I-V")
   sim.plot_band_diagram()

Solar Cell
----------

.. autofunction:: nanohubpadre.devices.solar_cell.create_solar_cell

.. warning::

   ``create_solar_cell()`` is experimental. PADRE cannot resolve solar-cell
   dark currents (I₀ ~ 10⁻²⁰ A) with the default cross-section. Band diagrams
   and carrier profiles are still valid.

**Example:**

.. code-block:: python

   from nanohubpadre import create_solar_cell

   sim = create_solar_cell(
       emitter_depth=0.5,
       base_thickness=200.0,
       emitter_doping=1e19,
       base_doping=1e16,
       device_type="n_on_p",
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_band_diagram()
   sim.plot_carriers(log_scale=True)

Customizing Generated Devices
------------------------------

The returned ``Simulation`` object can be further customized before running:

.. code-block:: python

   from nanohubpadre import create_pn_diode, Material, Models

   sim = create_pn_diode()

   # Override material properties
   sim.add_material(Material(name="silicon", taun0=1e-7, taup0=1e-7))

   # Override physical models
   sim.models = Models(temperature=350, srh=True, auger=True,
                       conmob=True, fldmob=True, print_models=True)

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_band_diagram()
