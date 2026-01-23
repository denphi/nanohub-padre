API Reference
=============

This section provides detailed API documentation for all PyPADRE classes.

Device Factory Functions
------------------------

.. automodule:: nanohubpadre.devices
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nanohubpadre.devices.pn_diode
   :members:
   :undoc-members:

.. automodule:: nanohubpadre.devices.mos_capacitor
   :members:
   :undoc-members:

.. automodule:: nanohubpadre.devices.mosfet
   :members:
   :undoc-members:

.. automodule:: nanohubpadre.devices.mesfet
   :members:
   :undoc-members:

.. automodule:: nanohubpadre.devices.bjt
   :members:
   :undoc-members:

.. automodule:: nanohubpadre.devices.schottky_diode
   :members:
   :undoc-members:

.. automodule:: nanohubpadre.devices.solar_cell
   :members:
   :undoc-members:

Simulation
----------

.. automodule:: nanohubpadre.simulation
   :members:
   :undoc-members:
   :show-inheritance:

Base Classes
------------

.. automodule:: nanohubpadre.base
   :members:
   :undoc-members:
   :show-inheritance:

Mesh
----

.. automodule:: nanohubpadre.mesh
   :members:
   :undoc-members:
   :show-inheritance:

Region
------

.. automodule:: nanohubpadre.region
   :members:
   :undoc-members:
   :show-inheritance:

Electrode
---------

.. automodule:: nanohubpadre.electrode
   :members:
   :undoc-members:
   :show-inheritance:

Doping
------

.. automodule:: nanohubpadre.doping
   :members:
   :undoc-members:
   :show-inheritance:

Contact
-------

.. automodule:: nanohubpadre.contact
   :members:
   :undoc-members:
   :show-inheritance:

Material
--------

.. automodule:: nanohubpadre.material
   :members:
   :undoc-members:
   :show-inheritance:

Models
------

.. automodule:: nanohubpadre.models
   :members:
   :undoc-members:
   :show-inheritance:

Solver
------

.. automodule:: nanohubpadre.solver
   :members:
   :undoc-members:
   :show-inheritance:

Log
---

.. automodule:: nanohubpadre.log
   :members:
   :undoc-members:
   :show-inheritance:

Options
-------

.. automodule:: nanohubpadre.options
   :members:
   :undoc-members:
   :show-inheritance:

Interface
---------

.. automodule:: nanohubpadre.interface
   :members:
   :undoc-members:
   :show-inheritance:

Regrid
------

.. automodule:: nanohubpadre.regrid
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
--------

.. automodule:: nanohubpadre.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Plot3D
------

.. automodule:: nanohubpadre.plot3d
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference
---------------

Device Factory Functions
~~~~~~~~~~~~~~~~~~~~~~~~

==========================  ========================================
Function                    Description
==========================  ========================================
``create_pn_diode``         PN junction diode
``create_mos_capacitor``    MOS capacitor for C-V analysis
``create_mosfet``           NMOS/PMOS transistor
``create_mesfet``           Metal-semiconductor FET
``create_bjt``              NPN/PNP bipolar transistor
``create_schottky_diode``   Schottky barrier diode
``create_solar_cell``       PN junction solar cell
==========================  ========================================

Aliases: ``pn_diode``, ``mos_capacitor``, ``mosfet``, ``mesfet``, ``bjt``,
``schottky_diode``, ``solar_cell``

Core Classes
~~~~~~~~~~~~

==================  ========================================
Class               Description
==================  ========================================
``Simulation``      Main simulation container
``Mesh``            Mesh definition with X/Y/Z lines
``XMesh``           X grid line specification
``YMesh``           Y grid line specification
``ZMesh``           Z grid plane specification
``Region``          Material region definition
``Electrode``       Electrode contact definition
``Doping``          Doping profile specification
``Contact``         Contact boundary conditions
``Material``        Material property customization
``Alloy``           Alloy material definition
==================  ========================================

Models and Solver
~~~~~~~~~~~~~~~~~

==================  ========================================
Class               Description
==================  ========================================
``Models``          Physical model configuration
``System``          Carrier and solver type selection
``Method``          Solver method parameters
``LinAlg``          Linear algebra solver options
==================  ========================================

Solution Control
~~~~~~~~~~~~~~~~

==================  ========================================
Class               Description
==================  ========================================
``Solve``           Solve command for bias conditions
``Log``             I-V and AC data logging
``Load``            Load previous solution
``Options``         Global simulation options
==================  ========================================

Output
~~~~~~

==================  ========================================
Class               Description
==================  ========================================
``Plot1D``          1D line plot output
``Plot2D``          2D contour/surface plot
``Contour``         Contour plot
``Vector``          Vector field plot
``Plot3D``          3D scatter plot output
==================  ========================================

Advanced
~~~~~~~~

==================  ========================================
Class               Description
==================  ========================================
``Interface``       Interface properties
``Surface``         Surface recombination
``Regrid``          Mesh refinement
``Adapt``           Adaptive mesh refinement
``Comment``         Comment line
``Title``           Title line
==================  ========================================

Common Parameters
-----------------

Position Parameters
~~~~~~~~~~~~~~~~~~~

Most classes support both index-based and coordinate-based positioning:

**Index-based** (for rectangular meshes):

* ``ix_low``, ``ix_high``: X index bounds
* ``iy_low``, ``iy_high``: Y index bounds

**Coordinate-based** (in microns):

* ``x_min``, ``x_max``: X coordinate bounds
* ``y_min``, ``y_max``: Y coordinate bounds
* ``z_min``, ``z_max``: Z coordinate bounds

Doping Parameters
~~~~~~~~~~~~~~~~~

* ``concentration``: Peak doping concentration (/cm³)
* ``junction``: Junction depth (microns)
* ``peak``: Peak position (microns)
* ``characteristic``: Characteristic length (microns)
* ``region``: Target region number(s)

Model Flags
~~~~~~~~~~~

* ``srh``: Shockley-Read-Hall recombination
* ``auger``: Auger recombination
* ``direct``: Radiative recombination
* ``bgn``: Band-gap narrowing
* ``impact``: Impact ionization
* ``conmob``: Concentration-dependent mobility
* ``fldmob``: Field-dependent mobility
* ``gatmob``: Gate-field mobility

Solve Parameters
~~~~~~~~~~~~~~~~

* ``initial``: Solve for equilibrium
* ``previous``: Use previous solution
* ``project``: Projected continuation
* ``v1``-``v10``: Electrode voltages
* ``vstep``: Voltage step size
* ``nsteps``: Number of steps
* ``electrode``: Electrode to sweep
* ``ac_analysis``: Enable AC analysis
* ``frequency``: AC frequency (Hz)
