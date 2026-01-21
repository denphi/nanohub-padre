API Reference
=============

This section provides detailed API documentation for all PyPADRE classes.

Simulation
----------

.. automodule:: pypadre.simulation
   :members:
   :undoc-members:
   :show-inheritance:

Base Classes
------------

.. automodule:: pypadre.base
   :members:
   :undoc-members:
   :show-inheritance:

Mesh
----

.. automodule:: pypadre.mesh
   :members:
   :undoc-members:
   :show-inheritance:

Region
------

.. automodule:: pypadre.region
   :members:
   :undoc-members:
   :show-inheritance:

Electrode
---------

.. automodule:: pypadre.electrode
   :members:
   :undoc-members:
   :show-inheritance:

Doping
------

.. automodule:: pypadre.doping
   :members:
   :undoc-members:
   :show-inheritance:

Contact
-------

.. automodule:: pypadre.contact
   :members:
   :undoc-members:
   :show-inheritance:

Material
--------

.. automodule:: pypadre.material
   :members:
   :undoc-members:
   :show-inheritance:

Models
------

.. automodule:: pypadre.models
   :members:
   :undoc-members:
   :show-inheritance:

Solver
------

.. automodule:: pypadre.solver
   :members:
   :undoc-members:
   :show-inheritance:

Log
---

.. automodule:: pypadre.log
   :members:
   :undoc-members:
   :show-inheritance:

Options
-------

.. automodule:: pypadre.options
   :members:
   :undoc-members:
   :show-inheritance:

Interface
---------

.. automodule:: pypadre.interface
   :members:
   :undoc-members:
   :show-inheritance:

Regrid
------

.. automodule:: pypadre.regrid
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
--------

.. automodule:: pypadre.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Plot3D
------

.. automodule:: pypadre.plot3d
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference
---------------

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
