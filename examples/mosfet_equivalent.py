#!/usr/bin/env python3
"""
Python equivalent of mosfet.in for PADRE simulation.

This script generates the same PADRE input deck as mosfet.in
NMOS MOSFET structure
"""

from nanohubpadre import (
    Simulation, Mesh, Region, Electrode, Doping, Contact,
    Models, System, Solve, Log, Plot3D, Load
)


def create_mosfet_simulation():
    """Create the MOSFET simulation equivalent to mosfet.in"""

    sim = Simulation(title="MOSFET - NMOS")

    # Mesh specification
    sim.mesh = Mesh(nx=51, ny=51)
    sim.mesh.add_x_mesh(1, 0)
    sim.mesh.add_x_mesh(15, 0.05, ratio=0.8)
    sim.mesh.add_x_mesh(26, 0.0625, ratio=1.25)
    sim.mesh.add_x_mesh(36, 0.075, ratio=0.8)
    sim.mesh.add_x_mesh(51, 0.125, ratio=1.25)
    sim.mesh.add_y_mesh(1, 0)
    sim.mesh.add_y_mesh(25, 0.068, ratio=0.8)
    sim.mesh.add_y_mesh(36, 0.0805, ratio=1.25)
    sim.mesh.add_y_mesh(46, 0.093, ratio=0.8)
    sim.mesh.add_y_mesh(51, 0.0942, ratio=1.25)

    # Regions
    # Substrate
    sim.add_region(Region(1, ix_low=1, ix_high=51, iy_low=1, iy_high=25, silicon=True))
    # Source
    sim.add_region(Region(2, ix_low=1, ix_high=15, iy_low=25, iy_high=46, silicon=True))
    # Drain
    sim.add_region(Region(3, ix_low=36, ix_high=51, iy_low=25, iy_high=46, silicon=True))
    # Channel
    sim.add_region(Region(4, ix_low=15, ix_high=36, iy_low=25, iy_high=46, silicon=True))
    # Gate oxide
    sim.add_region(Region(5, ix_low=15, ix_high=36, iy_low=46, iy_high=51, oxide=True))
    # Fillers
    sim.add_region(Region(6, ix_low=1, ix_high=15, iy_low=46, iy_high=51, oxide=True))
    sim.add_region(Region(7, ix_low=36, ix_high=51, iy_low=46, iy_high=51, oxide=True))

    # Electrodes
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=15, iy_low=46, iy_high=46))   # Source
    sim.add_electrode(Electrode(2, ix_low=36, ix_high=51, iy_low=46, iy_high=46))  # Drain
    sim.add_electrode(Electrode(3, ix_low=15, ix_high=36, iy_low=51, iy_high=51))  # Gate
    sim.add_electrode(Electrode(4, ix_low=1, ix_high=51, iy_low=1, iy_high=1))     # Substrate

    # Doping
    sim.add_doping(Doping(region=[2, 3], uniform=True, concentration=1e20, n_type=True))
    sim.add_doping(Doping(region=4, uniform=True, concentration=1e19, p_type=True))
    sim.add_doping(Doping(region=1, uniform=True, concentration=5e16, p_type=True))

    # Plot doping
    sim.add_command(Plot3D(doping=True, outfile="doping"))

    # Contact
    sim.add_contact(Contact(number=3, n_polysilicon=True))

    # Models and system
    sim.models = Models(temperature=300, bgn=True)
    sim.system = System(newton=True, carriers=1, electrons=True)

    # Solve initial
    sim.add_solve(Solve(initial=True, outfile="initsol"))
    sim.add_command(Plot3D(potential=True, electrons=True, outfile="equ"))

    # Solve for transfer characteristic
    sim.add_solve(Solve(v2=0, vstep=0.05, nsteps=1, electrode=2))
    sim.add_solve(Solve(v2=0.05))
    sim.add_log(Log(ivfile="idvg"))
    sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))
    sim.add_log(Log(off=True))

    # Load initial and sweep gate voltage
    sim.add_load(Load(infile="initsol"))
    sim.add_solve(Solve(v3=0.0, outfile="vgn0"))
    sim.add_solve(Solve(v3=0.0, vstep=0.0833333333333, nsteps=6, electrode=3))
    sim.add_solve(Solve(v3=0.5, outfile="vgn1"))
    sim.add_solve(Solve(v3=0.5, vstep=0.0833333333333, nsteps=6, electrode=3))
    sim.add_solve(Solve(v3=1.0, outfile="vgn2"))
    sim.add_solve(Solve(v3=1.0, vstep=0.0833333333333, nsteps=6, electrode=3))
    sim.add_solve(Solve(v3=1.5, outfile="vgn3"))
    sim.add_solve(Solve(v3=1.5, vstep=0.0833333333333, nsteps=6, electrode=3))
    sim.add_solve(Solve(v3=2.0, outfile="vgn4"))

    # Output characteristics at different gate voltages
    sim.add_load(Load(infile="vgn0"))
    sim.add_log(Log(ivfile="idvd"))
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))

    sim.add_load(Load(infile="vgn1"))
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))

    sim.add_load(Load(infile="vgn2"))
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))

    sim.add_load(Load(infile="vgn3"))
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))

    sim.add_load(Load(infile="vgn4"))
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))
    sim.add_log(Log(off=True))

    # Final plot
    sim.add_command(Plot3D(potential=True, electrons=True, outfile="wmc"))

    return sim


if __name__ == "__main__":
    sim = create_mosfet_simulation()
    deck = sim.generate_deck()
    print(deck)
