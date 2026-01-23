#!/usr/bin/env python3
"""
Python equivalent of pndiode.in for PADRE simulation.

This script generates the same PADRE input deck as pndiode.in
"""

from nanohubpadre import (
    Simulation, Mesh, Region, Electrode, Doping, Contact,
    Material, Models, System, Solve, Log, Plot1D, Options
)


def create_pndiode_simulation():
    """Create the PN diode simulation equivalent to pndiode.in"""

    sim = Simulation(title="pn diode (setup)")

    # Options
    sim.options = Options(postscript=True)

    # Mesh specification
    sim.mesh = Mesh(nx=200, ny=3, width=1, outfile="mesh")
    sim.mesh.add_x_mesh(1, 0, ratio=1)
    sim.mesh.add_x_mesh(100, 0.5, ratio=0.8)
    sim.mesh.add_x_mesh(200, 1.0, ratio=1.05)
    sim.mesh.add_y_mesh(1, 0, ratio=1)
    sim.mesh.add_y_mesh(3, 1, ratio=1)

    # Regions specification
    sim.add_region(Region(1, ix_low=1, ix_high=100, iy_low=1, iy_high=3, silicon=True))
    sim.add_region(Region(1, ix_low=100, ix_high=200, iy_low=1, iy_high=3, silicon=True))

    # Electrodes
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=1, iy_low=1, iy_high=3))
    sim.add_electrode(Electrode(2, ix_low=200, ix_high=200, iy_low=1, iy_high=3))

    # Doping specification
    sim.add_doping(Doping(region=1, p_type=True, concentration=1e17,
                          x_left=0, x_right=0.5, y_top=0, y_bottom=1, uniform=True))
    sim.add_doping(Doping(region=1, n_type=True, concentration=1e17,
                          x_left=0.5, x_right=1.0, y_top=0, y_bottom=1, uniform=True))

    # Plot doping profile
    sim.add_command(Plot1D(logarithm=True, doping=True, absolute=True,
                           x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           points=True, ascii=True, outfile="dop"))

    # Materials specification
    sim.add_material(Material(name="silicon", taun0=1e-6, taup0=1e-6,
                              trap_type="0", etrap=0))

    # Specify models
    sim.models = Models(srh=True, conmob=True, fldmob=True, impact=True, temperature=300)
    sim.system = System(electrons=True, holes=True, newton=True)

    # Solve for initial conditions
    sim.add_solve(Solve(initial=True))

    # Initial plots
    sim.add_command(Plot1D(potential=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="pot"))
    sim.add_command(Plot1D(band_val=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="vband"))
    sim.add_command(Plot1D(band_con=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="cband"))
    sim.add_command(Plot1D(qfn=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="qfn"))
    sim.add_command(Plot1D(qfp=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="qfp"))
    sim.add_command(Plot1D(electrons=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="ele"))
    sim.add_command(Plot1D(holes=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="hole"))
    sim.add_command(Plot1D(net_charge=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="ro"))
    sim.add_command(Plot1D(e_field=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="efield"))
    sim.add_command(Plot1D(recomb=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="recomb"))

    # Solve for applied bias
    sim.add_log(Log(ivfile="iv"))
    sim.add_solve(Solve(previous=True))
    sim.add_solve(Solve(project=True, vstep=0.03, nsteps=20, electrode=1))

    # Final plots
    sim.add_command(Plot1D(potential=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="potiv"))
    sim.add_command(Plot1D(band_val=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="vbiv"))
    sim.add_command(Plot1D(band_con=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="cbiv"))
    sim.add_command(Plot1D(qfn=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="qfniv"))
    sim.add_command(Plot1D(qfp=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="qfpiv"))
    sim.add_command(Plot1D(electrons=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="eleiv"))
    sim.add_command(Plot1D(holes=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="holeiv"))
    sim.add_command(Plot1D(net_charge=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="roiv"))
    sim.add_command(Plot1D(e_field=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="efieldiv"))
    sim.add_command(Plot1D(recomb=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="reiv"))
    sim.add_command(Plot1D(j_electr=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="jelectr"))
    sim.add_command(Plot1D(j_hole=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="jhole"))
    sim.add_command(Plot1D(j_total=True, x_start=0, x_end=1.0, y_start=0.5, y_end=0.5,
                           ascii=True, outfile="jtot"))

    return sim


if __name__ == "__main__":
    sim = create_pndiode_simulation()
    deck = sim.generate_deck()
    print(deck)
