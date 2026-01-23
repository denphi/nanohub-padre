#!/usr/bin/env python3
"""
Python equivalent of moscap.in for PADRE simulation.

This script generates the same PADRE input deck as moscap.in
MOS Capacitor structure with oxide-silicon-oxide
"""

from nanohubpadre import (
    Simulation, Mesh, Region, Electrode, Doping, Contact,
    Material, Models, System, Solve, Log, Plot1D
)


def create_moscap_simulation():
    """Create the MOS capacitor simulation equivalent to moscap.in"""

    sim = Simulation()

    # Mesh Specification
    sim.mesh = Mesh(nx=3, ny=41)
    sim.mesh.add_y_mesh(1, 0, ratio=1)
    sim.mesh.add_y_mesh(10, 0.002, ratio=0.8)
    sim.mesh.add_y_mesh(21, 0.032, ratio=1.25)
    sim.mesh.add_y_mesh(31, 0.062, ratio=0.8)
    sim.mesh.add_y_mesh(41, 0.064, ratio=1.25)
    sim.mesh.add_x_mesh(1, 0, ratio=1)
    sim.mesh.add_x_mesh(3, 1, ratio=1)

    # Regions specification
    sim.add_region(Region(1, ix_low=1, ix_high=3, iy_low=1, iy_high=10,
                          material="sio2", insulator=True))
    sim.add_region(Region(2, ix_low=1, ix_high=3, iy_low=10, iy_high=31,
                          material="silicon", semiconductor=True))
    sim.add_region(Region(3, ix_low=1, ix_high=3, iy_low=31, iy_high=41,
                          material="sio2", insulator=True))

    # Electrodes specification
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=3, iy_low=1, iy_high=1))
    sim.add_electrode(Electrode(2, ix_low=1, ix_high=3, iy_low=41, iy_high=41))

    # Doping specification
    sim.add_doping(Doping(region=2, p_type=True, concentration=1e18, uniform=True))

    # Contact specification
    sim.add_contact(Contact(all_contacts=True, neutral=True))
    sim.add_contact(Contact(number=1, n_polysilicon=True))
    sim.add_contact(Contact(number=2, n_polysilicon=True))

    # Material lifetime specification
    sim.add_material(Material(name="silicon"))
    sim.add_material(Material(name="sio2", permittivity=3.9, qf=0))

    # Specify models
    sim.models = Models(temperature=300, conmob=True, fldmob=True)
    sim.system = System(electrons=True, holes=True, newton=True)

    # Solve for initial conditions
    sim.add_solve(Solve(initial=True))
    sim.add_command(Plot1D(potential=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="pot"))
    sim.add_command(Plot1D(qfn=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="qfn"))
    sim.add_command(Plot1D(qfp=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="qfp"))
    sim.add_command(Plot1D(band_val=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="val"))
    sim.add_command(Plot1D(band_con=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="con"))
    sim.add_command(Plot1D(electrons=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="ele"))
    sim.add_command(Plot1D(holes=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="hole"))
    sim.add_command(Plot1D(net_charge=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="ro"))
    sim.add_command(Plot1D(e_field=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="efield"))

    # Solve for applied bias
    sim.add_solve(Solve(v1=0, vstep=-0.2, nsteps=10, electrode=[1, 2]))
    sim.add_log(Log(acfile="ac"))
    sim.add_solve(Solve(v1=-2.0, vstep=0.2, nsteps=20, electrode=[1, 2],
                        ac_analysis=True, frequency=1e7))
    sim.add_log(Log(off=True))

    # Final plots after bias
    sim.add_command(Plot1D(potential=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="potiv"))
    sim.add_command(Plot1D(qfn=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="qfniv"))
    sim.add_command(Plot1D(qfp=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="qfpiv"))
    sim.add_command(Plot1D(band_val=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="valiv"))
    sim.add_command(Plot1D(band_con=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="coniv"))
    sim.add_command(Plot1D(electrons=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="eleiv"))
    sim.add_command(Plot1D(holes=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="holeiv"))
    sim.add_command(Plot1D(net_charge=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="roiv"))
    sim.add_command(Plot1D(e_field=True, y_start=0, y_end=0.064, x_start=0.5, x_end=0.5,
                           ascii=True, outfile="efieldiv"))

    return sim


if __name__ == "__main__":
    sim = create_moscap_simulation()
    deck = sim.generate_deck()
    print(deck)
