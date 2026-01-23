#!/usr/bin/env python3
"""
Schottky Diode Example using Device Factory

Demonstrates how to create and simulate a Schottky barrier diode
using the create_schottky_diode factory function.
"""

from nanohubpadre import create_schottky_diode, Solve, Log, Plot1D


def main():
    # Create a Schottky diode with custom parameters
    sim = create_schottky_diode(
        length=2.0,                  # 2 micron device length
        width=1.0,                   # 1 micron width
        doping=1e16,                 # N-type doping: 1e16 cm^-3
        doping_type="n",             # N-type semiconductor
        workfunction=4.8,            # Metal workfunction (V)
        barrier_lowering=True,       # Enable image-force barrier lowering
        surf_rec=True,               # Enable surface recombination
        temperature=300,             # Room temperature
        srh=True,                    # Enable SRH recombination
        conmob=True,                 # Enable concentration-dependent mobility
        fldmob=True,                 # Enable field-dependent mobility
        title="Schottky Diode - I-V Characteristics"
    )

    # Solve for equilibrium
    sim.add_solve(Solve(initial=True))

    # Plot initial band diagram (vertical cut through device)
    sim.add_command(Plot1D(
        potential=True,
        y_start=0, y_end=1.0,
        x_start=1.0, x_end=1.0,
        ascii=True, outfile="pot_eq"
    ))
    sim.add_command(Plot1D(
        band_con=True,
        y_start=0, y_end=1.0,
        x_start=1.0, x_end=1.0,
        ascii=True, outfile="ec_eq"
    ))
    sim.add_command(Plot1D(
        band_val=True,
        y_start=0, y_end=1.0,
        x_start=1.0, x_end=1.0,
        ascii=True, outfile="ev_eq"
    ))
    sim.add_command(Plot1D(
        electrons=True,
        y_start=0, y_end=1.0,
        x_start=1.0, x_end=1.0,
        ascii=True, outfile="n_eq"
    ))

    # =========================================
    # Forward Bias Characteristics
    # =========================================
    # Enable I-V logging
    sim.add_log(Log(ivfile="iv_forward"))

    # Forward bias sweep: 0 to 0.5V
    sim.add_solve(Solve(previous=True))
    sim.add_solve(Solve(
        project=True,
        vstep=0.025,
        nsteps=20,
        electrode=1
    ))

    # Plot forward bias band diagram
    sim.add_command(Plot1D(
        band_con=True,
        y_start=0, y_end=1.0,
        x_start=1.0, x_end=1.0,
        ascii=True, outfile="ec_fwd"
    ))

    # =========================================
    # Reverse Bias Characteristics
    # =========================================
    sim.add_log(Log(off=True))

    # Reload equilibrium for reverse bias
    sim.add_solve(Solve(initial=True))

    sim.add_log(Log(ivfile="iv_reverse"))

    # Reverse bias sweep: 0 to -5V
    sim.add_solve(Solve(previous=True))
    sim.add_solve(Solve(
        project=True,
        vstep=-0.25,
        nsteps=20,
        electrode=1
    ))

    # Plot reverse bias band diagram
    sim.add_command(Plot1D(
        band_con=True,
        y_start=0, y_end=1.0,
        x_start=1.0, x_end=1.0,
        ascii=True, outfile="ec_rev"
    ))
    sim.add_command(Plot1D(
        e_field=True,
        y_start=0, y_end=1.0,
        x_start=1.0, x_end=1.0,
        ascii=True, outfile="efield_rev"
    ))

    sim.add_log(Log(off=True))

    # Generate and print the input deck
    print(sim.generate_deck())


if __name__ == "__main__":
    main()
