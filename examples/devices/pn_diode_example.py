#!/usr/bin/env python3
"""
PN Diode Example using Device Factory

Demonstrates how to create and simulate a PN junction diode using
the create_pn_diode factory function.
"""

from nanohubpadre import create_pn_diode, Solve, Log, Plot1D


def main():
    # Create a PN diode with custom parameters
    sim = create_pn_diode(
        length=1.0,              # 1 micron device length
        width=1.0,               # 1 micron width
        junction_position=0.5,   # Junction at center
        p_doping=1e17,           # P-type doping: 1e17 cm^-3
        n_doping=1e17,           # N-type doping: 1e17 cm^-3
        temperature=300,         # Room temperature
        srh=True,                # Enable SRH recombination
        conmob=True,             # Enable concentration-dependent mobility
        fldmob=True,             # Enable field-dependent mobility
        impact=True,             # Enable impact ionization
        title="PN Diode - Forward Bias Simulation"
    )

    # Solve for equilibrium (zero bias)
    sim.add_solve(Solve(initial=True))

    # Plot initial band diagram
    sim.add_command(Plot1D(
        potential=True,
        x_start=0, x_end=1.0,
        y_start=0.5, y_end=0.5,
        ascii=True, outfile="pot_eq"
    ))
    sim.add_command(Plot1D(
        band_con=True,
        x_start=0, x_end=1.0,
        y_start=0.5, y_end=0.5,
        ascii=True, outfile="ec_eq"
    ))
    sim.add_command(Plot1D(
        band_val=True,
        x_start=0, x_end=1.0,
        y_start=0.5, y_end=0.5,
        ascii=True, outfile="ev_eq"
    ))

    # Enable I-V logging
    sim.add_log(Log(ivfile="iv_forward"))

    # Forward bias sweep: 0 to 0.6V in 30mV steps
    sim.add_solve(Solve(previous=True))
    sim.add_solve(Solve(
        project=True,
        vstep=0.03,
        nsteps=20,
        electrode=1
    ))

    # Plot carrier distributions under forward bias
    sim.add_command(Plot1D(
        electrons=True,
        x_start=0, x_end=1.0,
        y_start=0.5, y_end=0.5,
        ascii=True, outfile="n_fwd"
    ))
    sim.add_command(Plot1D(
        holes=True,
        x_start=0, x_end=1.0,
        y_start=0.5, y_end=0.5,
        ascii=True, outfile="p_fwd"
    ))

    # Generate and print the input deck
    print(sim.generate_deck())


if __name__ == "__main__":
    main()
