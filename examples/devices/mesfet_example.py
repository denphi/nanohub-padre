#!/usr/bin/env python3
"""
MESFET Example using Device Factory

Demonstrates how to create and simulate a Metal-Semiconductor FET
using the create_mesfet factory function.
"""

from nanohubpadre import create_mesfet, Solve, Log, Plot3D


def main():
    # Create an n-channel MESFET with custom parameters
    sim = create_mesfet(
        channel_length=0.2,          # 200nm source-gate and gate-drain spacing
        gate_length=0.2,             # 200nm gate length
        device_width=0.6,            # 600nm total device width
        channel_depth=0.2,           # 200nm channel depth
        substrate_depth=0.8,         # 800nm substrate depth
        channel_doping=1e17,         # Channel doping: 1e17 cm^-3
        substrate_doping=1e17,       # Substrate doping: 1e17 cm^-3
        contact_doping=1e20,         # S/D contact doping: 1e20 cm^-3
        device_type="n",             # N-channel MESFET
        gate_workfunction=4.87,      # Schottky barrier height
        temperature=300,             # Room temperature
        bgn=True,                    # Enable band-gap narrowing
        conmob=True,                 # Enable concentration-dependent mobility
        fldmob=True,                 # Enable field-dependent mobility
        title="N-channel MESFET - Output Characteristics"
    )

    # Plot initial doping profile
    sim.add_command(Plot3D(doping=True, outfile="doping"))

    # Solve for equilibrium
    sim.add_solve(Solve(initial=True, outfile="initsol"))
    sim.add_command(Plot3D(potential=True, electrons=True, outfile="equilibrium"))

    # =========================================
    # Gate Bias Setup
    # =========================================
    # Apply negative gate bias to control channel
    sim.add_solve(Solve(v3=0, vstep=-0.1, nsteps=4, electrode=3))
    sim.add_command(Plot3D(potential=True, electrons=True, outfile="gate_bias"))

    # =========================================
    # Output Characteristic (Id vs Vds)
    # =========================================
    # Enable I-V logging
    sim.add_log(Log(ivfile="idvd"))

    # Drain voltage sweep: 0 to 2V
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))

    # Plot final state
    sim.add_command(Plot3D(potential=True, electrons=True, outfile="on_state"))

    # Generate and print the input deck
    print(sim.generate_deck())


if __name__ == "__main__":
    main()
