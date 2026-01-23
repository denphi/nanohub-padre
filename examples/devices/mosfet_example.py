#!/usr/bin/env python3
"""
MOSFET Example using Device Factory

Demonstrates how to create and simulate an NMOS transistor for
transfer and output characteristics using the create_mosfet factory function.
"""

from nanohubpadre import create_mosfet, Solve, Log, Plot3D, Load


def main():
    # Create an NMOS transistor with custom parameters
    sim = create_mosfet(
        channel_length=0.025,        # 25nm channel length
        gate_oxide_thickness=0.012,  # 12nm gate oxide
        junction_depth=0.018,        # 18nm junction depth
        device_width=0.125,          # 125nm device width
        device_depth=0.068,          # 68nm substrate depth
        channel_doping=1e19,         # Channel doping: 1e19 cm^-3
        substrate_doping=5e16,       # Substrate doping: 5e16 cm^-3
        source_drain_doping=1e20,    # S/D doping: 1e20 cm^-3
        device_type="nmos",          # NMOS transistor
        temperature=300,             # Room temperature
        bgn=True,                    # Enable band-gap narrowing
        carriers=1,                  # Single carrier (electrons)
        title="NMOS Transistor - DC Characteristics"
    )

    # Plot initial doping profile
    sim.add_command(Plot3D(doping=True, outfile="doping"))

    # Solve for equilibrium
    sim.add_solve(Solve(initial=True, outfile="initsol"))
    sim.add_command(Plot3D(potential=True, electrons=True, outfile="equilibrium"))

    # =========================================
    # Transfer Characteristic (Id vs Vgs)
    # =========================================
    # Small drain bias
    sim.add_solve(Solve(v2=0, vstep=0.05, nsteps=1, electrode=2))
    sim.add_solve(Solve(v2=0.05))

    # Enable I-V logging for transfer curve
    sim.add_log(Log(ivfile="idvg"))

    # Gate voltage sweep: 0 to 1.5V
    sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))
    sim.add_log(Log(off=True))

    # =========================================
    # Output Characteristics (Id vs Vds)
    # =========================================
    # Reload equilibrium and set up gate bias points
    sim.add_load(Load(infile="initsol"))

    # Save solutions at different gate voltages
    sim.add_solve(Solve(v3=0.0, outfile="vg0"))
    sim.add_solve(Solve(v3=0.0, vstep=0.1, nsteps=5, electrode=3))
    sim.add_solve(Solve(v3=0.5, outfile="vg1"))
    sim.add_solve(Solve(v3=0.5, vstep=0.1, nsteps=5, electrode=3))
    sim.add_solve(Solve(v3=1.0, outfile="vg2"))
    sim.add_solve(Solve(v3=1.0, vstep=0.1, nsteps=5, electrode=3))
    sim.add_solve(Solve(v3=1.5, outfile="vg3"))

    # Enable I-V logging for output curves
    sim.add_log(Log(ivfile="idvd"))

    # Drain sweeps at each gate voltage
    sim.add_load(Load(infile="vg0"))
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))

    sim.add_load(Load(infile="vg1"))
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))

    sim.add_load(Load(infile="vg2"))
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))

    sim.add_load(Load(infile="vg3"))
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))
    sim.add_log(Log(off=True))

    # Final plot showing on-state
    sim.add_command(Plot3D(potential=True, electrons=True, outfile="on_state"))

    # Generate and print the input deck
    print(sim.generate_deck())


if __name__ == "__main__":
    main()
