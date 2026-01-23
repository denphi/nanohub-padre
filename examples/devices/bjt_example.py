#!/usr/bin/env python3
"""
Bipolar Junction Transistor (BJT) Example using Device Factory

Demonstrates how to create and simulate an NPN BJT for
common-emitter characteristics using the create_bjt factory function.
"""

from nanohubpadre import create_bjt, Solve, Log, Plot1D, Load


def main():
    # Create an NPN BJT with custom parameters
    sim = create_bjt(
        emitter_width=1.0,          # 1 micron emitter width
        base_width=0.3,             # 300nm base width (thin for high gain)
        collector_width=2.0,        # 2 micron collector width
        device_depth=1.0,           # 1 micron device depth
        emitter_doping=1e20,        # N+ emitter: 1e20 cm^-3
        base_doping=1e17,           # P base: 1e17 cm^-3
        collector_doping=1e16,      # N- collector: 1e16 cm^-3
        device_type="npn",          # NPN transistor
        temperature=300,            # Room temperature
        srh=True,                   # Enable SRH recombination
        auger=True,                 # Enable Auger recombination
        bgn=True,                   # Enable band-gap narrowing
        conmob=True,                # Enable concentration-dependent mobility
        fldmob=True,                # Enable field-dependent mobility
        title="NPN BJT - Common Emitter Characteristics"
    )

    # Solve for equilibrium
    sim.add_solve(Solve(initial=True, outfile="initsol"))

    # Plot initial band diagram along device
    total_width = 1.0 + 0.3 + 2.0
    sim.add_command(Plot1D(
        band_con=True,
        x_start=0, x_end=total_width,
        y_start=0.5, y_end=0.5,
        ascii=True, outfile="ec_eq"
    ))
    sim.add_command(Plot1D(
        band_val=True,
        x_start=0, x_end=total_width,
        y_start=0.5, y_end=0.5,
        ascii=True, outfile="ev_eq"
    ))

    # =========================================
    # Common-Emitter Output Characteristics
    # =========================================
    # Forward bias the base-emitter junction
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=7, electrode=2))  # Vbe = 0.7V
    sim.add_solve(Solve(v2=0.7, outfile="vbe_07"))

    # Enable I-V logging for collector current
    sim.add_log(Log(ivfile="ic_vce"))

    # Collector voltage sweep (Vce = 0 to 3V)
    sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=30, electrode=3))

    # Plot carrier distributions in active mode
    sim.add_command(Plot1D(
        electrons=True,
        x_start=0, x_end=total_width,
        y_start=0.5, y_end=0.5,
        ascii=True, outfile="n_active"
    ))
    sim.add_command(Plot1D(
        holes=True,
        x_start=0, x_end=total_width,
        y_start=0.5, y_end=0.5,
        ascii=True, outfile="p_active"
    ))

    # =========================================
    # Gummel Plot (Ic, Ib vs Vbe)
    # =========================================
    sim.add_log(Log(off=True))
    sim.add_load(Load(infile="initsol"))

    # Enable logging for Gummel plot
    sim.add_log(Log(ivfile="gummel"))

    # Sweep Vbe with fixed Vce
    sim.add_solve(Solve(v3=2.0))  # Fixed Vce = 2V
    sim.add_solve(Solve(v2=0, vstep=0.05, nsteps=16, electrode=2))  # Vbe sweep

    sim.add_log(Log(off=True))

    # Generate and print the input deck
    print(sim.generate_deck())


if __name__ == "__main__":
    main()
