#!/usr/bin/env python3
"""
Solar Cell Example using Device Factory

Demonstrates how to create and simulate a silicon solar cell
using the create_solar_cell factory function.
"""

from nanohubpadre import create_solar_cell, Solve, Log, Plot1D


def main():
    # Create an N-on-P solar cell with custom parameters
    sim = create_solar_cell(
        emitter_depth=0.5,               # 500nm emitter junction depth
        base_thickness=200.0,            # 200 micron base thickness
        device_width=1.0,                # 1 micron width (1D-like)
        emitter_doping=1e19,             # N+ emitter: 1e19 cm^-3
        base_doping=1e16,                # P base: 1e16 cm^-3
        device_type="n_on_p",            # N-on-P structure
        temperature=300,                 # Room temperature
        srh=True,                        # Enable SRH recombination
        auger=True,                      # Enable Auger recombination
        conmob=True,                     # Enable concentration-dependent mobility
        fldmob=True,                     # Enable field-dependent mobility
        taun0=1e-5,                      # Electron lifetime: 10 microseconds
        taup0=1e-5,                      # Hole lifetime: 10 microseconds
        front_surface_velocity=1e4,      # Front SRV: 1e4 cm/s
        back_surface_velocity=1e7,       # Back SRV: 1e7 cm/s (BSF)
        title="Silicon Solar Cell - I-V Analysis"
    )

    # Solve for equilibrium (dark, zero bias)
    sim.add_solve(Solve(initial=True))

    # Plot doping profile
    total_depth = 0.5 + 200.0
    sim.add_command(Plot1D(
        doping=True,
        logarithm=True,
        absolute=True,
        y_start=0, y_end=total_depth,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="doping"
    ))

    # Plot equilibrium band diagram
    sim.add_command(Plot1D(
        band_con=True,
        y_start=0, y_end=10.0,  # Plot first 10 microns (junction region)
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="ec_eq"
    ))
    sim.add_command(Plot1D(
        band_val=True,
        y_start=0, y_end=10.0,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="ev_eq"
    ))

    # =========================================
    # Dark I-V Characteristic
    # =========================================
    sim.add_log(Log(ivfile="iv_dark"))

    # Forward bias sweep (typical solar cell operating range)
    sim.add_solve(Solve(previous=True))
    sim.add_solve(Solve(
        v1=0,
        vstep=0.05,
        nsteps=15,
        electrode=1
    ))

    sim.add_log(Log(off=True))

    # =========================================
    # Analysis at Maximum Power Point (approx)
    # =========================================
    # Reload equilibrium
    sim.add_solve(Solve(initial=True))

    # Bias to approximate Vmp (around 0.5V for silicon)
    sim.add_solve(Solve(v1=0, vstep=0.1, nsteps=5, electrode=1))

    # Plot carrier distributions at operating point
    sim.add_command(Plot1D(
        electrons=True,
        y_start=0, y_end=10.0,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="n_op"
    ))
    sim.add_command(Plot1D(
        holes=True,
        y_start=0, y_end=10.0,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="p_op"
    ))

    # Plot recombination rate
    sim.add_command(Plot1D(
        recomb=True,
        y_start=0, y_end=10.0,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="recomb"
    ))

    # Plot current densities
    sim.add_command(Plot1D(
        j_electr=True,
        y_start=0, y_end=10.0,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="jn"
    ))
    sim.add_command(Plot1D(
        j_hole=True,
        y_start=0, y_end=10.0,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="jp"
    ))

    # Generate and print the input deck
    print(sim.generate_deck())


if __name__ == "__main__":
    main()
