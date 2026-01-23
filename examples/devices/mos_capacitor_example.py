#!/usr/bin/env python3
"""
MOS Capacitor Example using Device Factory

Demonstrates how to create and simulate a MOS capacitor for C-V analysis
using the create_mos_capacitor factory function.
"""

from nanohubpadre import create_mos_capacitor, Solve, Log, Plot1D


def main():
    # Create a MOS capacitor with custom parameters
    sim = create_mos_capacitor(
        oxide_thickness=0.002,      # 2nm gate oxide
        silicon_thickness=0.03,     # 30nm silicon thickness
        device_width=1.0,           # 1 micron width
        substrate_doping=1e18,      # P-type substrate: 1e18 cm^-3
        substrate_type="p",         # P-type substrate
        oxide_permittivity=3.9,     # SiO2 permittivity
        gate_type="n_poly",         # N+ polysilicon gate
        temperature=300,            # Room temperature
        title="MOS Capacitor - C-V Analysis"
    )

    # Solve for equilibrium (zero gate bias)
    sim.add_solve(Solve(initial=True))

    # Plot initial band diagram (vertical cut)
    total_thickness = 0.002 + 0.03
    sim.add_command(Plot1D(
        potential=True,
        y_start=0, y_end=total_thickness,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="pot_eq"
    ))
    sim.add_command(Plot1D(
        band_con=True,
        y_start=0, y_end=total_thickness,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="ec_eq"
    ))
    sim.add_command(Plot1D(
        electrons=True,
        y_start=0, y_end=total_thickness,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="n_eq"
    ))

    # Sweep gate voltage into accumulation
    sim.add_solve(Solve(v1=0, vstep=-0.2, nsteps=10, electrode=1))

    # Enable AC analysis logging for C-V extraction
    sim.add_log(Log(acfile="cv_data"))

    # C-V sweep from accumulation to inversion
    sim.add_solve(Solve(
        v1=-2.0,
        vstep=0.2,
        nsteps=20,
        electrode=1,
        ac_analysis=True,
        frequency=1e6  # 1 MHz measurement frequency
    ))
    sim.add_log(Log(off=True))

    # Plot inversion condition
    sim.add_command(Plot1D(
        potential=True,
        y_start=0, y_end=total_thickness,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="pot_inv"
    ))
    sim.add_command(Plot1D(
        electrons=True,
        y_start=0, y_end=total_thickness,
        x_start=0.5, x_end=0.5,
        ascii=True, outfile="n_inv"
    ))

    # Generate and print the input deck
    print(sim.generate_deck())


if __name__ == "__main__":
    main()
