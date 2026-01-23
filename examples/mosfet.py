#!/usr/bin/env python3
"""
Example: NMOS transistor simulation using PyPADRE.

This example demonstrates a more complex MOSFET structure with
gate oxide, source/drain regions, and multiple bias conditions.
"""

from nanohubpadre import (
    Simulation, Mesh, Region, Electrode, Doping, Contact,
    Models, System, Method, Solve, Log, Interface, Regrid
)


def create_nmos():
    """Create an NMOS transistor simulation."""

    sim = Simulation(title="NMOS Transistor - 0.5um Channel")

    # ========== MESH ==========
    # Device dimensions:
    # - Channel length: 0.5 um (centered)
    # - Total width: 2 um
    # - Oxide thickness: 10 nm
    # - Substrate depth: 2 um

    sim.mesh = Mesh(nx=60, ny=30, outfile="nmos_mesh.pg")

    # X mesh: finer near channel region
    sim.mesh.add_x_mesh(node=1, location=0.0)
    sim.mesh.add_x_mesh(node=15, location=0.6, ratio=0.85)  # Source region
    sim.mesh.add_x_mesh(node=25, location=0.75)              # Channel start
    sim.mesh.add_x_mesh(node=35, location=1.25)              # Channel end
    sim.mesh.add_x_mesh(node=45, location=1.4, ratio=1.18)   # Drain region
    sim.mesh.add_x_mesh(node=60, location=2.0)

    # Y mesh: very fine at interface, coarser in bulk
    sim.mesh.add_y_mesh(node=1, location=-0.01)    # Gate (in oxide)
    sim.mesh.add_y_mesh(node=5, location=0.0)      # Si-SiO2 interface
    sim.mesh.add_y_mesh(node=15, location=0.2, ratio=0.7)   # Near surface
    sim.mesh.add_y_mesh(node=25, location=1.0, ratio=1.3)   # Bulk
    sim.mesh.add_y_mesh(node=30, location=2.0, ratio=1.4)

    # ========== REGIONS ==========
    # Gate oxide (top)
    sim.add_region(Region(
        number=1,
        material="sio2",
        insulator=True,
        ix_low=1, ix_high=60,
        iy_low=1, iy_high=5
    ))

    # Silicon substrate
    sim.add_region(Region(
        number=2,
        material="silicon",
        semiconductor=True,
        ix_low=1, ix_high=60,
        iy_low=5, iy_high=30
    ))

    # ========== DOPING ==========
    # P-type substrate
    sim.add_doping(Doping(
        uniform=True,
        p_type=True,
        concentration=1e17,
        region=[2]
    ))

    # N+ Source (Gaussian, lateral spread)
    sim.add_doping(Doping(
        gaussian=True,
        n_type=True,
        concentration=5e19,
        junction=0.2,
        peak=0.0,
        x_right=0.75,
        ratio_lateral=0.7,
        erfc_lateral=True,
        region=[2]
    ))

    # N+ Drain (Gaussian, lateral spread)
    sim.add_doping(Doping(
        gaussian=True,
        n_type=True,
        concentration=5e19,
        junction=0.2,
        peak=0.0,
        x_left=1.25,
        ratio_lateral=0.7,
        erfc_lateral=True,
        region=[2]
    ))

    # Channel implant (threshold adjust)
    sim.add_doping(Doping(
        gaussian=True,
        p_type=True,
        concentration=5e17,
        junction=0.1,
        peak=0.0,
        x_left=0.75,
        x_right=1.25,
        region=[2]
    ))

    # ========== ELECTRODES ==========
    # Gate (on top of oxide)
    sim.add_electrode(Electrode(
        number=1,
        x_min=0.6, x_max=1.4,
        iy_low=1, iy_high=1
    ))

    # Source contact
    sim.add_electrode(Electrode(
        number=2,
        x_min=0.0, x_max=0.4,
        iy_low=5, iy_high=5
    ))

    # Drain contact
    sim.add_electrode(Electrode(
        number=3,
        x_min=1.6, x_max=2.0,
        iy_low=5, iy_high=5
    ))

    # Substrate contact (back)
    sim.add_electrode(Electrode(
        number=4,
        ix_low=1, ix_high=60,
        iy_low=30, iy_high=30
    ))

    # ========== CONTACTS ==========
    # N+ polysilicon gate
    sim.add_contact(Contact(number=1, n_polysilicon=True))

    # Ohmic contacts for source, drain, substrate
    sim.add_contact(Contact(number=2, neutral=True))
    sim.add_contact(Contact(number=3, neutral=True))
    sim.add_contact(Contact(number=4, neutral=True))

    # ========== INTERFACE ==========
    # Si-SiO2 interface quality
    sim.add_interface(Interface(
        number=1,
        qf=1e10,      # Fixed charge
        s_n=1e4,      # Surface recombination
        s_p=1e4
    ))

    # ========== MODELS ==========
    sim.models = Models(
        temperature=300,
        srh=True,
        conmob=True,
        fldmob=True,
        gatmob=True,  # Gate-field mobility degradation
        e_region=[2],
        g_region=[2]
    )

    # ========== SYSTEM & METHOD ==========
    sim.system = System(carriers=2, newton=True)
    sim.method = Method(
        trap=True,
        a_trap=0.5,
        itlimit=50
    )

    # ========== LOG ==========
    sim.log = Log(ivfile="nmos_iv.log")

    # ========== SOLVE ==========
    # Equilibrium
    sim.add_solve(Solve(
        initial=True,
        outfile="nmos_eq"
    ))

    # Vgs sweep at Vds=0.05V (subthreshold to strong inversion)
    sim.add_solve(Solve(
        project=True,
        v1=0.0,   # Gate
        v2=0.0,   # Source (reference)
        v3=0.05,  # Drain
        v4=0.0,   # Substrate
        vstep=0.1,
        nsteps=15,
        electrode=1,
        outfile="nmos_vgs_a"
    ))

    # Vds sweep at Vgs=1.5V (output characteristics)
    sim.add_solve(Solve(
        project=True,
        v1=1.5,   # Gate
        v2=0.0,   # Source
        v3=0.0,   # Drain (start)
        v4=0.0,   # Substrate
        vstep=0.1,
        nsteps=30,
        electrode=3,
        outfile="nmos_vds_a"
    ))

    return sim


def main():
    sim = create_nmos()
    deck = sim.generate_deck()
    print(deck)

    # Write to file
    # sim.write_deck("nmos.inp")


if __name__ == "__main__":
    main()
