#!/usr/bin/env python3
"""
MESFET: Shockley's gradual-channel model vs. PADRE, and when a gate can pinch off.

Companion script to notebooks/07_MESFET.ipynb.

Device: n-Si channel 1e17 cm^-3, a = 0.2 um thick, on a semi-insulating
substrate; Schottky gate (workfunction 4.87 eV -> phi_B = 0.70 eV with PADRE's
chi = 4.17 eV). Electrodes: 1 = source, 2 = drain, 3 = gate.

Note on names: `channel_length` is the width of each n+ source/drain contact
region; the gated channel is device_width - 2*channel_length wide.

What this script checks
-----------------------
The textbook threshold is  V_T = V_bi - q N_D a^2 / (2 eps_s) ~ -2.5 V, below
which the channel should be fully depleted (I_D ~ 0).
  * Default gate (L_g = 0.2 um = a, L_g/a = 1): the gate CANNOT pinch the
    channel off; I_D drops only ~3x by -3.5 V. The source/drain potential
    reaches under the short gate (a 2-D effect, like DIBL).
  * Long gate (L_g = 1.2 um, L_g/a = 6): I_D drops ~10^4x, as the
    gradual-channel model assumes. Pinch-off sits nearer -3.5 V than -2.5 V
    because electrons spill into the isotype substrate, making the channel
    effectively thicker than a (V_P grows as a^2).

Run:  python mesfet_example.py
"""

import shutil

import numpy as np

from nanohubpadre import create_mesfet

Q, KT, NC, CHI = 1.602e-19, 0.025851, 3.2e19, 4.17
EPS_SI = 11.8 * 8.854e-14
ND, A = 1e17, 0.2e-4


def linear_current(gate_length, vgs):
    sim = create_mesfet(gate_length=gate_length, channel_length=0.2, device_width=0.4 + gate_length,
                        vgs=vgs, statistics="boltzmann",      # "fermi" can fail on long depleted gates
                        log_iv=True, vds_sweep=(0.0, 0.05, 0.05))
    if sim.run().returncode != 0:
        return float("nan")
    iv = sim.get_iv_data()
    vd, idr = iv.get_voltages(2), np.abs(iv.get_currents(2))
    return idr[np.argmax(vd)]


def main():
    vbi = (4.87 - CHI) - KT * np.log(NC / ND)
    vp = Q * ND * A ** 2 / (2 * EPS_SI)
    print(f"Textbook: V_bi = {vbi:.2f} V, V_P = {vp:.2f} V, V_T = V_bi - V_P = {vbi - vp:.2f} V")

    if shutil.which("padre") is None:
        print(create_mesfet(log_iv=True, vds_sweep=(0.0, 2.0, 0.1)).generate_deck())
        print("\nPADRE not found on PATH: showing the input deck only.")
        return

    for lg in (0.2, 1.2):
        i0, i_off = linear_current(lg, 0.0), linear_current(lg, -3.5)
        print(f"  L_g = {lg} um (L_g/a = {lg / 0.2:.0f}):  Id(Vgs = -3.5 V) / Id(0) = {i_off / i0:.1e}"
              f"   -> {'no' if i_off / i0 > 0.1 else 'clear'} pinch-off")
    print("  The gradual-channel model is the L_g/a -> infinity limit; design rules ask for L_g/a >= 3.")


if __name__ == "__main__":
    main()
