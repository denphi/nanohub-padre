#!/usr/bin/env python3
"""
NIN isotype diode: Ohm's law and Mott-Gurney space-charge-limited current vs. PADRE.

Companion script to notebooks/08_NIN_Diode.ipynb.

Device (factory default): 2 um long, n+ contacts 1e18 cm^-3, an 0.8 um n-
layer doped 1e14 cm^-3 in between. No p-n junction, so it must not rectify.

What this script checks
-----------------------
1. Symmetry: swapping the contacts leaves the device unchanged, so
   I(-V) = -I(V). PADRE: exact to < 0.01 %.
2. Ohm's law, I = q N_D mu A V / L, assumes the n- layer holds only its own
   electrons. It underestimates PADRE ~10x: the Debye length at 1e14
   (0.41 um) is half the layer width, so contact electrons flood the layer.
3. Mott-Gurney, J = (9/8) eps_s mu V^2 / L^3, assumes injected charge only,
   no diffusion and constant mobility. At 2 V PADRE is ~40 % below it:
   diffusion opposes the injection and velocity saturation sets in.

Run:  python nin_diode_example.py
"""

import shutil

import numpy as np

from nanohubpadre import create_nin_diode

Q, EPS_SI = 1.602e-19, 11.8 * 8.854e-14
MU_N, N_B, L_B, AREA = 1382.0, 1e14, 0.8e-4, 1e-8   # PADRE conmob at 1e14


def compare(name, textbook, padre, why):
    print(f"  {name:<26} textbook {textbook:10.4g}   PADRE {padre:10.4g}   "
          f"({(padre - textbook) / textbook * 100:+7.1f} %)  {why}")


def main():
    sim = create_nin_diode(log_iv=True, bias_sweep=(-2.0, 2.0, 0.1))
    if shutil.which("padre") is None:
        print(sim.generate_deck())
        print("\nPADRE not found on PATH: showing the input deck only.")
        return
    if sim.run().returncode != 0:
        raise SystemExit("PADRE failed; see .padre_run.log in " + sim.working_dir)

    iv = sim.get_iv_data()
    v, i = iv.get_voltages(1), iv.get_currents(1)
    o = np.argsort(v)
    v, i = v[o], i[o]
    i_p, i_m = abs(np.interp(2.0, v, i)), abs(np.interp(-2.0, v, i))
    print("NIN diode: textbook vs PADRE")
    print(f"  symmetry |I(+2 V)| vs |I(-2 V)|: {abs(i_p - i_m) / i_p * 100:.4f} % apart")
    i01 = abs(np.interp(0.1, v, i))
    compare("I(0.1 V) vs Ohm's law", Q * N_B * MU_N * AREA * 0.1 / L_B, i01, "contact spill-over (L_D ~ layer)")
    compare("I(2 V) vs Mott-Gurney", 9 / 8 * EPS_SI * MU_N * 4 / L_B ** 3 * AREA, i_p, "diffusion + velocity saturation")


if __name__ == "__main__":
    main()
