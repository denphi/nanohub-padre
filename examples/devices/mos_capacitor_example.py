#!/usr/bin/env python3
"""
MOS capacitor: textbook C-V formulas vs. PADRE.

Companion script to notebooks/06_MOS_Capacitor.ipynb (which also implements
the exact classical C-V curve point by point).

Device (factory default): n+ poly gate, 2 nm SiO2, 5 um p-type Si 1e16 cm^-3,
ohmic back contact, gate 0.999 um x 1 um (the reference mesh starts at
x = 0.001 um).

What this script checks
-----------------------
1. Accumulation capacitance vs. C_ox = eps_ox A / t_ox: PADRE is ~5 % below,
   because the accumulation layer has a finite thickness (a capacitor in
   series with the oxide). The exact classical theory agrees with PADRE.
2. High-frequency C_min vs. Sze's formula
       C_min = eps_ox / (t_ox + (eps_ox/eps_s) W_m),  W_m = sqrt(4 eps_s phi_F / q N_A):
   PADRE is ~3 % lower, because in strong inversion the surface potential
   rises a little past the textbook's 2 phi_F.
3. Low-frequency C-V. PADRE's small-signal (AC) solve is unreliable at the
   1 Hz needed for an inversion response (charge conservation C11 + C12 = 0
   fails badly), so we use the quasi-static curve C = dQ_gate/dV_gate from
   the DC solutions: it returns to ~C_ox in inversion, as theory says.

Run:  python mos_capacitor_example.py
"""

import os
import shutil
import warnings

import numpy as np

from nanohubpadre import create_mos_capacitor
from nanohubpadre.parser import parse_ac_file

Q, KT, NI = 1.602e-19, 0.025851, 9.963e9
EPS_SI, EPS_OX = 11.8 * 8.854e-14, 3.9 * 8.854e-14
NA, TOX = 1e16, 2e-7
AREA = 0.999e-4 * 1e-4


def compare(name, textbook, padre, why):
    print(f"  {name:<28} textbook {textbook:10.4g}   PADRE {padre:10.4g}   "
          f"({(padre - textbook) / textbook * 100:+6.1f} %)  {why}")


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # the 1 Hz AC warning: handled below
        sim = create_mos_capacitor(log_cv=True, log_cv_lf=True, vg_sweep=(-2.0, 2.0, 0.1))
    if shutil.which("padre") is None:
        print(sim.generate_deck())
        print("\nPADRE not found on PATH: showing the input deck only.")
        return
    if sim.run().returncode != 0:
        raise SystemExit("PADRE failed; see .padre_run.log in " + sim.working_dir)

    hf = parse_ac_file(os.path.join(sim.working_dir, "cv_data"))
    lf = parse_ac_file(os.path.join(sim.working_dir, "cv_lf_data"))
    vg, c_hf = hf.get_cv_data(gate_electrode=1)
    vq, c_qs = hf.get_quasistatic_cv(gate_electrode=1)

    cox = EPS_OX / TOX * AREA
    phi_f = KT * np.log(NA / NI)
    wm = np.sqrt(4 * EPS_SI * phi_f / (Q * NA))
    cmin = EPS_OX / (TOX + EPS_OX / EPS_SI * wm) * AREA

    print("MOS capacitor: textbook vs PADRE (capacitances in F)")
    compare("C accumulation (-2 V)", cox, c_hf[0], "finite accumulation-layer thickness")
    compare("C_min, high frequency", cmin, c_hf.min(), "psi_s exceeds 2 phi_F in inversion")
    compare("C quasi-static (+2 V)", cox, c_qs[-1], "inversion layer follows at low frequency")
    err = lf.conservation_error()
    print(f"  1 Hz AC solve: worst |C11+C12|/|C11| = {err.max():.0%} "
          f"-> do not use get_cv_data() on the LF file; use get_quasistatic_cv().")


if __name__ == "__main__":
    main()
