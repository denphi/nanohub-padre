#!/usr/bin/env python3
"""
Schottky barrier diode: thermionic-emission theory vs. PADRE.

Companion script to notebooks/03_Schottky_Diode.ipynb.

Device (factory default): n-Si, Nd = 1e16 cm^-3, 1 um thick; Schottky contact
on top (2 um wide, workfunction 4.8 eV), ohmic contact at the bottom.

What this script checks
-----------------------
1. Barrier height (Schottky-Mott rule)  phi_B = phi_m - chi.
   PADRE's electron affinity is 4.17 eV, not the textbook 4.05 eV, so
   phi_m = 4.8 eV gives phi_B = 0.63 eV (not 0.75 eV). That 0.12 eV is a
   factor ~100 in current. Real metal/Si contacts are Fermi-level pinned
   and do not follow phi_m at all; PADRE (like the textbook) does.
2. Reverse saturation current vs.
     - pure thermionic emission  I0 = A A** T^2 exp(-q phi_B / kT)
     - Crowell & Sze thermionic-emission-diffusion (TED), which divides by
       (1 + vR/vD) for diffusion through the depletion region.
   PADRE matches TED to ~0.1 %; pure TE overestimates by ~20 %.
3. Forward ideality factor: theory 1, PADRE ~1.00.

Not simulated: image-force barrier lowering. PADRE 2.4E's BARRIERL option
diverges together with the thermionic contact (surf.rec) and has no effect
without it, so create_schottky_diode(barrier_lowering=True) warns.

Run:  python schottky_diode_example.py
"""

import shutil

import numpy as np

from nanohubpadre import create_schottky_diode

Q, KT, NC, CHI = 1.602e-19, 0.025851, 3.2e19, 4.17   # PADRE's constants
A_RICH, T, VSAT, MU_1E16 = 110.0, 300.0, 1.035e7, 1190.5
EPS_SI = 11.8 * 8.854e-14
ND, PHI_M = 1e16, 4.8
AREA = 2e-4 * 1e-4                                     # 2 um x 1 um contact (cm^2)


def compare(name, textbook, padre, why):
    print(f"  {name:<30} textbook {textbook:10.4g}   PADRE {padre:10.4g}   "
          f"({(padre - textbook) / textbook * 100:+6.1f} %)  {why}")


def main():
    sim = create_schottky_diode(
        log_bands_eq=True,
        log_iv=True,
        forward_sweep=(0.0, 0.3, 0.02),
        reverse_sweep=(0.0, -1.0, -0.2),
    )

    if shutil.which("padre") is None:
        print(sim.generate_deck())
        print("\nPADRE not found on PATH: showing the input deck only.")
        return
    if sim.run().returncode != 0:
        raise SystemExit("PADRE failed; see .padre_run.log in " + sim.working_dir)

    print("Schottky diode: textbook vs PADRE")
    phib = PHI_M - CHI
    ec = sim.outputs.get("cbeq")                       # Ec - EF, metal at the first point
    compare("phi_B, textbook chi = 4.05 (eV)", PHI_M - 4.05, ec.y[0], "PADRE's chi is 4.17 eV")
    compare("phi_B, PADRE chi = 4.17 (eV)", phib, ec.y[0], "same rule, same constant")

    iv = sim.get_iv_data()
    v, i = iv.get_voltages(1), np.abs(iv.get_currents(1))
    k = np.argmin(abs(v + 1.0))
    i_te = AREA * A_RICH * T ** 2 * np.exp(-phib / KT)
    vbi = phib - KT * np.log(NC / ND)
    emax = np.sqrt(2 * Q * ND * (vbi + 1.0 - KT) / EPS_SI)
    vd = MU_1E16 * emax / np.sqrt(1 + (MU_1E16 * emax / VSAT) ** 2)
    vr = A_RICH * T ** 2 / (Q * NC)
    compare("|I(-1 V)| vs thermionic (A/um)", i_te, i[k], "ignores diffusion to the interface")
    compare("|I(-1 V)| vs Crowell-Sze (A/um)", i_te / (1 + vr / vd), i[k], "emission and diffusion in series")

    fwd = (v > 0.06) & (v < 0.24)
    n = 1 / (KT * np.polyfit(v[fwd], np.log(i[fwd]), 1)[0])
    compare("forward ideality factor", 1.0, n, "real diodes: 1.01-1.1 (lowering, tunnelling)")


if __name__ == "__main__":
    main()
