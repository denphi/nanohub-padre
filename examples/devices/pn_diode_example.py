#!/usr/bin/env python3
"""
PN junction diode: textbook theory vs. PADRE.

Companion script to notebooks/02_PN_Diode.ipynb (the notebook has the full
derivations, plots and exercises).

Device (factory default): silicon, 1 um long, junction in the middle,
Na = Nd = 1e17 cm^-3, SRH lifetimes 1 us, ohmic contacts at both ends.

What this script checks
-----------------------
1. Built-in potential   Vbi = kT/q ln(Na Nd / ni^2)
   Pure thermodynamics: with PADRE's ni (9.963e9 cm^-3, not the old
   textbook 1.5e10) it agrees to < 1 mV.
2. Forward current vs. the Shockley equation in its *short-base* form
       I = q A ni^2 (Dn/(Na Wp) + Dp/(Nd Wn)) (exp(qV/kT) - 1)
   The neutral regions (~0.45 um) are ~100x shorter than the diffusion
   lengths (~40 um), so the classic long-base formula would be ~90x off.
   Expect agreement within a few % at 0.3-0.7 V. Above ~0.75 V PADRE falls
   below theory: high injection and series resistance, which the formula
   omits.
3. Ideality factor n from the slope of ln I vs V: diffusion theory gives
   n = 1. PADRE gives ~1.006.

Run:  python pn_diode_example.py        (prints the deck if PADRE is absent)
"""

import shutil

import numpy as np

from nanohubpadre import create_pn_diode

# PADRE's own constants (printed by MODELS ... PRINT)
Q, KT, NI, EPS_SI = 1.602e-19, 0.025851, 9.963e9, 11.8 * 8.854e-14
MU_N, MU_P = 739.6, 294.4          # PADRE conmob mobilities at 1e17 cm^-3


def compare(name, textbook, padre, why):
    print(f"  {name:<28} textbook {textbook:10.4g}   PADRE {padre:10.4g}   "
          f"({(padre - textbook) / textbook * 100:+6.1f} %)  {why}")


def shockley_short_base(v, na=1e17, nd=1e17, length_um=1.0, area_cm2=1e-8):
    """Short-base Shockley current with bias-dependent neutral widths."""
    vbi = KT * np.log(na * nd / NI ** 2)
    w = np.sqrt(2 * EPS_SI / Q * (1 / na + 1 / nd) * (vbi - v))      # cm
    wp = length_um / 2 * 1e-4 - w * nd / (na + nd)
    wn = length_um / 2 * 1e-4 - w * na / (na + nd)
    i0 = Q * area_cm2 * NI ** 2 * (MU_N * KT / (na * wp) + MU_P * KT / (nd * wn))
    return i0 * (np.exp(v / KT) - 1)


def main():
    sim = create_pn_diode(
        log_bands_eq=True,                  # Ec/Ev along the device at 0 V
        log_iv=True,                        # record terminal I-V
        forward_sweep=(0.0, 0.8, 0.025),    # anode (electrode 1, p side) 0 -> 0.8 V
    )

    if shutil.which("padre") is None:
        print(sim.generate_deck())
        print("\nPADRE not found on PATH: showing the input deck only.")
        return

    result = sim.run()
    if result.returncode != 0:
        raise SystemExit("PADRE failed; see .padre_run.log in " + sim.working_dir)

    print("PN diode: textbook vs PADRE")

    # 1. built-in potential from the band bending
    ec = sim.outputs.get("cbeq")
    compare("Vbi (V)", KT * np.log(1e34 / NI ** 2), abs(ec.y[0] - ec.y[-1]),
            "same formula and n_i")

    # 2. forward current
    iv = sim.get_iv_data()
    v = iv.get_voltages(1)                  # electrode 1 is the swept anode
    i = np.abs(iv.get_currents(1))
    good = (v > 0) & ~iv.check_continuity(tolerance=0.01, warn=False)
    for vk in (0.3, 0.5, 0.7, 0.8):
        k = np.argmin(abs(v - vk))
        why = "within Shockley's assumptions" if vk < 0.75 else "high injection + series R"
        compare(f"I at {v[k]:.2f} V (A/um)", float(shockley_short_base(v[k])), i[k], why)

    # 3. ideality factor in the diffusion-dominated window
    fit = good & (v >= 0.3) & (v <= 0.6)
    n = 1 / (KT * np.polyfit(v[fit], np.log(i[fit]), 1)[0])
    compare("ideality factor n", 1.0, n, "pure diffusion current")


if __name__ == "__main__":
    main()
