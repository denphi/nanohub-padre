#!/usr/bin/env python3
"""
n-p-n bipolar transistor: Gummel-number theory vs. PADRE.

Companion script to notebooks/05_BJT.ipynb.

Device (factory default): n+ emitter 1e20 cm^-3 (1 um), p base 1e17 cm^-3
(0.5 um), n collector 1e16 cm^-3 (2 um). Electrodes: 1 = emitter,
2 = base (contact on top of the base), 3 = collector.

What this script checks
-----------------------
1. Collector current from the base Gummel number
       Ic = q A ni^2 Dn / (N_B W_B) exp(qVbe/kT),
   with W_B the *quasi-neutral* base width (0.38 um of the 0.5 um).
   Expect agreement within a few %.
2. Current gain beta = Ic/Ib. Textbooks set ni(emitter) = ni(base) and
   predict beta ~ 8500 for this device. At 1e20 cm^-3 band-gap narrowing
   (Slotboom: ~125 meV) raises the emitter's effective ni^2 ~124x, so the
   base current rises ~100x and beta ~ 85. PADRE includes BGN.
3. Ideality factors of Ic and Ib: 1 in diffusion theory.

Run:  python bjt_example.py
"""

import shutil

import numpy as np

from nanohubpadre import create_bjt

Q, KT, NI, EPS_SI = 1.602e-19, 0.025851, 9.963e9, 11.8 * 8.854e-14
NE, NB, NCOL = 1e20, 1e17, 1e16
AREA = 1e-8                                    # 1 um tall x 1 um deep (cm^2)


def bgn(n):
    """Slotboom-de Graaff band-gap narrowing (eV)."""
    x = np.log(n / 1e17)
    return 9e-3 * (x + np.sqrt(x * x + 0.5))


def compare(name, textbook, padre, why):
    print(f"  {name:<30} textbook {textbook:10.4g}   PADRE {padre:10.4g}   "
          f"({(padre - textbook) / textbook * 100:+7.1f} %)  {why}")


def main():
    sim = create_bjt(log_iv=True, iv_file="gummel", gummel_sweep=(0.3, 0.8, 0.025), gummel_vce=2.0)
    if shutil.which("padre") is None:
        print(sim.generate_deck())
        print("\nPADRE not found on PATH: showing the input deck only.")
        return
    if sim.run().returncode != 0:
        raise SystemExit("PADRE failed; see .padre_run.log in " + sim.working_dir)

    iv = sim.get_iv_data()
    vbe, ib, ic = iv.get_voltages(2), np.abs(iv.get_currents(2)), np.abs(iv.get_currents(3))
    k = np.argmin(abs(vbe - 0.6))

    # quasi-neutral base width at Vbe = 0.6 V, Vce = 2 V
    x_eb = np.sqrt(2 * EPS_SI * (KT * np.log(NE * NB / NI ** 2) - 0.6) / (Q * NB))
    w_cb = np.sqrt(2 * EPS_SI * (KT * np.log(NB * NCOL / NI ** 2) + 1.4) / Q * (1 / NB + 1 / NCOL))
    wb = 0.5e-4 - x_eb - w_cb * NCOL / (NB + NCOL)

    dn_b = 739.6 * KT                           # PADRE conmob, electrons in 1e17
    dp_e = 49.0 * KT                            # holes in 1e20 (Masetti fit)
    tau_e = 1 / (1 / 1e-7 + 2.8e-31 * NE ** 2)  # SRH || Auger in the emitter
    lp_e = np.sqrt(dp_e * tau_e)

    def gain(with_bgn):
        ni2_b = NI ** 2 * (np.exp(bgn(NB) / KT) if with_bgn else 1)
        ni2_e = NI ** 2 * (np.exp(bgn(NE) / KT) if with_bgn else 1)
        ic0 = Q * AREA * ni2_b * dn_b / (NB * wb)
        ib0 = Q * AREA * ni2_e * dp_e / (NE * lp_e) / np.tanh(1e-4 / lp_e)
        return ic0, ib0

    ic0, ib0 = gain(True)
    ic0_tb, ib0_tb = gain(False)
    m = (vbe >= 0.45) & (vbe <= 0.65)
    n_c = 1 / (KT * np.polyfit(vbe[m], np.log(ic[m]), 1)[0])
    n_b = 1 / (KT * np.polyfit(vbe[m], np.log(ib[m]), 1)[0])

    print(f"BJT: textbook vs PADRE  (quasi-neutral base width {wb * 1e4:.3f} um)")
    compare("Ic at 0.6 V (A/um)", ic0 * np.exp(0.6 / KT), ic[k], "base physics: simple and well modelled")
    compare("beta, textbook (no BGN)", ic0_tb / ib0_tb, ic[k] / ib[k], "BGN raises emitter ni^2 ~124x")
    compare("beta, with Slotboom BGN", ic0 / ib0, ic[k] / ib[k], "same physics as PADRE")
    compare("ideality n_C", 1.0, n_c, "diffusion")
    compare("ideality n_B", 1.0, n_b, "emitter injection dominates I_B")


if __name__ == "__main__":
    main()
