#!/usr/bin/env python3
"""
n-MOSFET: long-channel textbook theory vs. a 0.15 um device simulated in 2-D.

Companion script to notebooks/04_MOSFET.ipynb.

Device (factory default, the nanoHUB MOSFET reference): n+ poly gate,
tox = 2 nm, L = 0.15 um, 20 nm deep p-type channel layer 1e18 cm^-3 on a
5e16 cm^-3 substrate, n+ source/drain 2e20 cm^-3.
Electrodes: 1 = source, 2 = drain, 3 = gate, 4 = substrate.

What this script checks
-----------------------
1. Threshold voltage. The uniform-body formula
       V_T = V_FB + 2 phi_F + sqrt(2 q eps_s N_A 2 phi_F) / C_ox
   overestimates it: the 1e18 layer is thinner than the maximum depletion
   width (~35 nm), so the depletion region reaches the lightly doped
   substrate. And "V_T" depends on how you extract it (max-gm vs constant
   current): state the definition.
2. Subthreshold swing S = ln(10) kT/q (1 + C_d/C_ox): PADRE is within
   ~1 mV/dec here, since the 2 nm oxide keeps gate control strong.
3. DIBL: zero in long-channel theory; ~17 mV/V here (a 2-D effect).
4. Saturation current vs. the absolute velocity-saturation bound
   W C_ox v_sat (V_G - V_T): PADRE reaches roughly half of it.

Not in PADRE or the textbook: gate tunnelling through 2 nm SiO2, inversion
layer quantisation, poly depletion, source/drain series resistance.

Run:  python mosfet_example.py
"""

import shutil

import numpy as np

from nanohubpadre import create_mosfet

Q, KT, NI, EG = 1.602e-19, 0.025851, 9.963e9, 1.12
EPS_SI, EPS_OX = 11.8 * 8.854e-14, 3.9 * 8.854e-14
COX = EPS_OX / 2e-7                                   # F/cm^2 for 2 nm
L_UM = 0.15


def compare(name, textbook, padre, why):
    diff = f"({(padre - textbook) / textbook * 100:+6.1f} %)" if textbook else "   (n/a) "
    print(f"  {name:<32} textbook {textbook:10.4g}   PADRE {padre:10.4g}   {diff}  {why}")


def transfer(vds):
    sim = create_mosfet(log_iv=True, vgs_sweep=(0.0, 1.5, 0.025), vds=vds)
    if sim.run().returncode != 0:
        raise SystemExit("PADRE failed; see .padre_run.log in " + sim.working_dir)
    iv = sim.get_iv_data()
    vg, idr = iv.get_voltages(3), np.abs(iv.get_currents(2))
    _, u = np.unique(vg, return_index=True)
    return vg[u], idr[u]


def main():
    if shutil.which("padre") is None:
        print(create_mosfet(log_iv=True, vgs_sweep=(0.0, 1.5, 0.025), vds=0.05).generate_deck())
        print("\nPADRE not found on PATH: showing the input deck only.")
        return

    vg_lin, id_lin = transfer(0.05)
    vg_sat, id_sat = transfer(1.2)

    phi_f = KT * np.log(1e18 / NI)
    vfb = -(EG / 2 + phi_f)
    vt_uniform = vfb + 2 * phi_f + np.sqrt(2 * Q * EPS_SI * 1e18 * 2 * phi_f) / COX

    gm = np.gradient(id_lin, vg_lin)
    k = np.argmax(gm)
    vt_maxgm = vg_lin[k] - id_lin[k] / gm[k] - 0.025
    icc = 1e-7 / L_UM
    vt_cc_lin = np.interp(np.log(icc), np.log(id_lin), vg_lin)
    vt_cc_sat = np.interp(np.log(icc), np.log(id_sat), vg_sat)

    sub = (vg_lin < 0.5) & (id_lin > 0)
    swing = 1e3 * np.min(np.gradient(vg_lin[sub], np.log10(id_lin[sub]))[2:-2])
    w_dep = 40.5e-7                                   # step-profile depletion width at V_T (cm)
    swing_th = 1e3 * np.log(10) * KT * (1 + EPS_SI / w_dep / COX)

    print("MOSFET (L = 0.15 um): textbook vs PADRE")
    compare("V_T (V), uniform-body formula", vt_uniform, vt_maxgm, "shallow 1e18 layer; max-gm definition")
    compare("V_T (V), constant-current", vt_uniform, vt_cc_lin, "a different definition, a different V_T")
    compare("subthreshold swing (mV/dec)", swing_th, swing, "gate control still strong at 2 nm oxide")
    compare("DIBL (mV/V)", 0.0, 1e3 * (vt_cc_lin - vt_cc_sat) / 1.15, "drain field reaches the source barrier")
    bound = 1e-4 * COX * 1.03e7 * (1.5 - vt_maxgm)
    compare("Id(Vg=1.5, Vd=1.2) vs vsat bound", bound, id_sat[-1], "carriers at the source are below v_sat")
    print(f"  on/off ratio at Vd = 1.2 V: {id_sat[-1] / id_sat[0]:.1e}")


if __name__ == "__main__":
    main()
