#!/usr/bin/env python3
"""
Silicon solar cell structure, and an honest limit of the simulation.

Device (factory default): n+ emitter (Gaussian, 1e19 cm^-3 peak, 0.5 um deep)
on a 200 um p-type base doped 1e16 cm^-3, front and back surface
recombination velocities, SRH + Auger recombination.

What this script shows
----------------------
1. The structure: equilibrium band diagram of the shallow n+/p junction.
2. Why create_solar_cell() is marked *experimental*. The dark current of a
   good 200 um cell at small forward bias is ~1e-17 A per um of depth or
   less, which is at PADRE's numerical noise floor. You can SEE this with an
   exact identity: terminal currents must satisfy I1 + I2 = 0 (Kirchhoff).
   Where the imbalance exceeds ~1 %, the "current" is round-off, not physics.
   Increasing device_z_width does not help: it scales signal and noise
   together.

The textbook dark current, J0 = q ni^2 Dn / (N_A L_n), is the physics a
solar cell's open-circuit voltage depends on; this device cannot resolve it,
so do not read I-V, ideality or V_oc from it.

Run:  python solar_cell_example.py
"""

import shutil
import warnings

import numpy as np

from nanohubpadre import create_solar_cell


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # the factory's "experimental" warning
        sim = create_solar_cell(log_bands_eq=True, log_iv=True, forward_sweep=(0.0, 0.4, 0.05))

    if shutil.which("padre") is None:
        print(sim.generate_deck())
        print("\nPADRE not found on PATH: showing the input deck only.")
        return
    if sim.run().returncode != 0:
        raise SystemExit("PADRE failed; see .padre_run.log in " + sim.working_dir)

    ec = sim.outputs.get("cbeq")
    print(f"Band bending across the junction: {abs(ec.y[0] - ec.y[-1]):.3f} eV "
          f"(textbook Vbi for 1e19/1e16: {0.025851 * np.log(1e35 / 9.963e9 ** 2):.3f} V)")

    iv = sim.get_iv_data()
    v, i, err = iv.get_voltages(1), np.abs(iv.get_currents(1)), iv.continuity_error()
    print("\n  V (V)     |I| (A/um)   |I1+I2|/|I|")
    for vk, ik, ek in zip(v, i, err):
        tag = "  <- numerical noise, not physics" if ek > 0.01 else ""
        print(f"  {vk:5.2f}   {ik:11.3e}   {ek:10.1e}{tag}")


if __name__ == "__main__":
    main()
