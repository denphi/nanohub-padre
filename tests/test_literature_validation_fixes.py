"""
Regression tests for the 2026-09-30 physics validation against literature.

Each expectation was found by running PADRE 2.4E-r15 on nanoHUB and
comparing against textbook/literature references; the comments name the
observed failure that motivated the test.
"""

import os
import re
import warnings

import numpy as np
import pytest

from nanohubpadre.devices import create_mos_capacitor, create_mosfet
from nanohubpadre.parser import parse_ac_file

DATA = os.path.join(os.path.dirname(__file__), "data")


def deck_lines(sim):
    """Deck lines with PADRE continuation lines folded back in."""
    out = []
    for raw in sim.generate_deck().splitlines():
        if raw.startswith("+") and out:
            out[-1] += " " + raw.lstrip("+ ").strip()
        else:
            out.append(raw)
    return out


def ix_range(line):
    return (int(re.search(r"ix\.l=(\d+)", line).group(1)),
            int(re.search(r"ix\.h=(\d+)", line).group(1)))


class TestBiasRamp:
    def test_ramp_never_resolves_its_start_bias(self):
        """PROJ from two solutions at the same bias went to NaN: the MOS-cap
        LF ramp re-solved Vg=2.0 after the HF sweep and PADRE aborted
        ("Bias stack exceeded"), so the LF C-V file was never written."""
        sim = create_mos_capacitor(log_cv=True, log_cv_lf=True,
                                   vg_sweep=(-2.0, 2.0, 0.1))
        solves = [l for l in deck_lines(sim) if l.startswith("solve")]
        k = next(i for i, l in enumerate(solves) if "vg_ramp_lf" in l)
        first, ramp = solves[k - 1], solves[k]
        assert "prev" in first and "v1=1.8" in first
        assert "proj" in ramp and "v1=1.6" in ramp

    @pytest.mark.parametrize("vds", [0.5, 1.2])
    def test_drain_ramp_projects_after_first_step(self, vds):
        """An all-PREV drain ramp stalled at Vd=0.849 V ("Bias trapping
        below tolerances") at every step size down to 0.05 V."""
        sim = create_mosfet(log_iv=True, vgs_sweep=(0.0, 1.5, 0.1), vds=vds)
        solves = [l for l in deck_lines(sim) if l.startswith("solve")]
        k = next(i for i, l in enumerate(solves) if "vd_set" in l)
        first = solves[k - 1]
        assert "prev" in first and "vstep=" not in first and "v2=0 " not in first
        assert "proj" in solves[k]


class TestMosfetContacts:
    @pytest.mark.parametrize("length,width", [(0.15, 0.25), (1.0, 1.3)])
    def test_source_drain_contacts_clear_the_junction_column(self, length, width):
        """The junction column takes the channel's p-type doping; an ohmic
        contact there pinned n to ni^2/Na between channel and source.  The
        default device lost 4.2x linear current and an L=1 um device carried
        2e-11 A instead of 2.8e-5 A."""
        lines = deck_lines(create_mosfet(channel_length=length,
                                         device_width=width))
        src = next(l for l in lines if l.startswith("elec num=1"))
        drn = next(l for l in lines if l.startswith("elec num=2"))
        chan = next(l for l in lines
                    if l.startswith("region num=4"))
        ch_lo, ch_hi = ix_range(chan)
        assert ix_range(src)[1] < ch_lo
        assert ix_range(drn)[0] > ch_hi


class TestAcParser:
    """Fixtures are trimmed from real PADRE 2.4E runs of the default MOS-cap
    (1 um x 1 um, tox = 2 nm, p = 1e16)."""

    def test_capacitance_is_in_farads_per_micron(self):
        """The parser multiplied by 1e8 while the docs and plot labels said
        F/um, so C_acc came out as 1.64e-6 instead of 1.64e-14."""
        ac = parse_ac_file(os.path.join(DATA, "moscap_hf_1MHz.ac"))
        _, c = ac.get_cv_data(gate_electrode=1)
        c_ox = 3.9 * 8.854e-14 * 1e-8 / 2e-7    # F for a 1 um^2 gate
        assert c[0] == pytest.approx(1.6404790e-14)
        assert 0.9 < c[0] / c_ox < 1.0           # accumulation just below Cox

    def test_hf_solution_conserves_charge(self):
        ac = parse_ac_file(os.path.join(DATA, "moscap_hf_1MHz.ac"))
        assert ac.conservation_error().max() < 1e-4
        assert not ac.check_conservation(warn=False).any()

    def test_low_frequency_ac_is_flagged(self):
        """At 1 Hz in inversion C11 + C12 was off by up to 2800%."""
        ac = parse_ac_file(os.path.join(DATA, "moscap_lf_1Hz.ac"))
        err = ac.conservation_error()
        assert err[:2].max() < 1e-4              # accumulation: fine
        assert err[2:].min() > 0.3               # inversion: noise
        with pytest.warns(UserWarning, match="charge conservation"):
            ac.check_conservation()

    def test_quasistatic_cv_from_dc_charge(self):
        """dQ/dV of the DC gate charge: 1.63e-14 F in accumulation and
        1.69e-14 F in strong inversion (exact classical LF: 1.62e-14 and
        1.68e-14), independent of the broken 1 Hz AC solve."""
        ac = parse_ac_file(os.path.join(DATA, "moscap_lf_1Hz.ac"))
        v, c = ac.get_quasistatic_cv(gate_electrode=1)
        assert list(v) == [-2.0, -1.9, 0.3, 2.0]
        assert c[0] == pytest.approx(1.6346e-14, rel=1e-3)
        hf = parse_ac_file(os.path.join(DATA, "moscap_hf_1MHz.ac"))
        v, c = hf.get_quasistatic_cv(gate_electrode=1)
        assert c[-1] == pytest.approx(1.6925e-14, rel=1e-3)


class TestOutfileCollisions:
    """PADRE names each bias point of a stepped SOLVE by incrementing the
    last character of outf= (idvd, idve, idvf, idvg, ...).  The MOSFET Id-Vg
    sweep wrote its first solution over the idvg I-V log (Vg = 0 lost), the
    Id-Vd sweep its 4th (Vd = 0..0.15 V lost), the MESFET its first."""

    @staticmethod
    def _collisions(sim):
        from nanohubpadre.devices._common import padre_outfile_names
        lines = deck_lines(sim)
        logs = set()
        for l in lines:
            if l.startswith("log"):
                logs.update(v for k, v in (t.split("=", 1) for t in l.split()
                                           if "=" in t) if k in ("outf", "acfile"))
        hits = []
        for l in lines:
            m = re.search(r"outf=(\S+)", l)
            if l.startswith("solve") and m:
                n = int(re.search(r"nsteps=(\d+)", l).group(1)) if "nsteps=" in l else 0
                hits += sorted(logs.intersection(padre_outfile_names(m.group(1), n + 1)))
        return hits

    @pytest.mark.parametrize("make", [
        lambda: create_mosfet(log_iv=True, vgs_sweep=(0.0, 1.5, 0.025), vds=0.05),
        lambda: create_mosfet(log_iv=True, vds_sweep=(0.0, 1.5, 0.05), vgs=1.2),
        lambda: create_mosfet(log_iv=True, iv_file="idvd",
                              vds_sweep=(0.0, 1.5, 0.05), vgs=1.2),
        lambda: __import__("nanohubpadre.devices", fromlist=["x"]).create_mesfet(
            log_iv=True, vds_sweep=(0.0, 2.0, 0.1)),
    ])
    def test_no_sweep_overwrites_its_log(self, make):
        assert self._collisions(make()) == []


class TestFieldMobilityDrive:
    def test_pn_diode_uses_parallel_field_drive(self):
        """With PADRE's default qfb drive the diode carried 0.60-0.89x the
        Shockley current (n = 1.03); eoqf gives 0.98-1.02x (n = 1.005)."""
        from nanohubpadre.devices import create_pn_diode
        models = next(l for l in deck_lines(create_pn_diode())
                      if l.startswith("models"))
        assert "e.drive=eoqf" in models
        models = next(l for l in deck_lines(create_pn_diode(fldmob=False))
                      if l.startswith("models"))
        assert "e.drive" not in models

    def test_eoj_drive_warns(self):
        from nanohubpadre.models import Models
        with pytest.warns(UserWarning, match="not available"):
            Models(fldmob=True, e_drive="eoj")


class TestSilentPadreStops:
    @pytest.mark.parametrize("message", [
        " Bias stack (10 intermediate points) exceeded",
        " Bias trapping below tolerances",
        " Warning: nonconvergence in idirac",
    ])
    def test_run_reports_failure(self, tmp_path, message):
        """PADRE printed these, stopped, and exited 0; the run looked
        successful while the I-V/AC logs were truncated or never written."""
        from nanohubpadre.devices import create_pn_diode
        fake = tmp_path / "fakepadre"
        fake.write_text(f"#!/bin/sh\necho 'Solution for bias:'\necho '{message}'\nexit 0\n")
        fake.chmod(0o755)
        sim = create_pn_diode(log_iv=True, forward_sweep=(0.0, 0.2, 0.1))
        sim.working_dir = str(tmp_path)
        with pytest.warns(UserWarning, match="stopped before the end"):
            result = sim.run(padre_executable=str(fake), auto_output_dir=False)
        assert result.returncode != 0
