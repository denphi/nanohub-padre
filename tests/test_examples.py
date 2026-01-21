#!/usr/bin/env python3
"""
Unit tests for the Python equivalent input deck files.

These tests verify that the Python representations generate
PADRE input decks that are functionally equivalent to the original .in files.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.pndiode import create_pndiode_simulation
from examples.moscap import create_moscap_simulation
from examples.mesfet import create_mesfet_simulation
from examples.single_mosgap import create_single_mosgap_simulation
from examples.mosfet_equivalent import create_mosfet_simulation


class TestPNDiode(unittest.TestCase):
    """Test PN diode simulation generation."""

    def setUp(self):
        self.sim = create_pndiode_simulation()
        self.deck = self.sim.generate_deck()

    def test_has_title(self):
        """Test that the deck has the correct title."""
        self.assertIn("TITLE  pn diode (setup)", self.deck)

    def test_has_options(self):
        """Test that options are included."""
        self.assertIn("options po", self.deck)

    def test_mesh_dimensions(self):
        """Test mesh dimensions."""
        self.assertIn("mesh rect nx=200 ny=3", self.deck)
        self.assertIn("width=1", self.deck)

    def test_mesh_lines(self):
        """Test mesh line specifications."""
        self.assertIn("x.m n=1 l=0", self.deck)
        self.assertIn("x.m n=100 l=0.5", self.deck)
        self.assertIn("x.m n=200 l=1.0", self.deck)
        self.assertIn("y.m n=1 l=0", self.deck)
        self.assertIn("y.m n=3 l=1", self.deck)

    def test_regions(self):
        """Test region definitions."""
        self.assertIn("region", self.deck.lower())
        self.assertIn("silicon", self.deck.lower())
        self.assertIn("num=1", self.deck)

    def test_electrodes(self):
        """Test electrode definitions."""
        self.assertIn("elec num=1", self.deck)
        self.assertIn("elec num=2", self.deck)

    def test_doping(self):
        """Test doping profiles."""
        self.assertIn("dop", self.deck)
        self.assertIn("p.type", self.deck)
        self.assertIn("n.type", self.deck)
        self.assertIn("uniform", self.deck)

    def test_material(self):
        """Test material specification."""
        self.assertIn("material name=silicon", self.deck)
        self.assertIn("taun0=1e-06", self.deck)
        self.assertIn("taup0=1e-06", self.deck)

    def test_models(self):
        """Test model specification."""
        self.assertIn("models", self.deck)
        self.assertIn("srh", self.deck)
        self.assertIn("conmob", self.deck)
        self.assertIn("fldmob", self.deck)
        self.assertIn("impact", self.deck)
        self.assertIn("temp=300", self.deck)

    def test_system(self):
        """Test system specification."""
        self.assertIn("system", self.deck)
        self.assertIn("electrons", self.deck)
        self.assertIn("holes", self.deck)
        self.assertIn("newton", self.deck)

    def test_solve_init(self):
        """Test initial solve."""
        self.assertIn("solve init", self.deck)

    def test_solve_sweep(self):
        """Test voltage sweep."""
        self.assertIn("solve proj", self.deck)
        self.assertIn("vstep=0.03", self.deck)
        self.assertIn("nsteps=20", self.deck)

    def test_log(self):
        """Test log command."""
        self.assertIn("log outf=iv", self.deck)

    def test_plots(self):
        """Test plot commands."""
        self.assertIn("plot.1d pot", self.deck)
        self.assertIn("plot.1d band.val", self.deck)
        self.assertIn("plot.1d band.con", self.deck)
        self.assertIn("plot.1d qfn", self.deck)
        self.assertIn("plot.1d qfp", self.deck)
        self.assertIn("plot.1d ele", self.deck)
        self.assertIn("plot.1d hole", self.deck)

    def test_ends_with_end(self):
        """Test that deck ends with 'end'."""
        self.assertTrue(self.deck.strip().endswith("end"))


class TestMOSCAP(unittest.TestCase):
    """Test MOS capacitor simulation generation."""

    def setUp(self):
        self.sim = create_moscap_simulation()
        self.deck = self.sim.generate_deck()

    def test_mesh_dimensions(self):
        """Test mesh dimensions."""
        self.assertIn("mesh rect nx=3 ny=41", self.deck)

    def test_mesh_order(self):
        """Test that mesh lines are in correct order (y before x)."""
        y_pos = self.deck.find("y.m n=1")
        x_pos = self.deck.find("x.m n=1")
        self.assertLess(y_pos, x_pos, "Y mesh should come before X mesh")

    def test_regions(self):
        """Test region definitions for oxide-silicon-oxide structure."""
        self.assertIn("region ins", self.deck)
        self.assertIn("region semi", self.deck)
        self.assertIn("name=sio2", self.deck)
        self.assertIn("name=silicon", self.deck)

    def test_three_regions(self):
        """Test that there are 3 regions defined."""
        self.assertIn("num=1", self.deck)
        self.assertIn("num=2", self.deck)
        self.assertIn("num=3", self.deck)

    def test_electrodes(self):
        """Test electrode definitions."""
        self.assertIn("elec num=1", self.deck)
        self.assertIn("elec num=2", self.deck)

    def test_doping(self):
        """Test doping profile."""
        self.assertIn("dop", self.deck)
        self.assertIn("p.type", self.deck)
        self.assertIn("conc=1", self.deck)
        self.assertIn("reg=2", self.deck)

    def test_contacts(self):
        """Test contact specifications."""
        self.assertIn("contact all neutral", self.deck)
        self.assertIn("n.polysilicon", self.deck)

    def test_materials(self):
        """Test material specifications."""
        self.assertIn("material name=silicon", self.deck)
        self.assertIn("material name=sio2", self.deck)
        self.assertIn("permittivity=3.9", self.deck)
        self.assertIn("qf=0", self.deck)

    def test_models(self):
        """Test model specification."""
        self.assertIn("models", self.deck)
        self.assertIn("conmob", self.deck)
        self.assertIn("fldmob", self.deck)
        self.assertIn("temp=300", self.deck)

    def test_ac_analysis(self):
        """Test AC analysis setup."""
        self.assertIn("ac.analysis", self.deck)
        self.assertIn("freq=", self.deck)
        self.assertIn("log acf=ac", self.deck)

    def test_log_off(self):
        """Test log off command."""
        self.assertIn("log off", self.deck)

    def test_ends_with_end(self):
        """Test that deck ends with 'end'."""
        self.assertTrue(self.deck.strip().endswith("end"))


class TestMESFET(unittest.TestCase):
    """Test MESFET simulation generation."""

    def setUp(self):
        self.sim = create_mesfet_simulation()
        self.deck = self.sim.generate_deck()

    def test_mesh_dimensions(self):
        """Test mesh dimensions."""
        self.assertIn("mesh rect nx=61 ny=51", self.deck)

    def test_regions(self):
        """Test region definitions."""
        self.assertIn("region silicon num=1", self.deck)
        self.assertIn("region silicon num=2", self.deck)
        self.assertIn("region silicon num=3", self.deck)
        self.assertIn("region silicon num=4", self.deck)

    def test_electrodes(self):
        """Test electrode definitions (source, drain, gate)."""
        self.assertIn("elec num=1", self.deck)
        self.assertIn("elec num=2", self.deck)
        self.assertIn("elec num=3", self.deck)

    def test_doping_types(self):
        """Test both n-type and p-type doping."""
        self.assertIn("p.type", self.deck)
        self.assertIn("n.type", self.deck)

    def test_schottky_contact(self):
        """Test Schottky gate contact with work function."""
        self.assertIn("workfunction=4.87", self.deck)

    def test_neutral_contacts(self):
        """Test neutral contacts."""
        self.assertIn("contact all neutral", self.deck)

    def test_models(self):
        """Test model specification."""
        self.assertIn("models", self.deck)
        self.assertIn("bgn", self.deck)
        self.assertIn("conmob", self.deck)
        self.assertIn("fldmob", self.deck)

    def test_single_carrier(self):
        """Test single carrier (electron) system."""
        self.assertIn("carr=1", self.deck)
        self.assertIn("electrons", self.deck)

    def test_plot3d(self):
        """Test 3D plot commands."""
        self.assertIn("plot.3d doping", self.deck)
        self.assertIn("plot.3d poten elect", self.deck)

    def test_solve_sequences(self):
        """Test solve sequence."""
        self.assertIn("solve init", self.deck)
        self.assertIn("vstep=-0.1", self.deck)
        self.assertIn("vstep=0.1", self.deck)

    def test_ends_with_end(self):
        """Test that deck ends with 'end'."""
        self.assertTrue(self.deck.strip().endswith("end"))


class TestSingleMOSGap(unittest.TestCase):
    """Test single MOS gap simulation generation."""

    def setUp(self):
        self.sim = create_single_mosgap_simulation()
        self.deck = self.sim.generate_deck()

    def test_mesh_dimensions(self):
        """Test mesh dimensions."""
        self.assertIn("mesh rect nx=3 ny=60", self.deck)

    def test_two_regions(self):
        """Test oxide-silicon structure."""
        self.assertIn("region ins num=1", self.deck)
        self.assertIn("region semi num=2", self.deck)

    def test_electrodes(self):
        """Test electrode definitions."""
        self.assertIn("elec num=1", self.deck)
        self.assertIn("elec num=2", self.deck)

    def test_doping(self):
        """Test p-type doping."""
        self.assertIn("p.type", self.deck)
        self.assertIn("conc=1", self.deck)

    def test_contacts(self):
        """Test contact specifications."""
        self.assertIn("contact all neutral", self.deck)
        self.assertIn("n.polysilicon num=1", self.deck)

    def test_materials(self):
        """Test material specifications."""
        self.assertIn("material name=silicon", self.deck)
        self.assertIn("material name=sio2", self.deck)

    def test_srh_model(self):
        """Test SRH model is enabled."""
        self.assertIn("srh", self.deck)

    def test_ac_analysis(self):
        """Test AC analysis."""
        self.assertIn("ac.analysis", self.deck)
        self.assertIn("freq=", self.deck)

    def test_plot1d_quantities(self):
        """Test 1D plot quantities."""
        self.assertIn("plot.1d pot", self.deck)
        self.assertIn("plot.1d qfn", self.deck)
        self.assertIn("plot.1d qfp", self.deck)
        self.assertIn("plot.1d band.val", self.deck)
        self.assertIn("plot.1d band.con", self.deck)
        self.assertIn("plot.1d ele", self.deck)
        self.assertIn("plot.1d hole", self.deck)
        self.assertIn("plot.1d net.charge", self.deck)
        self.assertIn("plot.1d e.field", self.deck)

    def test_ends_with_end(self):
        """Test that deck ends with 'end'."""
        self.assertTrue(self.deck.strip().endswith("end"))


class TestMOSFET(unittest.TestCase):
    """Test MOSFET simulation generation."""

    def setUp(self):
        self.sim = create_mosfet_simulation()
        self.deck = self.sim.generate_deck()

    def test_has_title(self):
        """Test that the deck has the correct title."""
        self.assertIn("MOSFET - NMOS", self.deck)

    def test_mesh_dimensions(self):
        """Test mesh dimensions."""
        self.assertIn("mesh rect nx=51 ny=51", self.deck)

    def test_seven_regions(self):
        """Test that there are 7 regions (substrate, source, drain, channel, gate oxide, fillers)."""
        for i in range(1, 8):
            self.assertIn(f"num={i}", self.deck)

    def test_silicon_and_oxide_regions(self):
        """Test silicon and oxide materials."""
        self.assertIn("silicon", self.deck.lower())
        self.assertIn("oxide", self.deck.lower())

    def test_four_electrodes(self):
        """Test 4 electrodes (source, drain, gate, substrate)."""
        self.assertIn("elec num=1", self.deck)
        self.assertIn("elec num=2", self.deck)
        self.assertIn("elec num=3", self.deck)
        self.assertIn("elec num=4", self.deck)

    def test_doping_types(self):
        """Test n-type and p-type doping."""
        self.assertIn("n.type", self.deck)
        self.assertIn("p.type", self.deck)

    def test_high_doping_source_drain(self):
        """Test high doping for source/drain."""
        self.assertIn("conc=1", self.deck)  # 1e20

    def test_polysilicon_gate(self):
        """Test n-polysilicon gate contact."""
        self.assertIn("n.polysilicon num=3", self.deck)

    def test_bgn_model(self):
        """Test band-gap narrowing model."""
        self.assertIn("bgn", self.deck)

    def test_single_carrier(self):
        """Test single carrier system."""
        self.assertIn("carr=1", self.deck)

    def test_solve_with_outfile(self):
        """Test solve with output file."""
        self.assertIn("solve init outf=initsol", self.deck)

    def test_load_command(self):
        """Test load command."""
        self.assertIn("load inf=initsol", self.deck)
        self.assertIn("load inf=vgn0", self.deck)

    def test_multiple_gate_voltages(self):
        """Test sweeping gate voltage."""
        self.assertIn("outf=vgn0", self.deck)
        self.assertIn("outf=vgn1", self.deck)
        self.assertIn("outf=vgn2", self.deck)
        self.assertIn("outf=vgn3", self.deck)
        self.assertIn("outf=vgn4", self.deck)

    def test_iv_logging(self):
        """Test IV logging."""
        self.assertIn("log outf=idvg", self.deck)
        self.assertIn("log outf=idvd", self.deck)

    def test_plot3d(self):
        """Test 3D plot commands."""
        self.assertIn("plot.3d doping", self.deck)
        self.assertIn("plot.3d poten elect", self.deck)

    def test_ends_with_end(self):
        """Test that deck ends with 'end'."""
        self.assertTrue(self.deck.strip().endswith("end"))


class TestDeckStructure(unittest.TestCase):
    """Test general deck structure requirements."""

    def test_all_decks_end_with_end(self):
        """Test that all simulation decks end with 'end'."""
        simulations = [
            create_pndiode_simulation(),
            create_moscap_simulation(),
            create_mesfet_simulation(),
            create_single_mosgap_simulation(),
            create_mosfet_simulation(),
        ]
        for sim in simulations:
            deck = sim.generate_deck()
            self.assertTrue(deck.strip().endswith("end"),
                          f"{sim.__class__.__name__} deck should end with 'end'")

    def test_all_decks_have_mesh(self):
        """Test that all simulation decks have mesh definition."""
        simulations = [
            create_pndiode_simulation(),
            create_moscap_simulation(),
            create_mesfet_simulation(),
            create_single_mosgap_simulation(),
            create_mosfet_simulation(),
        ]
        for sim in simulations:
            deck = sim.generate_deck()
            self.assertIn("mesh", deck.lower(),
                         f"{sim.__class__.__name__} deck should have mesh")

    def test_all_decks_have_region(self):
        """Test that all simulation decks have region definition."""
        simulations = [
            create_pndiode_simulation(),
            create_moscap_simulation(),
            create_mesfet_simulation(),
            create_single_mosgap_simulation(),
            create_mosfet_simulation(),
        ]
        for sim in simulations:
            deck = sim.generate_deck()
            self.assertIn("region", deck.lower(),
                         f"{sim.__class__.__name__} deck should have region")

    def test_all_decks_have_solve(self):
        """Test that all simulation decks have solve command."""
        simulations = [
            create_pndiode_simulation(),
            create_moscap_simulation(),
            create_mesfet_simulation(),
            create_single_mosgap_simulation(),
            create_mosfet_simulation(),
        ]
        for sim in simulations:
            deck = sim.generate_deck()
            self.assertIn("solve", deck.lower(),
                         f"{sim.__class__.__name__} deck should have solve")

    def test_lowercase_commands(self):
        """Test that commands are lowercase (PADRE convention)."""
        simulations = [
            create_pndiode_simulation(),
            create_moscap_simulation(),
            create_mesfet_simulation(),
            create_single_mosgap_simulation(),
            create_mosfet_simulation(),
        ]
        uppercase_commands = ["MESH", "REGION", "ELEC", "DOP", "SOLVE", "MODELS"]
        for sim in simulations:
            deck = sim.generate_deck()
            lines = deck.split('\n')
            for line in lines:
                if line.strip() and not line.strip().startswith('TITLE'):
                    # Check that the first word (command) is lowercase
                    first_word = line.strip().split()[0] if line.strip().split() else ""
                    if first_word in uppercase_commands:
                        self.fail(f"Command '{first_word}' should be lowercase in deck")


if __name__ == "__main__":
    unittest.main()
