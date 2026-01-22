#!/usr/bin/env python3
"""
Unit tests for the MOSFET simulation.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.mosfet_equivalent import create_mosfet_simulation


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


if __name__ == "__main__":
    unittest.main()
