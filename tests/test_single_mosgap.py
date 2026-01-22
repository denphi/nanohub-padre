#!/usr/bin/env python3
"""
Unit tests for the single MOS gap simulation.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.single_mosgap import create_single_mosgap_simulation


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


if __name__ == "__main__":
    unittest.main()
