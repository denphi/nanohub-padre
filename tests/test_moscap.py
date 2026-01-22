#!/usr/bin/env python3
"""
Unit tests for the MOS capacitor simulation.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.moscap import create_moscap_simulation


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


if __name__ == "__main__":
    unittest.main()
