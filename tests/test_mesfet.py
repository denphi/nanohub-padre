#!/usr/bin/env python3
"""
Unit tests for the MESFET simulation.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.mesfet import create_mesfet_simulation


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


if __name__ == "__main__":
    unittest.main()
