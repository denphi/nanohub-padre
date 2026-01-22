#!/usr/bin/env python3
"""
Unit tests for the PN diode simulation.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.pndiode import create_pndiode_simulation


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


if __name__ == "__main__":
    unittest.main()
