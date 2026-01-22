#!/usr/bin/env python3
"""
Unit tests for general deck structure requirements.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.pndiode import create_pndiode_simulation
from examples.moscap import create_moscap_simulation
from examples.mesfet import create_mesfet_simulation
from examples.single_mosgap import create_single_mosgap_simulation
from examples.mosfet_equivalent import create_mosfet_simulation


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
