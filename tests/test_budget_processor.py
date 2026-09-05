"""
tests/test_budget_processor.py
==============================
Unit tests for inference-time budget enforcement in scripts/budget_processor.py.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.budget_processor import enforce_character_budget


class TestBudgetProcessor(unittest.TestCase):
    """Test character-level budget enforcement logic."""

    def test_no_think_block(self):
        text = 'Hello world! <tool_call>{"name": "read_file"}</tool_call>'
        res = enforce_character_budget(text, per_block_budget=100)
        self.assertEqual(res, text)

    def test_under_budget_think_block(self):
        text = "<think>Brief check.</think>\n<tool_call>{}</tool_call>"
        res = enforce_character_budget(text, per_block_budget=100)
        self.assertEqual(res, text)

    def test_over_budget_truncation(self):
        long_reasoning = "A" * 500
        text = f"<think>{long_reasoning}</think>\n<tool_call>{{}}</tool_call>"
        res = enforce_character_budget(text, per_block_budget=100)
        # Should be truncated to 100 chars of A's plus the closing tag
        self.assertIn("<think>", res)
        self.assertIn("</think>", res)
        inside = res[res.index("<think>") + 7 : res.index("</think>")]
        self.assertIn("[truncated by budget]", inside)
        self.assertTrue(inside.startswith("A" * 100))

    def test_episode_global_budget(self):
        # 2 blocks of 80 chars each, episode budget 100
        block1 = f"<think>{'B' * 80}</think>"
        block2 = f"<think>{'C' * 80}</think>"
        text = f"{block1}\n{block2}"
        res = enforce_character_budget(text, per_block_budget=200, episode_budget=100)

        # First block should get 80 chars, second block should only get remaining 20
        self.assertIn("B" * 80, res)
        self.assertIn("C" * 20, res)
        self.assertNotIn("C" * 21, res)

    def test_empty_string(self):
        self.assertEqual(enforce_character_budget("", per_block_budget=100), "")


if __name__ == "__main__":
    unittest.main()
