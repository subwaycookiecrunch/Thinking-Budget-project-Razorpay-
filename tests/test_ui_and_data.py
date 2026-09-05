"""
tests/test_ui_and_data.py
=========================
Integration tests for dataset schema integrity, demo traces, and app.py Gradio helpers.
"""
from __future__ import annotations

import os
import sys
import json
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app


class TestDataIntegrity(unittest.TestCase):
    """Verify all static datasets adhere to expected schema and invariants."""

    def test_cve_training_data_schema(self):
        path = os.path.join(ROOT, "data", "cve_training_data.json")
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 100)
        first = data[0]
        self.assertIn("cveId", first)
        self.assertIn("repo", first)
        self.assertIn("file", first)
        self.assertIn("features", first)
        self.assertIn("label", first)

    def test_code_snippets_schema(self):
        path = os.path.join(ROOT, "data", "code_snippets.json")
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)
        self.assertGreater(len(data), 2800)

    def test_transfer_episodes_schema(self):
        path = os.path.join(ROOT, "data", "transfer_episodes.json")
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 5)
        for ep in data:
            self.assertIn("domain", ep)
            self.assertIn("files", ep)

    def test_demo_traces_schema(self):
        path = os.path.join(ROOT, "data", "demo_traces.json")
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 6)
        for trace in data:
            self.assertIn("cve_id", trace)
            self.assertIn("policy", trace)
            self.assertIn("steps", trace)
            self.assertIn("metrics", trace)

    def test_red_team_results_schema(self):
        path = os.path.join(ROOT, "data", "red_team_results.json")
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertIn("attacks", data)
        honest = next((a for a in data["attacks"] if a["name"] == "honest metacognitive"), None)
        self.assertIsNotNone(honest)
        honest_score = honest["combined_reward"]
        for a in data["attacks"]:
            if a["name"] != "honest metacognitive":
                self.assertLess(a["combined_reward"], honest_score)


class TestAppHelpers(unittest.TestCase):
    """Verify app.py UI helpers operate correctly with valid and edge inputs."""

    def test_cve_dropdown_choices(self):
        choices = app.cve_dropdown_choices()
        self.assertIsInstance(choices, list)
        self.assertGreater(len(choices), 0)
        label, val = choices[0]
        self.assertIn("CVE-", val)

    def test_run_demo(self):
        choices = app.cve_dropdown_choices()
        cve_id = choices[0][1]
        u_summary, t_summary, u_steps, t_steps = app.run_demo(cve_id)
        self.assertIn("TOTAL SCORE:", u_summary)
        self.assertIn("TOTAL SCORE:", t_summary)
        self.assertGreater(len(u_steps), 0)
        self.assertGreater(len(t_steps), 0)

    def test_apply_budget_to_trace(self):
        traces = app.load_traces()
        trace = traces[0]
        capped = app.apply_budget_to_trace(trace, per_block_budget=50, episode_budget=100)
        self.assertIsNotNone(capped)
        self.assertEqual(len(capped["steps"]), len(trace["steps"]))

    def test_red_team_helpers(self):
        choices, default = app.red_team_attack_choices()
        self.assertGreater(len(choices), 0)
        self.assertIsNotNone(default)

        header, details = app.render_red_team_attack(default)
        self.assertIn("Score", header)
        self.assertTrue(len(details) > 0)

        # Empty/None selection
        h_none, d_none = app.render_red_team_attack(None)
        self.assertIn("Select an attack", h_none)

    def test_transfer_metrics_md(self):
        md = app.transfer_metrics_md()
        self.assertIn("Transfer to", md)
        self.assertIn("F1", md)


if __name__ == "__main__":
    unittest.main()
