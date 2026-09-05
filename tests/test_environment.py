"""
tests/test_environment.py
=========================
Comprehensive integration & unit tests for CodeReviewEnvironment and CodeReviewEnv.
"""
from __future__ import annotations

import os
import sys
import unittest
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from openenv.core.env_server import CallToolAction, ListToolsAction
from code_review_env.server.environment import CodeReviewEnvironment, EPISODES, BUGGY_EPISODES, CODE_SNIPPETS
from code_review_env import CodeReviewEnv, CodeReviewAction, CodeReviewObservation


class TestCodeReviewEnvironment(unittest.TestCase):
    """Test suite for CodeReviewEnvironment (MCP-based environment)."""

    def setUp(self):
        self.env = CodeReviewEnvironment()

    def test_episodes_data_loaded(self):
        """Data files must load episodes and code snippets."""
        self.assertGreaterEqual(len(EPISODES), 100)
        self.assertGreaterEqual(len(BUGGY_EPISODES), 50)
        self.assertGreaterEqual(len(CODE_SNIPPETS), 2000)

    def test_reset_curriculum(self):
        """Reset with easy, medium, and hard difficulty returns valid context."""
        for diff in ["easy", "medium", "hard"]:
            obs = self.env.reset(seed=42, difficulty=diff)
            self.assertFalse(obs.done)
            self.assertIn("context", obs.metadata)
            ctx = obs.metadata["context"]
            self.assertIn("SECURITY INVESTIGATION BRIEFING", ctx)
            self.assertIn("CVE:", ctx)
            self.assertIn("Files in patch", ctx)

    def test_list_tools(self):
        """Environment must expose exactly 6 MCP tools."""
        self.env.reset(seed=123, difficulty="easy")
        obs = self.env.step(ListToolsAction())
        tool_names = {t.name for t in obs.tools}
        expected = {
            "read_file",
            "search_code",
            "get_function_list",
            "flag_vulnerable",
            "skip_file",
            "submit_report",
        }
        self.assertEqual(tool_names, expected)

    def test_read_file_tool(self):
        """read_file returns code, reduces budget, and catches unknown files."""
        obs = self.env.reset(seed=42, difficulty="easy")
        ctx = obs.metadata["context"]
        files = re.findall(r'• (.+?)\s+\[', ctx)
        self.assertTrue(len(files) > 0)
        target = files[0]

        # Valid read
        step_res = self.env.step(CallToolAction(tool_name="read_file", arguments={"file_path": target}))
        text = str(step_res.result.data if hasattr(step_res.result, "data") else step_res.result)
        self.assertIn(f"=== {target} ===", text)
        self.assertIn("Budget:", text)

        # Invalid file
        bad_res = self.env.step(CallToolAction(tool_name="read_file", arguments={"file_path": "non_existent.c"}))
        bad_text = str(bad_res.result.data if hasattr(bad_res.result, "data") else bad_res.result)
        self.assertIn("ERROR:", bad_text)

    def test_search_code_tool(self):
        """search_code finds matches or returns no matches message."""
        self.env.reset(seed=42, difficulty="easy")
        res = self.env.step(CallToolAction(tool_name="search_code", arguments={"pattern": "include"}))
        text = str(res.result.data if hasattr(res.result, "data") else res.result)
        self.assertTrue("Found 'include'" in text or "No matches" in text)

    def test_flag_and_skip_tools(self):
        """flag_vulnerable and skip_file properly record decisions and reasoning."""
        obs = self.env.reset(seed=42, difficulty="easy")
        ctx = obs.metadata["context"]
        files = re.findall(r'• (.+?)\s+\[', ctx)

        # Flag first file
        f_res = self.env.step(CallToolAction(
            tool_name="flag_vulnerable",
            arguments={"file_path": files[0], "reasoning": "Detailed vulnerability analysis with copy_from_user"}
        ))
        f_text = str(f_res.result.data if hasattr(f_res.result, "data") else f_res.result)
        self.assertIn("FLAGGED:", f_text)

        # Skip second file if available
        if len(files) > 1:
            s_res = self.env.step(CallToolAction(
                tool_name="skip_file",
                arguments={"file_path": files[1], "reasoning": "Header file, safe"}
            ))
            s_text = str(s_res.result.data if hasattr(s_res.result, "data") else s_res.result)
            self.assertIn("SKIPPED:", s_text)

    def test_submit_report_and_completion(self):
        """submit_report ends investigation, computes metrics, and locks further actions."""
        obs = self.env.reset(seed=42, difficulty="easy")
        ctx = obs.metadata["context"]
        files = re.findall(r'• (.+?)\s+\[', ctx)

        for f in files:
            self.env.step(CallToolAction(tool_name="skip_file", arguments={"file_path": f, "reasoning": "skip"}))

        rep = self.env.step(CallToolAction(
            tool_name="submit_report",
            arguments={"summary": "CVE-2017-16028 buffer overflow at line 42 in parse function.", "confidence": "high"}
        ))
        text = str(rep.result.data if hasattr(rep.result, "data") else rep.result)
        self.assertIn("INVESTIGATION COMPLETE", text)
        self.assertIn("TOTAL SCORE:", text)

        # Post-done actions should be blocked
        post = self.env.step(CallToolAction(tool_name="read_file", arguments={"file_path": files[0]}))
        post_text = str(post.result.data if hasattr(post.result, "data") else post.result)
        self.assertIn("ERROR: Investigation is complete", post_text)


class TestCodeReviewEnvClient(unittest.TestCase):
    """Test suite for CodeReviewEnv client wrapper (used by inference.py)."""

    def test_sync_context_manager(self):
        """Client must work in a 'with' block using .sync()."""
        with CodeReviewEnv().sync() as env:
            res = env.reset(difficulty="easy", seed=42)
            self.assertFalse(res.done)
            self.assertIsInstance(res.observation, CodeReviewObservation)
            self.assertTrue(len(res.observation.file_path) > 0)

            step_res = env.step(CodeReviewAction(decision="skip", reasoning="safe"))
            self.assertIsInstance(step_res.observation, CodeReviewObservation)

    def test_full_episode_triage_loop(self):
        """Stepping through all files terminates episode and yields final F1 score."""
        with CodeReviewEnv().sync() as env:
            res = env.reset(difficulty="easy", seed=100)
            steps = 0
            while not res.done and steps < 50:
                steps += 1
                obs = res.observation
                decision = "flag" if obs.files_flagged < obs.review_budget else "skip"
                res = env.step(CodeReviewAction(decision=decision, reasoning="test decision"))
            self.assertTrue(res.done)
            self.assertGreaterEqual(res.observation.f1_score, 0.0)
            self.assertLessEqual(res.observation.f1_score, 1.0)


if __name__ == "__main__":
    unittest.main()
