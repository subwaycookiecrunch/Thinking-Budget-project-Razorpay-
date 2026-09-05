"""
tests/test_rubrics.py
=====================
Comprehensive unit tests for the 8 composable sub-rubrics and composite rubric in rubrics.py.
"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import rubrics


class TestRubricComponents(unittest.TestCase):
    """Test each sub-rubric individually on boundary and normal conditions."""

    def test_f1_score_rubric(self):
        rubric = rubrics.F1ScoreRubric()

        # No session
        obs = SimpleNamespace()
        self.assertEqual(rubric(None, obs), 0.0)

        # No bugs, none flagged -> perfect 1.0
        session = SimpleNamespace(flagged_files=[], bugs=[])
        self.assertEqual(rubric(None, SimpleNamespace(session=session)), 1.0)

        # No bugs, but flagged something -> 0.0
        session = SimpleNamespace(flagged_files=["a.c"], bugs=[])
        self.assertEqual(rubric(None, SimpleNamespace(session=session)), 0.0)

        # Perfect flag
        session = SimpleNamespace(flagged_files=["a.c"], bugs=["a.c"])
        self.assertAlmostEqual(rubric(None, SimpleNamespace(session=session)), 1.0)

        # Partial precision / recall
        session = SimpleNamespace(flagged_files=["a.c", "b.c"], bugs=["a.c", "c.c"])
        # TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5 -> F1=0.5
        self.assertAlmostEqual(rubric(None, SimpleNamespace(session=session)), 0.5)

    def test_report_quality_rubric(self):
        rubric = rubrics.ReportQualityRubric()

        # Empty
        self.assertEqual(rubric(None, SimpleNamespace(report="")), 0.0)
        self.assertEqual(rubric(None, SimpleNamespace()), 0.0)

        # CVE only
        self.assertAlmostEqual(rubric(None, SimpleNamespace(report="Found CVE-2021-44228")), 0.4)

        # CVE + vulnerability keyword
        self.assertAlmostEqual(rubric(None, SimpleNamespace(report="CVE-2021-44228 has buffer overflow")), 0.7)

        # Full details
        self.assertAlmostEqual(rubric(None, SimpleNamespace(report="CVE-2021-44228 has buffer overflow at function handle line 10")), 1.0)

    def test_investigation_efficiency_rubric(self):
        rubric = rubrics.InvestigationEfficiencyRubric()

        session = SimpleNamespace(invest_used=0, invest_budget=10)
        self.assertAlmostEqual(rubric(None, SimpleNamespace(session=session)), 1.0)

        session = SimpleNamespace(invest_used=10, invest_budget=10)
        self.assertAlmostEqual(rubric(None, SimpleNamespace(session=session)), 0.5)

        session = SimpleNamespace(invest_used=0, invest_budget=0)
        self.assertAlmostEqual(rubric(None, SimpleNamespace(session=session)), 1.0)

    def test_precision_bonus_rubric(self):
        rubric = rubrics.PrecisionBonusRubric()

        # Zero FP with flags -> 1.0
        session = SimpleNamespace(flagged_files=["bug.c"], bugs=["bug.c"])
        self.assertEqual(rubric(None, SimpleNamespace(session=session)), 1.0)

        # FP present -> 0.0
        session = SimpleNamespace(flagged_files=["safe.c"], bugs=["bug.c"])
        self.assertEqual(rubric(None, SimpleNamespace(session=session)), 0.0)

        # Zero flags -> 0.0
        session = SimpleNamespace(flagged_files=[], bugs=["bug.c"])
        self.assertEqual(rubric(None, SimpleNamespace(session=session)), 0.0)

    def test_calibration_rubric(self):
        rubric = rubrics.CalibrationRubric()

        # Exact match
        session = SimpleNamespace(budget_pairs=[("short", 40), ("medium", 150), ("long", 400)])
        self.assertAlmostEqual(rubric(None, SimpleNamespace(session=session)), 1.0)

        # Out of bounds with decay
        session = SimpleNamespace(budget_pairs=[("short", 200)])
        score = rubric(None, SimpleNamespace(session=session))
        self.assertLess(score, 1.0)
        self.assertGreaterEqual(score, 0.0)

    def test_difficulty_awareness_rubric(self):
        rubric = rubrics.DifficultyAwarenessRubric()

        # Correct direction
        session = SimpleNamespace(
            bugs=["bug.c"],
            difficulty_triples=[("long", 400, "bug.c"), ("short", 40, "safe.c")]
        )
        self.assertAlmostEqual(rubric(None, SimpleNamespace(session=session)), 1.0)

        # Inverted direction
        session = SimpleNamespace(
            bugs=["bug.c"],
            difficulty_triples=[("short", 40, "bug.c"), ("long", 400, "safe.c")]
        )
        self.assertAlmostEqual(rubric(None, SimpleNamespace(session=session)), 0.0)

    def test_coupling_rubric(self):
        rubric = rubrics.CouplingRubric()

        # 100% coupling
        session = SimpleNamespace(prediction_count=4, coupled_count=4)
        self.assertAlmostEqual(rubric(None, SimpleNamespace(session=session)), 1.0)

        # 0% coupling
        session = SimpleNamespace(prediction_count=4, coupled_count=0)
        self.assertAlmostEqual(rubric(None, SimpleNamespace(session=session)), 0.0)

    def test_thinking_budget_composite_rubric(self):
        rubric = rubrics.ThinkingBudgetRubric()

        # Check sub-rubric names
        children = dict(rubric.named_children())
        self.assertIn("env", children)
        self.assertIn("metacog", children)

        # Run composite forward
        session = SimpleNamespace(
            flagged_files=["bug.c"],
            bugs=["bug.c"],
            invest_used=2,
            invest_budget=10,
            thinking_trace=[{"file": "bug.c", "length": 400}],
            budget_pairs=[("long", 400)],
            difficulty_triples=[("long", 400, "bug.c")],
            prediction_count=1,
            coupled_count=1,
        )
        obs = SimpleNamespace(
            session=session,
            report="CVE-2021-44228 buffer overflow at line 20 function test",
        )
        total_score = rubric(None, obs)
        self.assertGreater(total_score, 0.70)
        self.assertLessEqual(total_score, 1.0)


if __name__ == "__main__":
    unittest.main()
