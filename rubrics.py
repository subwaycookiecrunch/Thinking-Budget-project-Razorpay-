"""Composable Rubric system for The Thinking Budget.

Uses OpenEnv's official Rubric API (modeled after nn.Module) to expose
the 6-component reward as introspectable, composable sub-rubrics.

This wraps the existing reward logic from environment.py and
metacognitive_reward.py into the framework's Rubric hierarchy,
enabling judges to inspect individual component scores.

Usage:
    rubric = ThinkingBudgetRubric()
    score = rubric(action, observation)
    # Inspect sub-scores:
    for name, child in rubric.named_children():
        print(f"{name}: {child.last_score}")
"""

from openenv.core.rubrics.base import Rubric
from code_review_env.scoring import report_quality
from metacognitive_reward import BAND_RANGES, _calibration_score
try:
    from openenv.core.rubrics.containers import WeightedSum
    WeightedRubric = WeightedSum
except ImportError:
    from openenv.core.rubrics.containers import WeightedRubric  # type: ignore
    WeightedSum = WeightedRubric


# ── Individual rubric components ──────────────────────────────────

class F1ScoreRubric(Rubric):
    """Precision × recall on vulnerability detection."""
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.0
        flagged = set(getattr(session, "flagged", getattr(session, "flagged_files", set())))
        bugs = set(session.bugs)
        if not bugs:
            return 1.0 if not flagged else 0.0
        tp = len(flagged & bugs)
        fp = len(flagged - bugs)
        fn = len(bugs - flagged)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class ReportQualityRubric(Rubric):
    """CVE ID + vuln type + code-level details mentioned in report."""
    def forward(self, action, observation) -> float:
        session = getattr(observation, "session", None)
        report = getattr(observation, "report", None)
        if report is None:
            report = getattr(session, "report", "")
        return report_quality(report, session)



class InvestigationEfficiencyRubric(Rubric):
    """Strategic use of investigation budget (fewer wasted steps)."""
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.0
        if hasattr(session, "files") and hasattr(session, "step_count"):
            return max(0.0, 1.0 - session.step_count / max(1, len(session.files) * 3))
        used = session.invest_used
        budget = session.invest_budget
        if budget == 0:
            return 1.0
        efficiency = 1.0 - (used / budget)
        return max(0.0, min(1.0, 0.5 + 0.5 * efficiency))


class ThinkingEfficiencyRubric(Rubric):
    """In-environment proxy for selective deep reasoning.
    
    Rewards the model for spending more thinking on buggy files
    and less on safe files.
    """
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.0
        if hasattr(session, "thinking_efficiency_score"):
            return session.thinking_efficiency_score()
        trace = getattr(session, 'thinking_trace', [])
        if not trace:
            return 0.5
        bugs = set(session.bugs)
        bug_lens = [t['length'] for t in trace if t.get('file') in bugs]
        safe_lens = [t['length'] for t in trace if t.get('file') not in bugs]
        bug_avg = sum(bug_lens) / len(bug_lens) if bug_lens else 0
        safe_avg = sum(safe_lens) / len(safe_lens) if safe_lens else 0
        if safe_avg == 0:
            return 1.0 if bug_avg > 0 else 0.5
        ratio = bug_avg / safe_avg
        return min(1.0, ratio / 6.0)


class PrecisionBonusRubric(Rubric):
    """Extra reward for zero false positives."""
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.0
        flagged = set(getattr(session, "flagged", getattr(session, "flagged_files", set())))
        bugs = set(session.bugs)
        return len(flagged & bugs) / len(flagged) if flagged else 0.0


class CalibrationRubric(Rubric):
    """Metacognitive calibration: does actual think length match predicted band?"""
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.5
        pairs = getattr(session, 'budget_pairs', [])
        if not pairs:
            return 0.5
        scores = [
            _calibration_score(pred, actual_len)
            if pred in BAND_RANGES and isinstance(actual_len, (int, float)) and actual_len >= 0
            else 0.0
            for pred, actual_len in pairs
        ]
        return sum(scores) / len(scores)



class DifficultyAwarenessRubric(Rubric):
    """Metacognitive difficulty awareness: long on bugs, short on safe."""
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.5
        triples = getattr(session, 'difficulty_triples', [])
        if not triples:
            return 0.5
        bugs = set(session.bugs)
        scores = []
        for pred, _len, filepath in triples:
            is_bug = filepath in bugs
            if pred == 'long' and is_bug:
                scores.append(1.0)
            elif pred == 'short' and not is_bug:
                scores.append(1.0)
            elif pred == 'medium':
                scores.append(0.5)
            else:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.5


class CouplingRubric(Rubric):
    """Action coupling: predictions followed by real tool calls.
    
    Returns a raw fraction in [0, 1]. The composite applies its gate once.
    Live execution is still required to verify successful tool actions.
    """
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.0
        preds = getattr(session, 'prediction_count', 0)
        coupled = getattr(session, 'coupled_count', 0)
        if preds <= 0:
            return 0.0
        return max(0.0, min(1.0, coupled / preds))


# ── Composite rubric ──────────────────────────────────────────────

class MetacognitiveCompositeRubric(Rubric):
    """Metacognitive score = (0.5·calibration + 0.5·difficulty) × (0.5 + 0.5·coupling).
    
    Orphan predictions receive a multiplicative penalty. This length-based
    proxy does not prove semantic reasoning quality or prevent every exploit.
    """
    def __init__(self):
        super().__init__()
        self.calibration = CalibrationRubric()
        self.difficulty_awareness = DifficultyAwarenessRubric()
        self.coupling = CouplingRubric()

    def forward(self, action, observation) -> float:
        session = getattr(observation, "session", None)
        if session is None or getattr(session, "prediction_count", 0) <= 0:
            return 0.0
        cal = self.calibration(action, observation)
        diff = self.difficulty_awareness(action, observation)
        coup = self.coupling(action, observation)
        base = 0.5 * cal + 0.5 * diff
        return base * (0.5 + 0.5 * coup)


class ThinkingBudgetRubric(Rubric):
    """Top-level composable rubric for The Thinking Budget.
    
    total = F1 · (0.50 · env_score + 0.30 · metacog_score + 0.20 · text_score)
    
    Each component is a proper OpenEnv Rubric subclass, enabling
    introspection via rubric.named_children() and rubric.last_score.
    """
    def __init__(self):
        super().__init__()
        # Environment components (50% of total)
        self.env = WeightedRubric(
            rubrics=[
                F1ScoreRubric(),
                ReportQualityRubric(),
                InvestigationEfficiencyRubric(),
                ThinkingEfficiencyRubric(),
                PrecisionBonusRubric(),
            ],
            weights=[0.35, 0.20, 0.15, 0.15, 0.15],
        )
        # Metacognitive components (30% of total)
        self.metacog = MetacognitiveCompositeRubric()

    def forward(self, action, observation) -> float:
        env_score = self.env(action, observation)
        metacog_score = self.metacog(action, observation)
        # text_score is computed separately from completion text
        # (not available in the Rubric action/observation interface)
        text_score = max(0.0, min(1.0, getattr(observation, "text_score", 0.0)))
        correctness = F1ScoreRubric()(action, observation)
        return correctness * (0.50 * env_score + 0.30 * metacog_score + 0.20 * text_score)
