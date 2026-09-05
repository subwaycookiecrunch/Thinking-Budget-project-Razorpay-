"""
code_review_env package
=======================
MCP security code review environment & evaluation client.
"""
from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from openenv.core.client_types import StepResult
from code_review_env.server.environment import (
    CodeReviewEnvironment,
    EPISODES,
    BUGGY_EPISODES,
    CODE_SNIPPETS,
    _risk_summary,
)


@dataclass
class CodeReviewAction:
    """Action emitted by agent triaging vulnerability files."""
    decision: str = "skip"  # 'flag' or 'skip'
    reasoning: str = ""

    def __post_init__(self):
        if not isinstance(self.decision, str):
            self.decision = str(self.decision)
        self.decision = self.decision.strip().lower()


@dataclass
class CodeReviewObservation:
    """Enriched observation for a specific file triage step."""
    cve_id: str = ""
    cvss_score: float = 0.0
    cve_description: str = ""
    repo_name: str = ""
    file_path: str = ""
    file_language: str = ""
    file_component: str = ""
    is_test_file: bool = False
    churn_score: float = 0.0
    complexity_score: float = 0.0
    todo_score: float = 0.0
    recency_score: float = 0.0
    risk_summary: str = ""
    review_budget: int = 5
    files_flagged: int = 0
    files_remaining: int = 0
    f1_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeReviewEnv:
    """
    Client and evaluation interface for the CodeReview environment.
    Supports both local execution and persistent session client patterns.
    """

    def __init__(self, base_url: Optional[str] = None, **kwargs):
        self.base_url = base_url or os.getenv("ENV_SERVER_URL", "http://127.0.0.1:7860")
        self._current_episode: Optional[Dict[str, Any]] = None
        self._files: List[Dict[str, Any]] = []
        self._current_index: int = 0
        self._budget: int = 5
        self._flagged: List[str] = []
        self._skipped: List[str] = []
        self._bugs: set = set()
        self._done: bool = False

    def sync(self) -> "CodeReviewEnv":
        """Return synchronous context manager interface."""
        return self

    def __enter__(self) -> "CodeReviewEnv":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._done = True

    def reset(self, difficulty: str = "easy", seed: Optional[int] = None, **kwargs) -> StepResult[CodeReviewObservation]:
        """Reset the environment to a new episode matching difficulty."""
        rng = random.Random(seed) if seed is not None else random.Random()

        # Filter by difficulty curriculum
        candidates = BUGGY_EPISODES if BUGGY_EPISODES else EPISODES
        if difficulty == "easy":
            pool = [e for e in candidates if len(e["files"]) <= 15]
        elif difficulty == "medium":
            pool = [e for e in candidates if 16 <= len(e["files"]) <= 29]
        elif difficulty == "hard":
            pool = [e for e in candidates if len(e["files"]) >= 30]
        else:
            pool = candidates

        if not pool:
            pool = candidates

        ep = dict(rng.choice(pool))
        files = [dict(f) for f in ep["files"]]
        rng.shuffle(files)

        self._current_episode = ep
        self._files = files
        self._current_index = 0
        self._flagged = []
        self._skipped = []
        self._bugs = {f["file"] for f in files if f.get("label") == 1}
        self._budget = min(len(files), max(ep.get("total_bugs", 1) * 2 + 3, 5))
        self._done = False

        obs = self._build_observation()
        return StepResult(observation=obs, reward=0.0, done=False)

    def _build_observation(self, final_f1: float = 0.0) -> CodeReviewObservation:
        ep = self._current_episode or {}
        cvss = float(ep.get("cvss", 0.0) or 0.0)

        if self._current_index < len(self._files):
            f = self._files[self._current_index]
            feat = f.get("features", [0, 0, 0, 0])
            risk = _risk_summary(f, cvss)
            file_path = f.get("file", "")
            file_lang = f.get("file_language", "?")
            file_comp = f.get("file_component", "?")
            is_test = bool(f.get("is_test_file", False))
            churn = float(feat[0])
            complexity = float(feat[1])
            todos = float(feat[2])
            recency = float(feat[3])
        else:
            file_path = ""
            file_lang = ""
            file_comp = ""
            is_test = False
            churn = complexity = todos = recency = 0.0
            risk = "Triage complete"

        return CodeReviewObservation(
            cve_id=ep.get("cve_id", ""),
            cvss_score=cvss,
            cve_description=ep.get("cve_description", ""),
            repo_name=ep.get("repo", ""),
            file_path=file_path,
            file_language=file_lang,
            file_component=file_comp,
            is_test_file=is_test,
            churn_score=churn,
            complexity_score=complexity,
            todo_score=todos,
            recency_score=recency,
            risk_summary=risk,
            review_budget=self._budget,
            files_flagged=len(self._flagged),
            files_remaining=max(0, len(self._files) - self._current_index),
            f1_score=final_f1,
            metadata={"total_files": len(self._files)},
        )

    def step(self, action: CodeReviewAction) -> StepResult[CodeReviewObservation]:
        """Process agent's decision on the current file."""
        if self._done or self._current_index >= len(self._files):
            obs = self._build_observation()
            return StepResult(observation=obs, reward=0.0, done=True)

        current_file = self._files[self._current_index]["file"]
        decision = (action.decision or "skip").lower()

        step_reward = 0.0
        if decision == "flag" and len(self._flagged) < self._budget:
            self._flagged.append(current_file)
            step_reward = 1.0 if current_file in self._bugs else -0.5
        else:
            self._skipped.append(current_file)
            step_reward = 0.1 if current_file not in self._bugs else -1.0

        self._current_index += 1

        if self._current_index >= len(self._files):
            self._done = True
            # Compute final episode metrics
            flagged_set = set(self._flagged)
            tp = len(flagged_set & self._bugs)
            fp = len(flagged_set - self._bugs)
            fn = len(self._bugs - flagged_set)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            obs = self._build_observation(final_f1=f1)
            return StepResult(observation=obs, reward=f1, done=True)
        else:
            obs = self._build_observation()
            return StepResult(observation=obs, reward=step_reward, done=False)


__all__ = [
    "CodeReviewAction",
    "CodeReviewObservation",
    "CodeReviewEnv",
    "CodeReviewEnvironment",
]
