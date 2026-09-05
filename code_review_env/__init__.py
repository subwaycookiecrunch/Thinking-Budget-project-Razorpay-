"""
code_review_env package
=======================
MCP security code review environment & evaluation client.
"""
from __future__ import annotations

import os
import json
import random
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from openenv.core.client_types import StepResult
from code_review_env.server.environment import (
    CodeReviewEnvironment,
    EPISODES,
    BUGGY_EPISODES,
    CODE_SNIPPETS,
    _risk_summary,
    review_budget,
)
from code_review_env.scoring import classification_metrics


@dataclass
class CodeReviewAction:
    """Action emitted by agent triaging vulnerability files."""
    decision: str = "skip"  # 'flag' or 'skip'
    reasoning: str = ""

    def __post_init__(self):
        if not isinstance(self.decision, str):
            raise ValueError("decision must be 'flag' or 'skip'")
        self.decision = self.decision.strip().lower()
        if self.decision not in {"flag", "skip"}:
            raise ValueError("decision must be 'flag' or 'skip'")
        if not isinstance(self.reasoning, str):
            raise ValueError("reasoning must be text")


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
    source_code: str = ""
    review_budget: int = 5
    files_flagged: int = 0
    files_remaining: int = 0
    f1_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeReviewEnv:
    """
    Local sequential evaluation interface for the CodeReview environment.
    Remote MCP sessions use openenv.core.generic_client.GenericEnvClient.
    """

    def __init__(self, base_url: Optional[str] = None, **kwargs):
        if base_url is not None:
            raise ValueError("CodeReviewEnv is local-only. Use GenericEnvClient for a remote /ws session.")
        self.base_url = None
        self._current_episode: Optional[Dict[str, Any]] = None
        self._files: List[Dict[str, Any]] = []
        self._current_index: int = 0
        self._budget: int = 5
        self._flagged: List[str] = []
        self._skipped: List[str] = []
        self._bugs: set = set()
        self._done: bool = False
        self._final_f1: float = 0.0

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
        candidates = EPISODES
        if difficulty == "easy":
            pool = [e for e in candidates if len(e["files"]) <= 15]
        elif difficulty == "medium":
            pool = [e for e in candidates if 16 <= len(e["files"]) <= 29]
        elif difficulty == "hard":
            pool = [e for e in candidates if len(e["files"]) >= 30]
        else:
            raise ValueError("difficulty must be easy, medium, or hard")

        if not candidates:
            raise RuntimeError("No episodes available")
        if not pool:
            pool = candidates

        ep = copy.deepcopy(rng.choice(pool))
        files = [dict(f) for f in ep["files"]]
        rng.shuffle(files)

        self._current_episode = ep
        self._files = files
        self._current_index = 0
        self._flagged = []
        self._skipped = []
        self._bugs = {f["file"] for f in files if f.get("label") == 1}
        self._budget = review_budget(len(files))
        self._final_f1 = 0.0
        self._done = False

        obs = self._build_observation()
        return StepResult(observation=obs, reward=0.0, done=False)

    def _build_observation(self, final_f1: Optional[float] = None) -> CodeReviewObservation:
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
            source_code=CODE_SNIPPETS.get(file_path, ""),
            review_budget=self._budget,
            files_flagged=len(self._flagged),
            files_remaining=max(0, len(self._files) - self._current_index),
            f1_score=self._final_f1 if final_f1 is None else final_f1,
            metadata={"total_files": len(self._files)},
        )

    def step(self, action: CodeReviewAction) -> StepResult[CodeReviewObservation]:
        """Process agent's decision on the current file."""
        if self._current_episode is None:
            raise RuntimeError("Call reset() before step()")
        if not isinstance(action, CodeReviewAction):
            raise TypeError("action must be CodeReviewAction")
        if self._done or self._current_index >= len(self._files):
            obs = self._build_observation()
            return StepResult(observation=obs, reward=0.0, done=True)

        current_file = self._files[self._current_index]["file"]
        decision = (action.decision or "skip").lower()

        step_reward = 0.0
        if decision == "flag" and len(self._flagged) < self._budget:
            self._flagged.append(current_file)
            step_reward = 0.0
        else:
            self._skipped.append(current_file)
            step_reward = 0.0

        self._current_index += 1

        if self._current_index >= len(self._files):
            self._done = True
            # Compute final episode metrics
            metrics = classification_metrics(self._flagged, self._bugs, [f["file"] for f in self._files])
            f1 = metrics["f1"]
            self._final_f1 = f1

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
