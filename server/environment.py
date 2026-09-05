"""Compatibility import. The canonical environment lives in code_review_env."""
from code_review_env.server.environment import (
    BUGGY_EPISODES, CODE_SNIPPETS, DATA_DIR, EPISODES,
    CodeReviewEnvironment, InvestigationObservation, InvestigationSession,
    InvestigationToolObservation, _load_data, _risk_summary, review_budget,
)
