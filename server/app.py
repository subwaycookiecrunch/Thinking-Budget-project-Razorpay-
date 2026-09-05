"""
OpenEnv server application for CodeReviewEnvironment.
=====================================================
Exposes CodeReviewEnvironment as an OpenEnv HTTP/WebSocket service
for remote agents and evaluation harnesses.
"""
from __future__ import annotations

import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server import CallToolAction
from code_review_env.server.environment import CodeReviewEnvironment, InvestigationObservation

app: FastAPI = create_app(
    env=lambda: CodeReviewEnvironment(),
    action_cls=CallToolAction,
    observation_cls=InvestigationObservation,
    env_name="CodeReviewEnv",
    max_concurrent_envs=8,
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
