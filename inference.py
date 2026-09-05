"""
CodeReviewEnv - LLM Inference Script
=====================================
Runs an LLM against the CodeReviewEnv using the OpenAI-compatible API.
Uses enriched observations for context-aware vulnerability triage.
"""
import os
import sys
from typing import List, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from openai import OpenAI
from code_review_env import CodeReviewEnv, CodeReviewAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy_key")

TASK_NAME = "code_review"
BENCHMARK = "code_review_env"

SYSTEM_PROMPT = """You are an expert security code reviewer. You are triaging files from a CVE vulnerability patch.

For each file, analyze the provided metrics and context:
- CVE description and CVSS severity
- File path, language, and component
- Code metrics: churn, complexity, TODO count, recency
- Risk summary

Respond with EXACTLY one word: 'flag' or 'skip'.
- flag: This file likely contains or is related to the vulnerability
- skip: This file is likely safe and not related to the vulnerability"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def clamp_score(raw: float) -> float:
    if raw <= 0.0:
        return 0.01
    if raw >= 1.0:
        return 0.99
    return raw


def parse_decision(text: str) -> str:
    text = (text or "").strip().lower()
    return "flag" if text == "flag" else "skip"


def build_prompt(obs) -> str:
    """Build a rich prompt from the enriched observation."""
    return (
        f"CVE: {obs.cve_id} (CVSS {obs.cvss_score})\n"
        f"Description: {obs.cve_description}\n"
        f"Repository: {obs.repo_name}\n\n"
        f"File: {obs.file_path}\n"
        f"Language: {obs.file_language} | Component: {obs.file_component}\n"
        f"Test file: {'Yes' if obs.is_test_file else 'No'}\n"
        f"Metrics: churn={obs.churn_score}, complexity={obs.complexity_score}, "
        f"todos={obs.todo_score}, recency={obs.recency_score}\n"
        f"Risk: {obs.risk_summary}\n\n"
        f"Budget remaining: {obs.review_budget - obs.files_flagged}/{obs.review_budget}\n"
        f"Files remaining: {obs.files_remaining}\n\n"
        f"Should we 'flag' or 'skip' this file? Answer with exactly one word."
    )


def run_task(env_url: str, difficulty: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.01

    log_start(task=f"{TASK_NAME}_{difficulty}", env=BENCHMARK, model=MODEL_NAME)

    try:
        with CodeReviewEnv(base_url=env_url).sync() as env:
            step_result = env.reset(difficulty=difficulty)
            obs = step_result.observation

            while not step_result.done:
                steps_taken += 1
                prompt = build_prompt(obs)

                decision = "skip"
                reasoning = ""
                error_msg = None

                try:
                    res = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=10,
                        temperature=0.1,
                    )
                    content = ""
                    if res.choices and res.choices[0].message and res.choices[0].message.content:
                        content = res.choices[0].message.content
                    decision = parse_decision(content)
                    reasoning = content
                except Exception as e:
                    error_msg = str(e).replace("\n", " ")
                    print(error_msg, file=sys.stderr, flush=True)

                action = CodeReviewAction(decision=decision, reasoning=reasoning)
                step_result = env.step(action)
                obs = step_result.observation

                reward = step_result.reward or 0.0
                rewards.append(reward)

                log_step(
                    step=steps_taken,
                    action=decision,
                    reward=reward,
                    done=step_result.done,
                    error=error_msg,
                )

            raw_score = getattr(obs, "f1_score", 0.0) or 0.0
            score = clamp_score(raw_score)
            success = raw_score > 0.0

    except Exception as e:
        print(str(e).replace("\n", " "), file=sys.stderr, flush=True)
        score = 0.01
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    env_url = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:7860")

    for difficulty in ["easy", "medium", "hard"]:
        run_task(env_url, difficulty)


if __name__ == "__main__":
    main()
