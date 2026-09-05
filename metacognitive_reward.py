"""
metacognitive_reward.py
========================
Calibrated metacognition as RL.

THE INNOVATION
--------------
Standard reasoning RL (GRPO, PPO over <think>...</think> tokens) treats the
model's reasoning as a black box: a roll-out is sampled, a final answer is
scored, gradients flow.  Whether the model "knew" the problem was hard
*before* engaging deep reasoning is never measured and never trained.

This module trains the meta-skill explicitly.  Before each reasoning block,
the agent must emit an explicit *budget prediction*:

    <budget_prediction>long</budget_prediction>
    <think> ... reasoning ... </think>
    <tool_call>{"name": "flag_vulnerable", ...}</tool_call>

The reward function rewards:
  1. **Calibration** — does the actual <think> length fall in the predicted band?
  2. **Difficulty awareness** — long predictions land on actually-vulnerable
     files; short predictions land on safe files.
  3. **Coupling** — predictions and actions are made on the same files
     (no orphan predictions).

This turns the reward signal from "did you reason well?" into "did you know
in advance how much reasoning the problem deserved, then deliver exactly
that much, on the right files?".  That is metacognitive *awareness*, not
just metacognitive *behavior*.

The contribution is the auxiliary objective; it is trainable on top of any
existing reasoning RL setup.  This module is self-contained and importable
from the main train_grpo.py.
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Predicted budget bands (tokens of <think> content, characters proxy) ──
# We use character-length bands as a coarse proxy for token count.  Three
# bands match the "short / medium / long" categorical the model emits.
BAND_RANGES = {
    "short":  (0,   80),
    "medium": (80,  250),
    "long":   (250, float("inf")),
}
BAND_TARGETS = {  # midpoints used to define monotonic difficulty signal
    "short":  40,
    "medium": 165,
    "long":   400,
}
BAND_ORDER = {"short": 0, "medium": 1, "long": 2}


# ── Regex patterns ────────────────────────────────────────────────────────
RE_PRED_THINK = re.compile(
    r"<budget_prediction>\s*(short|medium|long)\s*</budget_prediction>"
    r"\s*<think>(.*?)</think>",
    re.IGNORECASE | re.DOTALL,
)

RE_PRED_THINK_THEN_FLAG = re.compile(
    r"<budget_prediction>\s*(short|medium|long)\s*</budget_prediction>"
    r"\s*<think>(.*?)</think>"
    r"\s*<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)

RE_LOOSE_PRED = re.compile(
    r"<budget_prediction>\s*(short|medium|long)\s*</budget_prediction>",
    re.IGNORECASE,
)

RE_TOOL_CALL = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL,
)


# ── Metacognitive reward components ───────────────────────────────────────
@dataclass
class MetacogResult:
    calibration: float          # 0..1 — does actual length match predicted band?
    difficulty_awareness: float  # 0..1 — long preds on bugs, short on safe?
    coupling: float              # 0..1 — fraction of preds followed by a tool call
    n_predictions: int           # how many predictions the model emitted
    raw_score: float             # weighted aggregate (0..1)
    # Per-prediction trace for downstream calibration plots.  Each entry:
    #   (predicted_band, actual_think_length, label_int_or_None)
    # label = 1 if the file is in bug_files, 0 if safe, None if unknown.
    details: List[Tuple[str, int, Optional[int]]] = field(default_factory=list)


def _calibration_score(predicted: str, actual_len: int) -> float:
    """
    1.0 if `actual_len` is inside the predicted band; smooth decay outside.
    """
    lo, hi = BAND_RANGES[predicted]
    if lo <= actual_len < hi:
        return 1.0
    if actual_len < lo:
        return max(0.0, actual_len / max(1, lo))
    overshoot = (actual_len - hi) / max(1, hi)
    return max(0.0, 1.0 - overshoot)


def _difficulty_score(predicted: str, file_is_bug: Optional[bool]) -> float:
    """
    Reward "long on bugs, short on safe".  Medium is neutral.
    """
    if file_is_bug is None:
        return 0.5  # no ground truth — neutral
    if predicted == "long":
        return 1.0 if file_is_bug else 0.0
    if predicted == "short":
        return 1.0 if not file_is_bug else 0.0
    return 0.5  # medium


def _parse_file_tool(json_str: str):
    """Validate tool structure. Never execute model-produced Python or JSON."""
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return None
        call = data.get("function", data)
        if not isinstance(call, dict):
            return None
        name, args = call.get("name"), call.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            return None
        if name not in {"read_file", "get_function_list", "flag_vulnerable", "skip_file"}:
            return None
        path = args.get("file_path")
        if not isinstance(path, str) or not path.strip() or len(path) > 4096:
            return None
        if name in {"flag_vulnerable", "skip_file"}:
            reason = args.get("reasoning")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 16000:
                return None
        return name, path
    except (ValueError, TypeError):
        return None


def _extract_filepath_from_tool_call(json_str: str) -> Optional[str]:
    call = _parse_file_tool(json_str)
    return call[1] if call else None


def _extract_tool_name(json_str: str) -> Optional[str]:
    call = _parse_file_tool(json_str)
    return call[0] if call else None


def compute_metacognitive_reward(
    text: str,
    bug_files: Optional[set] = None,
    valid_files: Optional[set] = None,
) -> MetacogResult:
    """Score an auxiliary length-allocation proxy, not reasoning correctness.

    ``None`` means ground truth unavailable; an empty set means all files safe.
    Supply ``valid_files`` from the actual episode to reject hallucinated paths.
    Without that set, path membership cannot be verified from completion text.
    Coupling validates JSON/tool structure; only live execution proves success.
    """
    if not isinstance(text, str):
        return MetacogResult(0.0, 0.0, 0.0, 0, 0.0, [])
    n_preds = len(re.findall(r"<budget_prediction>", text, re.IGNORECASE))
    pairs = list(RE_PRED_THINK.finditer(text))
    if not pairs or n_preds == 0:
        return MetacogResult(0.0, 0.0, 0.0, n_preds, 0.0, [])

    # Invalid bands and predictions without a think block remain in denominator.
    calibration = sum(_calibration_score(m[1].lower(), len(m[2].strip()))
                      for m in pairs) / n_preds
    coupled = 0
    diff_sum = 0.0
    details: List[Tuple[str, int, Optional[int]]] = []
    seen_decisions = set()
    for match in RE_PRED_THINK_THEN_FLAG.finditer(text):
        pred, think, tool_json = match.groups()
        parsed = _parse_file_tool(tool_json)
        if parsed is None:
            continue
        tool, path = parsed
        if valid_files is not None and path not in valid_files:
            continue
        if tool in {"flag_vulnerable", "skip_file"}:
            if path in seen_decisions:
                continue
            seen_decisions.add(path)
        is_bug = None if bug_files is None else path in bug_files
        coupled += 1
        diff = _difficulty_score(pred.lower(), is_bug)
        # Correct length on a knowingly wrong decision earns no difficulty credit.
        if is_bug is not None and ((tool == "flag_vulnerable" and not is_bug)
                                   or (tool == "skip_file" and is_bug)):
            diff = 0.0
        diff_sum += diff
        details.append((pred.lower(), len(think.strip()), None if is_bug is None else int(is_bug)))

    difficulty = diff_sum / n_preds
    coupling = min(1.0, coupled / n_preds)
    raw = (0.5 * calibration + 0.5 * difficulty) * (0.5 + 0.5 * coupling)
    return MetacogResult(calibration, difficulty, coupling, n_preds,
                        max(0.0, min(1.0, raw)), details)


# ── System-prompt patch ───────────────────────────────────────────────────
METACOG_SYSTEM_PROMPT_ADDENDUM = """

CRITICAL — Metacognitive Format (REQUIRED):

Before EVERY <think> block, you MUST emit a budget prediction first:

    <budget_prediction>short|medium|long</budget_prediction>
    <think>
    ...your reasoning here...
    </think>
    <tool_call>{"name": "...", "arguments": {...}}</tool_call>

Budget bands:
  - short  : 0–80 characters of reasoning. Use for obviously safe files
             (test files, headers with no logic, boilerplate).
  - medium : 80–250 characters. Use when you need to verify but don't
             see strong red flags.
  - long   : 250+ characters. Use when you suspect the file is vulnerable
             and need to lay out the bug pattern (function name, unsafe
             operation, missing check, exploit path).

You will be SCORED on:
  1. Calibration — does the actual length of your <think> match the band
     you predicted?
  2. Difficulty awareness — do you predict 'long' on actually-vulnerable
     files and 'short' on safe ones?
  3. Coupling — every prediction must be followed by a real tool call
     against a file (no orphan predictions).

The optimal policy predicts BEFORE thinking, thinks the predicted amount,
and predicts longer for bugs.  Be honest about the difficulty.
"""


# ── CLI smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick sanity check
    sample = """
<budget_prediction>long</budget_prediction>
<think>
Looking at copy_from_user without a length check on the user-supplied size
parameter. This is a textbook integer overflow → heap overflow primitive.
The CVE description matches exactly. Function `do_ioctl_handler` in line 412.
</think>
<tool_call>{"name": "flag_vulnerable", "arguments": {"file_path": "drivers/foo.c", "reasoning": "ioctl bug"}}</tool_call>

<budget_prediction>short</budget_prediction>
<think>
Header.
</think>
<tool_call>{"name": "skip_file", "arguments": {"file_path": "include/foo.h", "reasoning": "header"}}</tool_call>
"""
    r = compute_metacognitive_reward(sample, bug_files={"drivers/foo.c"})
    print(f"calibration={r.calibration:.2f}  difficulty_awareness={r.difficulty_awareness:.2f}  "
          f"coupling={r.coupling:.2f}  n={r.n_predictions}  raw={r.raw_score:.3f}")
    assert r.raw_score > 0.7, f"smoke test failed: {r}"
    print("✅ smoke test passed")
