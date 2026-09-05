"""Experimental inference-time cap for tokenizers with atomic think tags.

Counts generated tokens only, with a global allowance across reasoning blocks.
The character helper edits recorded text only; it cannot measure compute savings
or show how a model would behave under a different generation constraint.
"""
from __future__ import annotations
import re
from typing import Optional

try:
    import torch
    from transformers import LogitsProcessor
except ImportError:
    torch = None
    LogitsProcessor = object


def _limit(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _atomic_tag_id(tokenizer, tag):
    ids = tokenizer.encode(tag, add_special_tokens=False)
    if len(ids) != 1 or tokenizer.decode(ids).strip() != tag:
        raise ValueError(f"{tag} must be one atomic token; split-tag tokenizers are unsupported")
    return int(ids[0])


class ThinkingBudgetProcessor(LogitsProcessor):
    """Force </think> when a per-block or generation-wide budget is exhausted.

    Rebuild state from the generated suffix on each call: correct across tensor
    allocation, beam reordering and repeated calls. Call reset() between runs.
    Explicit prompt_length is useful when attaching to an already-started run.
    Atomic special tokens are required; normal whitespace is never a delimiter.
    """
    def __init__(self, tokenizer, per_block_budget=400, episode_budget=None,
                 verbose=False, prompt_length=None, start_in_think=None):
        self.per_block_budget = _limit(per_block_budget, "per_block_budget")
        self.episode_budget = None if episode_budget is None else _limit(episode_budget, "episode_budget")
        self.open_id = _atomic_tag_id(tokenizer, "<think>")
        self.close_id = _atomic_tag_id(tokenizer, "</think>")
        if self.open_id == self.close_id:
            raise ValueError("Opening and closing think tags must be distinct")
        self.verbose = verbose
        self.prompt_length = None if prompt_length is None else _limit(prompt_length, "prompt_length")
        self.start_in_think = start_in_think
        self._initial_length = self.prompt_length

    def _sequence_state(self, sequence):
        if self._initial_length is None:
            self._initial_length = len(sequence)
        prefix = sequence[:self._initial_length]
        if self.start_in_think is None:
            last_open = max((i for i,t in enumerate(prefix) if t == self.open_id), default=-1)
            last_close = max((i for i,t in enumerate(prefix) if t == self.close_id), default=-1)
            active = last_open > last_close
        else:
            active = bool(self.start_in_think)
        block = total = 0
        for token in sequence[self._initial_length:]:
            if token == self.close_id:
                active = False
                block = 0
            elif token == self.open_id and not active:
                active = True
                block = 0
            elif active:
                # A nested open is content, not a chance to reset the allowance.
                block += 1
                total += 1
        return active, block, total

    def __call__(self, input_ids, scores):
        if torch is None:
            raise RuntimeError("Install torch and transformers to use the logits processor")
        for index, sequence in enumerate(input_ids.tolist()):
            active, block, total = self._sequence_state(sequence)
            if active and (block >= self.per_block_budget or
                           (self.episode_budget is not None and total >= self.episode_budget)):
                scores[index] = torch.full_like(scores[index], float("-inf"))
                scores[index, self.close_id] = 0
                if self.verbose:
                    print(f"[budget] forced close: block={block}, total={total}")
        return scores

    def reset(self):
        self._initial_length = self.prompt_length


def enforce_character_budget(text: str, per_block_budget=400, episode_budget=None) -> str:
    """Post-hoc text editing, not inference. Annotation sits outside the cap."""
    per_block_budget = _limit(per_block_budget, "per_block_budget")
    if episode_budget is not None:
        episode_budget = _limit(episode_budget, "episode_budget")
    spent = 0

    def cap_block(match):
        nonlocal spent
        content = match.group(1)
        cap = per_block_budget if episode_budget is None else min(per_block_budget, max(0, episode_budget - spent))
        kept = content[:cap]
        spent += len(kept)
        annotation = "[truncated by budget]" if len(kept) < len(content) else ""
        return f"<think>{kept}</think>{annotation}"

    return re.sub(r"<think>(.*?)(?:</think>|$)", cap_block, text, flags=re.DOTALL)
