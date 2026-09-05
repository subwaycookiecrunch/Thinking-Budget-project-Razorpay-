# Safeguards and failure boundaries

The product is a source-review assistant. It does not execute submitted code, modify repositories, approve releases, or perform payment actions. This document describes implemented boundaries, not a claim of production security certification.

## Product failure behavior

| Failure | Implemented response | Evidence |
|---|---|---|
| Invalid/oversized source bundle or unsafe path | Reject before review | `parse_patch`, input tests |
| Insufficient token, file, or source allowance | Keep file `deferred` with a reason | `review_patch`, coverage table |
| Request timeout or provider error | Mark file `needs_review`; retain unknown-usage reservation | Provider error handling and usage ledger |
| Malformed JSON or contradictory decision | Reject response as review output | `validate_review` |
| Quoted line absent from source | Reject finding; require human review | Exact line-and-quote validation |
| Model abstains | Explicit `needs_review` | Per-file result state |
| Two consecutive failed reviews | Open circuit; defer remaining work | Circuit event in audit |
| Provider reports more output than requested | Expose limit violation and stop further requests | Usage validation |
| No rule matches in offline mode | Label result as rule scan only | Offline mode and result provenance |

The UI's Failure lab exercises timeout, malformed output, and false-citation cases using explicitly injected failures. These simulations test the application's recovery path. They are not observations of a real provider outage.

Source can include misleading instructions, comments, and deceptive code. The model receives it as review material, but prompt instructions are not a proof of injection resistance. The executable boundary is that model output cannot run tools or code; it must pass a limited result schema and source quotation check. A valid quotation can still accompany a wrong interpretation.

## What the audit provides

Each export records run identity, mode, model, configuration, source hashes, routing, raw response, validation, usage, and final state. A hash chain lets a verifier detect inconsistent edits in the supplied event sequence. Anyone able to replace the entire file can recompute the chain. No external signature, trusted timestamp, or immutable storage is claimed.

Raw review responses and source quotations may contain user code. Treat downloaded audit files as source-bearing artifacts. The local UI does not automatically publish them.

## Research reward checks

`python scripts/red_team.py` exercises five constructed attack families and an intended honest fixture: flag-all, skip-all, orphan predictions, padding, and inverted allocation. The original harness approximated environment scoring locally, so its scores must not be described as execution through the full trainer. Inspect the current harness and artifact metadata before drawing a stronger conclusion.

No finite list of lower-scoring attacks proves a reward ungameable. The word-length objective can reward style rather than insight; positive labels are an imperfect proxy for difficulty; a correct action may have an unfaithful explanation. Reward scoring must keep these limitations visible.

The current research environment withholds intermediate labels, derives flag budget from patch size, validates costs before spending, disables unrestricted code mode, isolates SDK sessions through a factory, and gates auxiliary score with F1. These changes address concrete failure modes found in the audit. They do not establish production readiness or trained-policy robustness.

## Recovery narrative supported by the repository

1. Historical heuristic traces were labeled as model improvement. Current docs separate heuristic evaluation, pretrained runtime inference, and newly supplied training artifacts.
2. Historical truncation and tag analyses did not perform the claimed intervention. Those causal claims were withdrawn.
3. Intermediate tool replies revealed labels. Current replies record the decision and withhold ground truth until submission.
4. Investigation points could be overspent before rejection. Costs are now checked before mutation.
5. The runtime can return malformed model output. It is retained as failed evidence and leaves a human-review task; an empty findings list does not imply approval.

The newly supplied adapter and training logs are retained as real artifacts. Their existence does not validate the earlier generated comparison charts or demonstrate a trained-policy improvement. Training diagnostics and independent inference evaluation belong in the evidence report, including negative results.

## Verification

```bash
python -m pytest tests/ -q
python scripts/red_team.py
python metacognitive_reward.py
```

The test suite is the executable specification for tested boundaries. The pitch should show at least one failure and the resulting unresolved work, rather than only claiming recovery exists.
