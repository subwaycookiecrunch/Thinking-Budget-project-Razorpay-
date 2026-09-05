# The Thinking Budget: system design and evidence report

Razorpay Buildathon 2026 · Open Track · Local prototype

## Abstract

The Thinking Budget is a bounded code-review workflow. A deterministic policy allocates output allowances across source files; a local language model returns structured review candidates; exact source quotations are checked; failed and unreviewed files remain explicit. The research layer explores a calibration-and-action-coupling reward for training budget-aware agents. These are separate evidence tracks.

The checkout now contains a Qwen2.5-1.5B-Instruct LoRA adapter, checkpoints, and a 100-step training summary. Their presence establishes saved training artifacts, not successful learning. The product uses pretrained Qwen3:4b through Ollama and does not use that adapter. The allocation benchmark measures coverage of synthetic dataset labels under actual file-read budgets. It does not measure bug detection, learned reasoning, or production savings.

## 1. Problem and system boundary

A developer reviewing a batch of changes has finite review capacity. The useful output is a source-linked queue of concerns plus the work still unresolved. A short answer alone cannot communicate whether every important file was inspected.

The current system accepts source bundles, rather than fetching private repositories or executing code. Each model request reviews one file, so cross-file defects may be missed. It produces candidates for a human reviewer and never approves a release or changes payment state.

## 2. Architecture

```text
source bundle → input validation → deterministic routing → budget reservation
                                                        ↓
                                               local model / offline rules
                                                        ↓
                              structured response + exact source-line check
                                                        ↓
                                  findings + unresolved queue + audit export
```

`review_engine.py` separates three units: configured output-token allowances, provider-reported input/output tokens, and source characters admitted to review. They are not interchangeable. The router is hand-designed, not trained. The local product disables Qwen3 thinking mode; its allowance bounds generated review output, not a measured internal reasoning process.

A provider timeout may occur after tokens have been generated. The workflow conservatively consumes that request's reservation when usage is unknown. Invalid output and false citations produce `needs_review`. Two consecutive failed reviews open the circuit and defer remaining files. A hash-linked event sequence exposes accidental alteration of its contents; without external signing or anchoring it is not a tamper-proof record.

## 3. Reproducible prioritization benchmark

Protocol: `python benchmark.py`, or the default `python eval_baseline.py`. The artifact includes source hashes, seed, per-episode results, and the metric definition.

The dataset contains 150 synthetic CVE-themed episodes, 2,892 file rows, 319 positive-labeled rows, and 58 negative-only episodes. Paths repeat across episodes; seven paths have conflicting labels. Generated snippets are not guaranteed to implement the corresponding CVE. Consequently, the primary metric is **coverage recall: positive-labeled files actually read / all positive-labeled files**. Reading a file is not detecting its bug.

At a 50% per-episode allowance rounded upward, every budgeted policy reads 1,468 files:

| Policy, original features | Positive rows read | Coverage | 95% episode-bootstrap interval | Positive rows not read |
|---|---:|---:|---:|---:|
| Seeded random order | 159 / 319 | 49.84% | 44.31–55.32% | 160 |
| Risk ranked | 317 / 319 | 99.37% | 98.45–100% | 2 |
| Risk with exploration | 314 / 319 | 98.43% | 96.68–99.69% | 5 |
| Exhaustive, 2,892 reads | 319 / 319 | 100% | 100–100% | 0 |

Risk ranking reads 1,851,768 of 3,613,066 available source characters: 48.75% fewer than exhaustive reading. This is not a token, latency, price, or energy measurement. Shuffling features reduces risk-ranked coverage to 49.53%, showing dependence on the synthetic feature-label relationship. Strong original-feature performance should not be extrapolated to real repositories.

These intervals describe resampling sensitivity over the bundled episodes, with one seeded random ordering. All data was available during development and includes training data. This is not a held-out evaluation. The optional weighted triage-loss units are illustrative assumptions, not measured business costs.

## 4. Training artifacts: what is present

The supplied adapter configuration identifies `Qwen/Qwen2.5-1.5B-Instruct`, LoRA rank 16 and alpha 32. The adapter contains 392 BF16 tensors; the main file is 36,981,856 bytes. The training summary records 100 steps, mean reward 0.094827, early mean 0.101822 and late mean 0.104709, a difference of 0.002888.

A small reward difference is not a detection-accuracy result. The trace audit also finds severe limitations: 200 reward traces include only 13 non-null environment scores; action coupling is zero in every trace, and 130 traces have no budget prediction. These diagnostics do not support a successful budget-aware agent training claim. Consult the current training audit artifact for full counts and definitions.

The product's pretrained Qwen3:4b inference is separate. A captured local run, including raw responses and validation outcomes, demonstrates runtime execution. It is not an evaluation of the saved Qwen2.5 adapter.

## 5. Experimental reward objective

The proposed formatting objective associates a budget prediction with a reasoning block and following tool call. Character bands in `metacognitive_reward.py` are short `[0,80)`, medium `[80,250)`, and long `[250,9999)`. The implementation applies a smooth penalty outside a band.

```text
metacognitive = (0.5 × length_calibration + 0.5 × label_alignment)
                × (0.5 + 0.5 × action_coupling)
```

“Length calibration” means matching a categorical text-length band; it is not probability calibration. “Label alignment” rewards long text for positive-labeled files and short text for negative-labeled files; a vulnerability label is only a crude proxy for review difficulty. Safe code can require substantial effort to verify. Text length is not a direct measurement of cognition.

The environment's current final score gates auxiliary proxies with classification F1:

```text
env_score = F1 × (0.35 × F1 + 0.20 × report_structure
                 + 0.15 × step_efficiency + 0.15 × reasoning_length_proxy
                 + 0.15 × precision_bonus)
```

The trainer has a separate composite and fallback path. An environment-execution score and a text-only fallback score must not be pooled without reporting which path produced each reward. Saved checkpoints may precede the current hardened reward; compare the training configuration and source revision before reproducing.

## 6. Withdrawn historical conclusions

The older demonstration scripts generated heuristic traces labeled “trained” and “untrained.” The historical 6× allocation ratio, perfect F1, and apparent transfer gains do not establish model improvement. The new adapter does not retroactively validate those outputs.

The old truncation experiment changed displayed reasoning length while holding decisions fixed; constant F1 followed by construction. Ignoring a tag in a saved trace is not removing it during generation. Neither is a causal model ablation. Old calibration and training figures should be treated according to their generator and provenance, not their visual titles.

Five constructed reward attacks are useful unit scenarios, but a simplified reward fixture is not the full training environment. Lower scores on five examples are not proof of general robustness or immunity to reward hacking.

## 7. Related work

Adaptive budgets are established research directions. [TALE](https://arxiv.org/abs/2412.18547) estimates budgets for reasoning tasks. [s1](https://arxiv.org/abs/2501.19393) controls test-time compute through budget forcing. [SelfBudgeter](https://arxiv.org/abs/2505.11274) combines cost pre-estimation with budget-guided reinforcement learning. [BudgetThinker](https://arxiv.org/abs/2508.17196) combines remaining-budget signals with staged training.

This project's current claim is a review workflow with resource accounting, source evidence checks, and explicit handoff states. It does not claim to originate budget prediction, budget forcing, or adaptive reasoning training.

## 8. Experiments needed next

Freeze the policy before evaluation on independently labeled real pull requests. Compare the same model under equal total budgets with uniform allocation, risk allocation, fixed generation caps, and exhaustive review. Re-run actual inference for each condition. Report precision, recall, false positives, missed defects, abstentions, token usage, latency, and failure rate; include multiple seeds and paired uncertainty.

To claim learning, evaluate the base model and actual adapter on the same untouched inputs and verify tool execution. To claim business value, measure developer review time and accepted findings in a pilot. Those remain future validations.
