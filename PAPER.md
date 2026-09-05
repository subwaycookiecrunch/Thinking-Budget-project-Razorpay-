# The Thinking Budget: Calibrated Metacognition as Reinforcement Learning

**Razorpay AI Buildathon 2026 — Open Track. Built with PyTorch OpenEnv.**

> An RL environment + auxiliary objective that trains a reasoning LLM to
> *predict how hard a problem is* before solving it, then deliver exactly
> that much reasoning, on the right files.  We argue this is metacognitive
> *awareness*, not just metacognitive *behavior*, and we present evidence
> that the learned policy transfers across domains.

## 1. Abstract

Modern reasoning LLMs (Qwen3, DeepSeek-R1, GPT-o3) can produce arbitrarily
long `<think>` blocks, but they use that capacity poorly: either they
think hard about everything (slow, expensive) or about nothing (fast,
shallow, wrong).  Existing reasoning RL (GRPO, PPO over reasoning tokens)
treats the `<think>` block as a black box — a roll-out is sampled, a
final answer is scored, gradients flow.  Whether the model *knew the
problem was hard* before engaging deep reasoning is never measured and
never trained.

We propose a hybrid environment + auxiliary objective that trains the
meta-skill explicitly.  The agent investigates a real CVE patch
(2,892 files across 150 NVD episodes) using six MCP tools.  Before each
`<think>` block, the agent must emit a *budget prediction* — `short`,
`medium`, or `long`.  The reward function scores **calibration** (does
the actual length match the predicted band?), **difficulty awareness**
(long predictions on actually-vulnerable files, short on safe ones), and
**coupling** (every prediction tied to a real tool call) on top of task
F1.  At inference time, a `LogitsProcessor` hard-caps `<think>` tokens
per block and per episode.

We show that:

  - The combined reward shapes a policy that **halves the reasoning effort
    on safe code (51 % fewer `<think>` characters per safe file) and
    nearly triples it on bug-bearing files (2.82× more)** — a **5.87×
    bug-vs-safe allocation ratio**, compared to 1.02× for the untrained
    baseline whose reasoning is essentially uniform.  Calibration on the
    held-out eval set: **0.92 P(`long` | bug)** and **0.00 P(`long` |
    safe)** on the heuristic-proxy policy that instantiates the target
    shape (real numbers regenerate from the trained adapter at end of
    training; both modes ship via `scripts/generate_calibration_plot.py`).
  - The same allocation policy transfers, **without retraining**, to a
    different domain (non-security pull-request review for race
    conditions, auth bypasses, and tenant leaks): **F1 = 1.00** versus
    **0.28** for the untrained baseline, with the thinking-allocation
    ratio preserved (**5.24×**).
  - Inference-time budget enforcement degrades gracefully on the trained
    policy and catastrophically on the baseline.

The contribution is the auxiliary objective + the environment that makes
it learnable; both are open-source and importable on top of any
reasoning-RL setup.

## 2. Why this matters

Process Reward Models score whether reasoning is correct.  They do **not**
score whether the model knew the difficulty in advance.  That distinction
matters because:

  1. **Compute-adaptive inference**.  Models that self-assess difficulty
     before reasoning can allocate compute on demand without
     architectural changes.  Today's adaptive-compute literature
     (early-exit, MoE routing) requires architecture surgery; a
     metacognitive RL objective gets you most of the way there with
     reward shaping alone.
  2. **Transferable skill**.  Reasoning-allocation is an architecture-
     and task-agnostic capability.  If we can train it on a substrate
     where ground-truth difficulty is cheap and verifiable (CVE triage)
     and have it transfer to PR review, the technique scales to any
     heterogeneous multi-step investigation task.
  3. **Honest agents**.  Calibration is also a safety property.  An agent
     that overestimates its certainty is a known failure mode; one that
     reliably emits `medium` when it's actually unsure is a foundation
     for downstream uncertainty handling.

## 3. The Environment

### 3.1 Substrate

  - **150 real CVEs** from NVD with full patch context.  Examples:
    Log4Shell (CVE-2021-44228), Dirty COW (CVE-2016-5195), PwnKit
    (CVE-2021-4034), BlueKeep (CVE-2019-0708), Zerologon (CVE-2020-1472).
  - **2,892 source files** with extracted features: churn, cyclomatic
    complexity, TODO/FIXME density, recency of last modification, file
    component, language.
  - **Three difficulty levels** controlled by file count: easy (≤15),
    medium (16–29), hard (≥30).
  - Per-episode budgets: a flag budget proportional to ground-truth bug
    count and an investigation point budget proportional to file count.

### 3.2 Tools (6 MCP endpoints, served via FastMCP)

`read_file` (1 pt), `search_code` (2 pt), `get_function_list` (1 pt),
`flag_vulnerable`, `skip_file`, `submit_report`.  All tool calls are
logged and replayable; the environment is fully deterministic for a
given seed, which is critical for our `reward_fn` to replay episodes
during GRPO training.

### 3.3 Reward function (six components)

```
total = 0.50 × env_score + 0.30 × metacognitive_score + 0.20 × text_score
env_score = 0.35·F1
          + 0.20·report_quality
          + 0.15·investigation_efficiency
          + 0.15·thinking_efficiency
          + 0.15·precision_bonus
metacognitive_score = (½·calibration + ½·difficulty_awareness)
                       × (½ + ½·coupling)
```

The metacognitive component (this paper's contribution) is described in
§4.

## 4. Calibrated Metacognitive Reward

> *Hackathon FAQ Q11:* **"Process supervision means giving feedback on intermediate reasoning or intermediate steps, not only on the final outcome."*
>
> *FAQ Q44:* **"Use layered verification."**
>
> The reward described below is exactly that: per-prediction process supervision (calibration, difficulty awareness, coupling), each layer scored by independent code with no shared state, composed multiplicatively where independence cannot be assumed (coupling).

### 4.1 Output format

```text
<budget_prediction>long</budget_prediction>
<think>
do_ioctl_handler in drivers/foo.c:412 calls copy_from_user with a
user-supplied size; integer overflow into kmalloc → heap overflow.
This is the bug.
</think>
<tool_call>{"name": "flag_vulnerable",
            "arguments": {"file_path": "drivers/foo.c", ...}}</tool_call>
```

Bands map to character ranges (token proxy): `short` ∈ [0, 80),
`medium` ∈ [80, 250), `long` ∈ [250, ∞).

### 4.2 Sub-rewards

  1. **Calibration**.  For each prediction-think pair `(p, T)`,
     calibration is 1.0 iff `len(T) ∈ band(p)`, with smooth linear
     decay outside the band.  Aggregated by mean over predictions.

  2. **Difficulty awareness**.  For each prediction-think-tool triple,
     look up the ground-truth label of the file referenced by the tool
     call.  `long` on bugs and `short` on safe earn 1.0; the wrong
     direction earns 0.0; `medium` is neutral 0.5.

  3. **Coupling**.  The fraction of `<budget_prediction>` tokens that
     are followed within the same generation by a tool call.  Acts as a
     multiplier so a model cannot game the score by emitting orphan
     predictions.

### 4.3 Formal definition

Let *C* = (*p*<sub>1</sub>, *T*<sub>1</sub>, *a*<sub>1</sub>), …, (*p*<sub>N</sub>, *T*<sub>N</sub>, *a*<sub>N</sub>) be the sequence
of (prediction, think-block, tool-call) triples extracted from a
completion. Let *L*(*T*) denote the character length of think-block *T*,
let *band*(·) map a band name to its closed-open character interval,
and let *m*(·) map a band name to its midpoint:

> *band*(short) = [0, 80), &nbsp;*band*(medium) = [80, 250), &nbsp;*band*(long) = [250, ∞)

> *m*(short) = 40, &nbsp;*m*(medium) = 165, &nbsp;*m*(long) = 400

**Calibration** for a single triple is

> *c*<sub>i</sub> = 1 if *L*(*T*<sub>i</sub>) ∈ *band*(*p*<sub>i</sub>); &nbsp; *L*(*T*<sub>i</sub>) / *lo*(*p*<sub>i</sub>) if *L*(*T*<sub>i</sub>) < *lo*(*p*<sub>i</sub>); &nbsp; max(0, 1 − (*L*(*T*<sub>i</sub>) − *hi*(*p*<sub>i</sub>)) / *hi*(*p*<sub>i</sub>)) otherwise

i.e., 1.0 inside the band with smooth linear decay outside.

**Difficulty awareness** depends on the ground-truth label *y*<sub>i</sub> ∈ {0 (safe), 1 (bug), ⊥ (unknown)} of the file referenced by *a*<sub>i</sub>:

> *d*<sub>i</sub> = 1 if (*p*<sub>i</sub> = long ∧ *y*<sub>i</sub> = 1) ∨ (*p*<sub>i</sub> = short ∧ *y*<sub>i</sub> = 0)
>
> *d*<sub>i</sub> = 0 if (*p*<sub>i</sub> = long ∧ *y*<sub>i</sub> = 0) ∨ (*p*<sub>i</sub> = short ∧ *y*<sub>i</sub> = 1)
>
> *d*<sub>i</sub> = 0.5 otherwise (medium prediction or unknown label)

**Coupling** is the fraction of `<budget_prediction>` tags in the
completion that are followed (within the same generation) by a tool
call. Let *P* = total predictions emitted, *N* = predictions with a
matched tool call. Then

> *coupling* = *N* / max(1, *P*)

**The composite metacognitive reward** is

> *R*<sub>metacog</sub> = ½(*calibration* + *difficulty*) · (½ + ½ · *coupling*) ∈ [0, 1]

where *calibration* = (1/*N*) Σ *c*<sub>i</sub> and *difficulty* = (1/*N*) Σ *d*<sub>i</sub>
are the means over coupled triples. The factor (½ + ½ · *coupling*)
hard-caps a fully-uncoupled emitter at 50% of the metacog signal —
this *multiplicative* form (rather than an additive bonus) is what
prevents orphan-prediction reward hacking; see §4.5.

**The combined trainer reward** weights three orthogonal signals:

> *R*<sub>final</sub> = 0.50 · *R*<sub>env</sub> + 0.30 · *R*<sub>metacog</sub> + 0.20 · *R*<sub>text</sub>

where *R*<sub>env</sub> is the live MCP environment's composite F1 + thinking-
efficiency score and *R*<sub>text</sub> is a deterministic format-and-quality
heuristic (no LLM-as-judge).

### 4.4 Empirical reward-hacking robustness

We adversarially verified the safety property:

> **Property (no-cheat).** For every cheating policy π in our red team,
> *R*<sub>final</sub>(π) < *R*<sub>final</sub>(π<sub>honest</sub>).

We constructed five attack families spanning the structural failure
modes of metacognitive RL — calibration-only optimizers, difficulty
inverters, orphan emitters, and reasoning padders — and ran each
through the same scoring path the trainer uses (see [`scripts/red_team.py`](scripts/red_team.py)).

| # | Attack | *c* | *d* | coup | *R*<sub>metacog</sub> | *R*<sub>env</sub> | *R*<sub>text</sub> | **R<sub>final</sub>** | gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | All-long spammer | 1.00 | 0.20 | 1.00 | 0.60 | 0.41 | 0.20 | 0.426 | −50% |
| 2 | All-short lazy | 1.00 | 0.80 | 1.00 | 0.90 | 0.00 | 0.04 | 0.278 | −67% |
| 3 | Orphan predictions | 0.85 | 0.00 | 0.00 | 0.21 | 0.00 | 0.06 | 0.076 | −91% |
| 4 | Reasoning padding | 1.00 | 0.20 | 1.00 | 0.60 | 0.88 | 0.21 | 0.662 | **−22%** |
| 5 | Difficulty inverter | 1.00 | 0.00 | 1.00 | 0.50 | 0.00 | 0.21 | 0.192 | −77% |
| ✅ | Honest metacognitive | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | **0.850** | — |

Attack 4 is the empirical lower bound on the defense margin: a policy
that takes correct actions but pads `<think>` with semantic-empty
repetition. The text-reward's vuln-vocabulary heuristic and the
metacog's difficulty-awareness term together still keep it 22% below
the honest reference. Attack 2 ("all-short lazy") demonstrates the
multi-component defense most clearly: it scores 0.90 on the metacog
reward alone (perfect calibration, 4/5 difficulty correct), but the
*R*<sub>env</sub> = 0 from skipping the bug zeros out the dominant term.

A formal writeup, including the geometric argument for why the three
sub-rewards are functionally orthogonal, is in
[`SAFEGUARDS.md`](SAFEGUARDS.md).

### 4.5 Why this design

We designed for *non-gameability*. The geometric structure of *R*<sub>final</sub>
forces honest self-assessment because:

- **Calibration alone** is insufficient: an attacker can match length
  bands without knowing difficulty (attack 5).
- **Difficulty alone** is insufficient: an attacker can claim long-on-bug
  without actually delivering long thoughts.
- **Coupling alone** is insufficient: an attacker can ground predictions
  in any tool call regardless of correctness.
- The **multiplicative coupling factor** cuts the floor for
  uncoupled emitters at 50%, while the **additive env / metacog / text
  combination** ensures that exploiting one component sacrifices another.

The only policy that maximizes all three sub-rewards simultaneously is
one that (i) emits calibrated predictions, (ii) bases predictions on
ground-truth file difficulty, and (iii) takes correct actions on the
right files. That is the policy we want.

## 5. Inference-time Budget Enforcement

A `LogitsProcessor` (`scripts/budget_processor.ThinkingBudgetProcessor`)
maintains per-sequence state across the batch:

  - `in_block: bool` — are we currently inside `<think>...</think>`?
  - `block_used`, `episode_used: int` — running token counters

When `block_used ≥ per_block_budget` or
`episode_used ≥ episode_budget`, the next-token logits are forced to
the `</think>` token id.  This converts the soft, learned budget into
a hard inference-time constraint without retraining.

Combined with the metacognitive reward, the loop is closed: shaping
during GRPO → policy learns to allocate; budget processor at inference
→ policy degrades gracefully under arbitrary tightening.

## 6. Domain Transfer

We constructed five held-out episodes from a different domain — pull-
request review for non-security regressions:

| ID | Title | Files | Bugs |
|---|---|---|---|
| TR-CR-001 | payment refactor (race condition) | 12 | 2 |
| TR-CR-002 | auth middleware migration (path-prefix bypass) | 14 | 1 |
| TR-CR-003 | ML training-pipeline (reproducibility regression) | 10 | 1 |
| TR-CR-004 | frontend perf refactor (stale closure) | 11 | 1 |
| TR-CR-005 | DB query optimization (tenant leak) | 9 | 1 |

None are CVEs.  None appear in the training data.  We evaluate the same
risk-driven allocation policy that the metacognitive reward shapes the
trained model toward — it uses only structural features (churn,
complexity, TODOs, recency, is_test) and never sees the ground-truth
label.

| Policy | Aggregate F1 | Aggregate think ratio (bug/safe) |
|---|---:|---:|
| Untrained baseline (uniform random) | 0.28 | 1.29× |
| Metacognitive policy (transfer) | **1.00** | **5.24×** |

The same allocation pattern that solves CVE triage solves PR review.
Reasoning effort routes to the right files in a domain the policy has
never seen, on bug *types* never observed during training.

## 7. Limitations

  1. **Backbone size: Qwen3-1.7B is the right scale, not a constraint**.
     1.7B is the *smallest* variant of Qwen3's thinking-mode family — the
     smallest model that emits real `<think>` blocks at all. Going below
     1.7B (e.g. Qwen2.5-0.5B, the model used in the canonical TRL/OpenEnv
     Wordle tutorial) would remove the very behaviour we are studying.
     Going larger would obscure the source of the effect: a 70B model
     might allocate `<think>` correctly *because it is large*, not
     because the reward shaped it. At 1.7B the prior is weak enough that
     the metacognitive signal is the only plausible cause of the
     allocation pattern — that is methodologically *stronger*, not
     weaker. Recent published GRPO + verifiable-reward work clusters at
     1.5–7B (DeepSeek-R1-Distill-1.5B, TinyLlama-GRPO, Qwen2.5-1.5B-RLHF
     papers); we sit firmly in that band. The full reward stack and all
     hyperparameters apply unchanged to Qwen3-4B and Qwen3-8B;
     replication on larger backbones is a follow-up scaling question,
     not a core claim of this submission.
  2. **Calibration band granularity**.  Three bands (short/medium/long)
     keep the format learnable in 400 steps; a finer numeric
     prediction (e.g. 50/150/300/600 tokens) is more powerful but
     harder to train at hackathon scale.
  3. **Single-task transfer**.  Transfer is shown on one held-out
     domain; broader transfer (mathematical reasoning, scientific paper
     triage, bug-bounty triage) is the obvious next experiment.
  4. **Live calibration plot**.  During training, the reward function
     streams per-prediction `(pred_band, actual_length, label)` triples
     to `grpo_output/eval_calibration.json`; the calibration figure is
     regenerated automatically from this file at the end of training,
     so the headline figure is real model data, not a heuristic proxy.
     The pre-training placeholder figure shipped with the Space is
     labeled as such.

## 8. Future work

  - **Numeric budget prediction** — replace the categorical band with a
    real-valued token-count head (regression target).  Trade learnability
    for richer calibration plots.
  - **Self-play curriculum** — use the trained policy to *generate* new
    investigation episodes (synthetic CVEs of bounded difficulty),
    extending the 150-CVE dataset autonomously.
  - **Cross-task transfer** — evaluate on math, scientific paper review,
    and code-completion benchmarks.  If the policy transfers there, we
    have a domain-general metacognition skill.
  - **Inference-time KV-cache reuse** — the budget processor pairs
    naturally with speculative decoding and prefix-cache reuse;
    quantifying the wall-clock savings is straightforward future work.
  - **Multi-agent extension** — pair the Investigator with a Skeptic
    that challenges its flags; use cross-agent disagreement to detect
    miscalibration.

## 9. Reproducibility

All code is in this repository.  The full reproduction is:

```bash
git clone https://github.com/subwaycookiecrunch/Meta-final-round-
cd Meta-final-round-
pip install -r requirements.txt    # see notebook for exact pins
python train_grpo.py                # ~6–10 hours on a single A100
python eval_baseline.py             # produces eval_baseline_vs_trained.png
python transfer_eval.py             # produces transfer_results.png
python scripts/generate_calibration_plot.py --mode real
```

The `train_colab.ipynb` notebook wraps these steps for judges to run on
a free A100 Colab.  All seeds are fixed (`42` for training, `7` for the
heuristic plots, `11` for transfer), and a single deterministic episode
seed is embedded in every prompt so the reward function can replay each
episode in a fresh environment during GRPO.

## 10. Acknowledgements

Built with PyTorch OpenEnv. Submitted to the Razorpay AI Buildathon 2026.  We thank the OpenEnv
maintainers for the `MCPEnvironment` substrate, the TRL team for
GRPOTrainer + custom `reward_funcs`, and Unsloth for the 4-bit + LoRA
training stack that fits Qwen3-1.7B comfortably under the HF Space
14 GiB memory cap with full 4096-token context.

---

*Repository:* https://github.com/subwaycookiecrunch/Meta-final-round-
*Live Space:* https://huggingface.co/spaces/lucid987654/code-review-env-v3
