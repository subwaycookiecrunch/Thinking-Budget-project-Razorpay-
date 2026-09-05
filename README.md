---
title: "The Thinking Budget"
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: true
---

# The Thinking Budget

**Budget-aware code review for teams shipping AI-generated code.**

Give a reviewer a batch of source files and a fixed output budget. See where review effort goes, inspect findings against the source, and keep unfinished work in a human-review queue.

The product combines deterministic allocation with a model-based source reviewer. It exposes resource limits and failure states rather than implying that an incomplete review is complete. The bundled payment-idempotency example makes the workflow concrete; custom multi-file input lets a reviewer test it on different code.

**Razorpay Buildathon · Track 05 — Open Track.** Start with [the judge guide](JUDGES.md). The local submission kit contains [a five-minute pitch](docs/PITCH_SCRIPT.md), [application answer drafts](docs/SUBMISSION_FORM.md), and [the candid assessment](docs/BRUTAL_ASSESSMENT.md).

## Evidence first

| Layer | What it demonstrates | What it does not establish |
|---|---|---|
| Review lab | Source review, allocation, validated source quotations, unresolved work, and an audit export | Guaranteed defect detection or approval to ship |
| Real model mode | Inference from the selected local or explicitly configured model | A newly trained model or improved model weights |
| Offline mode | A deterministic rule-based review that works without model access | LLM inference |
| Synthetic benchmark | Executed file prioritization with actual bounded source reads | Bug detection, held-out production accuracy, billed token savings, or business ROI |
| Training research | Saved Qwen2.5 LoRA adapter, checkpoints, logs, environment, and reward | Verified adapter improvement, causal ablations, or generalization |

The checkout contains a real **Qwen2.5-1.5B-Instruct LoRA adapter and 100-step training artifacts**. The run does not establish improved model behavior: recorded action coupling is zero and most rewards lack an environment score. The product uses **pretrained Qwen3:4b**, separately, through Ollama. Historical “trained” heuristic traces and plots remain unsupported as model results. See [the evidence report](PAPER.md) and `python scripts/audit_training_evidence.py`.

## Try the product locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open the URL printed at startup. In Review lab:

1. Load the synthetic payment example or provide a source bundle.
2. Choose the review mode and budget.
3. Run the review; inspect the source evidence and files still needing human review.
4. Run a documented failure mode and inspect the audit trail.
5. Export the result for review.

The offline mode requires no API key. Local AI requires Ollama running with `qwen3:4b` available (`ollama pull qwen3:4b` if needed). Set `OLLAMA_MODEL` to select another installed model; the default endpoint is `http://127.0.0.1:11434`. Model output is untrusted: quoting a real source line confirms the quote exists, not that the model's conclusion is correct.

## Reproduce the evaluation

```bash
python -m pytest tests/ -q
python eval_baseline.py
```

The default evaluation regenerates:

- `grpo_output/benchmark_results.json`: protocol, policy summaries, uncertainty, and cost/error metrics.
- `grpo_output/benchmark_episodes.jsonl`: per-episode evidence.
- `grpo_output/benchmark_pareto.png`: the accuracy-versus-reading-cost comparison.

Read the generated protocol before quoting a number. The primary metric is **coverage of dataset-positive files actually read**, not defects detected. On 150 synthetic episodes and 2,892 file rows, risk ranking reads 317/319 positive rows under a 1,468-file allowance; seeded random order reads 159/319. Risk ranking reads 48.75% fewer source characters than exhaustive reading. Shuffling features drops its coverage to 49.53%, exposing dependence on synthetic signals. Source characters are not model tokens or measured API bills. All bundled data was available during policy development and includes training data.

The research reward smoke tests are separate:

```bash
python metacognitive_reward.py
python scripts/red_team.py
```

Five constructed reward attacks test a limited scoring fixture; they do not prove agent safety or a production training reward is ungameable.

## Why these implementation choices

AI is useful for interpreting source and explaining a suspected defect. Deterministic logic is better suited to counting resources, selecting within limits, parsing structured results, and checking whether cited evidence is present. The human reviewer decides whether a finding is valid and whether the change may ship.

The review result must distinguish “finding,” “reviewed without a validated finding,” and “still needs review.” Budget limits, provider failure, and invalid responses are meaningful outcomes. They belong in the same report as successful findings.

Adaptive reasoning budgets already have substantial prior work. This project's contribution is the bounded review workflow and its inspectable evidence. It makes no first-ever adaptive-thinking claim. See [related work and originality boundary](docs/BRUTAL_ASSESSMENT.md#related-work-and-originality-boundary).

## Repository map

| File | Purpose |
|---|---|
| `app.py` | Local interactive product and research views |
| `review_engine.py` | Source validation, routing, model review, resource accounting, and audit |
| `benchmark.py` | Reproducible deterministic file-coverage benchmark |
| `code_review_env/server/environment.py` | Six-tool OpenEnv investigation environment |
| `server/app.py` | Environment HTTP entry point |
| `metacognitive_reward.py` | Experimental budget calibration and action-coupling reward |
| `scripts/budget_processor.py` | Experimental local decoding budget processor |
| `train_grpo.py`, `train_sft_warmup.py` | Research training entry points; not required for the demo |
| `grpo_output/adapter_config.json`, `grpo_output/adapter_model.safetensors` | Saved Qwen2.5 research adapter, separate from product model |
| `data/` | Synthetic episodes, generated source snippets, and historical trace fixtures |
| `ENV.md`, `SAFEGUARDS.md`, `PAPER.md` | Contracts, failure boundaries, and research limitations |
| `docs/` | Submission kit and evidence assessment |

## Known limits

The benchmark uses synthetic vulnerabilities and generated source, with feature/template correlations that can favor hand-designed rules. There is no independently labeled production evaluation or measured developer-time study. Saved adapter weights are local artifacts and may be excluded by Git ignore rules; verify that any eventual submission actually includes or separately links the required weights. Character-based research diagnostics do not measure internal model cognition. Evidence checks are not semantic verification. Training and evaluation across truly held-out repositories remain future work.

This repository is being prepared locally. Historical GitHub and Hugging Face links are not presented as verified deployments of this build. Publishing, video upload, and final form submission remain the entrant's actions.
