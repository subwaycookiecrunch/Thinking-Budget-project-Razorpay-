# Judge guide — The Thinking Budget

**Track 05: Open Track.** A code-review product with an explicit review budget, source-grounded findings, and a visible queue of work still needing human attention.

The supplied brief asks for a real problem, a working product, meaningful AI, and evidence of value. This guide follows its four pillars. It does not assume an OpenEnv research-hackathon rubric.

## A five-minute inspection

| Time | Action | What to inspect |
|---|---|---|
| 0:00–0:45 | Open Review Lab and load the synthetic payment example | The user problem, model mode, source, and chosen budget are visible |
| 0:45–2:00 | Run the review | Inspect a finding's exact source evidence and the unresolved files |
| 2:00–2:45 | Reduce the budget and repeat | The allocation and incomplete work change explicitly |
| 2:45–3:30 | Trigger a documented provider failure | Failed work remains in the review queue; inspect the audit export |
| 3:30–4:15 | Change the input | Confirm the product computes a result from supplied source |
| 4:15–5:00 | Open benchmark output and limitations | Read FP/FN, cost units, protocol, and the synthetic-data boundary |

Use the selected model mode honestly. A saved model run is a replay; offline rules are deterministic. Neither should be presented as live LLM inference.

## Four pillars, concrete evidence

| Pillar | Product decision | Evidence | Remaining limit |
|---|---|---|---|
| **Problem Taste** | Help developers prioritize limited review effort without hiding unfinished work | Payment fixture, custom source input, per-file decisions | No production user study or measured time savings yet |
| **Build Quality** | Keep limits, evidence, outcome state, and export in one flow | `app.py`, review implementation, tests, audit JSON | Prototype; validation does not prove every finding correct |
| **AI Judgment** | Use the model for source interpretation; use code for allocation and evidence checks | Real model mode plus labeled offline fallback | No new model training or proven model superiority |
| **Failure Recovery** | Retain unsuccessful work as needs-review and expose the cause | Fault-injection demo and exported events | Tested failure cases are finite, not exhaustive |

## Reproduce, don't trust screenshots

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python app.py
```

In a second shell:

```bash
python eval_baseline.py
```

The evaluation produces `grpo_output/benchmark_results.json`, per-episode JSONL, and a Pareto plot. Its policy comparison uses the same deterministic source detector. It reports true/false positives, missed bugs, and source read cost. The generated metadata is the source of truth for policy settings, dataset identity, and uncertainty.

## What the numbers mean

The bundled CVE-themed files and transfer scenarios are synthetic. They illustrate a review-allocation problem; they are not patches scraped from the named projects. The development team has seen this data. “Synthetic benchmark” does not mean “held-out benchmark.”

The default benchmark measures executed rule policies. It does not load a trained adapter, and its cost metric is source characters read. Do not interpret a source-character reduction as an equivalent reduction in model tokens, API charges, wall-clock latency, or human review time.

Historical `trained`/`untrained` trace names are legacy labels for generated policies. Historical F1 improvements, calibration plots, and apparent tag-removal results do not prove learned behavior. No adapter weights are present in this checkout.

## What broke

The audit found a credibility failure: heuristic outputs were described as trained-model improvements, transfer scores disagreed across documents, and two proposed ablations did not perform the claimed intervention. The submission now separates evidence types and withdraws unsupported learned-model results. The product applies the same principle to runtime behavior: distinguish observed evidence, configured allowances, actual usage, and incomplete work.

See [the failure-recovery answer](docs/SUBMISSION_FORM.md#what-broke-and-how-you-got-out), [the assessment](docs/BRUTAL_ASSESSMENT.md), and [safeguards](SAFEGUARDS.md).

## Next evidence that would change the claim

A fair model comparison needs real completions under fixed and adaptive budgets, the same base model and test inputs, actual token/latency measurements, and labels independent of policy development. A training claim additionally needs a reproducible checkpoint and baseline-versus-adapter run. A business-value claim needs developer pilot data. Those are specific next validations, not completed achievements.
