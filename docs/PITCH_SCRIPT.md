# Five-minute pitch and demo script

Target duration: 4:40–4:55. Record at readable zoom. Use the exact final UI controls; select the real model mode only when it has been verified. Keep the model name and mode visible. Do not label offline output as AI inference.

## 0:00–0:35 — the problem

> “A developer is about to ship a payment refactor. There are several files to review, one limited review budget, and a dangerous ambiguity: does ‘no findings’ mean the code was reviewed, or that the reviewer ran out of time? The Thinking Budget makes that distinction visible. It spends review effort where risk indicators point, attaches findings to source evidence, and keeps unfinished work in a human-review queue.”

On screen: Review Lab, synthetic payment fixture. State that the fixture is synthetic. Avoid an opening tour of tabs or training equations.

## 0:35–1:30 — a working review

> “Here is the source we are reviewing. These are the budget and model settings. The router chooses how much output each selected file may use. The model analyzes code; deterministic checks handle the budget and verify whether its evidence exists in the source.”

Run the review. Show one finding, its exact evidence, and the associated file. Explain the plausible payment consequence in plain language. A model finding is a review hypothesis, not proof of exploitation or a guarantee of a defect.

> “The important output is this evidence queue. I can inspect the basis for a finding and see which files still need review.”

If using a previously captured real run to avoid waiting, say “This is a captured run from [model], with its configuration and output saved.” Never present a replay as live.

## 1:30–2:15 — change the budget

Lower the review budget and run again. Show selected files, usage, and pending human-review items. Do not promise identical findings: that is what the comparison is testing.

> “A smaller budget changes the amount of review available. The interface does not convert an unreviewed file into a clean bill of health. It leaves an explicit task for a person. Allocated output, reported usage, and code characters are different measurements; the audit preserves that distinction.”

## 2:15–3:00 — show failure recovery

Enable one documented fault-injection mode and run again.

> “Now I will force a provider failure. The review records the failure and marks the affected work as needing review. It does not fabricate a successful model answer. The offline mode remains available as a separately labeled deterministic tool.”

Show the failure event and unresolved file list. Export the audit JSON. Open enough of it to show mode, configuration, outcome, and evidence; avoid scrolling through pages of raw data.

## 3:00–3:45 — evidence, including the limits

Open the regenerated benchmark summary and its metric table.

> “This benchmark tests prioritization across 150 bundled synthetic episodes. It measures which positive-labeled files each policy actually reads under the same allowance, alongside missed positives and safe files reviewed. It measures file coverage, not bug detection. The shuffled-feature stress test shows the ranking depends strongly on those features. Actual model runs are shown separately.”

Read only the current values shown in `grpo_output/benchmark_results.json`. Do not memorize numbers from legacy plots. Point out missed positive files and shuffled-feature degradation, even if the original-feature result looks favorable.

## 3:45–4:25 — what broke

> “The hardest bug was in the claim itself. The first version's headline trained-model results came from heuristic traces. A supposed truncation ablation held decisions fixed, so its answer was predetermined. I removed those claims and rebuilt the submission around separate evidence types. That also changed the product: provenance, failures, and unreviewed work are now first-class outputs.”

Show the evidence limitations briefly. Discuss only recovery steps actually present in the final build.

## 4:25–4:55 — why this entry

> “This is an Open Track entry because the problem is developer verification capacity. AI handles the semantic code review; deterministic code controls resources and checks evidence. The next validation is an independently labeled set of real pull requests and developer review time. Today the repository gives you a working review flow, inspectable failures, and a reproducible evaluation. You can change the code input and test the limits yourself.”

End on the review result, repository URL, and model mode. Avoid “world first,” guaranteed savings, production-ready, or guaranteed safety.

## Recording fallback

If the live provider is unavailable, show a verified captured model run with provenance, then demonstrate the failure live. If no genuine model run exists, state that clearly and record the offline prototype; the lack of real inference remains a submission weakness. Do not substitute simulated text and call it model output.
