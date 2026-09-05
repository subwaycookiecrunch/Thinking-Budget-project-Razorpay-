# The hardest bug was in the evidence

The Thinking Budget began with an appealing research question: can a code-review agent spend less effort on routine files and more on difficult ones? The repository contained a tool environment, reward shaping, generated trajectories, and dramatic before-and-after charts.

An audit exposed a larger problem than UI polish. Heuristic trajectories were described as trained-model results. Transfer scores disagreed across documents. A supposed truncation ablation changed text length while keeping decisions fixed. A supposed tag-removal experiment merely ignored the tag when analyzing saved text. Those experiments could not support the conclusions attached to them.

The recovery was to separate the claims and build something a judge could interrogate.

The product now accepts an editable bundle of source files. A deterministic router allocates output allowances; a local model reviews each selected file; code validates the response format and checks each quoted source line. A missing, invalid, or unfinished review remains visible as work for a person. The audit preserves the model mode, limits, usage, raw response, validation outcome, and failure events.

That division of work is deliberate. The model interprets code. Ordinary code counts resources and checks quotations. A matching quote confirms where a finding points; it does not prove the model understood the code correctly. No finding automatically approves a file.

The runtime uses pretrained Qwen3:4b through Ollama. There is also an explicitly labeled offline rule scanner. It covers a narrow set of source patterns and is not presented as model inference.

The research branch now has genuine training artifacts: a Qwen2.5-1.5B-Instruct LoRA adapter, checkpoints, and a 100-step summary. This matters, but it does not repair the old evidence. Mean reward is about 0.095, the early-to-late difference is small, and recorded action coupling is zero. The right conclusion is that a training run produced artifacts while successful budget-aware behavior remains unproven. The product does not use that adapter.

The evaluation also became narrower and more useful. It measures which synthetic positive-labeled files a policy actually reads under a fixed read allowance. At the 50% allowance, risk ranking reads 317 of 319 positive rows while seeded random order reads 159. That sounds strong until the stress test: shuffle the synthetic features and risk-ranked coverage falls to about 50%. The mechanism depends on those signals. These are coverage results on data used during development, not bug-detection accuracy or production savings.

The most important demo now includes a failure. Force a timeout or false citation and the affected file stays in the human-review queue. The system does not replace that failure with fabricated success. A deadline-limited review should tell its user what remains undone.

The next work is an independent real-pull-request evaluation and a developer pilot. Until then, the useful claim is concrete: a local review workflow with visible budgets, source-linked candidates, and inspectable failures. That is a claim someone can test by changing the input.

[Product and setup](README.md) · [Evidence report](PAPER.md) · [Judge guide](JUDGES.md)
