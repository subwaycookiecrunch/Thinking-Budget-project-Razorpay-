---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags: [grpo, lora, research, code-review]
---

# Thinking Budget research adapter

A supplied LoRA adapter for **Qwen/Qwen2.5-1.5B-Instruct**, rank 16, alpha 32. Files include the main adapter and training checkpoints. This is separate from the product's pretrained Ollama `qwen3:4b` reviewer.

## Evidence status

The training summary records 100 steps and mean reward 0.094827. Early mean is 0.101822; late mean is 0.104709. These reward summaries do not establish detection improvement. Recorded action coupling is zero; most trace rewards have no environment score. Historical heuristic charts in this directory must not be treated as inference results from these weights.

## Inspect and evaluate

```bash
python scripts/audit_training_evidence.py
```

For a real baseline-versus-adapter evaluation, install the separately documented training/inference dependencies and make the base model available. This command loads the model and performs inference; it can require a compatible GPU, network access to obtain weights, and substantial runtime:

```bash
python eval_baseline.py --real --adapter grpo_output --seeds 42 137 256
```

The default `python eval_baseline.py` is a **deterministic file-prioritization benchmark**, not this adapter evaluation. It does not establish trained-model performance.

## Limits

No independent held-out model result, production accuracy, reliable tool-use improvement, or learned budget allocation is established. Check the audit artifact, source revision, and trainer configuration before reproducing. The saved adapter may precede the current hardened environment reward. Local weight files may be ignored by Git; public availability is not implied by this model card.

[System and evidence report](../PAPER.md) · [Product README](../README.md)
