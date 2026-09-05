# Submission assessment

Audit date: 5 September 2026. Assessment against the four pillars in the supplied Razorpay Open Track brief. These are qualitative judgments, not official scores or predictions of selection.

## The direct assessment

The original submission was a research demonstration presented as a completed model result. Its largest risk was credibility, followed by a weak user workflow. Another polished chart would not fix either. The strongest entry is now a bounded code-review product: give a developer a useful evidence queue within a chosen review budget, make the unreviewed work explicit, and let judges inspect how the system fails.

No one can promise a win, and the assumed participant count is not verified. The practical goal is a submission that survives a technically skeptical judge opening the repository and changing the demo input.

## What would have cost this entry

| Finding in the original checkout | Why it mattered | Required correction |
|---|---|---|
| README said a 1.7B model learned a 6× allocation ratio; artifacts were heuristic traces | The central achievement was unsupported by a reproducible adapter and inference run | Separate working product, synthetic benchmark, and unverified historical research |
| Transfer F1 was 0.67 in README and 1.00 in other documents | Contradictory numbers make every metric suspect | Use one freshly generated artifact with its exact protocol |
| “Tag removal” only ignored tags when analyzing saved traces | No causal intervention occurred | Withdraw the ablation claim |
| Truncation analysis held decisions fixed | Constant F1 was guaranteed by the script | Re-run inference under each cap before claiming accuracy effects |
| “No existing reasoning-RL work does this” | Closely related prior work already exists | Cite prior work; explain workflow contribution |
| `2 × files` investigation points with reads costing 1 | Reading every file once was still possible | Distinguish legacy tool budget from product review allocation |
| Red-team score used a simplified local reward | Five scripted examples did not prove production robustness | State scope and add direct failure tests |
| Judge guide followed an unrelated OpenEnv rubric | It answered the wrong competition | Map explicitly to Problem Taste, Build Quality, AI Judgment, Failure Recovery |
| A long first-person blog described unverified training outcomes | A persuasive story cannot substitute for provenance | Use only source-backed recovery events from this audit |

## The product worth pitching

**User:** developer or small engineering team reviewing a batch of changed code before release.

**Pain:** a large review queue competes for a fixed amount of model output, attention, and time. A terse report can conceal that important files never received review.

**Promise:** choose a review budget; inspect the allocation, source-grounded findings, and unresolved files; export the audit trail. The developer remains responsible for release decisions.

**Wedge:** a synthetic payment idempotency review provides a concrete consequence and understandable demo. It illustrates a code-review failure; it does not establish money recovered or payment fraud detection.

**Defensible differentiation:** one coherent workflow combines explicit allocation, actual source review, evidence validation, visible abstention, and reproducible failure handling. The project does not establish a new model capability or superiority over commercial code-review tools.

## Remaining gates before submission

1. Capture a successful real model review and save model identity, inputs, configuration, usage, findings, and validation results. An offline rule engine alone leaves meaningful AI insufficiently demonstrated.
2. Verify the public repository and public demo from a signed-out browser. A working local URL is not a working submission link.
3. Show a difficult negative case and a forced provider failure in the video. Do not edit either away.
4. Explain benchmark limitations before the judge asks: synthetic data, in-sample policy design, deterministic detector, and code characters rather than billed token savings.
5. Confirm an external reviewer can follow README setup and reproduce the evidence.

## Related work and originality boundary

These primary sources invalidate a claim that adaptive budgets or budget-aware training are new by themselves:

| Work | Relevant idea | What this submission can honestly claim |
|---|---|---|
| [TALE: Token-Budget-Aware LLM Reasoning](https://arxiv.org/abs/2412.18547) | Estimates a token budget and uses it to guide reasoning | A concrete review workflow and auditable resource policy |
| [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) | Controls reasoning length with budget forcing | Applying bounded review and explicit incomplete states to code inspection |
| [SelfBudgeter](https://arxiv.org/abs/2505.11274) | Pre-estimates reasoning cost and uses budget-guided reinforcement learning | No first-ever prediction-before-reasoning claim |
| [BudgetThinker](https://arxiv.org/abs/2508.17196) | Uses remaining-budget control tokens and staged SFT/RL | A product prototype, with future training treated as a hypothesis |

The distinction above is an inference from the papers' described methods and this repository's implementation. It is not a claim that competing products lack similar features.
