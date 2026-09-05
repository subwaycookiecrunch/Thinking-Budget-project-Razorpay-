# Environment and runtime contracts

The repository has two separate execution surfaces: the product review engine and the research OpenEnv environment. Their budgets and outputs are different.

## Product review engine

Source: `review_engine.py`. UI: `app.py`.

Input is a JSON source bundle or file blocks headed `=== relative/path.py ===`. The engine validates unique relative paths, rejects traversal/control characters, and limits input to 24 files, 16,000 characters per file, and 80,000 source characters overall. Source is data; it is never executed.

A `ReviewBudget` specifies output tokens, accepted source characters, file count, request timeout, and run duration. The risk policy chooses per-file output allowances; the uniform policy uses input order and a common allowance. The preview shows a worst-case reservation. Actual execution can reuse unspent reported tokens.

| Unit | Meaning |
|---|---|
| Output allowance | Requested maximum generated review tokens |
| Output tokens reported | Provider's measured generation count |
| Output tokens accounted | Reported generation, or the conservative reservation when usage is unknown |
| Input tokens reported | Provider count for its complete prompt |
| Source characters | User source admitted to requests; excludes prompt scaffolding |

Output allowance does not bound input tokens, hardware compute, energy, or billed cost. The current local provider is Ollama, default `qwen3:4b`, with thinking disabled. The saved custom Qwen2.5 adapter is not loaded by this product path.

A response must have a valid decision, bounded findings and summary, and exact single-line evidence for each finding. Outcomes are `flag`, `no_finding`, `needs_review`, or `deferred`. Invalid output, model abstention, and provider failure require human review. Two consecutive review failures open a circuit and defer subsequent files. Unknown usage consumes the request reservation. A provider-reported cap violation is exposed and stops further model requests.

The JSON export contains configuration, source hashes, raw response events, validation outcomes, counters, and a hash-linked event sequence. The Markdown export is a readable human handoff. The hash chain is locally verifiable but not externally signed.

## Research investigation environment

Source: `code_review_env/server/environment.py`. Server: `server/app.py`.

```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 7861
```

Use a persistent WebSocket or MCP session for a multi-action investigation. The server's factory creates an environment per SDK session. An environment instance holds one active episode; reset replaces it and discards the previous session. Do not multiplex independent investigations through one direct Python instance.

The public reset observation exposes context and session ID as serializable fields. It includes CVE-themed metadata, paths, and structural features. Ground-truth labels are retained for final scoring and withheld from intermediate flag/skip replies. These labels belong to generated scenarios; the code is not an extracted production CVE patch.

| Tool | Arguments | Investigation cost |
|---|---|---:|
| `read_file` | `file_path` | 1 |
| `search_code` | `pattern` | 2 |
| `get_function_list` | `file_path` | 1 |
| `flag_vulnerable` | `file_path`, `reasoning` | 0; consumes a flag slot |
| `skip_file` | `file_path`, `reasoning` | 0 |
| `submit_report` | `summary`, optional `confidence` | 0; ends episode |

There are six tools. `search_code` performs case-insensitive substring matching, not arbitrary regex or shell execution. The SDK code-execution surface is disabled.

For N files, investigation points are `2 × N`. A read costs one point, so every file can be read once; this budget limits repeated investigation, not initial coverage. Costs are checked before spending, preventing negative balances. The flag allowance is `min(N, max(5, ceil(0.4 × N)))`, based on observable patch size rather than hidden bug count.

The environment rejects missing files, duplicate/conflicting decisions, invalid bounded reasoning, and post-submission actions. Exhausted investigation tools return a warning; the caller must submit. There is no general hard global action limit in this class. Client timeouts and request limits belong to the caller or product layer.

Submission finalizes the research classification: undecided files are treated as unflagged for metrics. This differs from the product's explicit unresolved queue. Final precision/recall/F1 and composite reward are exposed structurally. Correct all-negative episodes receive F1=1 under the project's stated convention. See `code_review_env/scoring.py`.

```text
final = F1 × (0.35 × F1 + 0.20 × report_structure
              + 0.15 × step_efficiency + 0.15 × reasoning_length_proxy
              + 0.15 × precision_bonus)
```

Report and reasoning terms are heuristic proxies. The detection gate prevents those auxiliary terms alone from rescuing zero detection F1. This is not proof against all reward attacks. The training script has its own composite/fallback path; see [PAPER.md](PAPER.md).

## Reproducibility boundary

Seeded resets use a local random generator and copy episode data before injecting deceptive structural features. The source dataset is not modified by reset. Difficulty selects file-count bands: easy at most 15, medium 16–29, hard at least 30, with available-pool fallback.

The benchmark reads canonical `data/` files directly and does not call the research environment's reset. Product review uses user-provided source. Verify each surface independently; success on one is not validation of the other.
