"""Reproducible, label-blind file-prioritization benchmark (no model inference).

The unit of work is an actual bounded source read. A reviewed positive is
*covered*, not automatically detected: this benchmark cannot validate a
security finding or measure LLM reasoning tokens. Labels are only given to
the scorer after a policy returns its review plan.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean
from typing import Sequence

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "grpo_output" / "benchmark_results.json"


def _mask_comments(source: str) -> str:
    """Mask common comments while preserving offsets and quoted strings.

    This is a lightweight scanner, not a full cross-language parser. Preserved
    offsets let every diagnostic quote refer to the actual supplied source.
    """
    pattern = re.compile(r'''("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|(/\*[\s\S]*?\*/|//[^\n]*|\#[^\n]*)''')
    return pattern.sub(lambda m: m.group(1) if m.group(1) else re.sub(r"[^\n]", " ", m.group(2)), source)


def inspect_source(source: str, file_path: str) -> list[dict]:
    """Conservative source-pattern review with exact evidence, no dataset labels.

    Findings are candidates for human review. Missing findings do not establish
    safety; these rules cover a narrow set of security and reliability patterns.
    """
    code = _mask_comments(source)
    findings = []

    def add(rule: str, match: re.Match, severity: str, title: str, rationale: str) -> None:
        start = source.rfind("\n", 0, match.start()) + 1
        end = source.find("\n", match.start())
        end = len(source) if end < 0 else end
        quote = source[start:end].strip()
        findings.append({"rule_id": rule, "file_path": file_path,
                         "line": source.count("\n", 0, match.start()) + 1,
                         "severity": severity, "title": title, "evidence": quote,
                         "rationale": rationale, "source": "static_rule",
                         "confidence": "pattern_match; reachability requires review"})

    rules = [
        ("DYNAMIC_EVAL", r"\b(?:eval|exec)\s*\(\s*(?![\"'])\w+", "high",
         "Dynamic code execution", "A variable is passed to a code-execution primitive. Trace its origin; untrusted input can execute code."),
        ("SHELL_TRUE", r"\b(?:subprocess\.)?(?:run|Popen|call|check_output|check_call)\s*\([\s\S]{0,400}?\bshell\s*=\s*True", "high",
         "Shell interpretation enabled", "shell=True allows shell syntax in the command. Untrusted interpolation requires removal or a strict input boundary."),
        ("UNSAFE_PICKLE", r"\bpickle\.(?:load|loads)\s*\(", "high",
         "Unsafe object deserialization", "Pickle can execute code during loading. Verify the input cannot be controlled by an untrusted actor."),
        ("YAML_LOADER", r"\byaml\.load\s*\([^\n]*(?:Loader\s*=\s*(?:yaml\.)?(?:Loader|UnsafeLoader)|\)\s*$)", "high",
         "Potentially unsafe YAML loader", "Use safe_load or SafeLoader when parsing untrusted YAML; verify loader behavior for the installed version."),
        ("JWT_VERIFICATION_DISABLED", r"(?:[\"']verify_signature[\"']\s*:\s*False|\bverify\s*=\s*False|\bverify_signature\s*=\s*False)", "critical",
         "Token signature verification disabled", "Accepting token claims without signature verification can allow forged authentication claims."),
        ("AUTH_PREFIX", r"\b(?:path|pathname|url|request\.path|req\.path)\.(?:startswith|startsWith)\s*\(\s*[\"']/[^\"']*[\"']", "medium",
         "Prefix-based route check", "If this check bypasses authentication, a prefix also accepts unintended suffixes. Verify exact route boundaries and use explicit public-route matching."),
        ("MUTABLE_DEFAULT", r"\bdef\s+\w+\s*\([^\n)]*?=\s*(?:\[\]|\{\})", "medium",
         "Shared mutable default argument", "Python reuses this default object across calls. Mutation can leak state between requests; initialize inside the function instead."),
        ("DISABLED_TLS", r"\b(?:requests\.(?:get|post|put|delete)|httpx\.(?:get|post))\s*\([\s\S]{0,300}?\bverify\s*=\s*False", "high",
         "TLS certificate verification disabled", "The request accepts unverified server certificates, exposing traffic to interception on an untrusted network."),
    ]
    for rule, pattern, severity, title, rationale in rules:
        for match in re.finditer(pattern, code, re.MULTILINE):
            add(rule, match, severity, title, rationale)

    # Link interpolation to an actual query sink; a query-looking string alone
    # is insufficient. Safe parameterized execute(sql, values) does not match.
    query_assignment = re.compile(r"\b(\w+)\s*=\s*(?:f[\"']|`)[^\n]*(?:SELECT|INSERT|UPDATE|DELETE)[^\n]*(?:\{|\$\{)", re.I)
    for match in query_assignment.finditer(code):
        variable = re.escape(match.group(1))
        if re.search(rf"\.(?:execute|query|raw)\s*\(\s*{variable}\b", code[match.end():]):
            add("SQL_INTERPOLATION", match, "high", "Interpolated SQL reaches a query sink",
                "The query string incorporates a value before execution. Use bound parameters and verify whether the interpolated value is attacker controlled.")
    for match in re.finditer(r"\.(?:execute|query|raw)\s*\(\s*(?:f[\"']|`)[^\n]*(?:SELECT|INSERT|UPDATE|DELETE)[^\n]*(?:\{|\$\{)", code, re.I):
        add("SQL_INTERPOLATION", match, "high", "Interpolated SQL executed directly",
            "String interpolation reaches a SQL execution call. Bound parameters keep data separate from SQL syntax.")

    unique = {(f["rule_id"], f["line"]): f for f in findings}
    return sorted(unique.values(), key=lambda f: (f["line"], f["rule_id"]))


@dataclass(frozen=True)
class PublicFile:
    path: str
    churn: float
    complexity: float
    todos: float
    recency: float
    is_test: bool


def public_files(rows: list[dict]) -> tuple[PublicFile, ...]:
    """Whitelist observable fields; exclude labels, summaries and descriptions."""
    result = []
    for row in rows:
        values = list(row.get("features", [0, 0, 0, 0]))
        if len(values) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
            raise ValueError("Each file must have four finite numeric features")
        result.append(PublicFile(str(row["file"]), *map(float, values),
                                 bool(row.get("is_test_file", row.get("is_test", False)))))
    if len({f.path for f in result}) != len(result):
        raise ValueError("Duplicate paths inside an episode cannot be scored unambiguously")
    return tuple(result)


def feature_risk(file: PublicFile) -> float:
    """Fixed structural heuristic; weights are not learned or tuned in this run."""
    score = .45 * file.churn / 100 + .40 * file.complexity / 100
    score += .10 * file.todos / 20 + .05 * file.recency / 100
    return score * (.30 if file.is_test else 1.0)


def stable_seed(seed: int, episode_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()[:8], "big")


def plan_reviews(files: Sequence[PublicFile], policy: str, budget: int, seed: int) -> list[str]:
    """Choose files before seeing source or ground truth; never mutate input."""
    if budget < 0:
        raise ValueError("Budget must be nonnegative")
    ordered = sorted(files, key=lambda f: f.path)
    if policy == "risk_ranked":
        ordered.sort(key=lambda f: (-feature_risk(f), f.path))
    elif policy == "risk_with_exploration":
        ranked = sorted(ordered, key=lambda f: (-feature_risk(f), f.path))
        exploit_count = max(0, budget - max(1, budget // 4))
        tail = ranked[exploit_count:]
        random.Random(seed).shuffle(tail)
        ordered = ranked[:exploit_count] + tail
    elif policy == "random_order":
        random.Random(seed).shuffle(ordered)
    elif policy != "exhaustive":
        raise ValueError(f"Unknown policy: {policy}")
    return [file.path for file in ordered[:budget]]


class ReadWorkspace:
    """Bounded source-read tool with no access to scoring labels."""
    def __init__(self, files: Sequence[PublicFile], snippets: dict[str, str], budget: int):
        self._paths = frozenset(f.path for f in files)
        self._snippets = snippets
        self.budget = budget
        self.reads: list[dict] = []

    def read_file(self, path: str) -> str:
        if path not in self._paths:
            raise ValueError("Path is outside the episode")
        if len(self.reads) >= self.budget:
            raise RuntimeError("Read budget exhausted")
        code = self._snippets.get(path)
        self.reads.append({"path": path, "source_chars": len(code) if code is not None else 0,
                           "source_available": code is not None})
        return code or ""


def score_plan(rows: list[dict], reviewed: list[str], miss_cost: float = 5, review_cost: float = 1) -> dict:
    """Coverage metrics; FP here means a safe file reviewed, not a bug alarm."""
    labels = {row["file"]: int(row["label"]) for row in rows}
    if not all(value in (0, 1) for value in labels.values()):
        raise ValueError("Labels must be binary")
    selected = set(reviewed)
    if not selected <= labels.keys() or len(selected) != len(reviewed):
        raise ValueError("Review plan must contain unique known paths")
    tp = sum(labels[path] for path in selected)
    fp = len(selected) - tp
    fn = sum(labels.values()) - tp
    tn = len(labels) - tp - fp - fn
    return {"covered_positives": tp, "safe_reviews": fp, "missed_positives": fn,
            "safe_unreviewed": tn, "coverage_recall": tp / (tp + fn) if tp + fn else None,
            "review_precision": tp / (tp + fp) if tp + fp else 0.0,
            "selection_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "weighted_triage_loss": miss_cost * fn + review_cost * fp}


def run_episode(episode: dict, snippets: dict, policy: str, budget_fraction: float,
                seed: int = 17, feature_shift: bool = False, miss_cost: float = 5,
                review_cost: float = 1) -> dict:
    if not 0 <= budget_fraction <= 1:
        raise ValueError("Budget fraction must be between zero and one")
    rows = episode["files"]
    files = public_files(rows)
    episode_seed = stable_seed(seed, episode["episode_id"])
    if feature_shift:
        features = list(files)
        random.Random(episode_seed).shuffle(features)
        files = tuple(PublicFile(f.path, other.churn, other.complexity, other.todos,
                                 other.recency, other.is_test)
                      for f, other in zip(files, features))
    budget = len(files) if policy == "exhaustive" else math.ceil(len(files) * budget_fraction)
    plan = plan_reviews(files, policy, budget, episode_seed)
    workspace = ReadWorkspace(files, snippets, budget)
    for path in plan:
        workspace.read_file(path)
    metrics = score_plan(rows, plan, miss_cost, review_cost)
    return {"episode_id": episode["episode_id"], "domain": episode.get("domain", "synthetic-cve"),
            "policy": policy, "seed": seed, "budget_fraction": budget_fraction,
            "feature_condition": "shuffled_features" if feature_shift else "original_features",
            "n_files": len(files), "n_positives": sum(int(r["label"]) for r in rows),
            "read_budget": budget, "reads_used": len(workspace.reads),
            "source_chars_read": sum(x["source_chars"] for x in workspace.reads),
            "source_chars_available": sum(len(snippets.get(f.path, "")) for f in files),
            "missing_source_reads": sum(not x["source_available"] for x in workspace.reads),
            **metrics, "reviewed_files": workspace.reads,
            "missed_files": [r["file"] for r in rows if r["label"] == 1 and r["file"] not in plan],
            "safe_files_reviewed": [r["file"] for r in rows if r["label"] == 0 and r["file"] in plan]}


def aggregate(records: list[dict]) -> dict:
    tp = sum(r["covered_positives"] for r in records)
    fp = sum(r["safe_reviews"] for r in records)
    fn = sum(r["missed_positives"] for r in records)
    chars = sum(r["source_chars_read"] for r in records)
    total_chars = sum(r["source_chars_available"] for r in records)
    return {"n_episodes": len(records), "covered_positives": tp, "safe_reviews": fp,
            "missed_positives": fn, "coverage_recall": tp / (tp + fn) if tp + fn else 0,
            "review_precision": tp / (tp + fp) if tp + fp else 0,
            "selection_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "reads_used": sum(r["reads_used"] for r in records),
            "available_files": sum(r["n_files"] for r in records),
            "source_chars_read": chars, "source_chars_available": total_chars,
            "source_character_reduction": 1 - chars / total_chars if total_chars else None,
            "mean_weighted_triage_loss": mean(r["weighted_triage_loss"] for r in records),
            "missing_source_reads": sum(r["missing_source_reads"] for r in records)}


def bootstrap_interval(records: list[dict], metric: str, seed: int, samples: int = 1000) -> list[float]:
    """Episode bootstrap, not independent file bootstrap; descriptive only."""
    if not records:
        return [0.0, 0.0]
    rng = random.Random(seed)
    values = sorted(aggregate(rng.choices(records, k=len(records)))[metric] for _ in range(samples))
    return [values[int(.025 * (samples - 1))], values[int(.975 * (samples - 1))]]


def load_cve_episodes(path: Path = ROOT / "data" / "cve_training_data.json") -> list[dict]:
    groups: dict = defaultdict(list)
    for row in json.loads(path.read_text()):
        groups[(row["cveId"], row["repo"])].append(row)
    return [{"episode_id": f"{cve}:{repo}", "domain": "synthetic-cve", "files": rows}
            for (cve, repo), rows in sorted(groups.items())]


def dataset_audit(episodes: list[dict], snippets: dict) -> dict:
    rows = [r for episode in episodes for r in episode["files"]]
    paths = Counter(r["file"] for r in rows)
    labels_by_path: dict = defaultdict(set)
    for row in rows:
        labels_by_path[row["file"]].add(row["label"])
    return {"episodes": len(episodes), "file_rows": len(rows),
            "positive_rows": sum(r["label"] for r in rows),
            "negative_only_episodes": sum(not any(r["label"] for r in e["files"]) for e in episodes),
            "unique_source_paths": len(paths), "repeated_paths": sum(v > 1 for v in paths.values()),
            "conflicting_label_paths": sorted(p for p, labels in labels_by_path.items() if len(labels) > 1),
            "missing_source_paths": sorted(set(paths) - snippets.keys()),
            "snippet_validity": "Synthetic snippets; correspondence between positive labels and actual defects is unverified.",
            "split_status": "All bundled episodes, including training data. No independent held-out evaluation."}


def run_benchmark(episodes: list[dict] | None = None, snippets: dict | None = None,
                  seed: int = 17, bootstrap_samples: int = 1000,
                  budget_fractions: tuple = (.25, .5, .75, 1.0),
                  miss_cost: float = 5, review_cost: float = 1) -> dict:
    if miss_cost < 0 or review_cost < 0:
        raise ValueError("Costs must be nonnegative")
    episodes = load_cve_episodes() if episodes is None else episodes
    snippets = json.loads((ROOT / "data" / "code_snippets.json").read_text()) if snippets is None else snippets
    if not episodes or bootstrap_samples < 1:
        raise ValueError("Need at least one episode and one bootstrap sample")
    records, summaries = [], []
    for condition in (False, True):
        for policy in ("random_order", "risk_ranked", "risk_with_exploration", "exhaustive"):
            for fraction in ((1.0,) if policy == "exhaustive" else budget_fractions):
                batch = [run_episode(ep, snippets, policy, fraction, seed, condition, miss_cost, review_cost)
                         for ep in episodes]
                summary = {"policy": policy, "budget_fraction": fraction,
                           "feature_condition": "shuffled_features" if condition else "original_features",
                           **aggregate(batch)}
                summary["coverage_recall_ci95"] = bootstrap_interval(batch, "coverage_recall", seed, bootstrap_samples)
                summaries.append(summary)
                records.extend(batch)
    hashes = {}
    for path in [Path(__file__), ROOT / "data" / "cve_training_data.json", ROOT / "data" / "code_snippets.json"]:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"schema_version": 1, "evidence_type": "executed_deterministic_heuristic_prioritization",
            "model_inference": False, "trained_model_evaluated": False,
            "primary_metric": "coverage_recall: fraction of dataset-positive files actually read; not bug-detection recall",
            "cost_unit": "actual source characters returned by bounded read_file calls; not tokens, time, money, or energy",
            "seed": seed, "bootstrap_samples": bootstrap_samples,
            "uncertainty": "95% percentile bootstrap over episodes; one seeded random ordering. Descriptive sensitivity, not population generalization.",
            "weighted_cost_assumptions": {"missed_positive": miss_cost, "safe_file_review": review_cost,
                                          "unit": "illustrative relative units, not measured financial impact"},
            "limitations": ["Synthetic features may encode the dataset construction process.",
                            "Source reads do not imply correct security findings; no detection model is evaluated.",
                            "Code snippets may not faithfully implement the labeled CVE defect.",
                            "Feature shuffling is a synthetic distribution-shift stress test, not a new real-world domain.",
                            "No measured LLM token savings, latency improvement, training gain, or production generalization."],
            "dataset": dataset_audit(episodes, snippets), "source_sha256": hashes,
            "policies": summaries, "episodes": records}


def plot_benchmark(result: dict, output: Path) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    colors = {"risk_ranked": "#0d9488", "random_order": "#64748b", "risk_with_exploration": "#7c3aed"}
    for ax, condition, title in zip(axes, ["original_features", "shuffled_features"],
                                    ["Bundled synthetic features", "Stress test: shuffled features"]):
        for policy, color in colors.items():
            rows = [r for r in result["policies"] if r["policy"] == policy and r["feature_condition"] == condition]
            ax.plot([r["reads_used"] / r["available_files"] for r in rows],
                    [r["coverage_recall"] for r in rows], "o-", label=policy.replace("_", " "), color=color)
            ax.fill_between([r["reads_used"] / r["available_files"] for r in rows],
                            [r["coverage_recall_ci95"][0] for r in rows],
                            [r["coverage_recall_ci95"][1] for r in rows], color=color, alpha=.10)
        ax.set(title=title, xlabel="Fraction of files actually read", xlim=(0, 1.04), ylim=(0, 1.05))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Coverage of dataset-positive files")
    fig.suptitle("Thinking Budget · executable file-prioritization benchmark", fontweight="bold")
    fig.text(.5, .01, "Heuristic policies · synthetic training data · shaded 95% episode bootstrap · no LLM inference", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def write_results(result: dict, output: Path, plot: bool = True) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    episodes_path = output.with_name("benchmark_episodes.jsonl")
    episodes_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in result["episodes"]))
    summary = {key: value for key, value in result.items() if key != "episodes"}
    summary["episode_artifact"] = episodes_path.name
    summary["episode_artifact_sha256"] = hashlib.sha256(episodes_path.read_bytes()).hexdigest()
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if plot:
        plot_benchmark(result, output.with_name("benchmark_pareto.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--miss-cost", type=float, default=5)
    parser.add_argument("--review-cost", type=float, default=1)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(seed=args.seed, bootstrap_samples=args.bootstrap_samples,
                           miss_cost=args.miss_cost, review_cost=args.review_cost)
    write_results(result, args.output, plot=not args.no_plot)
    print(f"Executed heuristic prioritization on {result['dataset']['episodes']} synthetic episodes.")
    for row in result["policies"]:
        if row["feature_condition"] == "original_features" and (row["budget_fraction"] == .5 or row["policy"] == "exhaustive"):
            print(f"{row['policy']:24s} coverage={row['coverage_recall']:.3f} reads={row['reads_used']} "
                  f"missed={row['missed_positives']} safe_reviews={row['safe_reviews']}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
