"""
CodeReviewEnv v3 — MCP-based Security Code Investigator
========================================================
Multi-tool environment where the LLM investigates CVE vulnerabilities
by reading code, searching patterns, and producing triage reports.
"""
import json
import random
import uuid
import os
import re
from collections import defaultdict
from typing import Optional, Any

from fastmcp import FastMCP
from openenv.core.env_server import MCPEnvironment, Observation

# ── Data Loading ────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def _load_data():
    with open(os.path.join(DATA_DIR, "cve_training_data.json")) as f:
        raw = json.load(f)
    snippets_path = os.path.join(DATA_DIR, "code_snippets.json")
    if os.path.exists(snippets_path):
        with open(snippets_path) as f:
            snippets = json.load(f)
    else:
        snippets = {}

    groups = defaultdict(list)
    for s in raw:
        groups[(s["cveId"], s["repo"])].append(s)

    episodes = []
    for (cve_id, repo), files in groups.items():
        buggy = [f for f in files if f["label"] == 1]
        if not buggy and len(files) > 30:
            files = random.sample(files, 30)
        episodes.append({
            "cve_id": cve_id,
            "cvss": files[0].get("cvss", 0.0),
            "cve_description": files[0].get("cve_description", ""),
            "repo": repo,
            "files": files,
            "total_bugs": len(buggy),
        })
    return episodes, snippets

EPISODES, CODE_SNIPPETS = _load_data()
BUGGY_EPISODES = [e for e in EPISODES if e["total_bugs"] > 0]
print(f"CodeReviewEnv v3: {len(EPISODES)} episodes, {len(BUGGY_EPISODES)} with bugs, {len(CODE_SNIPPETS)} code snippets")


def _risk_summary(file_entry, cvss):
    feat = file_entry.get("features", [0, 0, 0, 0])
    churn, complexity, todos, recency = feat
    parts = []
    if churn > 50: parts.append("high churn")
    elif churn > 20: parts.append("moderate churn")
    if complexity > 60: parts.append("very high complexity")
    elif complexity > 30: parts.append("elevated complexity")
    if todos > 10: parts.append(f"{todos} TODO/FIXME markers")
    elif todos > 0: parts.append(f"{todos} TODOs")
    if recency > 50: parts.append("recently modified")
    if file_entry.get("is_test_file"): parts.append("test file")

    sev = "CRITICAL" if cvss >= 9.0 else "HIGH" if cvss >= 7.0 else "MEDIUM" if cvss >= 4.0 else "LOW"
    risk = f"[{sev} CVSS {cvss}] "
    risk += ("; ".join(parts) + ".") if parts else "No notable risk indicators."
    risk += f" Lang: {file_entry.get('file_language', '?')}."
    return risk


# ── Session State ───────────────────────────────────────────────────
class InvestigationSession:
    """Tracks one investigation episode."""
    def __init__(self, episode, budget):
        self.episode = episode
        self.files = {f["file"]: f for f in episode["files"]}
        self.bugs = {f["file"] for f in episode["files"] if f["label"] == 1}
        self.deceptive = {f["file"] for f in episode["files"]
                         if f.get("is_deceptive", False)}
        self.flagged = set()
        self.skipped = set()
        self.reads = set()
        self.searches = []
        self.flag_reasons = {}
        self.report = None
        self.budget = budget            # max flags
        self.invest_budget = len(episode["files"]) * 2  # investigation points
        self.invest_used = 0
        self.done = False
        self.step_count = 0
        self.thinking_cost = 0          # running total of reasoning chars

        # Thinking budget tracking (for Qwen3 thinking mode integration)
        # Tracks whether the agent applied deep reasoning to the RIGHT files
        self.deep_thinks_on_bugs = 0      # <think> blocks on actual buggy files
        self.deep_thinks_on_clean = 0     # <think> blocks on clean files (wasted)
        self.shallow_on_bugs = 0          # no <think> on buggy files (missed)
        self.reasoning_lengths = {}       # file -> reasoning length

    def use_invest(self, cost=1):
        self.invest_used += cost
        self.step_count += 1
        return self.invest_used <= self.invest_budget

    def record_reasoning(self, file_path, reasoning_text):
        """Track reasoning quality — did the agent think deeply on the right files?"""
        is_bug = file_path in self.bugs
        self.reasoning_lengths[file_path] = len(reasoning_text)
        # Reasoning > 100 chars counts as 'deep thinking'
        is_deep = len(reasoning_text) > 100
        if is_deep and is_bug:
            self.deep_thinks_on_bugs += 1
        elif is_deep and not is_bug:
            self.deep_thinks_on_clean += 1
        elif not is_deep and is_bug:
            self.shallow_on_bugs += 1

    def thinking_efficiency_score(self):
        """How well did the agent allocate its thinking budget?
        Perfect score: deep thinking on ALL buggy files, shallow on clean files."""
        total_decisions = len(self.flagged) + len(self.skipped)
        if total_decisions == 0:
            return 0.0
        # Reward deep thinking on bugs, penalize wasted deep thinking on clean
        if len(self.bugs) == 0:
            return 1.0 if self.deep_thinks_on_clean == 0 else 0.5
        bug_coverage = self.deep_thinks_on_bugs / max(1, len(self.bugs))
        waste_penalty = self.deep_thinks_on_clean / max(1, total_decisions)
        return max(0, bug_coverage - 0.5 * waste_penalty)


class CodeReviewEnvironment(MCPEnvironment):
    """MCP-based security code investigation environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("CodeReviewInvestigator")
        self._sessions = {}  # session_id -> InvestigationSession
        self._current_session_id = None

        # ── Tool: read_file ──────────────────────────────
        @mcp.tool()
        def read_file(file_path: str) -> str:
            """Read the source code of a file in the CVE patch. Costs 1 investigation point."""
            s = self._get_session()
            if s.done:
                return "ERROR: Investigation is complete. No more actions allowed."
            if file_path not in s.files:
                return f"ERROR: '{file_path}' not found in this patch. Use list shown at reset."

            if not s.use_invest(1):
                return "WARNING: Investigation budget exhausted. Submit your report."

            s.reads.add(file_path)
            code = CODE_SNIPPETS.get(file_path, "// [source code not available]")
            f = s.files[file_path]
            feat = f.get("features", [0, 0, 0, 0])
            risk = _risk_summary(f, float(s.episode.get("cvss", 0)))

            return (
                f"=== {file_path} ===\n"
                f"Language: {f.get('file_language', '?')} | "
                f"Component: {f.get('file_component', '?')}\n"
                f"Metrics: churn={feat[0]}, complexity={feat[1]}, "
                f"todos={feat[2]}, recency={feat[3]}\n"
                f"Risk: {risk}\n"
                f"{'─'*50}\n"
                f"{code}\n"
                f"{'─'*50}\n"
                f"[Budget: {s.invest_budget - s.invest_used} investigation points remaining | "
                f"Flags: {len(s.flagged)}/{s.budget} | "
                f"Thinking cost: {s.thinking_cost} chars]"
            )

        # ── Tool: search_code ────────────────────────────
        @mcp.tool()
        def search_code(pattern: str) -> str:
            """Search for a text pattern across all files. Costs 2 investigation points. Returns matching files."""
            s = self._get_session()
            if s.done:
                return "ERROR: Investigation is complete."

            if not s.use_invest(2):
                return "WARNING: Investigation budget exhausted. Submit your report."

            s.searches.append(pattern)
            pat = pattern.lower()
            matches = []

            for fpath, code in CODE_SNIPPETS.items():
                if fpath not in s.files:
                    continue
                if pat in code.lower() or pat in fpath.lower():
                    lines = [i+1 for i, l in enumerate(code.split('\n'))
                             if pat in l.lower()]
                    matches.append((fpath, lines[:3]))

            if not matches:
                return f"No matches for '{pattern}' in patch files."

            result = f"Found '{pattern}' in {len(matches)} file(s):\n"
            for fpath, lines in matches[:10]:
                line_str = ", ".join(str(l) for l in lines) if lines else "in path"
                result += f"  • {fpath} (line{'s' if len(lines)>1 else ''}: {line_str})\n"
            if len(matches) > 10:
                result += f"  ... and {len(matches)-10} more files\n"
            return result

        # ── Tool: get_function_list ──────────────────────
        @mcp.tool()
        def get_function_list(file_path: str) -> str:
            """List functions/methods in a file with complexity indicators. Costs 1 point."""
            s = self._get_session()
            if s.done:
                return "ERROR: Investigation is complete."
            if file_path not in s.files:
                return f"ERROR: '{file_path}' not found."

            if not s.use_invest(1):
                return "WARNING: Budget exhausted."

            code = CODE_SNIPPETS.get(file_path, "")
            f = s.files[file_path]
            lang = f.get("file_language", "")

            # Extract function-like patterns
            funcs = []
            if lang in ("C", "C++", "C/C++ Header"):
                for m in re.finditer(r'(?:static\s+)?(?:int|void|struct\s+\w+\s*\*?|char\s*\*?|bool)\s+(\w+)\s*\(', code):
                    funcs.append(m.group(1))
            elif lang == "Python":
                for m in re.finditer(r'def\s+(\w+)\s*\(', code):
                    funcs.append(m.group(1))
            elif lang in ("JavaScript", "TypeScript"):
                for m in re.finditer(r'function\s+(\w+)\s*\(', code):
                    funcs.append(m.group(1))
            elif lang == "Go":
                for m in re.finditer(r'func\s+(\w+)\s*\(', code):
                    funcs.append(m.group(1))
            elif lang == "Java":
                for m in re.finditer(r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\(', code):
                    funcs.append(m.group(1))

            if not funcs:
                funcs = ["(no functions detected)"]

            feat = f.get("features", [0, 0, 0, 0])
            result = f"Functions in {file_path}:\n"
            for fn in funcs:
                result += f"  • {fn}() — complexity≈{feat[1]}, churn={feat[0]}\n"
            return result

        # ── Tool: flag_vulnerable ────────────────────────
        @mcp.tool()
        def flag_vulnerable(file_path: str, reasoning: str) -> str:
            """Flag a file as containing/related to the vulnerability. Provide detailed reasoning."""
            s = self._get_session()
            if s.done:
                return "ERROR: Investigation is complete."
            if file_path not in s.files:
                return f"ERROR: '{file_path}' not found."
            if file_path in s.flagged:
                return f"Already flagged: {file_path}"
            if file_path in s.skipped:
                return f"Already skipped: {file_path}. Cannot change decision."

            if len(s.flagged) >= s.budget:
                return (f"OVER BUDGET: Cannot flag more files. "
                        f"You've used all {s.budget} flags. "
                        f"Consider submitting your report.")

            s.step_count += 1
            s.flagged.add(file_path)
            s.flag_reasons[file_path] = reasoning
            s.record_reasoning(file_path, reasoning)  # Track thinking quality
            s.thinking_cost += len(reasoning)
            is_bug = file_path in s.bugs

            status = "✓ CORRECT — this file IS vulnerable" if is_bug else "✗ INCORRECT — this file was safe"
            remaining = s.budget - len(s.flagged)

            undecided = len(s.files) - len(s.flagged) - len(s.skipped)
            return (
                f"FLAGGED: {file_path}\n"
                f"Result: {status}\n"
                f"Flags remaining: {remaining}/{s.budget} | "
                f"Files undecided: {undecided}\n"
                f"Tip: Use submit_report when ready to conclude."
            )

        # ── Tool: skip_file ──────────────────────────────
        @mcp.tool()
        def skip_file(file_path: str, reasoning: str) -> str:
            """Mark a file as safe / not related to the vulnerability. Provide reasoning."""
            s = self._get_session()
            if s.done:
                return "ERROR: Investigation is complete."
            if file_path not in s.files:
                return f"ERROR: '{file_path}' not found."
            if file_path in s.skipped:
                return f"Already skipped: {file_path}"
            if file_path in s.flagged:
                return f"Already flagged: {file_path}. Cannot change decision."

            s.step_count += 1
            s.skipped.add(file_path)
            s.record_reasoning(file_path, reasoning)  # Track thinking quality
            s.thinking_cost += len(reasoning)
            is_bug = file_path in s.bugs

            status = "✓ CORRECT — this file is safe" if not is_bug else "✗ MISSED — this file WAS vulnerable!"
            undecided = len(s.files) - len(s.flagged) - len(s.skipped)
            return (
                f"SKIPPED: {file_path}\n"
                f"Result: {status}\n"
                f"Files undecided: {undecided}\n"
            )

        # ── Tool: submit_report ──────────────────────────
        @mcp.tool()
        def submit_report(summary: str, confidence: str = "medium") -> str:
            """Submit your final triage report to end the investigation. Include what you found and why."""
            s = self._get_session()
            if s.done:
                return "ERROR: Investigation already complete."

            s.done = True
            s.report = summary
            s.step_count += 1

            # Auto-skip undecided files
            for fpath in list(s.files.keys()):
                if fpath not in s.flagged and fpath not in s.skipped:
                    s.skipped.add(fpath)

            # Compute metrics
            tp = len(s.flagged & s.bugs)
            fp = len(s.flagged - s.bugs)
            fn = len(s.bugs - s.flagged)
            tn = len(s.skipped - s.bugs)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            # Score the report quality
            report_score = self._score_report(summary, s)

            # Investigation efficiency
            max_steps = len(s.files) * 3
            efficiency = max(0, 1.0 - (s.step_count / max_steps))

            # Thinking budget efficiency (Qwen3 integration)
            # Measures whether the agent reasoned deeply on the RIGHT files
            thinking_eff = s.thinking_efficiency_score()

            # Compute composite reward (5 components)
            total_reward = (
                0.35 * f1 +                                        # Correct flag/skip decisions
                0.20 * report_score +                              # Report quality
                0.15 * efficiency +                                # Investigation efficiency
                0.15 * thinking_eff +                              # Thinking budget allocation
                0.15 * (1.0 if tp > 0 and fp == 0 else prec)      # Precision bonus
            )

            # Count deceptive traps triggered (agent flagged a safe-but-suspicious file)
            traps_triggered = len(s.flagged & s.deceptive)
            traps_avoided = len(s.deceptive - s.flagged)

            result = (
                f"{'='*60}\n"
                f"  INVESTIGATION COMPLETE — {s.episode['cve_id']}\n"
                f"{'='*60}\n"
                f"  Repository: {s.episode['repo']}\n"
                f"  CVSS: {s.episode.get('cvss', 0)}\n\n"
                f"  Vulnerability Detection:\n"
                f"    Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}\n"
                f"    TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}\n\n"
                f"  Investigation Quality:\n"
                f"    Files read: {len(s.reads)}/{len(s.files)}\n"
                f"    Searches: {len(s.searches)}\n"
                f"    Steps taken: {s.step_count}\n"
                f"    Report quality: {report_score:.2f}\n"
                f"    Efficiency: {efficiency:.2f}\n"
                f"    Thinking efficiency: {thinking_eff:.2f}\n"
                f"      Deep on bugs: {s.deep_thinks_on_bugs} | "
                f"Shallow on bugs: {s.shallow_on_bugs} | "
                f"Wasted deep: {s.deep_thinks_on_clean}\n"
                f"    Deceptive files: {traps_triggered} traps triggered, "
                f"{traps_avoided} correctly avoided"
                f" (of {len(s.deceptive)} total)\n"
                f"    Total thinking cost: {s.thinking_cost} chars\n\n"
                f"  TOTAL SCORE: {total_reward:.3f}\n"
                f"{'='*60}"
            )
            return result

        super().__init__(mcp)

    def _get_session(self) -> InvestigationSession:
        sid = self._current_session_id
        if sid not in self._sessions:
            raise RuntimeError("No active session. Call reset() first.")
        return self._sessions[sid]

    def _score_report(self, report, session):
        """Score the quality of the triage report."""
        if not report or len(report.strip()) < 20:
            return 0.0

        score = 0.0
        report_lower = report.lower()

        # Mentions CVE ID
        if session.episode["cve_id"].lower() in report_lower:
            score += 0.15

        # Mentions vulnerability type keywords
        vuln_keywords = ["buffer", "overflow", "injection", "xss", "privilege",
                         "escalation", "denial", "service", "bypass", "traversal",
                         "use-after-free", "integer", "authentication", "crypto"]
        matches = sum(1 for k in vuln_keywords if k in report_lower)
        score += min(0.2, matches * 0.05)

        # Mentions specific files
        files_mentioned = sum(1 for f in session.flagged if f.split('/')[-1].lower() in report_lower)
        if files_mentioned > 0:
            score += 0.2

        # Provides reasoning (length heuristic)
        if len(report) > 100:
            score += 0.15
        if len(report) > 300:
            score += 0.1

        # Mentions code-level details
        code_terms = ["function", "variable", "parameter", "return", "input",
                      "validation", "check", "boundary", "size", "length"]
        code_matches = sum(1 for t in code_terms if t in report_lower)
        score += min(0.2, code_matches * 0.05)

        return min(1.0, score)

    def reset(self, seed=None, episode_id=None, difficulty=None, **kwargs) -> Observation:
        if seed is not None:
            random.seed(seed)

        difficulty = str(difficulty).lower() if difficulty else random.choice(["easy", "medium", "hard"])
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"

        size_fn = {
            "easy": lambda e: len(e["files"]) <= 15,
            "medium": lambda e: 15 < len(e["files"]) < 30,
            "hard": lambda e: len(e["files"]) >= 30,
        }[difficulty]

        candidates = [e for e in BUGGY_EPISODES if size_fn(e)]
        if not candidates:
            candidates = BUGGY_EPISODES or EPISODES
        ep = random.choice(candidates)

        # Inject synthetic bugs if needed
        if ep["total_bugs"] == 0:
            ep = dict(ep)
            files = [dict(f) for f in ep["files"]]
            n = max(1, len(files) // 5)
            for idx in random.sample(range(len(files)), min(n, len(files))):
                files[idx]["label"] = 1
            ep["files"] = files
            ep["total_bugs"] = sum(1 for f in files if f["label"] == 1)

        # ── Inject deceptive files ──────────────────────────────
        # Deceptive files are SAFE files with artificially high risk
        # features (high churn, high complexity).  They look like
        # bugs to any policy that thresholds on features alone.
        # The agent MUST actually read and reason about the code.
        n_safe = sum(1 for f in ep["files"] if f["label"] == 0)
        n_deceptive = max(1, n_safe // 5)  # ~20% of safe files become traps
        safe_indices = [i for i, f in enumerate(ep["files"]) if f["label"] == 0
                       and not f.get("is_test_file")]
        if safe_indices:
            for idx in random.sample(safe_indices, min(n_deceptive, len(safe_indices))):
                f = ep["files"][idx]
                # Inflate features to look suspicious
                orig_feat = list(f.get("features", [0, 0, 0, 0]))
                f["features"] = [
                    max(orig_feat[0], random.randint(60, 90)),   # high churn
                    max(orig_feat[1], random.randint(55, 85)),   # high complexity
                    orig_feat[2],
                    max(orig_feat[3], random.randint(80, 100)),  # recently modified
                ]
                f["is_deceptive"] = True

        random.shuffle(ep["files"])
        budget = min(len(ep["files"]), max(ep["total_bugs"] * 2 + 3, 5))

        sid = episode_id or str(uuid.uuid4())
        self._current_session_id = sid
        self._sessions[sid] = InvestigationSession(ep, budget)

        # Build file listing
        file_list = []
        for f in ep["files"]:
            feat = f.get("features", [0, 0, 0, 0])
            file_list.append(
                f"  • {f['file']}  [{f.get('file_language','?')}] "
                f"complexity={feat[1]}, churn={feat[0]}"
            )

        context = (
            f"{'='*60}\n"
            f"  SECURITY INVESTIGATION BRIEFING\n"
            f"{'='*60}\n"
            f"  CVE: {ep['cve_id']} (CVSS: {ep.get('cvss', 0)})\n"
            f"  Repository: {ep['repo']}\n"
            f"  Description: {ep.get('cve_description', 'N/A')}\n\n"
            f"  Files in patch ({len(ep['files'])}):\n"
            + "\n".join(file_list) + "\n\n"
            f"  Your budget: {budget} flags\n"
            f"  Investigation points: {len(ep['files']) * 2}\n\n"
            f"  MISSION: Investigate which files contain or are related to\n"
            f"  the vulnerability. Use tools to read code, search patterns,\n"
            f"  then flag/skip files. Submit a triage report when done.\n"
            f"{'='*60}"
        )

        return Observation(done=False, reward=None, metadata={"context": context})

    def _step_impl(self, action, timeout_s=None, **kwargs) -> Observation:
        # Non-MCP actions fall through here
        s = self._get_session()
        return Observation(
            done=s.done,
            reward=None,
            metadata={"message": "Use the MCP tools (read_file, search_code, flag_vulnerable, skip_file, submit_report)."}
        )

    @property
    def state(self):
        from openenv.core.env_server import State
        s = self._sessions.get(self._current_session_id)
        if not s:
            return State()
        return State(
            episode_id=self._current_session_id,
            step_count=s.step_count,
        )
