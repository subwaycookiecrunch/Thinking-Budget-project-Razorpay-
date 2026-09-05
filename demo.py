"""
CodeReviewEnv v3 — Demo: Multi-Tool Security Investigation
===========================================================
Shows how different agent strategies interact with the
MCP-based environment using tools to read code, search,
and triage CVE vulnerabilities.
"""
# NOTE: This demo uses heuristic agent strategies to demonstrate the environment.
# The smart-investigator agent approximates the behavior learned by GRPO training.
# For actual trained model inference, see inference.py with a GPU-trained checkpoint.

import sys
import os
import re
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from openenv.core.env_server import CallToolAction, ListToolsAction
from code_review_env.server.environment import CodeReviewEnvironment


def extract_text(obs):
    """Extract plain text from MCP observation."""
    if hasattr(obs, 'result') and obs.result:
        r = obs.result
        if hasattr(r, 'data'):
            return str(r.data)
        if hasattr(r, 'content') and r.content:
            return str(r.content[0].text) if r.content else str(r)
        return str(r)
    if hasattr(obs, 'metadata') and obs.metadata:
        return obs.metadata.get('context', str(obs))
    return str(obs)


def extract_files_from_context(context_text):
    """Parse file paths from the briefing context."""
    return re.findall(r'• (.+?)\s+\[', context_text)


def call_tool(env, name, args=None):
    """Call an MCP tool and return the text result."""
    obs = env.step(CallToolAction(tool_name=name, arguments=args or {}))
    return extract_text(obs)


# ── Agent: Blind Skip (worst baseline) ──────────────────────────
def heuristic_agent_blind_skip(env, files):
    """Skips everything without reading. Should score terribly."""
    for f in files:
        call_tool(env, "skip_file", {"file_path": f, "reasoning": "skipping"})
    return call_tool(env, "submit_report", {
        "summary": "Skipped everything.",
        "confidence": "low"
    })


# ── Agent: Flag Everything (dumb baseline) ──────────────────────
def heuristic_agent_flag_all(env, files):
    """Flags everything without reading. Wastes budget."""
    for f in files:
        result = call_tool(env, "flag_vulnerable", {
            "file_path": f,
            "reasoning": "flagging everything"
        })
        if "OVER BUDGET" in result:
            call_tool(env, "skip_file", {"file_path": f, "reasoning": "out of budget"})

    return call_tool(env, "submit_report", {
        "summary": "Flagged as many files as budget allowed.",
        "confidence": "low"
    })


# ── Agent: Read-Then-Decide (smart heuristic) ──────────────────
def heuristic_agent_smart_investigator(env, files, cve_desc):
    """Reads code, searches for patterns, then makes informed decisions."""

    # Step 1: Search for vulnerability-related patterns
    vuln_keywords = ["overflow", "buffer", "malloc", "free", "input",
                     "user", "copy", "unsafe", "inject", "query",
                     "exec", "eval", "system", "privilege", "auth"]
    
    suspicious_files = set()
    for kw in vuln_keywords[:3]:  # limit searches to save budget
        result = call_tool(env, "search_code", {"pattern": kw})
        if "Found" in result:
            for m in re.findall(r'• (.+?)\s+\(', result):
                suspicious_files.add(m)

    # Step 2: Read suspicious files first
    read_files = set()
    risky_files = set()
    
    for f in list(suspicious_files)[:5]:
        code = call_tool(env, "read_file", {"file_path": f})
        read_files.add(f)
        if any(w in code.lower() for w in ["bug", "overflow", "unsafe", "vuln", "todo"]):
            risky_files.add(f)

    # Step 3: Read any unread files with high complexity
    for f in files:
        if f not in read_files and len(read_files) < len(files) * 0.6:
            code = call_tool(env, "read_file", {"file_path": f})
            read_files.add(f)
            if any(w in code.lower() for w in ["bug", "overflow", "free", "input", "user"]):
                risky_files.add(f)

    # Step 4: Flag risky files, skip safe ones
    flagged = []
    for f in files:
        if f in risky_files:
            result = call_tool(env, "flag_vulnerable", {
                "file_path": f,
                "reasoning": f"Code contains patterns matching vulnerability indicators. "
                             f"File was identified through code analysis and pattern search."
            })
            if "OVER BUDGET" not in result:
                flagged.append(f)
            else:
                call_tool(env, "skip_file", {"file_path": f, "reasoning": "budget exhausted"})
        elif f not in risky_files:
            call_tool(env, "skip_file", {
                "file_path": f,
                "reasoning": "No vulnerability indicators found in code review."
            })

    # Step 5: Submit detailed report
    report = (
        f"Security Triage Report for {cve_desc[:80]}\n"
        f"Files investigated: {len(read_files)}/{len(files)}\n"
        f"Searches performed: pattern matching for vulnerability indicators\n"
        f"Files flagged: {', '.join(flagged) if flagged else 'none'}\n"
        f"Assessment: Based on code analysis, the vulnerability appears to involve "
        f"unsafe input handling and insufficient validation in the flagged files."
    )
    return call_tool(env, "submit_report", {"summary": report, "confidence": "medium"})


# ── Main Demo ───────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  CodeReviewEnv v3 — Multi-Tool Security Investigation Demo")
    print("=" * 70)

    agents = {
        "blind-skip": heuristic_agent_blind_skip,
        "flag-all": heuristic_agent_flag_all,
        "smart-investigator": heuristic_agent_smart_investigator,
    }

    results = {}
    for name, agent_fn in agents.items():
        env = CodeReviewEnvironment()
        obs = env.reset(seed=42, difficulty="easy")
        context = obs.metadata["context"]
        files = extract_files_from_context(context)

        # Extract CVE description
        desc_match = re.search(r'Description: (.+)', context)
        cve_desc = desc_match.group(1) if desc_match else ""

        print(f"\n{'─' * 70}")
        print(f"  Agent: {name}")
        print(f"{'─' * 70}")
        
        # Show briefing for first agent only
        if name == "blind-skip":
            print(context[:600])
            print("  ...")

        if name == "smart-investigator":
            result = agent_fn(env, files, cve_desc)
        else:
            result = agent_fn(env, files)

        # Parse scores from result
        f1_m = re.search(r'F1: ([\d.]+)', result)
        score_m = re.search(r'TOTAL SCORE: ([\d.]+)', result)
        report_m = re.search(r'Report quality: ([\d.]+)', result)

        f1 = float(f1_m.group(1)) if f1_m else 0.0
        total = float(score_m.group(1)) if score_m else 0.0
        report_q = float(report_m.group(1)) if report_m else 0.0

        results[name] = {"f1": f1, "total": total, "report": report_q}
        print(f"  F1: {f1:.3f} | Total Score: {total:.3f} | Report: {report_q:.2f}")
        
        if name == "smart-investigator":
            print(f"\n  Full result:\n{result}")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'AGENT COMPARISON':^70}")
    print(f"{'=' * 70}")
    print(f"{'Agent':<25} {'F1':>6} {'Total':>8} {'Report':>8}")
    print(f"{'─' * 70}")
    for name, r in results.items():
        print(f"{name:<25} {r['f1']:>6.3f} {r['total']:>8.3f} {r['report']:>8.2f}")
    
    print(f"\n{'=' * 70}")
    print("  Key insight: The smart-investigator agent reads code and")
    print("  searches patterns before making decisions, producing better")
    print("  F1 scores and higher-quality reports than blind strategies.")
    print("  An LLM trained with GRPO would learn even better investigation")
    print("  strategies — reading the RIGHT files, flagging the RIGHT code.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
