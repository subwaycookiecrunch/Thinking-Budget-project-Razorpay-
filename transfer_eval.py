#!/usr/bin/env python3
"""Structural-prioritization illustration on five authored transfer scenarios.
No trained model, sampled reasoning lengths, or independent held-out claim.
"""
import argparse
import json
from pathlib import Path
from benchmark import ROOT, run_benchmark, write_results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--episodes', type=Path, default=ROOT / 'data' / 'transfer_episodes.json')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--metrics', type=Path, default=ROOT / 'grpo_output' / 'transfer_metrics.json')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    episodes = [{**e, 'episode_id': e['task_id']} for e in json.loads(args.episodes.read_text())]
    result = run_benchmark(episodes=episodes, snippets={}, seed=args.seed, bootstrap_samples=1000)
    result['evidence_type'] = 'authored_transfer_metadata_prioritization'
    result['cost_unit'] = 'Selected-file counts only; source unavailable in this metadata fixture'
    result['dataset']['split_status'] = 'Five hand-authored scenarios seen during development; not independent held-out data'
    result['limitations'].append('Summaries and labels are excluded from policy inputs; no source code is read in this metadata-only transfer suite.')
    # Dataset provenance must identify the file actually evaluated.
    import hashlib
    result['source_sha256'] = {str(args.episodes.relative_to(ROOT)) if args.episodes.is_relative_to(ROOT) else str(args.episodes): hashlib.sha256(args.episodes.read_bytes()).hexdigest()}
    output = args.metrics
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = result.pop('episodes')
    rows_path = output.with_name('transfer_episodes.jsonl')
    rows_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    result['episode_artifact'] = rows_path.name
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {output}: metadata coverage only, no model inference or bug-detection measurement.')


if __name__ == '__main__':
    main()
