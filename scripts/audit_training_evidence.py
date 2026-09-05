#!/usr/bin/env python3
"""Audit saved training artifacts without loading model code or changing weights."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import struct

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory: Path) -> dict:
    config_path = directory / 'adapter_config.json'
    weights_path = directory / 'adapter_model.safetensors'
    state_path = directory / 'checkpoint-100' / 'trainer_state.json'
    traces_path = directory / 'trace_log.jsonl'
    config = json.loads(config_path.read_text())
    with weights_path.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if header_size > 16_000_000:
            raise ValueError('Invalid safetensors header size')
        header = json.loads(stream.read(header_size))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    state = json.loads(state_path.read_text())
    logs = [r for r in state.get('log_history', []) if 'reward' in r]
    traces = [json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()]
    rewards = [row['reward'] for row in logs]
    split = len(rewards) // 3
    early, late = rewards[:split], rewards[2 * split:]
    stats = json.loads((directory / 'training_stats.json').read_text())
    sources = [config_path, weights_path, state_path, traces_path, directory / 'training_stats.json']
    return {
        'evidence_type': 'saved_training_artifact_audit',
        'model': config['base_model_name_or_path'], 'adapter_present': True,
        'adapter_tensor_count': len(tensors), 'adapter_bytes': weights_path.stat().st_size,
        'adapter_dtypes': dict(Counter(t['dtype'] for t in tensors.values())),
        'trainer_global_step': state['global_step'], 'logged_reward_steps': len(logs),
        'trainer_input_tokens_seen': state.get('num_input_tokens_seen'),
        'mean_logged_reward': statistics.mean(rewards),
        'early_mean': statistics.mean(early), 'late_mean': statistics.mean(late),
        'early_step_range': [logs[0]['step'], logs[split - 1]['step']],
        'late_step_range': [logs[2 * split]['step'], logs[-1]['step']],
        'early_to_late_reward_delta': statistics.mean(late) - statistics.mean(early),
        'reported_training_mean_matches_log': abs(stats['mean_reward'] - statistics.mean(rewards)) < 1e-7,
        'mean_completion_clipped_ratio': statistics.mean(r['completions/clipped_ratio'] for r in logs),
        'reward_traces': len(traces),
        'traces_with_environment_score': sum(t.get('env_score') is not None for t in traces),
        'traces_with_no_budget_prediction': sum(t.get('metacog', {}).get('n_predictions', 0) == 0 for t in traces),
        'traces_with_zero_coupling': sum(t.get('metacog', {}).get('coupling') == 0 for t in traces),
        'traces_with_zero_metacognitive_score': sum(t.get('metacog_score') == 0 for t in traces),
        'interpretation': [
            'Adapter tensors and optimizer-step logs support that a training run was saved.',
            'Training reward is not held-out performance or causal proof of learned allocation.',
            'Missing environment scores and zero action coupling limit conclusions about tool-use learning.',
            'Budget labels and text lengths are surface diagnostics, not access to internal cognition.',
            'Hardware type and training wall time are not independently verified by this audit.'
        ],
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
    }


def plot(directory, output):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.loads((directory / 'checkpoint-100' / 'trainer_state.json').read_text())['log_history']
    rows = [r for r in rows if 'reward' in r]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([r['step'] for r in rows], [r['reward'] for r in rows], color='#0d9488', alpha=.7)
    axes[0].set(title='Actual saved training rewards', xlabel='Optimizer step', ylabel='Logged training reward')
    axes[1].plot([r['step'] for r in rows], [r['completions/clipped_ratio'] for r in rows], color='#d97706', alpha=.7)
    axes[1].set(title='Completions reaching generation cap', xlabel='Optimizer step', ylabel='Clipped completion fraction', ylim=(0, 1.03))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Qwen2.5-1.5B LoRA · saved checkpoint evidence', fontweight='bold')
    fig.text(.5, .015, 'Training diagnostics only. No held-out efficacy or GRPO improvement established.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, default=ROOT / 'grpo_output')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    result = audit(args.directory)
    output = args.directory / 'training_evidence_audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    if not args.no_plot:
        plot(args.directory, args.directory / 'training_curves.png')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
