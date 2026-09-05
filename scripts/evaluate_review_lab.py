"""Execute the actual review lab across fixtures and matched file allowances.

Ground truth is used only after each inference run. File-level localization is
not semantic adjudication: a finding can name the right file for a wrong reason.
The full findings are retained so a reviewer can check that distinction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from review_engine import MODEL, ReviewBudget, SourceFile, review_patch


def score_run(report, annotations):
    targets={f['path'] for f in annotations if f.get('expected_issue')}
    flagged={f['path'] for f in report['files'] if f['status']=='flag'}
    tp,fp,fn=len(targets & flagged),len(flagged-targets),len(targets-flagged)
    return {'target_files_flagged':tp,'control_files_flagged':fp,'target_files_not_flagged':fn,
            'file_localization_precision':tp/(tp+fp) if tp+fp else None,
            'file_localization_recall':tp/(tp+fn) if tp+fn else None,
            'expected_issues':{f['path']:f['expected_issue'] for f in annotations if f.get('expected_issue')},
            'missed_target_files':sorted(targets-flagged), 'flagged_control_files':sorted(flagged-targets)}


def run_matrix(cases,mode,strategies,file_limits,tokens,output,timeout=60):
    result={'schema_version':1,'evidence_type':'executed_review_lab_fixture_matrix',
            'mode':mode,'model':MODEL if mode=='ollama' else 'deterministic-rules-v1',
            'protocol':'Same fixture source, sorted path input order, output allowance and engine; vary routing and max_files.',
            'source_order':'Lexicographic file path order, as in a directory listing.',
            'limitations':['Hand-authored development fixtures, not a held-out test set.',
                           'Localization metrics only: inspect findings to determine whether the intended defect was identified.',
                           'Duplicate safe controls across checkout/fixed fixtures are not independent examples.',
                           'Runs are sequential with model caching; timing is descriptive, not a controlled latency comparison.'],
            'engine_sha256':hashlib.sha256((ROOT/'review_engine.py').read_bytes()).hexdigest(),
            'fixture_sha256':hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
            'output_token_allowance':tokens,'runs':[]}
    output.parent.mkdir(parents=True,exist_ok=True)
    for limit in file_limits:
        for strategy in strategies:
            for case in cases:
                files=[SourceFile(f['path'],f['content']) for f in sorted(case['files'],key=lambda f:f['path'])]
                report=review_patch(files,ReviewBudget(output_tokens=tokens,max_files=limit,timeout_seconds=timeout),mode=mode,strategy=strategy)
                entry={'case_id':case['id'],'strategy':strategy,'max_files':limit,
                       'score':score_run(report,case['files']),'report':report}
                result['runs'].append(entry)
                # Save every completed run, including failures, before continuing.
                output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
                print(case['id'],strategy,limit,report['summary'],flush=True)
    result['complete']=True
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['offline','ollama'],default='offline')
    p.add_argument('--strategies',nargs='+',choices=['adaptive','uniform'],default=['adaptive','uniform'])
    p.add_argument('--file-limits',nargs='+',type=int,default=[2,12])
    p.add_argument('--tokens',type=int,default=2400)
    p.add_argument('--timeout',type=int,default=60)
    p.add_argument('--output',type=Path,default=ROOT/'grpo_output/review_lab_evaluation.json')
    a=p.parse_args()
    cases=json.loads((ROOT/'examples/review_cases.json').read_text())
    run_matrix(cases,a.mode,a.strategies,a.file_limits,a.tokens,a.output,a.timeout)
    print(f'Results: {a.output}')


if __name__=='__main__':
    main()
