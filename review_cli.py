"""Run the same bounded review engine used by the UI, with inspectable exports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from review_engine import ReviewBudget, parse_patch, review_patch, verify_audit

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='checkout',help='Bundled sample ID: checkout, checkout_fixed, tenant')
    parser.add_argument('--input',type=Path,help='JSON or === path === source bundle; never executed')
    parser.add_argument('--mode',choices=['ollama','offline'],default='offline')
    parser.add_argument('--strategy',choices=['adaptive','uniform'],default='adaptive')
    parser.add_argument('--tokens',type=int,default=2400)
    parser.add_argument('--max-files',type=int,default=12)
    parser.add_argument('--input-chars',type=int,default=24000)
    parser.add_argument('--timeout',type=int,default=60)
    parser.add_argument('--run-seconds',type=int,default=240)
    parser.add_argument('--fault',choices=['none','timeout','malformed','evidence'],default='none')
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/review.json')
    parser.add_argument('--verify',type=Path,help='Verify an existing report audit chain, without inference')
    args=parser.parse_args()
    if args.verify:
        report=json.loads(args.verify.read_text())
        if not verify_audit(report.get('events',[])):
            parser.exit(2,'Audit verification failed.\n')
        print('Audit chain verified; authenticity requires a trusted retained hash.')
        return
    if args.input:
        source=args.input.read_text()
    else:
        cases=json.loads((ROOT/'examples/review_cases.json').read_text())
        case=next((c for c in cases if c['id']==args.case),None)
        if not case:
            parser.error('Unknown case')
        source=json.dumps(case)
    try:
        files=parse_patch(source)
        budget=ReviewBudget(args.tokens,args.input_chars,args.max_files,args.timeout,args.run_seconds)
        report=review_patch(files,budget,mode=args.mode,strategy=args.strategy,fault=args.fault,
            on_progress=lambda r: print(f"{r['files'][-1]['path']}: {r['files'][-1]['status']}",flush=True))
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'summary':report['summary'],'usage':report['usage']},indent=2))
    print(f'Audit: {args.output}')


if __name__=='__main__':
    main()
