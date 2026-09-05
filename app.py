"""Local review lab. Importing the app never starts training or inference."""
from __future__ import annotations
import html
import json
import os
from pathlib import Path
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr
from review_engine import MODEL, ReviewBudget, ollama_status, parse_patch, review_patch, route_file

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "grpo_output"
EXPORTS = ROOT / ".cache" / "reviews"


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


CASES = load_json(ROOT / "examples/review_cases.json", [])


def load_case(case_id):
    case = next((c for c in CASES if c["id"] == case_id), None)
    if not case:
        return "", "Paste source below. Files are never executed."
    return "\n\n".join(f"=== {f['path']} ===\n{f['content'].rstrip()}" for f in case["files"]), case["description"]


def model_status_html():
    s = ollama_status()
    return f'<div class="runtime"><i class="status-dot {"ready" if s["ready"] else ""}"></i>{html.escape(s["message"])} <span class="mono">{html.escape(MODEL)}</span></div>'


def render_plan(text, tokens=2400, max_files=12, input_chars=24000, strategy="adaptive"):
    try:
        files = parse_patch(text)
    except ValueError as exc:
        return f'<div class="plan-note">{html.escape(str(exc))}</div>'
    pairs = [(f, route_file(f)) for f in files]
    if strategy == "adaptive":
        pairs.sort(key=lambda p: (-p[1]["risk_score"], p[0].path))
    remaining, used_chars, accepted, bars = int(tokens), 0, 0, []
    uniform = max(256, int(tokens) // min(len(files), int(max_files)))
    for f, r in pairs:
        cap = min(remaining, r["requested_tokens"] if strategy == "adaptive" else uniform)
        allowed = cap >= 256 and accepted < int(max_files) and used_chars + len(f.content) <= int(input_chars)
        if allowed:
            remaining -= cap
            used_chars += len(f.content)
            accepted += 1
        bars.append(f'<div class="allocation-row"><span class="allocation-path">{html.escape(f.path)}</span><div class="bar-track"><i style="width:{min(100, cap / 768 * 100) if allowed else 2}%" class="{r["tier"] if allowed else "deferred"}"></i></div><span class="mono allocation-value">{str(cap) + " max" if allowed else "deferred"}</span></div>')
    return '<div class="allocation"><div class="eyebrow">OUTPUT ALLOCATION · PREVIEW</div>' + ''.join(bars) + '<p class="plan-note">Worst-case reservation. Unused tokens return to the pool. Routing is deterministic, not learned.</p></div>'


def metric_cards(items):
    return '<div class="metric-grid">' + ''.join(f'<div class="metric"><span>{a}</span><strong>{b}</strong><small>{c}</small></div>' for a,b,c in items) + '</div>'


EMPTY_METRICS = metric_cards([("FINDINGS", "—", "Source-cited candidates"), ("REVIEWED", "—", "Files actually analyzed"), ("OUTPUT TOKENS", "—", "Reported by local model"), ("HUMAN REVIEW", "—", "Unresolved work stays visible")])


def render_metrics(report):
    s,u = report["summary"],report["usage"]
    output_label = f'{u["output_tokens_reported"]:,}'
    if u["unknown_usage_calls"]:
        output_label += ' known'
    cards = metric_cards([("FINDINGS", str(s["findings"]), "Source-cited candidates"),
        ("REVIEWED", f'{s["reviewed_files"]}/{s["total_files"]}', "Files actually analyzed"),
        ("OUTPUT TOKENS", output_label if report["mode"] == "ollama" else "No AI", "Unknown usage charged at full cap" if u["unknown_usage_calls"] else "Provider-reported · not estimated"),
        ("HUMAN REVIEW", str(s["needs_review"]), "Deferred, failed, or uncertain")])
    scope = "Local AI review" if report["mode"] == "ollama" else "Offline rules · no model inference"
    status = "Limits respected" if s["budget_respected"] else "Provider limit violation · stopped"
    return cards + f'<div class="run-strip">{scope} · {status} · {report["duration_ms"]/1000:.1f}s · <span class="mono">run {report["run_id"]}</span></div>'


def render_findings(report):
    cards=[]
    for file in report["files"]:
        for finding in file["findings"]:
            e={k:html.escape(str(v)) for k,v in finding.items()}
            cards.append(f'<article class="finding"><div class="finding-top"><span class="severity {e["severity"]}">{e["severity"]}</span><span class="mono">{html.escape(file["path"])}:{e["line"]}</span><span class="evidence-tag">QUOTE VERIFIED</span></div><h3>{e["title"]}</h3><p>{e["explanation"]}</p><pre>{e["quote"]}</pre></article>')
    if not cards:
        cards.append('<div class="empty-state"><b>No validated findings in this run.</b><p>Check coverage and unresolved work below. An empty result does not certify safety.</p></div>')
    unresolved=[f for f in report["files"] if f["status"] in {"deferred","needs_review"}]
    if unresolved:
        cards.append('<div class="handoff"><div class="eyebrow">HUMAN REVIEW REQUIRED</div>' + ''.join(f'<p><b>{html.escape(f["path"])}</b><br>{html.escape(f["summary"])}</p>' for f in unresolved) + '</div>')
    return ''.join(cards)


def file_table(report):
    statuses={"flag":"Finding","no_finding":"No finding","needs_review":"Needs review","deferred":"Deferred"}
    return [[f["path"],statuses[f["status"]],f["tier"],f["output_cap"] if report["mode"]=="ollama" else 0,f["reported_tokens"],f["source_chars"]," · ".join(f["signals"])] for f in report["files"]]


def handoff_markdown(report):
    lines=["# The Thinking Budget — review handoff", "", f"Run: {report['run_id']} · Mode: {report['mode']} · Model: {report['model']}", f"Patch SHA-256: {report['patch_hash']}", "", "Findings are candidates, not confirmed vulnerabilities. No file is automatically approved.", ""]
    for file in report["files"]:
        lines += [f"## {file['path']} — {file['status']}", "", file["summary"], ""]
        for f in file["findings"]:
            lines += [f"- Line {f['line']} [{f['severity']}]: {f['title']}", f"  {f['explanation']}", ""]
    lines += ["## Usage", "", "```json", json.dumps(report["usage"], indent=2), "```", "", "## Limits", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines)


def run_review(text, mode, tokens, max_files, input_chars, strategy, fault, progress=gr.Progress()):
    try:
        files=parse_patch(text)
        budget=ReviewBudget(output_tokens=int(tokens), max_files=int(max_files), input_chars=int(input_chars))
    except (ValueError,TypeError) as exc:
        raise gr.Error(str(exc)) from exc
    progress(0,desc="Validating source and reserving budget")
    report=review_patch(files,budget,mode=mode,strategy=strategy,fault=fault,
        on_progress=lambda r: progress(len(r["files"])/len(files),desc=f'Reviewed {r["files"][-1]["path"]}'))
    EXPORTS.mkdir(parents=True,exist_ok=True)
    jp,mp=EXPORTS/f'{report["run_id"]}.json',EXPORTS/f'{report["run_id"]}.md'
    jp.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    mp.write_text(handoff_markdown(report))
    return render_metrics(report),render_findings(report),file_table(report),report,str(jp),str(mp)


def benchmark_md():
    d=load_json(RESULTS/"benchmark_results.json")
    if not d:
        return "Run `python benchmark.py` to generate prioritization evidence."
    lines=[f'### {d["dataset"]["episodes"]} episodes. Every file counted.', '', 'Executed deterministic prioritization on synthetic data. **Coverage measures positive-labeled files read, not defects detected.** These are not LLM results.', '', '| Policy at 50% read allowance | Files read | Positive-file coverage | Missed positives | Safe files read |','|---|---:|---:|---:|---:|']
    for r in d["policies"]:
        if r["feature_condition"]=="original_features" and (r["budget_fraction"]==.5 or r["policy"]=="exhaustive"):
            ci=r["coverage_recall_ci95"]
            lines.append(f'| {r["policy"].replace("_"," ")} | {r["reads_used"]:,} | {r["coverage_recall"]:.1%} ({ci[0]:.1%}–{ci[1]:.1%}) | {r["missed_positives"]} | {r["safe_reviews"]} |')
    lines += ['', '**Limits:** synthetic features may encode dataset construction; some snippets do not faithfully implement the labeled defect. All bundled episodes include training data. Shuffled features test dependence on those signals.', '', 'Reproduce: `python benchmark.py` · Row-level audit: `grpo_output/benchmark_episodes.jsonl`']
    return '\n'.join(lines)


def live_evidence_md():
    r=load_json(RESULTS/"live_review_checkout.json")
    if not r or r.get("mode")!="ollama":
        return 'No captured model run yet. Use **Review lab → Local AI** to create real inference evidence.'
    s,u=r["summary"],r["usage"]
    return f'### Captured local model run\n\n**{r["model"]}**, pretrained; no project fine-tuning. Checkout fixture: **{s["reviewed_files"]}/{s["total_files"]} files reviewed**, **{s["findings"]} validated finding candidates**, **{s["needs_review"]} unresolved files**.\n\nProvider reported **{u["output_tokens_reported"]:,} output tokens**, **{u["input_tokens_reported"]:,} input tokens**; wall time **{r["duration_ms"]/1000:.1f}s**. A small synthetic fixture demonstrates execution, not production accuracy or RL gains.\n\nA valid quote proves the cited line exists, not that the interpretation is correct.'


def run_failure_lab(kind):
    source,_=load_case("checkout")
    r=review_patch(parse_patch(source),mode="offline",fault=kind)
    return render_metrics(r),render_findings(r),r


THEME=gr.themes.Base(primary_hue="teal",secondary_hue="slate",neutral_hue="slate",font=["Inter","ui-sans-serif","system-ui"],font_mono=["ui-monospace","monospace"])
# Keep our light product palette consistent even when the OS requests dark mode.
_palette=THEME.to_dict()["theme"]
THEME.set(**{key:_palette[key[:-5]] for key in _palette if key.endswith("_dark") and key[:-5] in _palette})
with gr.Blocks(title="The Thinking Budget · Review Lab",analytics_enabled=False) as app:
    gr.HTML('''<div class="masthead"><div class="brand"><span class="brand-mark">tb</span> THE THINKING BUDGET</div><div class="edition">OPEN TRACK / BUILD 01</div></div><section class="hero"><div class="eyebrow">CODE REVIEW, WITH A COMPUTE BUDGET</div><h1>Spend compute where<br><em>failure costs more.</em></h1><p>Prioritize a patch. Let local AI inspect the risky files. Keep every limit,<br class="desktop-break"> finding, and unfinished review visible.</p><div class="hero-tags"><span>Local inference</span><span>Source-cited findings</span><span>Explicit human handoff</span></div></section>''')
    runtime=gr.HTML('<div class="runtime">Checking local model availability…</div>')
    with gr.Tabs():
        with gr.Tab("Review lab"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5,min_width=340):
                    gr.Markdown("### 01 / Load a patch")
                    picker=gr.Dropdown(choices=[(c["title"],c["id"]) for c in CASES],value="checkout",label="Sample case")
                    initial_source,initial_description=load_case("checkout")
                    case_description=gr.Markdown(initial_description,elem_classes=["subtle"])
                    source_input=gr.Textbox(value=initial_source,label="Source files · editable",lines=15,max_lines=25,elem_id="source-editor")
                    gr.Markdown("File blocks: `=== path/to/file.py ===`, or JSON with `path` and `content`. Source is never executed.",elem_classes=["subtle"])
                with gr.Column(scale=4,min_width=320):
                    gr.Markdown("### 02 / Set your limits")
                    mode=gr.Radio(choices=[("Local AI · Qwen","ollama"),("Offline · static rules","offline")],value="ollama",label="Reviewer")
                    tokens=gr.Slider(256,8192,value=2400,step=128,label="Total output-token allowance",info="Enforced per request. Input characters have a separate cap.")
                    with gr.Row():
                        max_files=gr.Slider(1,24,value=12,step=1,label="Max files to review")
                        input_chars=gr.Slider(512,80000,value=24000,step=512,label="Source-character allowance")
                    strategy=gr.Radio(choices=[("Risk-prioritized","adaptive"),("Uniform · input order","uniform")],value="adaptive",label="Allocation strategy")
                    allocation=gr.HTML(render_plan(initial_source))
                    with gr.Accordion("Inject a failure",open=False):
                        fault=gr.Radio(choices=[("None","none"),("Timeout once","timeout"),("Bad JSON once","malformed"),("False citation once","evidence")],value="none",label="First review only · explicitly simulated")
                    run_button=gr.Button("Review this patch  →",variant="primary",size="lg")
                    gr.Markdown("No AI available? Choose offline rules. Model failures never silently become successful AI reviews.",elem_classes=["subtle"])
            gr.HTML('<div class="section-divider"><span>03 / REVIEW RESULTS</span><span>Evidence first. Human decision last.</span></div>')
            metrics=gr.HTML(EMPTY_METRICS)
            findings=gr.HTML('<div class="empty-state"><b>Your review starts here.</b><p>Load a patch, set a budget, and run. Actual findings and unresolved work appear here.</p></div>')
            coverage=gr.Dataframe(headers=["File","Outcome","Allocation","Output cap","Actual output tokens","Source chars","Routing signals"],datatype=["str","str","str","number","number","number","str"],interactive=False,label="File coverage · no silent skips",wrap=True)
            with gr.Row():
                json_download=gr.File(label="Complete audit · JSON",interactive=False)
                md_download=gr.File(label="Reviewer handoff · Markdown",interactive=False)
            with gr.Accordion("Inspect the hash-linked audit",open=False):
                audit=gr.JSON(label="Usage, failures, evidence and routing")
                gr.Markdown("Hashes detect changes relative to a retained record. This is not a signed external audit log.",elem_classes=["subtle"])
            picker.change(load_case,[picker],[source_input,case_description])
            for control in (source_input,tokens,max_files,input_chars,strategy):
                control.change(render_plan,[source_input,tokens,max_files,input_chars,strategy],[allocation])
            run_button.click(run_review,[source_input,mode,tokens,max_files,input_chars,strategy,fault],[metrics,findings,coverage,audit,json_download,md_download],concurrency_limit=1,concurrency_id="model_review",api_name="review_patch")
        with gr.Tab("Evidence"):
            gr.Markdown("## Show the work, including the limits\n\nActual model execution and deterministic prioritization are separate evidence types.")
            captured=gr.Markdown(live_evidence_md())
            gr.Button("Refresh captured evidence",size="sm").click(live_evidence_md,outputs=[captured])
            gr.Markdown(benchmark_md())
            figure=RESULTS/"benchmark_pareto.png"
            gr.Image(value=str(figure) if figure.exists() else None,label="Coverage vs file-read allowance · heuristic experiment")
            with gr.Accordion("What the old numbers cannot prove",open=False):
                gr.Markdown("Original 6× thinking ratio, perfect F1, calibration charts, and tag-removal ablations used scripted or label-dependent behavior. They do not prove trained-model improvement. Truncating a saved trace does not rerun a model. These claims are excluded from submission evidence. See `docs/BRUTAL_ASSESSMENT.md`.")
        with gr.Tab("Failure lab"):
            gr.Markdown("## Break the review. Keep the boundary.\n\nInject one failure. The file stays open for human review; the run continues within limits. This drill uses deterministic rules and simulated failures, with no model call.")
            failure_kind=gr.Radio(choices=[("Provider timeout","timeout"),("Malformed JSON","malformed"),("Invented source citation","evidence")],value="evidence",label="Failure scenario")
            break_button=gr.Button("Run failure drill  →",variant="primary")
            fail_metrics,fail_findings=gr.HTML(EMPTY_METRICS),gr.HTML()
            fail_audit=gr.JSON(label="Failure drill audit")
            break_button.click(run_failure_lab,[failure_kind],[fail_metrics,fail_findings,fail_audit])
        with gr.Tab("Design decisions"):
            gr.Markdown('''## AI analyzes. Code enforces.

| Responsibility | Choice | Reason |
|---|---|---|
| Find semantic defects | Local pretrained Qwen | Retries, ownership and trust boundaries need context |
| Prioritize files | Visible deterministic signals | Cheap, inspectable, never given answer labels |
| Limit generation | Output cap + conservative usage ledger | Unknown usage consumes the full reservation |
| Validate findings | Structured output + exact line/quote match | Invalid evidence cannot become an accepted finding |
| Recover | Explicit human handoff; circuit after repeated failures | Uncertainty stays visible |
| Approve a patch | Human reviewer | No output can certify arbitrary code as safe |

### Research path, clearly separated

The repository also contains an OpenEnv environment, experimental metacognitive rewards,
SFT/GRPO scripts, and a token-budget processor. A Qwen2.5-1.5B LoRA checkpoint and 100-step
training logs are supplied. The audit found zero action coupling in 200 saved reward traces.
Training happened; useful learned allocation has not been established. The live product uses
a separate pretrained model and deterministic routing.

### What broke

The audit found label leakage during decisions, missing observations over HTTP,
unrestricted inherited execution, and simulated metrics presented as learned behavior.
These failures drove separation between source, labels, inference, controls, and evidence.

### Run locally

```bash
python -m pip install -r requirements.txt
ollama pull qwen3:4b
python app.py
python -m pytest tests -q
python benchmark.py
```

No training starts on boot. Source stays on the local model endpoint by default.
Requested review exports are stored under `.cache/reviews/`.
''')
    gr.HTML('<footer class="footer"><b>THE THINKING BUDGET</b><span>Razorpay AI Buildathon · Open Track · Local prototype</span></footer>')
    app.load(model_status_html,outputs=[runtime])


if __name__=="__main__":
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "7860"))
    css_text = (ROOT/"ui/style.css").read_text() if (ROOT/"ui/style.css").exists() else ""
    launched = False
    for p in range(port, port + 10):
        try:
            app.queue(default_concurrency_limit=1).launch(
                server_name=host,
                server_port=p,
                ssr_mode=False,
                share=False,
                theme=THEME,
                css=css_text,
                show_error=False,
                footer_links=[],
                blocked_paths=[str(ROOT/".git"), str(ROOT/".venv")]
            )
            launched = True
            break
        except OSError:
            continue

