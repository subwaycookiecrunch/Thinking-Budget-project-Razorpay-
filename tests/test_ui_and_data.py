"""Product integration: editable fixtures, escaped output, exports, honest labels."""
import json
from pathlib import Path
import app
from review_engine import parse_patch, review_patch


def test_fixtures_are_parseable_without_ground_truth():
    assert len(app.CASES)>=3
    for case in app.CASES:
        source,description=app.load_case(case["id"])
        files=parse_patch(source)
        assert len(files)==len(case["files"])
        assert "expected_issue" not in source
        assert description


def test_no_network_required_for_failure_drill():
    metrics,findings,report=app.run_failure_lab("evidence")
    assert report["summary"]["needs_review"]==1
    assert report["usage"]["model_calls"]==0
    assert "HUMAN REVIEW REQUIRED" in findings
    assert "no model inference" in metrics


def test_model_text_is_escaped_in_html():
    source,_=app.load_case("checkout")
    report=review_patch(parse_patch(source),mode="offline")
    report["files"][0]["findings"]=[{"title":"<script>alert(1)</script>","line":1,"quote":"<img src=x onerror=alert(1)>","severity":"high","explanation":"<b>untrusted</b>"}]
    rendered=app.render_findings(report)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;img" in rendered


def test_review_exports_match_rendered_run(tmp_path,monkeypatch):
    monkeypatch.setattr(app,"EXPORTS",tmp_path)
    source,_=app.load_case("checkout")
    metrics,findings,table,report,jp,mp=app.run_review(source,"offline",2400,2,24000,"adaptive","none")
    assert json.loads(Path(jp).read_text())["run_id"]==report["run_id"]
    assert report["run_id"] in Path(mp).read_text()
    assert len(table)==6
    assert report["summary"]["needs_review"]==4


def test_plan_changes_actual_file_allowance():
    source,_=app.load_case("checkout")
    assert "deferred" in app.render_plan(source,256,1,24000)
    assert "deterministic, not learned" in app.render_plan(source)


def test_benchmark_does_not_claim_model_detection():
    md=app.benchmark_md()
    assert "not defects detected" in md
    assert "not LLM results" in md
