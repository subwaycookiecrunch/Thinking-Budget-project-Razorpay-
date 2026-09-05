"""Regressions for real resource limits, evidence rejection, and failure handoff."""
import copy
import json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import patch

import pytest
from review_engine import (ProviderFailure, ReviewBudget, SourceFile, parse_patch,
                           review_patch, route_file, validate_review, verify_audit)


def response(tokens=30, decision="no_finding", findings=None):
    return {"text": json.dumps({"decision": decision, "summary": "Reviewed visible source.",
                               "findings": findings or []}), "output_tokens": tokens,
            "input_tokens": 100, "stop_reason": "stop"}


def test_labels_never_enter_source_or_routing():
    raw=[{"path":"x.py","content":"x = 1", "expected_issue":"secret answer", "label":1}]
    files=parse_patch(json.dumps(raw))
    assert files == [SourceFile("x.py","x = 1")]
    raw[0]["label"]=0
    assert route_file(files[0]) == route_file(parse_patch(json.dumps(raw))[0])


@pytest.mark.parametrize("value", ["", "garbage", '[{"path":"../secret","content":"x"}]',
    '[{"path":"/etc/passwd","content":"x"}]', '[{"path":"x","content":5}]',
    '[{"path":"x","content":"a"},{"path":"x","content":"b"}]'])
def test_reject_invalid_bundles(value):
    with pytest.raises(ValueError):
        parse_patch(value)


def test_global_budget_holds_and_unknown_usage_is_not_reclaimed():
    files=[SourceFile(f"a{i}.py", "x = 1") for i in range(4)]
    def failed(*args):
        raise ProviderFailure("Timed out")
    result=review_patch(files,ReviewBudget(output_tokens=512),mode="ollama",provider=failed)
    assert result["usage"]["output_tokens_accounted"]==512
    assert result["usage"]["model_calls"]==2
    assert result["summary"]["needs_review"]==4
    assert result["summary"]["budget_respected"]
    assert any(e["kind"]=="circuit_opened" for e in result["events"])


def test_actual_unused_tokens_return_to_pool():
    files=[SourceFile(f"x{i}.py","x = 1") for i in range(3)]
    caps=[]
    def review(file,cap,timeout):
        caps.append(cap)
        return response(30)
    r=review_patch(files,ReviewBudget(output_tokens=320),mode="ollama",provider=review)
    assert caps==[256,256,256]
    assert r["usage"]["output_tokens_accounted"]==90
    assert r["summary"]["reviewed_files"]==3


def test_bad_json_consumes_reported_tokens_and_hands_off():
    def bad(*args):
        return {**response(200),"text":"{unfinished"}
    r=review_patch([SourceFile("x.py","x = 1")],mode="ollama",provider=bad)
    assert r["files"][0]["status"]=="needs_review"
    assert r["usage"]["output_tokens_accounted"]==200
    assert r["summary"]["findings"]==0
    assert any(e["kind"]=="review_response" for e in r["events"])


def test_provider_violating_cap_is_reported_not_hidden():
    r=review_patch([SourceFile("x.py","x = 1"),SourceFile("y.py","y = 1")],
                  ReviewBudget(output_tokens=256),mode="ollama",provider=lambda *a:response(300))
    assert r["usage"]["output_tokens_reported"]==300
    assert not r["summary"]["budget_respected"]
    assert r["files"][1]["status"]=="deferred"


def test_input_budget_never_silently_truncates_source():
    called=[]
    def reviewer(file,*args):
        called.append(file.path)
        return response()
    r=review_patch([SourceFile("payments/large.py","x"*600),SourceFile("small.py","x = 1")],
                  ReviewBudget(input_chars=512),mode="ollama",provider=reviewer)
    assert called==["small.py"]
    assert r["usage"]["input_chars"]==5
    assert r["files"][0]["status"]=="deferred"


def test_context_overflow_is_deferred_before_any_model_call():
    def should_not_call(*args):
        raise AssertionError("Oversized source must not reach the model")
    r=review_patch([SourceFile("x.py","x\n"*3000)],mode="ollama",provider=should_not_call)
    assert r['files'][0]['status']=='deferred'
    assert r['usage']['model_calls']==0
    assert 'context bound' in r['files'][0]['summary']


def test_exact_evidence_required():
    f=SourceFile("x.py","first\nsecond")
    finding={"line":1,"quote":"second","title":"Issue","severity":"high","explanation":"Reason"}
    with pytest.raises(ValueError,match="Evidence quote"):
        validate_review(response(decision="flag",findings=[finding])["text"],f)
    finding["line"]=2
    assert validate_review(response(decision="flag",findings=[finding])["text"],f)["decision"]=="flag"


@pytest.mark.parametrize("fault",["timeout","malformed","evidence"])
def test_injected_failures_make_no_model_request_and_resume(fault):
    calls=[]
    def reviewer(file,*args):
        calls.append(file.path)
        return response()
    r=review_patch([SourceFile("a.py","a = 1"),SourceFile("b.py","b = 1")],
                  mode="ollama",provider=reviewer,fault=fault)
    assert calls==["b.py"]
    assert r["files"][0]["status"]=="needs_review"
    assert r["files"][1]["status"]=="no_finding"
    assert r["usage"]["unknown_usage_calls"]==0


def test_audit_chain_detects_mutation_and_reordering():
    r=review_patch([SourceFile("x.py","x = 1")],mode="offline")
    assert verify_audit(r["events"])
    changed=copy.deepcopy(r["events"])
    changed[1]["path"]="different.py"
    assert not verify_audit(changed)
    assert not verify_audit(r["events"][::-1])


def test_static_rules_keep_execution_as_data():
    with patch("builtins.eval",side_effect=AssertionError("Must not execute source")):
        r=review_patch([SourceFile("unsafe.py","result = eval(user_input)")],mode="offline")
    assert r["summary"]["findings"]==1
    assert r["usage"]["model_calls"]==0
    assert r["usage"]["output_tokens_reported"]==0


@pytest.mark.parametrize("kwargs",[{"output_tokens":0},{"output_tokens":True},{"max_files":0},{"timeout_seconds":-1}])
def test_invalid_limits_fail_before_execution(kwargs):
    with pytest.raises(ValueError):
        ReviewBudget(**kwargs)
