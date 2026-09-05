"""Validate benchmark provenance boundaries and actual read accounting."""
import copy
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from benchmark import public_files, plan_reviews, ReadWorkspace, run_episode, inspect_source


def rows():
    return [{"file":"risk.py","features":[80,90,2,80],"label":1},
            {"file":"safe.py","features":[2,1,0,3],"label":0}]


def test_changing_labels_cannot_change_policy_plan():
    data=rows()
    before=public_files(data)
    for r in data:
        r['label']=1-r['label']
        r['secret_ground_truth']='not public'
    assert before==public_files(data)
    assert plan_reviews(before,'risk_ranked',1,17)==['risk.py']


def test_read_budget_and_episode_boundary():
    workspace=ReadWorkspace(public_files(rows()),{'risk.py':'abcd','safe.py':'x'},1)
    with pytest.raises(ValueError):
        workspace.read_file('/etc/passwd')
    assert workspace.read_file('risk.py')=='abcd'
    with pytest.raises(RuntimeError):
        workspace.read_file('safe.py')
    assert workspace.reads==[{'path':'risk.py','source_chars':4,'source_available':True}]


def test_coverage_is_not_mislabeled_as_detection():
    episode={'episode_id':'test','files':rows()}
    result=run_episode(episode,{'risk.py':'a','safe.py':'bb'},'risk_ranked',.5)
    assert result['covered_positives']==1
    assert result['source_chars_read']==1
    assert result['reads_used']==1
    assert 'f1' not in result and 'detection_recall' not in result
    assert episode=={'episode_id':'test','files':rows()}


def test_rule_evidence_is_real_and_parameter_binding_is_not_injection():
    source='db.execute("SELECT * FROM users WHERE id = ?", (user_id,))'
    assert not inspect_source(source,'query.py')
    unsafe='db.execute(f"SELECT * FROM users WHERE id = {user_id}")'
    findings=inspect_source(unsafe,'query.py')
    assert any(f['rule_id']=='SQL_INTERPOLATION' for f in findings)
    for f in findings:
        assert unsafe.splitlines()[f['line']-1].strip()==f['evidence']


def test_missing_source_is_counted_explicitly():
    r=run_episode({'episode_id':'x','files':rows()},{},'exhaustive',1)
    assert r['missing_source_reads']==2
    assert r['source_chars_read']==0
    assert r['covered_positives']==0
    assert r['missed_positives']==1
