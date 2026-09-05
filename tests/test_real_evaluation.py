import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from eval_baseline import adapter_model_name, parse_tool_calls, observation_text
from types import SimpleNamespace


def test_valid_json_missing_wrapper_can_be_recovered_but_not_incomplete_json():
    assert parse_tool_calls('<tool_call>{"name":"read_file","arguments":{"file_path":"a.py"}}')[0]['wrapper_repaired']
    assert not parse_tool_calls('<tool_call>{"name":"read_file","arguments":')
    assert not parse_tool_calls('<tool_call>{"name":"read_file"} garbage')


def test_closed_wrapper_recorded_without_repair():
    call=parse_tool_calls('<tool_call>{"name":"read_file","arguments":{}}</tool_call>')[0]
    assert not call['wrapper_repaired']


def test_missing_adapter_cannot_fall_back_to_fake_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        adapter_model_name(tmp_path)


def test_serialized_observation_contains_actual_source():
    assert observation_text(SimpleNamespace(result={'data':'source code'}))=='source code'
    assert observation_text(SimpleNamespace(context='briefing',metadata={}))=='briefing'
