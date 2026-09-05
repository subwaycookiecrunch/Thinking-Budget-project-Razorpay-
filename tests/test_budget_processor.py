import re
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from scripts.budget_processor import ThinkingBudgetProcessor, enforce_character_budget


class Tokenizer:
    def encode(self,text,add_special_tokens=False):
        return [1] if text=="<think>" else [2]
    def decode(self,ids):
        return "<think>" if ids==[1] else "</think>"


def test_character_caps_include_global_pool_and_zero():
    text='<think>'+('x'*80)+'</think> mid <think>'+('y'*80)+'</think>'
    result=enforce_character_budget(text,100,100)
    blocks=re.findall(r'<think>(.*?)</think>',result)
    assert [len(b) for b in blocks]==[80,20]
    assert '[truncated by budget]' in result
    assert enforce_character_budget('<think>abc</think>',10,0)=='<think></think>[truncated by budget]'


def test_unclosed_block_is_capped_and_closed():
    assert enforce_character_budget('before <think>abcdef',3)=='before <think>abc</think>[truncated by budget]'


def test_plain_text_unchanged():
    assert enforce_character_budget('ordinary text',0)=='ordinary text'


@pytest.mark.parametrize('cap',[-1,True,1.5])
def test_invalid_character_limit(cap):
    with pytest.raises(ValueError):
        enforce_character_budget('text',cap)


def test_prompt_examples_not_counted_as_generated_thinking():
    p=ThinkingBudgetProcessor(Tokenizer(),2,3,prompt_length=5)
    assert p._sequence_state([1,8,8,2,1,7,7])==(True,2,2)


def test_nested_open_cannot_reset_allowance():
    p=ThinkingBudgetProcessor(Tokenizer(),2,3,prompt_length=1)
    assert p._sequence_state([1,7,1,7])==(True,3,3)


def test_budget_spans_blocks_and_calls_are_idempotent():
    p=ThinkingBudgetProcessor(Tokenizer(),10,3,prompt_length=1)
    seq=[1,7,7,2,1,7]
    assert p._sequence_state(seq)==(True,1,3)
    assert p._sequence_state(seq)==(True,1,3)


def test_split_tag_tokenizer_rejected():
    class Split(Tokenizer):
        def encode(self,*args,**kwargs):return [1,2,3]
    with pytest.raises(ValueError,match='atomic'):
        ThinkingBudgetProcessor(Split())


def test_real_logits_force_only_closing_tag():
    torch=pytest.importorskip('torch')
    p=ThinkingBudgetProcessor(Tokenizer(),2,prompt_length=1)
    logits=torch.zeros((1,10))
    result=p(torch.tensor([[1,4,5]]),logits)
    assert torch.isfinite(result).sum().item()==1
    assert result[0,2]==0
