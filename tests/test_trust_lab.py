"""Offline tests: labels never influence score selection or AMD prompt."""
import json
from types import SimpleNamespace
from pathlib import Path
import pandas as pd
import pytest
from factorymind.trust_lab import (load_scores, select_for_review, comparison,
                                   case_payload, request_amd_hearing, render_hearing)


def frame():
    return pd.DataFrame([
        {'cycle_id': n, 'fold': 1 if n<=10 else 2, 'actual': 90 if n in (1,11) else 100,
         'predicted':100, 'p_optimal_uncalibrated': .35 + (.02*n),
         'switch_fraction': .02*(n%5), 'max_p_optimal_drop': .01*n,
         'challenge_priority': .5, 'min_stressed_p_optimal': .1,
         'worst_case_sensor':'PS2','worst_case_gain_pct':2.0,'scenarios_tested':50,
         'base_p_optimal': .35 + (.02*n)} for n in range(1,21)])


def test_review_policy_ignores_truth():
    a=frame()
    original=select_for_review(a,'Confidence-only',20)
    b=a.copy()
    b['actual']=100
    assert select_for_review(b,'Confidence-only',20)==original
    assert len(original)==4


def test_compare_total_budget():
    result=comparison(frame(),20)
    assert len(result)==4
    assert result['Cycles reviewed'].tolist()==[4]*4
    assert result.loc[0,'False-green caught']==2


def test_payload_hides_truth_and_id():
    f=frame().iloc[0]
    p=case_payload(f,{1})
    text=json.dumps(p)
    assert 'actual' not in text and 'cycle_id' not in text and 'Small lag' not in text
    assert p['review_selected_by_confidence_only'] is True


def test_load_invalid_data(tmp_path):
    p=tmp_path/'scores.csv'
    x=frame();x.loc[0,'predicted']=90;x.to_csv(p,index=False)
    with pytest.raises(ValueError, match='predicted-optimal'):load_scores(p)


def test_amd_accepts_only_bounded_json_and_known_model():
    class FakeCompletion:
        model='self-dploy/DeepSeek-V4-Flash'
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            'evidence_ids':['base_score','switch_fraction'],
            'action_ids':['verify_sensor_trace'], 'verdict':'review_the_prediction'})))]
    class FakeClient:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: FakeCompletion()))
    class FakeConfig:
        provider='amd_tokenfactory'; model='DeepSeek-V4-Flash'; temperature=0.; top_p=1.; max_tokens=300
        def client(self):return FakeClient()
        def matches_model(self, returned):return returned=='self-dploy/DeepSeek-V4-Flash'
    payload=case_payload(frame().iloc[0],{1})
    report=request_amd_hearing(payload,config=FakeConfig())
    assert report['selection']['verdict']=='review_the_prediction'
    assert '37.0%' in render_hearing(report,payload)['evidence'][0]
