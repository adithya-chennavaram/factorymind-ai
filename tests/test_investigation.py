import json
from types import SimpleNamespace
import httpx
import openai
import pytest
from factorymind.core import predict_cycle, investigate, investigation_prompt

@pytest.fixture(autouse=True)
def managed_provider_environment(monkeypatch):
    """These fixtures simulate a managed API response regardless of the user's shell."""
    for key, value in {
        "LLM_PROVIDER": "amd_tokenfactory",
        "LLM_BASE_URL": "https://developer.amd.com.cn/radeon/api/v1",
        "LLM_MODEL": "DeepSeek-V4-Flash",
        "LLM_API_KEY": "rc-" + "a" * 48,
        "LLM_TIMEOUT_SECONDS": "60",
        "LLM_MAX_TOKENS": "300",
        "LLM_TEMPERATURE": "0",
        "LLM_TOP_P": "1",
    }.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def case(trained):
    return predict_cycle(*trained, 1)


def fake_client(monkeypatch, content, finish="stop", exception=None):
    captured = {}
    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=self)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def create(self, **kwargs):
            captured.update(kwargs)
            if exception: raise exception
            return SimpleNamespace(model="DeepSeek-V4-Flash", usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30), choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))])
    monkeypatch.setattr(openai, "OpenAI", Client)
    return captured


def test_valid_bounded_plan(case, monkeypatch):
    plan = {"evidence_sensors": ["PS1"], "interpretation": "insufficient_evidence", "human_checks": ["verify_prediction"]}
    captured = fake_client(monkeypatch, json.dumps(plan))
    result = investigate(case)
    assert str(case["sensor_evidence"]["PS1"]["mean"]) in result
    assert "score is uncalibrated" in result
    assert captured["max_retries"] == 0
    assert captured["timeout"].read == 60
    assert "dataset_label" not in investigation_prompt(case)[1]["content"]


@pytest.mark.parametrize("content,finish", [
    ("", "stop"), ("not json", "stop"),
    ('{"evidence_sensors":["fake"],"interpretation":"insufficient_evidence","human_checks":["verify_prediction"]}', "stop"),
    ('{"evidence_sensors":["PS1"],"interpretation":"insufficient_evidence","human_checks":["start_machine"]}', "stop"),
    ('{"evidence_sensors":["PS1"],"interpretation":"insufficient_evidence","human_checks":["verify_prediction"],"invented_measurement":99}', "stop"),
    ("{}", "length")])
def test_unsafe_or_incomplete_plan_rejected(case, monkeypatch, content, finish):
    fake_client(monkeypatch, content, finish)
    with pytest.raises(ValueError): investigate(case)


def test_provider_timeout(case, monkeypatch):
    fake_client(monkeypatch, None, exception=openai.APITimeoutError(request=httpx.Request("POST", "http://localhost")))
    with pytest.raises(openai.APITimeoutError): investigate(case)


def test_contradicting_interpretation(case, monkeypatch):
    wrong = "non_optimal_prediction" if case["predicted_valve_condition_code"] == 100 else "optimal_prediction"
    fake_client(monkeypatch, json.dumps({"evidence_sensors": ["PS1"], "interpretation": wrong, "human_checks": ["verify_prediction"]}))
    with pytest.raises(ValueError, match="contradicts"): investigate(case)


@pytest.mark.parametrize("claim", ["Valve seal is ruptured", "Shut down the machine immediately", "Savings are $10,000"])
def test_unsupported_diagnosis_claims_never_reach_display(case, monkeypatch, claim):
    plan = {"evidence_sensors": ["PS1"], "interpretation": "insufficient_evidence",
            "human_checks": ["verify_prediction"], "diagnosis": claim}
    fake_client(monkeypatch, json.dumps(plan))
    from factorymind.core import run_investigation
    result = run_investigation(case)
    assert result["provider_mode"] == "fallback"
    assert claim not in result["investigation"]
    assert result["requires_human_approval"] and result["equipment_control"] is False


def test_exact_factual_render_and_no_implicit_approval(case, monkeypatch):
    from factorymind import reviews
    def forbidden(*args, **kwargs): raise AssertionError("Investigation must never write a human decision")
    monkeypatch.setattr(reviews, "save_review", forbidden)
    plan = {"evidence_sensors": ["PS2", "VS1"], "interpretation": "insufficient_evidence", "human_checks": ["verify_prediction"]}
    fake_client(monkeypatch, json.dumps(plan))
    from factorymind.core import run_investigation
    result = run_investigation(case)
    assert result["served_by"] == "amd_tokenfactory"
    assert result["telemetry"]["generated_tokens"] == 30
    for sensor in plan["evidence_sensors"]:
        e = case["sensor_evidence"][sensor]
        assert f"{sensor}: mean {e['mean']} {e['unit']}; standard deviation {e['std']} {e['unit']}" in result["investigation"]
    assert "PS1:" not in result["investigation"]
    assert "No physical cause is established" in result["investigation"]
    assert result["requires_human_approval"] is True


def test_response_model_mismatch_rejected(case, monkeypatch):
    from factorymind.providers import ProviderConfig
    from factorymind.core import run_investigation
    fake_client(monkeypatch, '{}')  # fake returns DeepSeek-V4-Flash, not the configured model
    config = ProviderConfig("amd_tokenfactory", "https://developer.amd.com.cn/radeon/api/v1", "other-model", "rc-" + "a" * 48)
    result = run_investigation(case, config=config)
    assert result["failure"] == "model_mismatch"
    assert result["served_by"] == "rule_based"
    assert result["model"] is None


def test_generation_settings_reach_client(case, monkeypatch):
    from factorymind.providers import ProviderConfig
    content = json.dumps({'evidence_sensors': ['PS1'], 'interpretation': 'insufficient_evidence', 'human_checks': ['verify_prediction']})
    captured = fake_client(monkeypatch, content)
    config = ProviderConfig('amd_tokenfactory', 'https://developer.amd.com.cn/radeon/api/v1', 'DeepSeek-V4-Flash', 'rc-' + 'a' * 48, temperature=.2, max_tokens=400, top_p=.9)
    investigate(case, config=config)
    assert (captured['temperature'], captured['max_tokens'], captured['top_p']) == (.2, 400, .9)
