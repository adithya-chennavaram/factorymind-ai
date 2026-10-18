"""Offline, no-secret tests for the official AMD ACT III managed model gateway."""
import json
from types import SimpleNamespace
import httpx
import pytest
from factorymind import providers, core
from factorymind.providers import ProviderConfig, ProviderConfigError, check_connection

API_URL = "https://developer.amd.com.cn/radeon/api/v1"
TEST_KEY = "rc-" + "a" * 48  # synthetic, never a working secret


def config():
    return ProviderConfig("amd_tokenfactory", API_URL, "DeepSeek-V4-Flash", TEST_KEY)


def test_documented_managed_profile_and_no_key_leak():
    c = config()
    assert c.provider == "amd_tokenfactory"
    assert c.api_key not in repr(c)
    assert c.api_key not in json.dumps(c.public())
    assert API_URL not in json.dumps(c.public())
    assert providers.hardware_status(c)["hardware_verified"] is False
    assert providers.hardware_status(c)["status"] == "managed_amd_api_documented"


@pytest.mark.parametrize("bad", ["", "untrusted_provider", "rc-abcdef", "rc-" + "g"*48])
def test_reject_bad_amd_team_keys(bad):
    with pytest.raises(ProviderConfigError):
        ProviderConfig("amd_tokenfactory", API_URL, "DeepSeek-V4-Flash", bad)


@pytest.mark.parametrize("bad_url", [
    "http://developer.amd.com.cn/radeon/api/v1",
    "https://developer.amd.com.cn/radeon/api/v1?token=secret",
    "https://evil.example/radeon/api/v1",
    "https://developer.amd.com.cn/radeon/api",
])
def test_pinned_endpoint_does_not_allow_key_exfiltration(bad_url):
    with pytest.raises(ProviderConfigError):
        ProviderConfig("amd_tokenfactory", bad_url, "DeepSeek-V4-Flash", TEST_KEY)


def test_documented_env_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "amd_tokenfactory")
    monkeypatch.setenv("LLM_API_KEY", TEST_KEY)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    c = ProviderConfig.from_env()
    assert c.model == "DeepSeek-V4-Flash"
    assert c.base_url == API_URL


def test_managed_preflight_only_fetches_models(monkeypatch):
    seen = []
    original = httpx.Client
    def handler(request):
        seen.append((request.method, str(request.url), request.headers.get("Authorization")))
        return httpx.Response(200, json={"object": "list", "data": [{"id": "DeepSeek-V4-Flash"}]})
    monkeypatch.setattr(providers.httpx, "Client", lambda **kwargs:
                        original(transport=httpx.MockTransport(handler), **kwargs))
    report = check_connection(config())
    assert report["ready"] is True
    assert len(seen) == 1
    assert seen[0][0] == "GET" and seen[0][1] == API_URL + "/models"
    assert seen[0][2] == "Bearer " + TEST_KEY
    assert TEST_KEY not in json.dumps(report)


def test_missing_model_not_ready(monkeypatch):
    original = httpx.Client
    monkeypatch.setattr(providers.httpx, "Client", lambda **kwargs:
        original(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data":[{"id":"some-other-model"}]})), **kwargs))
    report = check_connection(config())
    assert not report["ready"] and report["failure"] == "model_not_available"


def test_tokenfactory_investigation_uses_actual_provider(trained, monkeypatch):
    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def chat(self): raise AssertionError("not used")
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    assert kwargs["model"] == "DeepSeek-V4-Flash"
                    assert kwargs["response_format"] == {"type": "json_object"}
                    assert kwargs["messages"][0]["role"] == "system"
                    return SimpleNamespace(model="DeepSeek-V4-Flash",
                        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps({
                            "evidence_sensors": ["PS1", "FS1"], "interpretation": "insufficient_evidence",
                            "human_checks": ["verify_prediction", "review_trace"]
                        })))], usage=SimpleNamespace(prompt_tokens=150, completion_tokens=48))
    monkeypatch.setattr(ProviderConfig, "client", lambda self: FakeClient())
    report = core.run_investigation(core.predict_cycle(*trained, 1), config=config())
    assert report["provider_mode"] == "llm"
    assert report["served_by"] == "amd_tokenfactory"
    assert "AMD Token Factory" in report["provider_label"]
    assert report["model"] == "DeepSeek-V4-Flash"
    assert not report["hardware"]["hardware_verified"]
    assert report["telemetry"]["generated_tokens"] == 48
    assert TEST_KEY not in json.dumps(report)
    from factorymind.benchmark import evaluate_quality
    assert evaluate_quality(report)["exact_evidence_catalog_render"]


def test_documented_deepseek_weight_alias_is_accepted_only_for_target_model():
    c = config()
    assert c.matches_model("DeepSeek-V4-Flash")
    assert c.matches_model("V4-Flash-0731")
    assert c.matches_model("self-dploy/DeepSeek-V4-Flash")
    assert not c.matches_model("self-dploy/DeepSeek-V4.1-Flash")
    assert not c.matches_model("self-dploy/other-model")
    assert not c.matches_model("DeepSeek-V4.1-Flash")
    assert not c.matches_model("completely-different-model")
    other = ProviderConfig("amd_tokenfactory", API_URL, "DeepSeek-V4.1-Flash", TEST_KEY)
    assert not other.matches_model("V4-Flash-0731")
    assert not other.matches_model("self-dploy/DeepSeek-V4-Flash")


def test_tokenfactory_records_actual_mismatched_model_without_accepting_it(trained, monkeypatch):
    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(model="unrecognized-provider-model",
                        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="{}"))],
                        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=58))
    monkeypatch.setattr(ProviderConfig, "client", lambda self: FakeClient())
    report = core.run_investigation(core.predict_cycle(*trained, 1), config=config())
    assert report["provider_mode"] == "fallback"
    assert report["failure"] == "model_mismatch"
    assert report["served_by"] == "rule_based"
    assert report["model"] is None
    assert report["telemetry"]["response_model"] == "unrecognized-provider-model"
    assert report["telemetry"]["generated_tokens"] == 58
    assert TEST_KEY not in json.dumps(report)


def test_tokenfactory_documented_weight_alias_end_to_end(trained, monkeypatch):
    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(model="V4-Flash-0731",
                        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps({
                            "evidence_sensors": ["PS1"], "interpretation": "insufficient_evidence",
                            "human_checks": ["verify_prediction"]
                        })))], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=58))
    monkeypatch.setattr(ProviderConfig, "client", lambda self: FakeClient())
    report = core.run_investigation(core.predict_cycle(*trained, 1), config=config())
    assert report["provider_mode"] == "llm"
    assert report["served_by"] == "amd_tokenfactory"
    assert report["model"] == "V4-Flash-0731"


def test_observed_amd_managed_gateway_alias_end_to_end(trained, monkeypatch):
    """Only the exact observed gateway name is allowed, without disabling verification."""
    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    assert kwargs["model"] == "DeepSeek-V4-Flash"
                    return SimpleNamespace(
                        model="self-dploy/DeepSeek-V4-Flash",
                        choices=[SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content=json.dumps({
                                "evidence_sensors": ["PS1"],
                                "interpretation": "insufficient_evidence",
                                "human_checks": ["verify_prediction"]
                            })))],
                        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=58))
    monkeypatch.setattr(ProviderConfig, "client", lambda self: FakeClient())
    report = core.run_investigation(core.predict_cycle(*trained, 1), config=config())
    assert report["provider_mode"] == "llm"
    assert report["served_by"] == "amd_tokenfactory"
    assert report["model"] == "self-dploy/DeepSeek-V4-Flash"
    assert report["telemetry"]["response_model"] == "self-dploy/DeepSeek-V4-Flash"
    assert report["telemetry"]["finish_reason"] == "stop"
    assert report["telemetry"]["generated_tokens"] == 58
    assert TEST_KEY not in json.dumps(report)


def test_managed_provider_is_default(monkeypatch):
    for name in ('LLM_PROVIDER', 'LLM_BASE_URL', 'LLM_MODEL'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('LLM_API_KEY', TEST_KEY)
    c = ProviderConfig.from_env()
    assert c.provider == 'amd_tokenfactory' and c.base_url == API_URL


def test_missing_key_uses_offline_fallback_without_network(trained, monkeypatch):
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    monkeypatch.delenv('LLM_PROVIDER', raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail('No network client may be opened without a valid key')
    monkeypatch.setattr(ProviderConfig, 'client', forbidden)
    report = core.run_investigation(core.predict_cycle(*trained, 1))
    assert report['served_by'] == 'rule_based'
    assert report['failure'] == 'configuration'
    assert report['requires_human_approval'] and not report['equipment_control']
    assert report['hardware']['hardware_verified'] is False


def test_unknown_provider_is_rejected():
    with pytest.raises(ProviderConfigError):
        ProviderConfig('unsupported_provider', API_URL, 'DeepSeek-V4-Flash', TEST_KEY)
