"""Synthetic metadata tests never constitute real hardware evidence."""
from datetime import datetime, timezone, timedelta
import json
from types import SimpleNamespace
import httpx
import pytest
from factorymind import providers, core
from factorymind.providers import ProviderConfig, ProviderConfigError, hardware_status, check_connection


@pytest.fixture
def amd():
    return ProviderConfig("amd_tokenfactory", "https://developer.amd.com.cn/radeon/api/v1", "DeepSeek-V4-Flash", "rc-" + "a" * 48)


def test_no_implicit_remote_profile(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://gpu.example/v1")
    with pytest.raises(ProviderConfigError): ProviderConfig.from_env()


@pytest.mark.parametrize("url", ["http://gpu.example/v1", "https://user:secret@gpu.example/v1", "https://gpu.example/v1?token=secret", "file:///v1", "https://gpu.example/v1#secret"])
def test_insecure_or_credential_urls_rejected(url):
    with pytest.raises(ProviderConfigError):
        ProviderConfig("amd_tokenfactory", url, "model", "rc-" + "a" * 48)


@pytest.mark.parametrize("key", ["", "invalid-placeholder", "local-only-placeholder"])
def test_remote_key_required(key):
    with pytest.raises(ProviderConfigError):
        ProviderConfig("amd_tokenfactory", "https://developer.amd.com.cn/radeon/api/v1", "model", key)


@pytest.mark.parametrize("timeout", [0, -1, float('nan'), float('inf'), 121])
def test_timeout_bounds(timeout):
    with pytest.raises(ProviderConfigError): ProviderConfig("amd_tokenfactory", "https://developer.amd.com.cn/radeon/api/v1", "DeepSeek-V4-Flash", "rc-" + "a" * 48, timeout)


def test_nonsecret_metadata(amd):
    assert amd.api_key not in repr(amd)
    assert amd.api_key not in json.dumps(amd.public())
    assert amd.base_url not in json.dumps(amd.public())
    assert hardware_status(amd)["status"] == "managed_amd_api_documented"


def mock_http(monkeypatch, handler):
    real_client = httpx.Client
    monkeypatch.setattr(providers.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))


def test_auth_health_and_model_checks(amd, monkeypatch):
    seen = []
    def handler(request):
        seen.append(request)
        valid = request.headers.get("Authorization") == "Bearer " + amd.api_key
        if request.url.path == "/health": return httpx.Response(200)
        return httpx.Response(200, json={"data": [{"id": amd.model}]}) if valid else httpx.Response(401)
    mock_http(monkeypatch, handler)
    result = check_connection(amd)
    assert result["ready"]
    assert len(seen) == 1
    assert result["hardware"]["hardware_verified"] is False


def test_connection_timeout_redacts_error(amd, monkeypatch):
    def handler(request): raise httpx.ReadTimeout("sensitive token details")
    mock_http(monkeypatch, handler)
    result = check_connection(amd)
    assert result["failure"] == "timeout"
    assert "sensitive" not in json.dumps(result)


def test_fallback_never_claims_amd(amd, trained, monkeypatch):
    def fail(*args, **kwargs): raise TimeoutError("secret details")
    monkeypatch.setattr(core, "investigate", fail)
    result = core.run_investigation(core.predict_cycle(*trained, 1), config=amd)
    assert result["served_by"] == "rule_based"
    assert result["model"] is None
    assert result["hardware"]["gpu_details"] is None
    assert result["failure"] == "timeout"
    assert "AMD" not in result["provider_label"]


def test_provider_http_auth_failure_is_sanitized(amd, monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(401, text="private auth details"))
    result = check_connection(amd)
    assert not result["ready"] and result["failure"] == "authentication"
    assert "private auth" not in json.dumps(result)


def test_auth_pass_but_model_missing_is_not_ready(amd, monkeypatch):
    def handler(request):
        auth = request.headers.get("Authorization")
        if request.url.path == "/health": return httpx.Response(200)
        if auth != "Bearer " + amd.api_key: return httpx.Response(403)
        return httpx.Response(200, json={"data": [{"id": "other-model"}]})
    mock_http(monkeypatch, handler)
    result = check_connection(amd)
    assert not result["ready"] and not result["checks"]["model_available"]


def test_amd_connection_does_not_follow_redirects(amd, monkeypatch):
    seen = []
    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(302, headers={"Location": "https://different-origin.example/v1/models"})
    mock_http(monkeypatch, handler)
    assert not check_connection(amd)["ready"]
    assert set(seen) == {"developer.amd.com.cn"}


@pytest.mark.parametrize('key,value', [('LLM_TEMPERATURE','nan'), ('LLM_TEMPERATURE','3'), ('LLM_MAX_TOKENS','0'), ('LLM_MAX_TOKENS','1.5'), ('LLM_TOP_P','0')])
def test_invalid_generation_environment_refused(monkeypatch, key, value):
    monkeypatch.setenv("LLM_API_KEY", "rc-" + "a" * 48)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError): ProviderConfig.from_env()


def test_mocked_managed_success_retains_actual_provider(amd, trained, monkeypatch):
    from factorymind.core import predict_cycle, run_investigation
    import json
    def handler(request):
        assert request.headers['authorization'] == 'Bearer ' + amd.api_key
        body = json.loads(request.content)
        assert body['model'] == amd.model
        return httpx.Response(200, json={'id': 'synthetic-managed-completion', 'object': 'chat.completion', 'created': 0,
            'model': amd.model, 'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant',
                'content': json.dumps({'evidence_sensors': ['PS1'], 'interpretation': 'insufficient_evidence', 'human_checks': ['verify_prediction']})}}],
            'usage': {'prompt_tokens': 120, 'completion_tokens': 30, 'total_tokens': 150}})
    from openai import OpenAI
    monkeypatch.setattr(ProviderConfig, "client", lambda self: OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    result = run_investigation(predict_cycle(*trained, 1), config=amd)
    assert result['provider_mode'] == 'llm' and result['served_by'] == 'amd_tokenfactory'
    assert result['provider_label'] == 'AMD Token Factory · AMD-hosted API (documented)'
    assert not result['hardware']['hardware_verified']
    assert result['model'] == amd.model and result['telemetry']['generated_tokens'] == 30
    from factorymind.benchmark import evaluate_quality
    assert evaluate_quality(result)['exact_evidence_catalog_render'] is True
