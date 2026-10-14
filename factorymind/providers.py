"""Validated provider settings and non-secret attribution for each request.

Hardware records are operator-supplied evidence, not remote attestation. No profile
name or successful HTTP response is sufficient to prove AMD inference.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from pydantic import BaseModel, ConfigDict, Field


class ProviderConfigError(ValueError):
    pass


class ModelMismatchError(ValueError):
    pass


class GPUDevice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    architecture: str = Field(pattern=r"^gfx[0-9a-f]+$")
    total_vram_gib: float = Field(gt=0, le=1024, allow_inf_nan=False)
    free_vram_gib: float = Field(ge=0, le=1024, allow_inf_nan=False)


class HardwareEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    collected_at_utc: datetime
    deployment_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    image_digest: str = Field(pattern=r"^vllm/vllm-openai-rocm@sha256:[a-f0-9]{64}$")
    hip_version: str = Field(min_length=1, max_length=80)
    torch_version: str = Field(min_length=1, max_length=80)
    vllm_version: str = Field(min_length=1, max_length=80)
    gpu_compute_check: bool
    devices: list[GPUDevice] = Field(min_length=1, max_length=16)


def loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 60
    disable_thinking: bool = False
    hardware_evidence_path: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    temperature: float = 0
    max_tokens: int = 300
    top_p: float = 1

    def __post_init__(self):
        if self.provider not in {"ollama", "amd_vllm", "amd_tokenfactory"}:
            raise ProviderConfigError("LLM_PROVIDER must be ollama, amd_vllm or amd_tokenfactory")
        u = urlsplit(self.base_url)
        if (u.scheme not in {"http", "https"} or not u.hostname or u.username or u.password
                or u.query or u.fragment or u.path.rstrip("/") not in {"/v1", "/radeon/api/v1"}):
            raise ProviderConfigError("Use an HTTP(S) API v1 URL without credentials, queries or fragments")
        if self.provider == "amd_tokenfactory" and self.base_url.rstrip("/") != "https://developer.amd.com.cn/radeon/api/v1":
            raise ProviderConfigError("AMD Token Factory requires the official documented HTTPS API address")
        if self.provider != "amd_tokenfactory" and u.path.rstrip("/") != "/v1":
            raise ProviderConfigError("Ollama and self-hosted vLLM require a /v1 API URL")
        if u.scheme == "http" and not loopback(u.hostname):
            raise ProviderConfigError("Remote endpoints require HTTPS; HTTP is allowed only over local loopback/tunnel")
        if self.provider == "ollama" and not loopback(u.hostname):
            raise ProviderConfigError("The ollama profile is local-only; configure amd_vllm explicitly for the remote server")
        if not self.model or len(self.model) > 200 or any(c.isspace() for c in self.model):
            raise ProviderConfigError("An explicit valid model name is required")
        if self.provider == "amd_vllm" and (len(self.api_key.strip()) < 16 or self.api_key.lower() in {
                "local-only-placeholder", "your-api-key-here"}):
            raise ProviderConfigError("AMD vLLM requires a private API key of at least 16 characters")
        if self.provider == "amd_tokenfactory" and not re.fullmatch(r"rc-[0-9a-fA-F]{48}", self.api_key):
            raise ProviderConfigError("AMD Token Factory requires a privately supplied rc- team API key")
        if self.provider == "amd_tokenfactory" and self.disable_thinking:
            raise ProviderConfigError("Self-hosted disable_thinking setting is unsupported by Token Factory")
        if not math.isfinite(self.timeout_seconds) or not 1 <= self.timeout_seconds <= 120:
            raise ProviderConfigError("Timeout must be between 1 and 120 seconds")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ProviderConfigError("Temperature must be between 0 and 2")
        if not isinstance(self.max_tokens, int) or not 64 <= self.max_tokens <= 1024:
            raise ProviderConfigError("Generated token limit must be between 64 and 1024")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ProviderConfigError("Top-p must be greater than 0 and at most 1")
        if self.model_revision and not re.fullmatch(r"[a-f0-9]{40}", self.model_revision):
            raise ProviderConfigError("Model revision must be an immutable 40-character commit")

    @classmethod
    def from_env(cls):
        provider = os.getenv("LLM_PROVIDER", "ollama")
        default_base = {"ollama": "http://localhost:11434/v1",
                        "amd_tokenfactory": "https://developer.amd.com.cn/radeon/api/v1"}
        default_model = {"ollama": "llama3.2", "amd_tokenfactory": "DeepSeek-V4-Flash"}
        base = os.getenv("LLM_BASE_URL", default_base.get(provider, ""))
        model = os.getenv("LLM_MODEL", default_model.get(provider, ""))
        key = os.getenv("LLM_API_KEY", "ollama" if provider == "ollama" else "")
        try:
            return cls(provider, base.rstrip("/"), model, key,
                       float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
                       os.getenv("LLM_DISABLE_THINKING", "0") == "1", os.getenv("LLM_HARDWARE_EVIDENCE"),
                       os.getenv("LLM_MODEL_ID"), os.getenv("LLM_MODEL_REVISION"),
                       float(os.getenv("LLM_TEMPERATURE", "0")), int(os.getenv("LLM_MAX_TOKENS", "300")),
                       float(os.getenv("LLM_TOP_P", "1")))
        except (ValueError, TypeError) as exc:
            if isinstance(exc, ProviderConfigError):
                raise
            raise ProviderConfigError("Invalid provider settings") from None

    @property
    def endpoint_id(self):
        # No hostname, URL or token is exposed in the UI or logs.
        return hashlib.sha256(self.base_url.encode()).hexdigest()[:16]

    def generation(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens, "top_p": self.top_p,
                "response_format": "json_object", "disable_thinking": self.disable_thinking}

    def public(self):
        return {"requested_provider": self.provider, "requested_model": self.model,
                "endpoint_id": self.endpoint_id, "timeout_seconds": self.timeout_seconds,
                "transport": "loopback_or_ssh_tunnel" if loopback(urlsplit(self.base_url).hostname) else "https",
                "model_weights": {"id": self.model_id, "revision": self.model_revision, "source": "operator_configured_not_server_attested"},
                "max_retries": 0, "generation": self.generation()}

    def matches_model(self, returned):
        aliases = {self.model}
        if self.provider == "ollama":
            aliases.add(self.model.removesuffix(":latest"))
            aliases.add(self.model.removesuffix(":latest") + ":latest")
        if self.provider == "amd_tokenfactory" and self.model == "DeepSeek-V4-Flash":
            # The official Token Factory manual lists this exact weight identifier
            # for the DeepSeek-V4-Flash catalog model. Do not accept arbitrary names.
            aliases.add("V4-Flash-0731")
            # Observed ACT III managed gateway response. Match the entire
            # reported identifier; never accept arbitrary self-dploy/* models.
            aliases.add("self-dploy/DeepSeek-V4-Flash")
        return isinstance(returned, str) and returned in aliases

    def client(self):
        from openai import OpenAI
        # Disable environment proxies and redirects: do not send credentials to another origin.
        return OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=0,
                      timeout=httpx.Timeout(self.timeout_seconds, connect=min(5, self.timeout_seconds)),
                      http_client=httpx.Client(trust_env=False, follow_redirects=False))


def hardware_status(config: ProviderConfig, returned_model: str | None = None) -> dict:
    result = {"status": "not_applicable" if config.provider == "ollama" else "unverified",
              "gpu_details": None, "hardware_verified": False,
              "note": "Client responses cannot attest GPU execution."}
    if config.provider == "amd_tokenfactory":
        result.update(status="managed_amd_api_documented",
                      note="AMD Token Factory documentation states shared models run on AMD GPUs; this request does not include independently verified GPU hardware details.")
    if config.provider != "amd_vllm" or not config.hardware_evidence_path:
        return result
    try:
        raw = Path(config.hardware_evidence_path).read_bytes()
        evidence = HardwareEvidence.model_validate_json(raw)
        age = (datetime.now(timezone.utc) - evidence.collected_at_utc).total_seconds()
        if evidence.schema_version != 1 or not 0 <= age <= 86400 or not evidence.gpu_compute_check:
            raise ValueError("Unusable hardware record")
        if any(d.free_vram_gib > d.total_vram_gib or "AMD" not in d.name.upper() for d in evidence.devices):
            raise ValueError("Invalid GPU record")
        expected = "factorymind-" + evidence.deployment_id
        if config.model != expected or (returned_model is not None and returned_model != expected):
            raise ValueError("Deployment alias mismatch")
        result.update(status="operator_evidence_matched", gpu_details=evidence.model_dump(mode="json"),
                      evidence_sha256=hashlib.sha256(raw).hexdigest(),
                      note="Recent operator-supplied ROCm compute evidence matches the deployment alias; not cryptographic attestation or proof of this request's GPU execution.")
    except (OSError, ValueError, TypeError):
        result["status"] = "invalid_or_stale_evidence"
    return result


def error_category(exc: Exception) -> str:
    from openai import APITimeoutError, APIConnectionError, AuthenticationError, RateLimitError, APIStatusError
    if isinstance(exc, ProviderConfigError): return "configuration"
    if isinstance(exc, (APITimeoutError, httpx.TimeoutException, TimeoutError)): return "timeout"
    if isinstance(exc, AuthenticationError): return "authentication"
    if isinstance(exc, RateLimitError): return "rate_limited"
    if isinstance(exc, (APIConnectionError, httpx.ConnectError)): return "connection"
    if isinstance(exc, ModelMismatchError): return "model_mismatch"
    if isinstance(exc, APIStatusError): return "provider_http_error"
    if isinstance(exc, ValueError): return "invalid_response"
    return "provider_error"


def check_connection(config: ProviderConfig) -> dict:
    """Readiness + configured model + negative auth checks, with no inference."""
    result = {**config.public(), "ready": False, "checks": {}, "failure": None,
              "hardware": hardware_status(config), "server_metadata": {}}
    try:
        with httpx.Client(timeout=httpx.Timeout(min(config.timeout_seconds, 10), connect=5),
                          trust_env=False, follow_redirects=False) as client:
            headers = {"Authorization": "Bearer " + config.api_key}
            if config.provider == "amd_tokenfactory":
                # Managed service: no /health route or credential-negative probes.
                # GET /models is the documented lightweight authenticated preflight.
                models = client.get(config.base_url + "/models", headers=headers)
                result["checks"]["authenticated_models"] = models.status_code == 200
                if models.status_code == 200:
                    ids = [m.get("id") for m in models.json().get("data", [])]
                    result["checks"]["model_available"] = any(config.matches_model(m) for m in ids)
                else:
                    result["checks"]["model_available"] = False
                    result["failure"] = "authentication" if models.status_code in (401, 403) else ("rate_limited" if models.status_code == 429 else "provider_http_error")
                result["ready"] = all(result["checks"].values())
                if not result["ready"] and result["failure"] is None:
                    result["failure"] = "model_not_available"
                return result
            if config.provider == "amd_vllm":
                # An open /health is allowed by vLLM; /v1/models must enforce authentication.
                url = config.base_url + "/models"
                for name, test_headers in [("unauthenticated_rejected", {}),
                                            ("incorrect_key_rejected", {"Authorization": "Bearer factorymind-invalid-probe-key"})]:
                    response = client.get(url, headers=test_headers)
                    result["checks"][name] = response.status_code in (401, 403)
                health = client.get(config.base_url.removesuffix("/v1") + "/health", headers=headers)
            else:
                health = client.get(config.base_url.removesuffix("/v1") + "/api/version")
                if health.status_code == 200:
                    version = health.json().get("version")
                    result["checks"]["ollama_version"] = isinstance(version, str) and bool(version)
                    result["server_metadata"]["ollama_version"] = version
                    tags = client.get(config.base_url.removesuffix("/v1") + "/api/tags")
                    if tags.status_code == 200:
                        for entry in tags.json().get("models", []):
                            if config.matches_model(entry.get("name")):
                                details = entry.get("details", {})
                                result["server_metadata"].update(model_digest=entry.get("digest"),
                                    parameter_size=details.get("parameter_size"), quantization=details.get("quantization_level"))
            result["checks"]["health"] = health.status_code == 200
            models = client.get(config.base_url + "/models", headers=headers)
            result["checks"]["authenticated_models"] = models.status_code == 200
            if models.status_code == 200:
                ids = [m.get("id") for m in models.json().get("data", [])]
                result["checks"]["model_available"] = any(config.matches_model(m) for m in ids)
            else:
                result["checks"]["model_available"] = False
                result["failure"] = "authentication" if models.status_code in (401, 403) else "provider_http_error"
            result["ready"] = all(result["checks"].values())
            if not result["ready"] and result["failure"] is None:
                result["failure"] = "connection_checks_failed"
    except Exception as exc:
        result["failure"] = error_category(exc)
    return result
