"""Validated provider settings and non-secret attribution for each request.

Managed-service provenance is not independent physical-GPU attestation.
"""
from dataclasses import dataclass, field
import hashlib
import math
import os
import re
import httpx


class ProviderConfigError(ValueError):
    pass


class ModelMismatchError(ValueError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 60
    temperature: float = 0
    max_tokens: int = 300
    top_p: float = 1

    def __post_init__(self):
        if self.provider != "amd_tokenfactory":
            raise ProviderConfigError("LLM_PROVIDER must be amd_tokenfactory")
        if self.base_url.rstrip("/") != "https://developer.amd.com.cn/radeon/api/v1":
            raise ProviderConfigError("AMD Token Factory requires the official documented HTTPS API address")
        if not self.model or len(self.model) > 200 or any(c.isspace() for c in self.model):
            raise ProviderConfigError("An explicit valid model name is required")
        if not re.fullmatch(r"rc-[0-9a-fA-F]{48}", self.api_key):
            raise ProviderConfigError("AMD Token Factory requires a privately supplied rc- team API key")
        if not math.isfinite(self.timeout_seconds) or not 1 <= self.timeout_seconds <= 120:
            raise ProviderConfigError("Timeout must be between 1 and 120 seconds")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ProviderConfigError("Temperature must be between 0 and 2")
        if not isinstance(self.max_tokens, int) or not 64 <= self.max_tokens <= 1024:
            raise ProviderConfigError("Generated token limit must be between 64 and 1024")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ProviderConfigError("Top-p must be greater than 0 and at most 1")

    @classmethod
    def from_env(cls):
        try:
            return cls(
                provider=os.getenv("LLM_PROVIDER", "amd_tokenfactory"),
                base_url=os.getenv("LLM_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1").rstrip("/"),
                model=os.getenv("LLM_MODEL", "DeepSeek-V4-Flash"),
                api_key=os.getenv("LLM_API_KEY", ""),
                timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
                top_p=float(os.getenv("LLM_TOP_P", "1")))
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
                "response_format": "json_object"}

    def public(self):
        return {"requested_provider": self.provider, "requested_model": self.model,
                "endpoint_id": self.endpoint_id, "timeout_seconds": self.timeout_seconds,
                "transport": "https",
                "max_retries": 0, "generation": self.generation()}

    def matches_model(self, returned):
        aliases = {self.model}
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
    return {"status": "managed_amd_api_documented", "gpu_details": None,
            "hardware_verified": False,
            "note": "AMD Token Factory documentation describes AMD-hosted models; GPU hardware is not independently attested."}


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
    """Authenticated model readiness with no inference."""
    result = {**config.public(), "ready": False, "checks": {}, "failure": None,
              "hardware": hardware_status(config), "server_metadata": {}}
    try:
        with httpx.Client(timeout=httpx.Timeout(min(config.timeout_seconds, 10), connect=5),
                          trust_env=False, follow_redirects=False) as client:
            headers = {"Authorization": "Bearer " + config.api_key}
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
    except Exception as exc:
        result["failure"] = error_category(exc)
    return result
