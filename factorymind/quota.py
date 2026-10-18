"""Small process/session guard for public demonstration LLM requests.

This is quota protection, not authentication or a security boundary. Process
state resets when Streamlit restarts and each browser session has its own
counter.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import MutableMapping


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    reason: str | None = None
    retry_after_seconds: int = 0
    session_count: int = 0
    process_count: int = 0


class InvestigationQuota:
    _lock = threading.Lock()
    _process_count = 0

    def __init__(self, cooldown_seconds: float = 8.0, session_limit: int = 8,
                 process_limit: int = 60):
        if cooldown_seconds < 0 or session_limit < 1 or process_limit < 1:
            raise ValueError("Quota limits must be positive")
        self.cooldown_seconds = float(cooldown_seconds)
        self.session_limit = int(session_limit)
        self.process_limit = int(process_limit)

    @classmethod
    def from_env(cls) -> "InvestigationQuota":
        try:
            return cls(
                cooldown_seconds=float(os.getenv("FACTORYMIND_LLM_COOLDOWN_SECONDS", "8")),
                session_limit=int(os.getenv("FACTORYMIND_LLM_SESSION_LIMIT", "8")),
                process_limit=int(os.getenv("FACTORYMIND_LLM_PROCESS_CAP", "60")),
            )
        except (TypeError, ValueError):
            return cls()

    @classmethod
    def reset_process_count(cls) -> None:
        with cls._lock:
            cls._process_count = 0

    def attempt(self, session: MutableMapping[str, object], now: float | None = None) -> QuotaDecision:
        current = time.monotonic() if now is None else float(now)
        count = int(session.get("_factorymind_llm_count", 0))
        last = session.get("_factorymind_llm_last_at")
        process_count = type(self)._process_count
        if count >= self.session_limit:
            return QuotaDecision(False, "session_limit", session_count=count, process_count=process_count)
        if isinstance(last, (int, float)):
            remaining = self.cooldown_seconds - (current - float(last))
            if remaining > 0:
                return QuotaDecision(False, "cooldown", max(1, int(remaining + 0.999)), count, process_count)
        with self._lock:
            if type(self)._process_count >= self.process_limit:
                return QuotaDecision(False, "process_cap", session_count=count, process_count=type(self)._process_count)
            type(self)._process_count += 1
            process_count = type(self)._process_count
        session["_factorymind_llm_count"] = count + 1
        session["_factorymind_llm_last_at"] = current
        return QuotaDecision(True, session_count=count + 1, process_count=process_count)


def quota_message(decision: QuotaDecision) -> str:
    if decision.reason == "cooldown":
        return f"Please wait about {decision.retry_after_seconds} seconds before another AMD request."
    if decision.reason == "session_limit":
        return "This browser session has reached its demonstration request limit. Predictions, charts and the rule-based review remain available."
    if decision.reason == "process_cap":
        return "The shared demonstration quota is temporarily paused. Predictions, charts and the rule-based review remain available."
    return "The AMD request is temporarily unavailable. The local evidence workflow remains available."
