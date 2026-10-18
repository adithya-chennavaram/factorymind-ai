from factorymind.quota import InvestigationQuota


def test_session_cooldown_and_limit():
    InvestigationQuota.reset_process_count()
    quota = InvestigationQuota(cooldown_seconds=5, session_limit=2, process_limit=10)
    session = {}
    assert quota.attempt(session, now=100).allowed
    blocked = quota.attempt(session, now=102)
    assert not blocked.allowed and blocked.reason == "cooldown"
    assert quota.attempt(session, now=106).allowed
    blocked = quota.attempt(session, now=112)
    assert not blocked.allowed and blocked.reason == "session_limit"


def test_process_cap_is_shared_between_gate_instances():
    InvestigationQuota.reset_process_count()
    first = InvestigationQuota(cooldown_seconds=0, session_limit=5, process_limit=1)
    second = InvestigationQuota(cooldown_seconds=0, session_limit=5, process_limit=1)
    assert first.attempt({}, now=1).allowed
    blocked = second.attempt({}, now=2)
    assert not blocked.allowed and blocked.reason == "process_cap"
