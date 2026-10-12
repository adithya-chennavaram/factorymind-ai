"""Read-only host inventory/telemetry and private Compose lifecycle on an approved host.

Standard library only. Never provisions an instance, changes drivers or prints secrets.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent


def observed(command, timeout=20):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        # Failed command diagnostics may contain host details: retain only the status.
        return {"ok": p.returncode == 0, "returncode": p.returncode,
                "stdout": p.stdout if p.returncode == 0 else None}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": type(exc).__name__, "stdout": None}


def memory_snapshot(root=Path('/sys/class/drm')):
    """Linux amdgpu sysfs values are bytes; never guess AMD SMI output units."""
    rows = []
    for card in sorted(root.glob('card[0-9]*')):
        if not re.fullmatch(r'card\d+', card.name):
            continue
        device = card / 'device'
        try:
            if (device / 'vendor').read_text().strip().lower() != '0x1002':
                continue
            used = int((device / 'mem_info_vram_used').read_text())
            total = int((device / 'mem_info_vram_total').read_text())
            if not 0 <= used <= total or total <= 0:
                continue
            rows.append({'card': card.name, 'used_bytes': used, 'total_bytes': total,
                         'free_bytes': total - used})
        except (OSError, ValueError):
            continue
    return rows


def preflight():
    commands = {'amd_smi_version': ['amd-smi', 'version', '--json'],
                'amd_smi_inventory': ['amd-smi', 'static', '--json'],
                'amd_smi_memory': ['amd-smi', 'metric', '--mem-usage', '--json'],
                'rocminfo': ['rocminfo'], 'docker': ['docker', 'version'],
                'compose': ['docker', 'compose', 'version']}
    checks = {name: observed(command) for name, command in commands.items()}
    devices = [Path('/dev/kfd'), *Path('/dev/dri').glob('renderD*')]
    access = {str(p): p.exists() and os.access(p, os.R_OK | os.W_OK) for p in devices}
    return {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'commands': checks,
            'device_read_write_access': access, 'vram': memory_snapshot(),
            'ready_for_manual_compatibility_review': len(devices) > 1 and all(access.values()) and all(c['ok'] for c in checks.values()),
            'note': 'Inventory is not allocation approval. Confirm exactly one allocated GPU and driver/image compatibility; do not mask a larger allocation.'}


def monitor(deployment_id, duration, interval):
    if not re.fullmatch('[a-f0-9]{32}', deployment_id) or not 1 <= duration <= 3600 or not .2 <= interval <= 10:
        raise ValueError('Invalid telemetry bounds or deployment ID')
    samples = []
    until = time.monotonic() + duration
    while time.monotonic() < until:
        samples.append({'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                        'vram': memory_snapshot(),
                        'amd_smi': observed(['amd-smi', 'metric', '--gpu', '0', '--mem-usage', '--usage', '--json'], timeout=5)})
        time.sleep(min(interval, max(0, until - time.monotonic())))
    return {'schema_version': 1, 'deployment_id': deployment_id, 'samples': samples,
            'note': 'Host-wide sampled VRAM, not per-request allocation or exact peak. AMD SMI raw units retained. Clock alignment and single-device mapping require operator verification.'}


def lifecycle(action, env_file, authorized=False):
    if action == 'start' and not authorized:
        raise ValueError('Start requires explicit confirmation of an already authorized host/budget')
    if action == 'start' and len(os.environ.get('VLLM_API_KEY', '').strip()) < 16:
        raise ValueError('A private API key is required in the environment')
    base = ['docker', 'compose', '--project-name', 'factorymind-amd', '--env-file', str(env_file),
            '-f', str(ROOT / 'deploy/compose.amd.yaml')]
    # Preserve cleanup ability after the operator unsets the secret. Placeholder is
    # used only for interpolation during down, never to start an endpoint.
    if action == 'stop':
        os.environ.setdefault('VLLM_API_KEY', 'unused-during-container-removal')
    commands = {'start': ['up', '-d', '--wait', '--wait-timeout', '600'], 'stop': ['down', '--timeout', '30']}
    if action == 'health':
        result = observed(['docker', 'inspect', '--format', '{{.State.Health.Status}}', 'factorymind-vllm'])
        result['ok'] = result['ok'] and (result['stdout'] or '').strip() == 'healthy'
    else:
        result = observed(base + commands[action], timeout=660 if action == 'start' else 60)
        # Compose output can contain expanded values; don't export it.
        result['stdout'] = None
        if action == 'start' and not result['ok']:
            cleanup = observed(base + commands['stop'], timeout=60)
            result['cleanup_ok'] = cleanup['ok']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'monitor', 'start', 'health', 'stop'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deployment-id')
    parser.add_argument('--duration', type=int, default=120)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--env-file', type=Path, default=ROOT / 'artifacts/amd-compose.env')
    parser.add_argument('--authorized-host', action='store_true', help='Confirm owner already authorized this host and budget; this flag does not obtain approval')
    args = parser.parse_args()
    try:
        if args.action == 'preflight': result = preflight()
        elif args.action == 'monitor': result = monitor(args.deployment_id or '', args.duration, args.interval)
        else: result = lifecycle(args.action, args.env_file, args.authorized_host)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
            print(f'Evidence saved to {args.output}; review locally before sharing.')
        else: print(json.dumps(result, indent=2))
        if result.get('ok') is False or result.get('ready_for_manual_compatibility_review') is False:
            raise SystemExit(1)
    except (OSError, ValueError, KeyError):
        raise SystemExit('Host operation refused or unavailable; inspect configuration locally. No resource was provisioned.') from None


if __name__ == '__main__':
    main()
