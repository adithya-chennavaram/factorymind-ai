# AMD Instinct Developer Cloud — gated deployment runbook

> **Current owner update — Token Factory:** The main Intelligent Industry track uses organizer-issued AMD-hosted model API access at https://notebooks.amd.com/tokenfactory. The registered email is not recognized; organizer escalation is pending. Self-hosted GPU deployment is superseded for this track. Retain the instructions below as historical/optional preparation only; do not provision or deploy. Token Factory integration is not yet implemented. Follow `docs/HANDOFF.md` (or `HANDOFF.md` from this directory) for the current next steps.


**Status: prepared locally; AMD access pending.** GPU model, available VRAM, ROCm/driver versions and container restrictions are unknown. No cloud hardware, image tag/digest, model revision or runtime result has been selected or invented. No account, instance, paid resource or public endpoint was created.

Requested promotional credits: **$100, approval pending**. Maximum out-of-pocket spend: **$0 without explicit approval**. Desired allocation: **exactly one MI300X / gfx942**, advertised 192 GB VRAM, 20 vCPUs, 240 GB RAM and compatible Ubuntu/Linux. This is a request, not allocated hardware. Refuse eight-GPU offers; do not mask a larger allocation with a visibility variable. The collector/planner reject more than one visible device, while the owner must verify the billed allocation itself is single-GPU.

Before provisioning, inspect actual account pricing, credit approval/expiry/coverage, minimum billing period, GPU count, storage charges and shutdown/deletion semantics. Record the approved cap and deletion deadline. No provisioning has been authorized by this preparation work.

The owner must explicitly approve the allocation and permitted credit/billing usage before any GPU host is provisioned or used. Instructions below apply only to an already approved host. Do not paste credentials into chat, tracked files, URLs or terminal command arguments.

## References and selection policy

Checked October 12, 2026:

- [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html): advertised offerings do not establish our actual allocation.
- [vLLM ROCm installation and GPU compatibility](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/): select a release compatible with the observed architecture and host driver. A successful PyTorch check does not prove that every vLLM kernel/model configuration works.
- [Official Docker deployment](https://docs.vllm.ai/en/stable/deployment/docker/): candidate repository `vllm/vllm-openai-rocm`. Pin an actual tested digest, not a floating `latest` tag. Do not use a CUDA image or assume CPU fallback counts as AMD inference.
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct): instruction models with Apache-2.0 model cards and documented vLLM usage. Review the exact downloaded revision and license before redistribution.

The preferred model is Qwen2.5-7B-Instruct. Our **conditional candidates** are Qwen2.5-7B-Instruct if 75% of measured free VRAM is at least 24 GiB, or Qwen2.5-1.5B-Instruct if that budget is at least 8 GiB. Below that, the planner refuses. These are conservative project admission estimates, not measured memory footprints or vendor requirements. The planner uses measured free memory, not advertised capacity. It reserves headroom, caps vLLM at 80% of total VRAM, uses one GPU, FP16, 4,096 context tokens and two concurrent sequences. Final suitability still requires a successful model load and bounded-investigation evaluation. No actual model is selected until a real inventory is supplied. A smaller selection records its memory-budget reason in the plan. If 7B fits but fails actual loading/quality tests, preserve the sanitized failure evidence, diagnose configuration first, and document any smaller-model change; never silently switch models.

## 1. Observe the approved host

On Linux, collect non-secret evidence using available tools:

```bash
mkdir -p evidence
python3 scripts/amd_host.py preflight --output evidence/host-preflight.json
```

The read-only preflight captures `amd-smi version`, `static`, memory metrics, `rocminfo`, Docker/Compose versions and device read/write permissions. Missing tools or access fail readiness; no driver installation or permission changes are attempted. Review locally before sharing because driver inventory can include machine identifiers. Never install a new ROCm driver over the provider image without verified compatibility. See [AMD SMI CLI documentation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html); command availability must be checked against the allocated version. Obtain actual allocation type, GPU partition, VRAM, host driver/ROCm version and Docker/device-access restrictions from the approved account. Do not assume full-device memory, exclusive access, or a particular Instinct model from the cloud marketing page. Limit the host firewall to authorized SSH sources; do not open ports 8000, 8001 or 8501 publicly.

Select an exact candidate tag from the upstream release documentation **after** checking these observations. Verify its host-driver support. Pull that reviewed image only on the approved host, then record its immutable digest:

```bash
export VLLM_CANDIDATE='vllm/vllm-openai-rocm:<reviewed-compatible-tag>'
docker pull "$VLLM_CANDIDATE"
docker image inspect "$VLLM_CANDIDATE" --format '{{index .RepoDigests 0}}' > evidence/image-digest.txt
read -r VLLM_IMAGE < evidence/image-digest.txt
export VLLM_IMAGE
```

`<reviewed-compatible-tag>` is an explicit placeholder; there is no preselected tag in this project. If the host requires broader container permissions, review the upstream recipe and host policy before changing them. Never add `--privileged` to get past a driver problem.

## 2. Test that image on the actual GPU

After the owner authorizes transferring the necessary project source to the host, run the collector inside the exact image. It imports the image's PyTorch/vLLM, requires HIP, measures free/total VRAM and executes a tiny matrix multiplication on the single visible GPU. It does not install software or start an inference server.

```bash
docker run --rm --group-add=video --device /dev/kfd --device /dev/dri \
  -v "$PWD/scripts/collect_amd_hardware.py:/probe.py:ro" \
  --entrypoint python3 "$VLLM_IMAGE" /probe.py \
  --image-digest "$VLLM_IMAGE" > evidence/amd_hardware.json
```

This uses PyTorch's `torch.cuda` API on ROCm; the collector explicitly requires `torch.version.hip`, so a CUDA/CPU runtime is rejected. The resulting JSON records observed names, architectures, memory, versions, compute-check outcome, capture time and a unique deployment ID. Unsupported architectures or unusable evidence stop automatic planning. Our conservative planner currently covers Instinct `gfx90a`, `gfx942` and `gfx950`; any other architecture needs separate review.

GPU enumeration must remain identical between the probe and server launch. If the allocation changes, is partitioned differently or restarts, recollect. Free memory is a snapshot, so rerun immediately before launch on a shared host.

## 3. Produce a plan without launching anything

Copy only reviewed, non-secret hardware evidence to the Mac's `artifacts/amd_hardware.json`. In the project environment:

```bash
source .venv/bin/activate
python scripts/prepare_amd.py --hardware artifacts/amd_hardware.json
```

This writes `artifacts/amd_plan.json` containing the measured GPU, tested image digest, conditional model selection and memory settings. It cannot create cloud resources or run Docker. It rejects absent/stale/failed compute evidence and insufficient free VRAM.

Review the selected model card, then take the immutable 40-character commit from its official repository. Generate the complete non-secret Compose configuration:

```bash
python scripts/prepare_amd.py --hardware artifacts/amd_hardware.json \
  --model-revision <reviewed-model-commit> \
  --compose-env artifacts/amd-compose.env
```

The generated env file intentionally excludes the API token. The image is pinned to the digest that passed the probe; the model revision is explicit. Fresh evidence is required (within 24 hours), but freshness alone does not establish continued allocation identity.

## 4. Launch privately, only after approval

On the approved Linux host, use the same hardware record/plan and non-secret env file. Generate a private random token in the owner's secret manager and enter it at a hidden prompt. Do not run `docker compose config` with a real token: expanded configuration can print it.

```bash
# Linux bash; keep shell tracing disabled.
read -rsp 'Private vLLM API key: ' VLLM_API_KEY
export VLLM_API_KEY
# Use a random value of at least 16 characters, not a placeholder.
python3 scripts/amd_host.py start --authorized-host
python3 scripts/amd_host.py health
```

The `--authorized-host` flag acknowledges prior owner authorization; it does not grant approval. Start waits up to 600 seconds for the authenticated Docker health check and attempts container cleanup on failure. Recollect hardware and regenerate the plan immediately before launch; a prepared env file is not a freshness check.

The template binds host port 8000 to `127.0.0.1`, passes the API key via environment, exposes one selected GPU, uses an authenticated container health check, and never automatically pulls a different image (`pull_policy: never`). Model weights may download on first launch; this occurs only on the approved host within the approved budget. Docker administrators can inspect environment variables, so the host must be trusted. No API key is baked into the image or printed by project tooling.

Record startup logs showing the actual model revision, vLLM/ROCm versions, loaded GPU backend and successful loading. Observe `amd-smi`/`rocm-smi` while a real product investigation runs. Save sanitized GPU utilization and request evidence. A tiny compute probe alone is **not** proof that vLLM served the demonstrated request on AMD.

Docker is not installed in the local Mac workspace, so the Compose YAML was syntax-checked locally but no container launch has been validated. Actual Docker/ROCm validation is an account-access milestone.

## 5. Connect through SSH and configure the same application

```bash
ssh -o ExitOnForwardFailure=yes -N -L 127.0.0.1:8000:127.0.0.1:8000 USER@APPROVED_AMD_HOST
```

On the Mac, set values from the actual plan, not the placeholders below:

```bash
source .venv/bin/activate
export LLM_PROVIDER=amd_vllm
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL='factorymind-<deployment-id-from-hardware-record>'
export LLM_MODEL_ID='<selected-model-repository>'
export LLM_MODEL_REVISION='<reviewed-model-commit>'
export LLM_HARDWARE_EVIDENCE="$PWD/artifacts/amd_hardware.json"
export LLM_TIMEOUT_SECONDS=60
export LLM_DISABLE_THINKING=0
export LLM_TEMPERATURE=0
export LLM_MAX_TOKENS=300
export LLM_TOP_P=1
# macOS zsh: enter the same private token without echoing it.
read -rs 'LLM_API_KEY?Private vLLM API key: '
export LLM_API_KEY
python scripts/probe_llm.py --require-amd-evidence --output artifacts/amd_connection.json
```

The `LLM_MODEL` alias contains the deployment ID so a response can be matched against the inventory, while `LLM_MODEL_ID` and `LLM_MODEL_REVISION` identify the operator-configured weights separately. Alias matching is provenance linkage, not cryptographic attestation; do not reuse aliases across deployments.

The connection test checks live health, model availability, the real API key, and rejection of both missing and incorrect API keys. Failure stops the probe before inference. It then runs the **same application investigation** and validates the response. Timeouts, auth errors, rate limits, mismatched model names, truncated JSON and unsupported claims are recorded without raw exception text or secrets. No hidden retry to local Ollama exists. Redirect following and environment proxies are disabled; TLS verification remains enabled. Non-loopback plain HTTP is refused.

Restart the local applications with this environment:

```bash
streamlit run app.py
# Separate terminal, same environment:
uvicorn factorymind.api:app --host 127.0.0.1 --port 8001
```

`GET /health` checks classifier readiness. `GET /provider/health` separately checks the inference connection/auth/model and returns 503 on failure. LLM downtime therefore does not imply missing classifier assets. These local app endpoints are not a production-authenticated public service.

## 6. Benchmark and compare, without fabricated AMD results

Before the benchmark, in a second terminal **on the approved host**, sample GPU memory/activity over the test window (use the real deployment ID):

```bash
python3 scripts/amd_host.py monitor --deployment-id <actual-32-hex-id> \
  --duration 120 --interval 1 --output evidence/gpu-telemetry.json
```

The monitor records timestamped AMD SMI memory/activity JSON and Linux amdgpu sysfs used/total VRAM in bytes. Missing measurements remain absent. Samples are host-wide, not per-request allocations or exact peak usage. Synchronize host/Mac clocks, verify one GPU and observe vLLM activity; alias/time matching is not physical attestation. Extend the bounded duration if the benchmark is slower. Transfer only reviewed non-secret telemetry to the Mac after collection.

With the AMD environment active:

```bash
python scripts/benchmark_inference.py --output artifacts/amd_benchmark.json
python scripts/benchmark_inference.py --compare artifacts/ollama_benchmark.json artifacts/amd_benchmark.json \
  --output artifacts/inference_comparison.json
```

Attach the real overlapping samples after both commands finish:

```bash
python scripts/benchmark_inference.py --attach-gpu-telemetry artifacts/amd_benchmark.json artifacts/gpu-telemetry.json \
  --output artifacts/amd_benchmark_with_memory.json
```

The attachment rejects wrong deployment IDs and missing overlapping single-GPU byte samples. Keep original and enriched records. Mechanical quality checks record accepted schema, exact factual/catalog rendering, human-approval requirement and absence of equipment control. Expert usefulness/inspection relevance remains a separate pending human review; neither valid JSON nor classifier agreement proves a correct diagnosis.

The AMD benchmark requires matching recent operator GPU evidence. It records six identical held-out cases, one separately recorded warmup, three sequential repeats, returned model, provider/endpoint ID, configured weights, GPU evidence, token usage, application end-to-end latency, throughput and all failure categories. A fallback counts as failure, not successful AMD output. Missing token usage remains null. See `BENCHMARKS.md` for methodology and the actual Ollama baseline.

For a local baseline, launch a fresh shell with `LLM_PROVIDER=ollama`, `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=llama3.2`, `LLM_API_KEY=ollama`; unset remote model/evidence variables. No application logic changes are necessary.

## Evidence and completion boundary

The UI says **Local Ollama**, **vLLM · GPU unverified**, **vLLM · matching operator AMD evidence**, or **Rule-based fallback** for the actual result. Fallbacks never inherit a GPU claim. `hardware_verified` stays false because client software cannot remotely attest this physical fact. The owner/engineer must review the server allocation, compute probe, model-loading logs, GPU activity during inference and matched request evidence before claiming verified AMD use. The UI uses the narrower evidence label even after a match.

## 7. Cleanup and billing verification

On the host:

```bash
python3 scripts/amd_host.py stop
```

Close the SSH tunnel with Ctrl-C. After preserving reviewed evidence, remove the project model cache if it will no longer be used (`docker volume ls --filter label=com.docker.compose.project=factorymind-amd`, then remove only the confirmed project cache volume). Do not remove unrelated volumes. The default stop preserves weights to avoid accidental destruction; it does not authorize continued storage charges.

Through the approved AMD account, explicitly terminate/delete the test instance, remove unnecessary attached billable disks/snapshots/reserved IPs, and verify the resource list and billing meter. Record termination time and final charges/credit use. Revoke the temporary API token. Provider-specific deletion buttons/commands and prices must be checked against the actual account; they are not invented here.

Stop the server and SSH tunnel after testing. `docker compose ... down` stops the container but does **not** stop instance billing; shut down the cloud instance through the approved provider and confirm costs have stopped. Preserve reviewed evidence before deleting storage. No public publication, credential sharing or competition submission is automated.
