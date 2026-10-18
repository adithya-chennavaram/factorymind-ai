# FactoryMind AI: AMD ACT III Token Factory integration

This uses the official ACT III team's **shared managed model API**. It does **not** require AMD GPU cloud credits, provisioning, vLLM, ROCm, or a Docker instance.

## Confirmed from the team's Token Factory manual

- API URL: `https://developer.amd.com.cn/radeon/api/v1`
- Authentication: `Authorization: Bearer <team-key>` (the key looks like `rc-` plus 48 hex characters)
- Available model list: authenticated `GET /models` (model catalog changes)
- Chat: `POST /chat/completions` with OpenAI-compatible messages and `response_format={"type":"json_object"}`
- Recommended first model: `DeepSeek-V4-Flash` (text, no thinking by default). Another catalog choice: `Qwen3.8-Flash-Next` (reasoning tokens may consume `max_tokens`).
- Rate limits are shared across the team. Typical maximums: 30 RPM per key, 20 RPM per account, 8 concurrent requests. Daily quota is returned by the platform; do not assume a fixed amount.
- API-generated answers are development/trial use only; avoid sensitive data. Model selection and output limitations follow the manual.

## Add the key without putting it in a file or shell history

Open a fresh **Mac Terminal** window and run:

```bash
cd factorymind-ai
source .venv/bin/activate
export LLM_PROVIDER=amd_tokenfactory
export LLM_BASE_URL=https://developer.amd.com.cn/radeon/api/v1
export LLM_MODEL=DeepSeek-V4-Flash
export LLM_TIMEOUT_SECONDS=60
export LLM_MAX_TOKENS=300
export LLM_TEMPERATURE=0
read -s LLM_API_KEY
export LLM_API_KEY
printf '\nAPI key loaded privately (not displayed).\n'
```

When `read -s` waits for input, paste the private team key and press Return. Nothing will be echoed. Never put the key into a source file, commit, issue, screenshot or chat. Use that same terminal for subsequent commands. **Use another terminal for the second server and repeat the exports there**, or start just Streamlit first.

## First test (does use one hosted model request)

```bash
python -m scripts.probe_llm --cycle 213
```

Inspect its stdout for `ready: true`, `investigation_passed: true` and open `artifacts/provider_probe.json` to confirm `served_by: amd_tokenfactory` and the actual returned model. Do not publish that probe until it is reviewed and sanitized. The model inventory preflight uses only `GET /models`, with no invalid-key probes or self-hosted `/health` call.

Start the app using the same environment:

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

If using the API in another terminal, repeat the environment exports and private key input there before starting `uvicorn factorymind.api:app --host 127.0.0.1 --port 8001`.

## What we can and cannot claim

Successful `amd_tokenfactory` output plus the documentation supports the description **AMD-managed hosted inference**. The API does not independently report a GPU model/VRAM/ROCm version. Do not fabricate those details or label requests as self-hosted vLLM. A fallback is explicitly `rule_based` and **is not** counted as AMD inference.

## Failures

- `authentication`: wrong or inactive team key; do not paste it anywhere to debug.
- `model_not_available`: selected model retired; check the catalog in the portal or `GET /models` with your private key.
- `rate_limited`: shared quota; wait for the documented `Retry-After` and avoid automatic high-volume requests.
- `connection`/`timeout`: check network and retry once after a short interval.
- `invalid_response`: model did not return valid bounded JSON; preserve the safe fallback and test a different model or generation budget.

Benchmark sequentially with one or two cases first to respect team quota. Never present client-side hardware inventory as AMD GPU execution evidence. Do not commit model artifacts or user secrets.
