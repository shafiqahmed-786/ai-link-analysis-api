# LLM Research, Cost Analysis and Model Selection (Groq)

**Research date:** 2026-11-25
**Scope:** choose the LLM for an API that reads one website (cleaned text, ~6K tokens) and returns a ~3.5K-token structured JSON analysis (brand, competitors, 5 search queries, SEO, 5 services, audience).
**Provider:** Groq (OpenAI-compatible API). This replaces the previous Google Gemini integration, which was retired after persistent `401/403` authentication failures.

> **How to read this document.** Every price, context window and capability below was read from a
> **Groq documentation page** on the research date. Anything I could *not* confirm from an official page is
> listed in [§7 Open verification items](#7-open-verification-items) rather than stated as fact.
> **No benchmark was run.** Speed and accuracy statements are qualitative judgements and are labelled as such.
> Prices change often — re-check the linked pages before relying on a number.

---

## 1. Sources

| Provider | Page | How it was accessed |
|---|---|---|
| Groq | <https://console.groq.com/docs/structured-outputs> | Fetched in full (strict vs best-effort modes, supported models) |
| Groq | <https://console.groq.com/docs/model/openai/gpt-oss-120b> | Fetched in full (pricing, context, limits, speed) |
| Groq | <https://console.groq.com/docs/model/openai/gpt-oss-20b> | Fetched in full (pricing, context, limits, speed) |
| Groq | <https://console.groq.com/docs/model/qwen/qwen3.8-27b> | Fetched in full (pricing, context, limits, speed) |
| Groq | <https://console.groq.com/docs/openai> · <https://console.groq.com/docs/api-reference> | OpenAI-compatibility and chat-completions reference |
| Tavily | <https://docs.tavily.com/documentation/api-credits.md> | Official credit docs (unchanged; optional research) |

Third-party price aggregators were **not** used for the figures below.

---

## 2. Candidate models

Prices are USD per **1 million tokens**, standard (non-batch, non-cached) rates, read from the Groq model pages.

| Criteria | **GPT-OSS 120B** | **GPT-OSS 20B** | **Qwen 3.8 27B** |
|---|---|---|---|
| Model ID | **`openai/gpt-oss-120b`** | `openai/gpt-oss-20b` | `qwen/qwen3.8-27b` (preview) |
| Input price | **$0.15** | **$0.075** | $0.80 |
| Output price | **$0.60** | **$0.30** | $4.00 |
| Cached input | $0.075 | $0.0375 | — |
| Context window | 131,072 | 131,072 | 131,072 |
| Max output tokens | 65,536 | 65,536 | 16,384 |
| Structured JSON | JSON Object + JSON Schema modes; **strict mode supported** | JSON Object + JSON Schema modes; **strict mode supported** | JSON Object + JSON Schema modes; strict mode supported |
| Reported token speed | ~500 tps | ~1000 tps | ~450+ tps |
| Modality | Text in / text out | Text in / text out | Text + images in / text out |
| Status | Generally available | Generally available | Preview |
| Verdict | **Recommended primary** | **Recommended fallback** | Higher-cost alternative |

### Groq structured-output modes (from the official docs)

* **Strict mode (`strict: true`)** — constrained decoding guarantees the output matches the schema exactly.
  Requirements: **all fields `required`** and every object sets **`additionalProperties: false`**. Supported on
  `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` (and `meta-models/muse-spark-open`).
* **Best-effort mode (`strict: false`)** — valid JSON, schema adherence not guaranteed; broader model support.
* **JSON Object mode** — guarantees a valid JSON object but not a schema; used as the fallback when a schema is rejected.

This project requests **strict mode** and, if the provider rejects the schema (HTTP 400 mentioning the schema),
falls back to JSON Object mode with the schema embedded in the prompt — Pydantic still validates every answer
(see §5 and `app/services/llm_service.py`).

---

## 3. Evaluation against the assignment criteria

| Criterion | Assessment |
|---|---|
| **Accuracy** | The task is summarisation + structured reasoning over supplied text, not frontier reasoning. GPT-OSS 120B is a high-capability model (MMLU ≈ 90.0% per Groq's model page) at a low price; 20B is a cheaper, faster option. Accuracy is protected mainly by *design*, not model choice: evidence-only prompting, strict JSON Schema, Pydantic validation, and server-side provenance checks (see README). |
| **Speed** | Groq's inference is fast (120B ≈ 500 tps, 20B ≈ 1000 tps, *reported, not measured here*). Latency is dominated by *output* generation (~5K tokens) plus the website fetch. `LLM_TIMEOUT` defaults to 90 s. |
| **Context window** | All candidates offer 131,072 tokens; the app sends ≈10K. Context size does not discriminate. |
| **Structured output** | All three support JSON-Schema-constrained output, and all three support **strict mode**. The request uses `response_format: {type: "json_schema", json_schema: {name, strict: true, schema}}` with a schema derived from the Pydantic models and adapted to strict requirements. |
| **Cost** | See §4. At ≈10K in / 5K out the candidates cost between $0.00225 and $0.028 per analysis. |
| **Ease of Python integration / reliability** | Groq exposes an OpenAI-compatible REST API; this project calls it with HTTPX directly (key in an `Authorization: Bearer` header, never a URL). No vendor SDK dependency. No reliability data was measured. |
| **Fit for an assessment** | Very low per-analysis cost (a fraction of a cent to ~3 cents) and a generous free tier make the full flow runnable by the author and reviewers. |

---

## 4. Cost estimate per analysis

**Formula:** `cost = (input_tokens / 1,000,000 × input_price) + (output_tokens / 1,000,000 × output_price)`

**Assumptions (estimates, not measurements):**

| Item | Tokens | Basis |
|---|---|---|
| Cleaned page text | ≈ 6,000 | `LLM_PROMPT_MAX_CHARS=24000` ÷ ≈4 chars/token |
| Metadata, headings, JSON-LD, links | ≈ 1,500 | bounded by extractor caps |
| System prompt + task instructions | ≈ 1,500 | fixed text |
| JSON Schema | ≈ 1,000 | strict schema (conservatively counted as input) |
| **Total input** | **≈ 10,000** | one main call |
| JSON output | ≈ 3,500 | five sections with bounded list sizes |
| Reasoning tokens | ≈ 1,500 | billed as output (included in `completion_tokens`) |
| **Total output** | **≈ 5,000** | |

| Model | Input cost | Output cost | **Per analysis** | **Per 1,000 analyses** |
|---|---|---|---|---|
| **GPT-OSS 120B — primary** ($0.15 / $0.60) | $0.00150 | $0.00300 | **$0.00450** | $4.50 |
| GPT-OSS 20B — fallback ($0.075 / $0.30) | $0.00075 | $0.00150 | **$0.00225** | $2.25 |
| Qwen 3.8 27B ($0.80 / $4.00) | $0.00800 | $0.02000 | **$0.02800** | $28.00 |

Notes:

* **Worst case** (one repair call for invalid JSON) is ≈ 2× the figures above. A repeat of the same URL within `CACHE_TTL_SECONDS` costs **$0**.
* **Search API cost (reported separately, optional):** Tavily's `basic` search costs 1 credit; this app issues 2 searches per uncached analysis = **2 credits**.
* The running app can compute an **estimate** from real token usage if you set `LLM_INPUT_PRICE_PER_MTOK` / `LLM_OUTPUT_PRICE_PER_MTOK`
  (`metadata.llm.estimated_cost_usd`). It is an estimate from configured prices, **not billing data**.
* **Expected latency:** 15–45 s end-to-end is an *assumption* (not measured): a ≤20 s fetch budget plus generation of ≈5K tokens.

---

## 5. Final recommendation

| | |
|---|---|
| **Primary provider** | Groq |
| **Primary model** | `openai/gpt-oss-120b` (default in `app/config.py`; override with `LLM_MODEL`) |
| **Fallback** | `openai/gpt-oss-20b` — same API, ~half the price, ~2× reported speed; a config-only change |
| **Budget / higher-quality alternative** | `qwen/qwen3.8-27b` (preview; higher price, multimodal) |
| **Structured output** | Strict JSON Schema (`strict: true`) with a JSON-Object-mode + prompt-schema fallback |
| **Estimated input / output tokens** | ≈ 10,000 / ≈ 5,000 |
| **Estimated cost per analysis** | **≈ $0.0045** (120B) → **≈ $0.00225** (20B) — *estimates* |
| **Expected latency** | 15–45 s (*assumption, not measured*) |

**Why GPT-OSS 120B**

1. **Right quality tier for the job** — a high-capability model for competitor/market reasoning, at a very low price.
2. **Native strict structured output** — constrained decoding guarantees schema adherence; the schema is derived from the existing Pydantic models.
3. **Ample context** (131K tokens) — page text never has to be aggressively cut.
4. **Low, predictable cost** — a fraction of a cent to ~3 cents per analysis, with a free tier for the author/reviewers.
5. **Simple, vendor-neutral integration** — OpenAI-compatible REST over HTTPX, no SDK dependency.

**Limitations and trade-offs**

* **No quality benchmark** was run; model quality on real-world sites should be spot-checked before relying on it.
* **Strict-mode schema dialect:** the schema is transformed to satisfy Groq's strict requirements (all fields required,
  `additionalProperties: false`, unsupported length/pattern keywords removed). If a future model rejects the schema, the code
  retries once in JSON-Object mode with the schema in the prompt and Pydantic still validates the answer.
* **Only the Groq adapter is implemented.** The provider abstraction (`LLMProvider` + `PROVIDER_REGISTRY`) makes adding another
  provider a small, isolated change.
* **Token speed figures are Groq's published numbers**, not measured in this project.

---

## 6. Cost-optimisation strategies (implemented unless noted)

| Strategy | Where / how |
|---|---|
| **HTML cleaning** | Scripts, styles, nav, footer, cookie banners, hidden elements removed before any text reaches the LLM (`html_parser.py`) |
| **Content de-duplication** | Duplicate lines/headings removed; boilerplate lines (© / "all rights reserved") dropped |
| **Token limits** | Extraction cap `MAX_CONTENT_LENGTH`; prompt cap `LLM_PROMPT_MAX_CHARS` (24K chars ≈ 6K tokens); output cap `LLM_MAX_OUTPUT_TOKENS`; bounded list sizes in the schema |
| **Prompt compression** | Only evidence-bearing fields are sent (title, description, headings, JSON-LD, service hints, text); link lists are capped |
| **Appropriate model selection** | Mid-tier GPT-OSS model on Groq's fast inference; cheaper `openai/gpt-oss-20b` available by config |
| **Caching** | In-memory TTL cache keyed by normalized URL (`CACHE_TTL_SECONDS`, default 900 s) — repeats cost $0. *Per-process; use Redis for multi-instance deployments (not implemented).* |
| **Avoiding repeated calls** | Exactly one main LLM call; at most one repair call; no retry on auth/invalid-request errors; `Retry-After` honoured |
| **Structured output** | Strict schema-constrained JSON avoids prose and reduces repair calls |
| **Selective external research** | Optional; skipped entirely without `TAVILY_API_KEY`; capped at 2 searches per uncached analysis |
| **Abuse protection** | Per-client rate limit (`RATE_LIMIT_PER_MINUTE`) so a public deployment cannot burn the LLM budget |
| *Not implemented* | Groq prompt caching (discounted cached-input rate) and batch APIs — useful only at much higher volume |

---

## 7. Open verification items

Things **not** confirmed from an official page during this research. Do not treat them as established:

1. Real per-request latency and output quality on the target websites were **not measured** (no live call was run in the authoring sandbox).
2. Whether the live Groq API accepts the full strict schema for `openai/gpt-oss-120b` on the first try is untested; the JSON-Object-mode
   fallback in §5 covers the likely failure mode.
3. Account-specific model permissions and rate limits depend on the Groq account/plan and were not checked.
4. Free-tier rate limits and any sign-up credits were not confirmed on an official page.
5. Tavily pay-as-you-go pricing was not confirmed on an official page.
6. `qwen/qwen3.8-27b` is a **preview** model; its availability and limits may change.