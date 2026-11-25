# LLM Research, Cost Analysis and Model Selection

**Research date:** 2026-11-25
**Scope:** choose the LLM for an API that reads one website (cleaned text, ~6K tokens) and returns a ~3.5K-token structured JSON analysis (brand, competitors, 5 search queries, SEO, 5 services, audience).

> **How to read this document.** Every price, context window and capability below was read from a
> **provider's own documentation** on the research date. Anything I could *not* confirm from an official page is
> listed in [§7 Open verification items](#7-open-verification-items) rather than stated as fact.
> **No benchmark was run.** Speed and accuracy statements are qualitative judgements and are labelled as such.
> Prices change often — re-check the linked pages before relying on a number (two of the prices below change within five weeks of the research date).

---

## 1. Sources

| Provider | Page | How it was accessed |
|---|---|---|
| Google | <https://ai.google.dev/gemini-api/docs/pricing> (redirects to `aistudio.google.com/docs/pricing`) | Fetched in full |
| Google | <https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash> | Official Google Cloud guide; read from search result |
| Google | <https://ai.google.dev/gemini-api/docs/generate-content/structured-output> | Official docs; read from search result (REST fields) |
| OpenAI | <https://developers.openai.com/api/docs/pricing.md> | Official pricing table; read from search result |
| OpenAI | <https://developers.openai.com/api/docs/models/gpt-6.1-terra.md> | Official model page; read from search result |
| Anthropic | <https://platform.claude.com/docs/en/models/sonnet-5-5/overview> | Fetched in full |
| Anthropic | <https://platform.claude.com/docs/en/models/haiku-5-5/overview> | Fetched in full |
| Anthropic | <https://platform.claude.com/docs/en/build-with-claude/structured-outputs> | Official docs; read from search result |
| Tavily | <https://docs.tavily.com/documentation/api-credits.md> | Official credit docs; read from search result |

Third-party price aggregators appeared in search results and disagreed with each other; **none were used** for the figures below.

---

## 2. Candidate models

Prices are USD per **1 million tokens**, standard (non-batch, non-cached) rates.

| Criteria | **OpenAI** | **Google Gemini** | **Anthropic Claude** |
|---|---|---|---|
| Model (mid-tier candidate) | GPT-6.1 Terra | **Gemini 3.8 Flash** | Claude Sonnet 5.5 |
| Model ID | `gpt-6.1-terra` | `gemini-3.8-flash` | `claude-sonnet-5-5` |
| Input price | $1.00 | **$0.75** until 2026-12-31, then **$1.50** | $2.00 |
| Output price | $5.00 | **$3.75** until 2026-12-31, then **$7.50** | $10.00 |
| Context window | 1,050,000 tokens | 1,048,576 tokens | 1,000,000 tokens |
| Max output | 128,000 | 65,536 | 128,000 |
| Structured JSON output | Yes — Structured Outputs (JSON Schema) listed as supported | Yes — `responseMimeType` + `responseJsonSchema` | Yes — `output_config.format` JSON Schema (documented as a stable feature) |
| Free tier for the API | Not stated on the pages read | **Yes** (listed on the pricing page) | Not stated on the pages read |
| Response speed | *Not measured.* Reasoning effort is configurable, which trades latency for quality | *Not measured.* `thinking_level` low/medium/high; this project uses `low` | *Not measured* |
| Accuracy for business/website analysis | *Qualitative:* strong general reasoning expected at this tier; not benchmarked here | *Qualitative:* same expectation for a "Flash"-tier flagship; not benchmarked here | *Qualitative:* same expectation; not benchmarked here |
| Python integration | Official SDK; plain REST also trivial | Official SDK; **plain REST used here** (no SDK dependency) | Official SDK; plain REST also trivial |
| Verdict | **Recommended fallback** | **Recommended primary** | Strong alternative, 2× the primary's current cost |

### Cheaper tiers (priced from the same official pages)

| Model | Input | Output | Context | Note |
|---|---|---|---|---|
| Gemini 3.5 Flash-Lite | $0.30 | $2.50 | — | Free tier available; same API as the primary → config-only cost-down |
| GPT-6 Luna | $0.10 | $0.50 | *not checked* | Cheapest OpenAI tier |
| Claude Haiku 5.5 | $0.10 (≤100K-token prompts), $0.50 above | $0.50, $2.50 above | 1,000,000 | 128K max output; the page notes a tokenizer change that can raise token counts vs older Haiku models |

Other Google prices seen on the official page: Gemini 3.1 Flash-Lite $0.25 / $1.50; Gemini 4 Hydrogen $1.50 / $7.50 (free tier);
Gemini 4 Argon $2.00 / $10.00 until 2026-11-30, then $4.00 / $20.00 (no free tier).

---

## 3. Evaluation against the assignment criteria

| Criterion | Assessment |
|---|---|
| **Accuracy** | The task is summarisation + structured reasoning over supplied text, not frontier reasoning. A mid-tier flagship (Flash / Terra / Sonnet) is the sensible quality tier; the cheapest tiers (Luna, Haiku, Flash-Lite) are a plausible cost-down but their quality on competitor/market reasoning was **not tested**. Accuracy is protected mainly by *design*, not model choice: evidence-only prompting, Pydantic validation, and server-side provenance checks (see README). |
| **Speed** | Latency is dominated by *output* generation (~5K tokens incl. thinking) plus the website fetch. **Not measured.** `thinking_level=low` is used to keep it down; `LLM_TIMEOUT` defaults to 90 s. |
| **Context window** | All candidates offer ≈1M tokens; the app sends ≈10K. Context size does not discriminate. |
| **Structured output** | All three support JSON-Schema-constrained output. Gemini's REST field names were confirmed on the official page; the code *also* validates every answer with Pydantic and falls back to a prompt-embedded schema if a provider rejects the schema. |
| **Cost** | See §4. At ≈10K in / 5K out all candidates cost between $0.0035 and $0.07 per analysis. |
| **Ease of Python integration / reliability** | All three are simple REST + JSON. Gemini is called with HTTPX directly (key in a header, never a URL). No reliability data was measured. |
| **Fit for an assessment** | Gemini 3.8 Flash has a **free tier**, so the author and reviewers can run the project without a paid account. |

---

## 4. Cost estimate per analysis

**Formula:** `cost = (input_tokens / 1,000,000 × input_price) + (output_tokens / 1,000,000 × output_price)`

**Assumptions (estimates, not measurements):**

| Item | Tokens | Basis |
|---|---|---|
| Cleaned page text | ≈ 6,000 | `LLM_PROMPT_MAX_CHARS=24000` ÷ ≈4 chars/token |
| Metadata, headings, JSON-LD, links | ≈ 1,500 | bounded by extractor caps |
| System prompt + task instructions | ≈ 1,500 | fixed text |
| JSON Schema | ≈ 1,000 | `responseJsonSchema` (conservatively counted as input) |
| **Total input** | **≈ 10,000** | one main call |
| JSON output | ≈ 3,500 | five sections with bounded list sizes |
| Thinking tokens (`low`) | ≈ 1,500 | billed as output |
| **Total output** | **≈ 5,000** | |

| Model | Input cost | Output cost | **Per analysis** | **Per 1,000 analyses** |
|---|---|---|---|---|
| **Gemini 3.8 Flash — until 2026-12-31** ($0.75 / $3.75) | $0.00750 | $0.01875 | **$0.02625** | $26.25 |
| Gemini 3.8 Flash — from 2027-01-01 ($1.50 / $7.50) | $0.01500 | $0.03750 | **$0.05250** | $52.50 |
| Gemini 3.5 Flash-Lite ($0.30 / $2.50) | $0.00300 | $0.01250 | $0.01550 | $15.50 |
| GPT-6.1 Terra ($1.00 / $5.00) — *fallback* | $0.01000 | $0.02500 | $0.03500 | $35.00 |
| Claude Sonnet 5.5 ($2.00 / $10.00) | $0.02000 | $0.05000 | $0.07000 | $70.00 |
| Claude Haiku 5.5 ($0.10 / $0.50, ≤100K prompt) | $0.00100 | $0.00250 | $0.00350 | $3.50 |
| GPT-6 Luna ($0.10 / $0.50) | $0.00100 | $0.00250 | $0.00350 | $3.50 |

Notes:

* **Worst case** (one repair call for invalid JSON) is ≈ 2× the figures above. A repeat of the same URL within `CACHE_TTL_SECONDS` costs **$0**.
* **Search API cost (reported separately, optional):** Tavily's `basic` search costs 1 credit; this app issues 2 searches per uncached analysis = **2 credits**. The free plan includes 1,000 credits per month (≈500 analyses). Pay-as-you-go pricing was *not* confirmed on an official page in this research — check Tavily's pricing page.
* The running app can compute an **estimate** from real token usage if you set `LLM_INPUT_PRICE_PER_MTOK` / `LLM_OUTPUT_PRICE_PER_MTOK`
  (`metadata.llm.estimated_cost_usd`). It is an estimate from configured prices, **not billing data**.
* **Expected latency:** 15–45 s end-to-end is an *assumption* (not measured): a ≤20 s fetch budget plus generation of ≈5K tokens.

---

## 5. Final recommendation

| | |
|---|---|
| **Primary provider** | Google Gemini |
| **Primary model** | `gemini-3.8-flash` (default in `app/config.py`; override with `LLM_MODEL`) |
| **Fallback (cross-vendor)** | OpenAI `gpt-6.1-terra` — same price class (~$0.035/analysis), strict JSON-Schema structured output, protects against a single-vendor outage |
| **Budget option** | `gemini-3.5-flash-lite` (config-only change, ≈$0.0155/analysis) or Claude Haiku 5.5 / GPT-6 Luna (≈$0.0035) — acceptable if quality testing on real sites is satisfactory |
| **Estimated input / output tokens** | ≈ 10,000 / ≈ 5,000 |
| **Estimated cost per analysis** | **≈ $0.026** (promo price, until 2026-12-31) → **≈ $0.053** from 2027-01-01 — *estimates* |
| **Expected latency** | 15–45 s (*assumption, not measured*) |

**Why Gemini 3.8 Flash**

1. **Right quality tier for the job** — a flagship Flash model rather than a lite tier, for competitor/market reasoning, at a mid-range price.
2. **Free API tier** — the author and reviewers can run the full flow at no cost.
3. **Native structured output** with JSON Schema over plain REST, so the integration needs no vendor SDK.
4. **Ample context** (≈1M tokens) — page text never has to be aggressively cut.
5. Per-analysis cost is a few cents even after the price change; the difference to the cheapest tier is ~$0.02–0.05 per analysis, which is negligible at assessment volumes.

**Limitations and trade-offs**

* **Price change:** `gemini-3.8-flash` doubles on 2027-01-01. Budget accordingly or switch to `gemini-3.5-flash-lite`.
* **Free-tier terms:** the official pricing page indicates free-tier traffic may be used to improve Google's products. That is acceptable for public website content but not for confidential material; use a paid key for anything private.
* **Only the Gemini adapter is implemented.** The provider abstraction (`LLMProvider` + `PROVIDER_REGISTRY`) makes adding the fallback a small, isolated change; the fallback is a *recommendation*, not shipped code.
* **No quality benchmark** was run; model quality on real-world sites should be spot-checked before relying on it.
* **`thinkingLevel` / schema dialect:** request fields follow Google's REST docs but were not exercised against the live API during development (no network access in the authoring sandbox). The code retries without the schema if the provider rejects it; set `LLM_THINKING_LEVEL=` (blank) if a model rejects the thinking parameter.

---

## 6. Cost-optimisation strategies (implemented unless noted)

| Strategy | Where / how |
|---|---|
| **HTML cleaning** | Scripts, styles, nav, footer, cookie banners, hidden elements removed before any text reaches the LLM (`html_parser.py`) |
| **Content de-duplication** | Duplicate lines/headings removed; boilerplate lines (© / "all rights reserved") dropped |
| **Token limits** | Extraction cap `MAX_CONTENT_LENGTH`; prompt cap `LLM_PROMPT_MAX_CHARS` (24K chars ≈ 6K tokens); output cap `LLM_MAX_OUTPUT_TOKENS`; bounded list sizes in the schema |
| **Prompt compression** | Only evidence-bearing fields are sent (title, description, headings, JSON-LD, service hints, text); link lists are capped |
| **Appropriate model selection** | Mid-tier Flash model + `thinking_level=low`; lite tiers available by config |
| **Caching** | In-memory TTL cache keyed by normalized URL (`CACHE_TTL_SECONDS`, default 900 s) — repeats cost $0. *Per-process; use Redis for multi-instance deployments (not implemented).* |
| **Avoiding repeated calls** | Exactly one main LLM call; at most one repair call; no retry on auth/validation-of-request errors; `Retry-After` honoured |
| **Structured output** | Schema-constrained JSON avoids prose and reduces repair calls |
| **Selective external research** | Optional; skipped entirely without `TAVILY_API_KEY`; capped at 2 searches per uncached analysis |
| **Abuse protection** | Per-client rate limit (`RATE_LIMIT_PER_MINUTE`) so a public deployment cannot burn the LLM budget |
| *Not implemented* | Provider prompt caching and batch APIs (discounted rates) — useful only at much higher volume |

---

## 7. Open verification items

Things **not** confirmed from an official page during this research. Do not treat them as established:

1. Gemini 3.8 Flash's context/output limits were read from Google's *Cloud* developer guide, not from the Gemini API model page itself (`ai.google.dev/gemini-api/docs/models/...` was not fetched).
2. Claude structured-output support was confirmed as a documented stable feature; **per-model** support for Sonnet 5.5 / Haiku 5.5 was not individually confirmed.
3. GPT-6 Luna's context window and structured-output support were not checked.
4. Free-tier rate limits and whether Anthropic/OpenAI offer sign-up credits were not checked.
5. Tavily pay-as-you-go pricing was not confirmed on an official page.
6. No latency or quality measurements exist for any model.
7. Whether the live Gemini API accepts every keyword in the generated schema (and `thinkingLevel: "low"`) is untested; the fallbacks described in §5 cover the likely failure modes.
