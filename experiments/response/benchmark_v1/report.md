# Response benchmark v1: prompt-only models on the tutoring response task

Generated 2026-11-02T21:20:25+00:00 by `tools.response.benchmark report` (benchmark_v1, prompt_v1, validator_v1). Every number below is an automated check from `packages.response.validate` on one sampled response per case; no human rating is included. Read the limitations before using any table.

## Models

| Model | Provider | Model ID | Reasoning | Temp. | Price in/out ($/MTok) | Cases |
|---|---|---|---|---|---|---|
| Deterministic fallback (floor) | fallback | `-` | n/a | default | free | 107 |
| OpenAI GPT-6.1 Sol (frontier) | openai_compatible | `gpt-6.1-sol` | reasoning_effort=low | default | 2.0 / 10.0 | 107 |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 | openai_compatible | `accounts/fireworks/models/qwen3p8-27b` | reasoning_effort=none | 0 | GPU 8.0/h x1 | 0 |
| Anthropic Claude Fable 5.1 | openai_compatible | `claude-fable-5-1` | provider default (thinking always on) | default | 10.0 / 50.0 | 0 |
| Anthropic Claude Opus 5.5 | openai_compatible | `claude-opus-5-5` | provider default (adaptive thinking, effort medium) | default | 4.0 / 20.0 | 0 |
| Anthropic Claude Sonnet 5.5 | openai_compatible | `claude-sonnet-5-5` | provider default (adaptive thinking) | default | 2.0 / 10.0 | 0 |
| Anthropic Claude Haiku 5.5 | openai_compatible | `claude-haiku-5-5` | provider default (adaptive thinking, effort medium) | default | 0.1 / 0.5 | 0 |
| Google Gemini 4 Argon (flagship) | openai_compatible | `gemini-4-argon` | reasoning_effort=low | default | 2.0 / 10.0 | 0 |
| OpenAI GPT-6 Astra (flagship per the models page) | openai_compatible | `gpt-6-astra` | reasoning_effort=low | default | 10.0 / 50.0 | 0 |
| OpenAI GPT-6.1 Terra | openai_compatible | `gpt-6.1-terra` | reasoning_effort=none | default | 1.0 / 5.0 | 0 |
| Qwen3.8 Max on Fireworks serverless (hosted product built on Qwen3.8-2.4T-A95B) | openai_compatible | `accounts/fireworks/models/qwen3p8-max` | reasoning_effort=none | 0 | 2.0 / 6.0 | 0 |
| OpenAI gpt-oss-120b on Fireworks serverless (Apache-2.0 open weights) | openai_compatible | `accounts/fireworks/models/gpt-oss-120b` | reasoning_effort=low | 0 | 0.15 / 0.6 | 0 |
| Qwen3.5-9B Q4_K_M, local llama.cpp (Qwen3.8 architecture family proxy) | openai_compatible | `qwen3.5-9b` | enable_thinking=false via chat_template_kwargs | 0 | free | 0 |
| Qwen2.5-7B-Instruct Q4_K_M, local llama.cpp (the PRD's original base model) | openai_compatible | `qwen2.5-7b-instruct` | not applicable (no thinking mode) | 0 | free | 0 |
| Llama-3.1-8B-Instruct Q4_K_M, local llama.cpp | openai_compatible | `llama-3.1-8b-instruct` | not applicable (no thinking mode) | 0 | free | 0 |

Entries with 0 cases did not run: hosted entries wait for an API key in the keys file, local entries wait for weights, and disabled entries record why. Prices are standard-tier list prices per million tokens on the date in the config.

## Case sets

| Case set | Cases used | Note |
|---|---:|---|
| b_fixtures | 9 |  |
| smoke | 8 |  |
| mathdial_test | 90 | stratified sample 90 of 1618 |
| probability_v1 | 0 | optional, not present at evaluation/response/probability_cases_v1.jsonl |

Total: 107 cases. A case is one TeachingPlan; the model sees only the compiled prompt (plan projection plus recent dialogue).

## Results by model

Denominator per cell is shown explicitly. `action_faithful` is counted only where the plan carries the content its heuristic needs (MathDial HINT and EXPLAIN_CONCEPT plans have no permitted hint or approved facts, so they are not applicable). "Delivered as model" means all applicable checks passed; "fallback needed" means the serving pipeline (PRD 14.1) would repair, fall back or error instead.

| Model | N | All applicable pass | Delivered as model | Fallback needed | schema_valid | action_faithful | math_supported | disclosure_allowed | length_valid | question_count_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic fallback (floor) | 107 | 43/107 (40%) | 43/107 (40%) | 64/107 (60%) | 43/107 (40%) | 43/47 (91%) | 107/107 (100%) | 107/107 (100%) | 107/107 (100%) | 107/107 (100%) |
| OpenAI GPT-6.1 Sol (frontier) | 107 | 40/107 (37%) | 40/107 (37%) | 67/107 (63%) | 107/107 (100%) | 22/47 (47%) | 51/107 (48%) | 100/107 (93%) | 107/107 (100%) | 107/107 (100%) |

| Model | Errors | Repair | Fallback | Unavailable | p50 ms | p95 ms | In tok | Out tok | Reasoning tok | Words | Over 384 visible | <think> leaks | Cost $ | $/case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic fallback (floor) | 64 | 0 | 0 | 64 | 0 | 0 | 0.0 | 0.0 | 0.0 | 43.5 | 0 | 0 | 0.0 | 0.0 |
| OpenAI GPT-6.1 Sol (frontier) | 0 | 0 | 67 | 0 | 2572 | 4578 | 583.4 | 81.5 | 3.3 | 49.8 | 0 | 0 | 0.212 | 0.001982 |

Diagnostics on the strict checks (counts of failing cases): 

| Model | math failures where every flagged number came from the learner's message | disclosure flags whose fingerprint already appears in the stem or learner text |
|---|---:|---:|
| Deterministic fallback (floor) | 0 of 0 | 0 of 0 |
| OpenAI GPT-6.1 Sol (frontier) | 45 of 56 | 7 of 7 |

## Results by move

| Model / move | N | All applicable pass | Delivered as model | Fallback needed | schema_valid | action_faithful | math_supported | disclosure_allowed | length_valid | question_count_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic fallback (floor) / CHALLENGE | 2 | 2/2 (100%) | 2/2 (100%) | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| Deterministic fallback (floor) / CHECK_MASTERY | 3 | 3/3 (100%) | 3/3 (100%) | 0/3 (0%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| Deterministic fallback (floor) / EXPLAIN_CONCEPT | 32 | 2/32 (6%) | 2/32 (6%) | 30/32 (94%) | 2/32 (6%) | 2/2 (100%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| Deterministic fallback (floor) / HINT | 32 | 2/32 (6%) | 2/32 (6%) | 30/32 (94%) | 2/32 (6%) | 2/2 (100%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| Deterministic fallback (floor) / PRACTICE | 2 | 2/2 (100%) | 2/2 (100%) | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| Deterministic fallback (floor) / PROBE | 32 | 28/32 (88%) | 28/32 (88%) | 4/32 (12%) | 28/32 (88%) | 28/32 (88%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| Deterministic fallback (floor) / REFLECT | 2 | 2/2 (100%) | 2/2 (100%) | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| Deterministic fallback (floor) / WORKED_EXAMPLE | 2 | 2/2 (100%) | 2/2 (100%) | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / CHALLENGE | 2 | 2/2 (100%) | 2/2 (100%) | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / CHECK_MASTERY | 3 | 3/3 (100%) | 3/3 (100%) | 0/3 (0%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / EXPLAIN_CONCEPT | 32 | 13/32 (41%) | 13/32 (41%) | 19/32 (59%) | 32/32 (100%) | 2/2 (100%) | 13/32 (41%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / HINT | 32 | 13/32 (41%) | 13/32 (41%) | 19/32 (59%) | 32/32 (100%) | 2/2 (100%) | 14/32 (44%) | 28/32 (88%) | 32/32 (100%) | 32/32 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / PRACTICE | 2 | 2/2 (100%) | 2/2 (100%) | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / PROBE | 32 | 5/32 (16%) | 5/32 (16%) | 27/32 (84%) | 32/32 (100%) | 9/32 (28%) | 13/32 (41%) | 29/32 (91%) | 32/32 (100%) | 32/32 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / REFLECT | 2 | 2/2 (100%) | 2/2 (100%) | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / WORKED_EXAMPLE | 2 | 0/2 (0%) | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |

## Results by case set

| Model / case set | N | All applicable pass | Delivered as model | Fallback needed | schema_valid | action_faithful | math_supported | disclosure_allowed | length_valid | question_count_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic fallback (floor) / b_fixtures | 9 | 9/9 (100%) | 9/9 (100%) | 0/9 (0%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) |
| Deterministic fallback (floor) / mathdial_test | 90 | 26/90 (29%) | 26/90 (29%) | 64/90 (71%) | 26/90 (29%) | 26/30 (87%) | 90/90 (100%) | 90/90 (100%) | 90/90 (100%) | 90/90 (100%) |
| Deterministic fallback (floor) / smoke | 8 | 8/8 (100%) | 8/8 (100%) | 0/8 (0%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / b_fixtures | 9 | 8/9 (89%) | 8/9 (89%) | 1/9 (11%) | 9/9 (100%) | 8/9 (89%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / mathdial_test | 90 | 26/90 (29%) | 26/90 (29%) | 64/90 (71%) | 90/90 (100%) | 8/30 (27%) | 34/90 (38%) | 83/90 (92%) | 90/90 (100%) | 90/90 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / smoke | 8 | 6/8 (75%) | 6/8 (75%) | 2/8 (25%) | 8/8 (100%) | 6/8 (75%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |

## Failure examples (automated checks)

**Deterministic fallback (floor), `schema_valid` (64 failures):**

- `mathdial_test/mathdial:train:1012:t00` (EXPLAIN_CONCEPT): ""
- `mathdial_test/mathdial:train:1288:t10` (HINT): ""

**Deterministic fallback (floor), `action_faithful` (4 failures):**

- `mathdial_test/mathdial:train:1209:t08` (PROBE): ""
- `mathdial_test/mathdial:train:1552:t10` (PROBE): ""

**OpenAI GPT-6.1 Sol (frontier), `action_faithful` (25 failures):**

- `b_fixtures/teaching_plan_worked_example` (WORKED_EXAMPLE): "**Worked example:** A different coin has probability \( \tfrac12 \) of tails. For one toss, an **outcome** is the result you observe. “Tails” is one possible outcome; \( \tfrac12 \) tells us its probability, not a separate outcome. A **sample space** lists all possible outcomes. To build one, start with tails, then che..."
- `mathdial_test/mathdial:train:1760:t14` (PROBE): "Let’s check the 48 handfuls: if each handful adds 4 strawberries to the basket, how many strawberries would 48 handfuls put in the basket, and how does that compare with its capacity of 60?"

**OpenAI GPT-6.1 Sol (frontier), `math_supported` (56 failures):**

- `mathdial_test/mathdial:train:1684:t08` (HINT): "The 3 tells us how many eggs each mother laid—not how many eggs make one chick. Each hatched egg produces one new dove. If 45 eggs hatched, how many new doves would there be?" Unsupported numbers: ['45']; from the learner's own message: ['45'].
- `mathdial_test/mathdial:train:1042:t12` (EXPLAIN_CONCEPT): "The key distinction is **number of coins** versus **value of coins**. Multiplying by $0.25 or $0.10 gives the coins’ value in dollars. This question asks how many coins remain, so each quarter and each dime counts as **one coin**, regardless of its value. After setting aside 28 quarters, Gloria has: - **42 quarters** -..." Unsupported numbers: ['0.25', '0.10']; from the learner's own message: ['0.25', '0.10'].

**OpenAI GPT-6.1 Sol (frontier), `disclosure_allowed` (7 failures):**

- `mathdial_test/mathdial:train:1110:t08` (HINT): "Rounding changes a number only slightly; 13.6 is not a rounded version of 68. Also, your 25 tractor-days already accounts for all 5 days, so dividing by 5 again counts the time twice. To check your proposed rate, how many total acres would be planted at 13.6 acres per tractor-day over 25 tractor-days?" Fingerprints: ['68']; already in the stem or learner text: True.
- `mathdial_test/mathdial:train:2061:t10` (HINT): "Now compare the third candidate’s 300 votes with John’s 150 votes. What subtraction would tell you how many more votes the third candidate received?" Fingerprints: ['150']; already in the stem or learner text: True.

## Spend

| Ledger entry | Actual USD |
|---|---:|
| gpt-6.1-sol | 0.2120 |
| gpt-6.1-sol-temp0-probe | 0.0000 |
| **Total** | **0.2120** |

Cap: $25.00; token costs are computed from each response's `usage` times list prices; GPU costs are wall time from the create request to the verified deletion times the list price (an upper bound).

## Deployments

No deployment was created in the recorded runs.

## Notes from the 2026-11-02 run (hand-written; included verbatim by `benchmark report`)

**Scope.** Xueming chose `gpt-6.1-sol` as the frontier model and `Qwen3.8-27B` as the open
model; the cheaper OpenAI tier, other Fireworks serverless models and new local downloads
were dropped. The deterministic fallback is the floor. Anthropic and Google entries are
configured with verified prices and wait for keys.

**gpt-6.1-sol.** Ran on all 107 cases (0 errors) through the OpenAI Chat Completions API
with `reasoning_effort: low` (the lowest level its model page lists) and
`max_completion_tokens: 1024`. A probe with `temperature: 0` returned HTTP 400
("Unsupported value: 'temperature' does not support 0 with this model. Only the default (1)
value is supported."), so sampling ran at the default temperature and is not deterministic.
Reasoning tokens are recorded separately (mean 3.7 per case, maximum 102); no visible reply
exceeded the PRD 14.1 cap of 384 tokens and no length or question-count check failed.

**Qwen3.8-27B (D10 base) was not benchmarked.** `GET
/v1/accounts/fireworks/models/qwen3p8-27b` reports `supportsServerless: false`,
`supportsLora: true`, `supervisedLoraTunable: true` and `supportsServerlessTraining: false`,
so the model needs a dedicated deployment to benchmark and to serve (and D10's "serverless
training pool" claim deserves a check against that last flag). The validated deployment
shapes for it are 1x H100 80 GB BF16 (`accounts/fireworks/deploymentShapes/sva2n3mh`,
context 131072), 1x B200 BF16 and 2x B300 BF16; the entry uses the 1x H100 shape at the
list price of $8.00 per GPU hour with scale-to-zero after 5 minutes and a 75-minute hard
limit ($10 reserve). The free dry run (`validateOnly=true`) on account
`accounts/xueming-r9e8nrrxg6ml` returned HTTP 400 `code 9: "payment method is required"`,
so no deployment was created, the account's deployment list is empty (verified by GET with
`showDeleted=true`) and GPU spend is $0. Once a payment method is on the account, one
command runs the pilot, the full set and the deletion: `uv run python -m
tools.response.benchmark run --models qwen3.8-27b-fw-h100 --manage-deployment`.

**Reading the automated results.** Two of the three strict checks over-report on dialogue
data: `math_supported` accepts only numbers from the plan's approved content, so a tutor
who echoes the learner's own intermediate value fails it, and `disclosure_allowed` flags an
answer fingerprint even when the problem stem or the learner already stated that number.
The per-model diagnostics table counts both situations. `action_faithful` requires a PROBE
to restate every number in the stem, which diagnostic questions about one step rarely do.
Treat "delivered as model" as a lower bound on acceptable responses and the review sheet as
the way to measure the gap; a human read of the quoted failures is in the final report.

**Case sets.** B's 9 valid plan fixtures and the 8 smoke cases are tiny and all about one
coin problem. The MathDial sample is 30 turns per mapped move (90 of 1,618 test turns, at
most one turn per problem and move); its plans carry no approved hint, facts or example.
`evaluation/response/probability_cases_v1.jsonl` did not exist when this ran; it is listed
in the config and will be included by a rerun without a code change.

## Limitations

- Automated checks only. `math_supported` accepts a number only if it appears in the plan's approved content (or is a small integer); numbers the learner wrote, intermediate results and LaTeX-escaped percentages count as unsupported, so it over-reports math failures on dialogue data and never proves reasoning correct.
- `disclosure_allowed` matches answer fingerprints anywhere in the text, including values that the problem stem or the learner already stated.
- `action_faithful` is a heuristic: a PROBE must restate every stem number, a WORKED_EXAMPLE every example number, and a HINT must stay within 25 words of the permitted hint. Natural tutor replies fail it without being unfaithful.
- MathDial is GSM8K-style arithmetic with LLM-simulated students, not probability; its plans carry mapped (unaudited) moves and no approved hint, facts or example.
- Samples are small (see the case-set table); differences of a few cases are noise. Hosted models with a default temperature of 1 are not deterministic, so a rerun gives different texts.
- Latency is one client on a home connection with concurrency per entry as configured, not the PRD 14.3 serving measurement.

Review sheet: `review/review_sheet_blinded.csv` (40 cases, 62 responses); key in `review/review_key_do_not_open_before_rating.csv`.
