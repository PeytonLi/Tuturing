# Response benchmark v1: prompt-only models on the tutoring response task

Generated 2026-11-02T22:44:44+00:00 by `tools.response.benchmark report` (benchmark_v1, prompt_v1, validator_v2). Every number below is an automated check from `packages.response.validate` on one sampled response per case; no human rating is included. Read the limitations before using any table.

## Models

| Model | Provider | Model ID | Reasoning | Temp. | Price in/out ($/MTok) | Cases |
|---|---|---|---|---|---|---|
| Deterministic fallback (floor) | fallback | `-` | n/a | default | free | 162 |
| OpenAI GPT-6.1 Sol (frontier) | openai_compatible | `gpt-6.1-sol` | reasoning_effort=low | default | 2.0 / 10.0 | 162 |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 | openai_compatible | `accounts/fireworks/models/qwen3p8-27b` | reasoning_effort=none | 0 | GPU 8.0/h x1 | 162 |
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
| probability_v1 | 55 |  |

Total: 162 cases. A case is one TeachingPlan; the model sees only the compiled prompt (plan projection plus recent dialogue).

## Results by model

Denominator per cell is shown explicitly. `action_faithful` is counted only where the plan carries the content its heuristic needs (MathDial HINT and EXPLAIN_CONCEPT plans have no permitted hint or approved facts, so they are not applicable). "Delivered as model" means all applicable checks passed; "fallback needed" means the serving pipeline (PRD 14.1) would repair, fall back or error instead.

| Model | N | All applicable pass | Delivered as model | Fallback needed | schema_valid | action_faithful | math_supported | disclosure_allowed | length_valid | question_count_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic fallback (floor) | 162 | 98/162 (60%) | 98/162 (60%) | 64/162 (40%) | 98/162 (60%) | 98/102 (96%) | 162/162 (100%) | 162/162 (100%) | 162/162 (100%) | 162/162 (100%) |
| OpenAI GPT-6.1 Sol (frontier) | 162 | 73/162 (45%) | 73/162 (45%) | 89/162 (55%) | 162/162 (100%) | 63/102 (62%) | 96/162 (59%) | 150/162 (93%) | 162/162 (100%) | 162/162 (100%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 | 162 | 57/162 (35%) | 57/162 (35%) | 105/162 (65%) | 162/162 (100%) | 44/102 (43%) | 100/162 (62%) | 145/162 (90%) | 159/162 (98%) | 142/162 (88%) |

| Model | Errors | Repair | Fallback | Unavailable | p50 ms | p95 ms | In tok | Out tok | Reasoning tok | Words | Over 384 visible | <think> leaks | Cost $ | $/case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic fallback (floor) | 64 | 0 | 0 | 64 | 0 | 0 | 0.0 | 0.0 | 0.0 | 30.7 | 0 | 0 | 0.0 | 0.0 |
| OpenAI GPT-6.1 Sol (frontier) | 0 | 0 | 89 | 0 | 2645 | 5666 | 470.1 | 88.3 | 10.1 | 51.5 | 0 | 0 | 0.2953 | 0.001823 |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 | 0 | 5 | 100 | 0 | 1289 | 3363 | 528.2 | 95.1 | 0.0 | 68.1 | 0 | 0 | 0.9963 | 0.00615 |

Diagnostics on the strict checks (counts of failing cases): 

| Model | math failures where every flagged number came from the learner's message | disclosure flags whose fingerprint already appears in the stem or learner text | item-fidelity failures (validator_v2, digit-independent, part of action_faithful) |
|---|---:|---:|---:|
| Deterministic fallback (floor) | 0 of 0 | 0 of 0 | 0/77 (0%) |
| OpenAI GPT-6.1 Sol (frontier) | 53 of 66 | 12 of 12 | 5/111 (5%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 | 52 of 62 | 14 of 17 | 11/111 (10%) |

Item fidelity applies to PRACTICE, CHALLENGE, CHECK_MASTERY and HINT plans with a problem stem and to PROBE plans with a stem and a problem instance; the denominator is that count among records that produced a text (an unavailable fallback has none). A failure means the reply dropped the issued problem or introduced another scenario. Validator versions in the records: Deterministic fallback (floor): validator_v2, OpenAI GPT-6.1 Sol (frontier): validator_v2, Qwen3.8-27B (D10 base) on a Fireworks on-demand H100: validator_v2.

| Model / case set | Item fidelity pass |
|---|---:|
| Deterministic fallback (floor) / b_fixtures | 6/6 (100%) |
| Deterministic fallback (floor) / mathdial_test | 26/26 (100%) |
| Deterministic fallback (floor) / probability_v1 | 40/40 (100%) |
| Deterministic fallback (floor) / smoke | 5/5 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / b_fixtures | 6/6 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / mathdial_test | 58/60 (97%) |
| OpenAI GPT-6.1 Sol (frontier) / probability_v1 | 37/40 (92%) |
| OpenAI GPT-6.1 Sol (frontier) / smoke | 5/5 (100%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / b_fixtures | 3/6 (50%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / mathdial_test | 57/60 (95%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / probability_v1 | 36/40 (90%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / smoke | 4/5 (80%) |

## Results by move

| Model / move | N | All applicable pass | Delivered as model | Fallback needed | schema_valid | action_faithful | math_supported | disclosure_allowed | length_valid | question_count_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic fallback (floor) / CHALLENGE | 4 | 4/4 (100%) | 4/4 (100%) | 0/4 (0%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) |
| Deterministic fallback (floor) / CHECK_MASTERY | 12 | 12/12 (100%) | 12/12 (100%) | 0/12 (0%) | 12/12 (100%) | 12/12 (100%) | 12/12 (100%) | 12/12 (100%) | 12/12 (100%) | 12/12 (100%) |
| Deterministic fallback (floor) / EXPLAIN_CONCEPT | 38 | 8/38 (21%) | 8/38 (21%) | 30/38 (79%) | 8/38 (21%) | 8/8 (100%) | 38/38 (100%) | 38/38 (100%) | 38/38 (100%) | 38/38 (100%) |
| Deterministic fallback (floor) / HINT | 42 | 12/42 (29%) | 12/42 (29%) | 30/42 (71%) | 12/42 (29%) | 12/12 (100%) | 42/42 (100%) | 42/42 (100%) | 42/42 (100%) | 42/42 (100%) |
| Deterministic fallback (floor) / PRACTICE | 13 | 13/13 (100%) | 13/13 (100%) | 0/13 (0%) | 13/13 (100%) | 13/13 (100%) | 13/13 (100%) | 13/13 (100%) | 13/13 (100%) | 13/13 (100%) |
| Deterministic fallback (floor) / PROBE | 40 | 36/40 (90%) | 36/40 (90%) | 4/40 (10%) | 36/40 (90%) | 36/40 (90%) | 40/40 (100%) | 40/40 (100%) | 40/40 (100%) | 40/40 (100%) |
| Deterministic fallback (floor) / REFLECT | 8 | 8/8 (100%) | 8/8 (100%) | 0/8 (0%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |
| Deterministic fallback (floor) / WORKED_EXAMPLE | 5 | 5/5 (100%) | 5/5 (100%) | 0/5 (0%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / CHALLENGE | 4 | 3/4 (75%) | 3/4 (75%) | 1/4 (25%) | 4/4 (100%) | 3/4 (75%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / CHECK_MASTERY | 12 | 10/12 (83%) | 10/12 (83%) | 2/12 (17%) | 12/12 (100%) | 10/12 (83%) | 12/12 (100%) | 12/12 (100%) | 12/12 (100%) | 12/12 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / EXPLAIN_CONCEPT | 38 | 17/38 (45%) | 17/38 (45%) | 21/38 (55%) | 38/38 (100%) | 7/8 (88%) | 18/38 (47%) | 38/38 (100%) | 38/38 (100%) | 38/38 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / HINT | 42 | 17/42 (40%) | 17/42 (40%) | 25/42 (60%) | 42/42 (100%) | 7/12 (58%) | 23/42 (55%) | 38/42 (90%) | 42/42 (100%) | 42/42 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / PRACTICE | 13 | 13/13 (100%) | 13/13 (100%) | 0/13 (0%) | 13/13 (100%) | 13/13 (100%) | 13/13 (100%) | 13/13 (100%) | 13/13 (100%) | 13/13 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / PROBE | 40 | 8/40 (20%) | 8/40 (20%) | 32/40 (80%) | 40/40 (100%) | 13/40 (32%) | 18/40 (45%) | 35/40 (88%) | 40/40 (100%) | 40/40 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / REFLECT | 8 | 5/8 (62%) | 5/8 (62%) | 3/8 (38%) | 8/8 (100%) | 8/8 (100%) | 5/8 (62%) | 5/8 (62%) | 8/8 (100%) | 8/8 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / WORKED_EXAMPLE | 5 | 0/5 (0%) | 0/5 (0%) | 5/5 (100%) | 5/5 (100%) | 2/5 (40%) | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / CHALLENGE | 4 | 1/4 (25%) | 1/4 (25%) | 3/4 (75%) | 4/4 (100%) | 1/4 (25%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / CHECK_MASTERY | 12 | 6/12 (50%) | 6/12 (50%) | 6/12 (50%) | 12/12 (100%) | 6/12 (50%) | 11/12 (92%) | 11/12 (92%) | 12/12 (100%) | 12/12 (100%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / EXPLAIN_CONCEPT | 38 | 22/38 (58%) | 22/38 (58%) | 16/38 (42%) | 38/38 (100%) | 7/8 (88%) | 23/38 (61%) | 35/38 (92%) | 36/38 (95%) | 36/38 (95%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / HINT | 42 | 10/42 (24%) | 10/42 (24%) | 32/42 (76%) | 42/42 (100%) | 1/12 (8%) | 21/42 (50%) | 38/42 (90%) | 42/42 (100%) | 38/42 (90%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / PRACTICE | 13 | 10/13 (77%) | 10/13 (77%) | 3/13 (23%) | 13/13 (100%) | 10/13 (77%) | 13/13 (100%) | 13/13 (100%) | 13/13 (100%) | 12/13 (92%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / PROBE | 40 | 5/40 (12%) | 5/40 (12%) | 35/40 (88%) | 40/40 (100%) | 9/40 (22%) | 21/40 (52%) | 36/40 (90%) | 40/40 (100%) | 31/40 (78%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / REFLECT | 8 | 1/8 (12%) | 1/8 (12%) | 7/8 (88%) | 8/8 (100%) | 7/8 (88%) | 3/8 (38%) | 3/8 (38%) | 8/8 (100%) | 5/8 (62%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / WORKED_EXAMPLE | 5 | 2/5 (40%) | 2/5 (40%) | 3/5 (60%) | 5/5 (100%) | 3/5 (60%) | 4/5 (80%) | 5/5 (100%) | 4/5 (80%) | 4/5 (80%) |

## Results by case set

| Model / case set | N | All applicable pass | Delivered as model | Fallback needed | schema_valid | action_faithful | math_supported | disclosure_allowed | length_valid | question_count_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic fallback (floor) / b_fixtures | 9 | 9/9 (100%) | 9/9 (100%) | 0/9 (0%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) |
| Deterministic fallback (floor) / mathdial_test | 90 | 26/90 (29%) | 26/90 (29%) | 64/90 (71%) | 26/90 (29%) | 26/30 (87%) | 90/90 (100%) | 90/90 (100%) | 90/90 (100%) | 90/90 (100%) |
| Deterministic fallback (floor) / probability_v1 | 55 | 55/55 (100%) | 55/55 (100%) | 0/55 (0%) | 55/55 (100%) | 55/55 (100%) | 55/55 (100%) | 55/55 (100%) | 55/55 (100%) | 55/55 (100%) |
| Deterministic fallback (floor) / smoke | 8 | 8/8 (100%) | 8/8 (100%) | 0/8 (0%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / b_fixtures | 9 | 8/9 (89%) | 8/9 (89%) | 1/9 (11%) | 9/9 (100%) | 8/9 (89%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / mathdial_test | 90 | 26/90 (29%) | 26/90 (29%) | 64/90 (71%) | 90/90 (100%) | 8/30 (27%) | 34/90 (38%) | 83/90 (92%) | 90/90 (100%) | 90/90 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / probability_v1 | 55 | 33/55 (60%) | 33/55 (60%) | 22/55 (40%) | 55/55 (100%) | 41/55 (75%) | 45/55 (82%) | 50/55 (91%) | 55/55 (100%) | 55/55 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / smoke | 8 | 6/8 (75%) | 6/8 (75%) | 2/8 (25%) | 8/8 (100%) | 6/8 (75%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / b_fixtures | 9 | 4/9 (44%) | 4/9 (44%) | 5/9 (56%) | 9/9 (100%) | 5/9 (56%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 8/9 (89%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / mathdial_test | 90 | 29/90 (32%) | 29/90 (32%) | 61/90 (68%) | 90/90 (100%) | 4/30 (13%) | 44/90 (49%) | 84/90 (93%) | 90/90 (100%) | 80/90 (89%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / probability_v1 | 55 | 19/55 (35%) | 19/55 (35%) | 36/55 (65%) | 55/55 (100%) | 30/55 (55%) | 39/55 (71%) | 44/55 (80%) | 53/55 (96%) | 47/55 (85%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / smoke | 8 | 5/8 (62%) | 5/8 (62%) | 3/8 (38%) | 8/8 (100%) | 5/8 (62%) | 8/8 (100%) | 8/8 (100%) | 7/8 (88%) | 7/8 (88%) |

## Failure examples (automated checks)

**Deterministic fallback (floor), `schema_valid` (64 failures):**

- `mathdial_test/mathdial:train:1012:t00` (EXPLAIN_CONCEPT): ""
- `mathdial_test/mathdial:train:1288:t10` (HINT): ""

**Deterministic fallback (floor), `action_faithful` (4 failures):**

- `mathdial_test/mathdial:train:1209:t08` (PROBE): ""
- `mathdial_test/mathdial:train:1552:t10` (PROBE): ""

**OpenAI GPT-6.1 Sol (frontier), `action_faithful` (39 failures):**

- `b_fixtures/teaching_plan_worked_example` (WORKED_EXAMPLE): "**Worked example:** A different coin has probability \( \tfrac12 \) of tails. For one toss, an **outcome** is the result you observe. “Tails” is one possible outcome; \( \tfrac12 \) tells us its probability, not a separate outcome. A **sample space** lists all possible outcomes. To build one, start with tails, then che..."
- `mathdial_test/mathdial:train:1760:t14` (PROBE): "Let’s check the 48 handfuls: if each handful adds 4 strawberries to the basket, how many strawberries would 48 handfuls put in the basket, and how does that compare with its capacity of 60?"

**OpenAI GPT-6.1 Sol (frontier), `math_supported` (66 failures):**

- `mathdial_test/mathdial:train:1684:t08` (HINT): "The 3 tells us how many eggs each mother laid—not how many eggs make one chick. Each hatched egg produces one new dove. If 45 eggs hatched, how many new doves would there be?" Unsupported numbers: ['45']; from the learner's own message: ['45'].
- `mathdial_test/mathdial:train:1042:t12` (EXPLAIN_CONCEPT): "The key distinction is **number of coins** versus **value of coins**. Multiplying by $0.25 or $0.10 gives the coins’ value in dollars. This question asks how many coins remain, so each quarter and each dime counts as **one coin**, regardless of its value. After setting aside 28 quarters, Gloria has: - **42 quarters** -..." Unsupported numbers: ['0.25', '0.10']; from the learner's own message: ['0.25', '0.10'].

**OpenAI GPT-6.1 Sol (frontier), `disclosure_allowed` (12 failures):**

- `mathdial_test/mathdial:train:1110:t08` (HINT): "Rounding changes a number only slightly; 13.6 is not a rounded version of 68. Also, your 25 tractor-days already accounts for all 5 days, so dividing by 5 again counts the time twice. To check your proposed rate, how many total acres would be planted at 13.6 acres per tractor-day over 25 tractor-days?" Fingerprints: ['68']; already in the stem or learner text: True.
- `mathdial_test/mathdial:train:2061:t10` (HINT): "Now compare the third candidate’s 300 votes with John’s 150 votes. What subtraction would tell you how many more votes the third candidate received?" Fingerprints: ['150']; already in the stem or learner text: True.

**OpenAI GPT-6.1 Sol (frontier), `item_fidelity` (5 failures):**

- `mathdial_test/mathdial:test:0454:t08` (PROBE): "Dividing \(3/4\) by 3 finds one of three equal pieces of that portion. To find what remains, we need to account for the whole cake. If a whole cake is cut into 12 equal pieces, how many pieces represent the whole, and how would you use that to write the \(1\) in \(1-\frac13-\frac14\) as a fraction with denominator 12?" Reasons: reply reuses 0 of 13 stem words and none of its numbers; a PROBE must refer to the issued problem.
- `mathdial_test/mathdial:train:0516:t02` (PROBE): "A calculator can check the arithmetic, but we also need to check what each number represents. In your final sum, \(3.75 + 11.25 + 11.25\), which numbers represent sharks, and which represent other fish?" Reasons: reply reuses 2 of 13 stem words and none of its numbers; a PROBE must refer to the issued problem.

**Qwen3.8-27B (D10 base) on a Fireworks on-demand H100, `action_faithful` (58 failures):**

- `b_fixtures/teaching_plan_check_mastery` (CHECK_MASTERY): "Let’s check your understanding of sample spaces. Consider a standard six-sided die. List the complete sample space for a single roll. Then, identify the subset of outcomes that represent rolling an even number. Please attempt this."
- `b_fixtures/teaching_plan_challenge` (CHALLENGE): "Let’s test your understanding of sample spaces. Consider a standard six-sided die. The sample space is {1, 2, 3, 4, 5, 6}. Now, imagine a modified die where the face labeled "6" is replaced by a face labeled "7". The sample space becomes {1, 2, 3, 4, 5, 7}. If you roll this modified die once, what is the probability of..."

**Qwen3.8-27B (D10 base) on a Fireworks on-demand H100, `math_supported` (62 failures):**

- `mathdial_test/mathdial:train:1684:t08` (HINT): "That reasoning contains a logical error. Each hatched egg becomes one new dove, not three. You already calculated that 45 eggs hatched. Therefore, there are 45 new doves. What is the correct total number of doves Giselle has now?" Unsupported numbers: ['45', '45']; from the learner's own message: ['45', '45'].
- `mathdial_test/mathdial:train:1677:t08` (PROBE): "I see you still have 14 keys left, but you are still using 45 minutes for the cleaning time. How many minutes does it take to clean just those 14 remaining keys?" Unsupported numbers: ['45']; from the learner's own message: ['45'].

**Qwen3.8-27B (D10 base) on a Fireworks on-demand H100, `disclosure_allowed` (17 failures):**

- `mathdial_test/mathdial:train:2061:t10` (HINT): "That is correct. James got 700, leaving 300 for the third candidate. Now, compare the third candidate’s 300 votes to John’s 150 votes. How many more votes did the third candidate receive?" Fingerprints: ['150']; already in the stem or learner text: True.
- `mathdial_test/mathdial:train:1110:t04` (PROBE): "I notice a small inconsistency in your final step. You calculated 1700 divided by 25, which equals 68. However, you then stated the result is 13.6. Can you explain how you arrived at 13.6 from that division?" Fingerprints: ['68']; already in the stem or learner text: True.

**Qwen3.8-27B (D10 base) on a Fireworks on-demand H100, `length_valid` (3 failures):**

- `probability_v1/original_v1:c08d-287` (EXPLAIN_CONCEPT): "We must stay on the current topic: the multiplication rule and probability trees. Bayes’ theorem is a separate concept and is not part of this specific teaching plan. Let’s focus on how probability trees work using the approved facts. A probability tree is a diagram that maps out all possible outcomes of a sequence of ..."
- `probability_v1/original_v1:p13-d` (EXPLAIN_CONCEPT): "Conditional probability adjusts the sample space. Imagine a standard deck of 52 cards. If you ask, "What is the probability of drawing a King?", the denominator is 52. There are 4 Kings, so the probability is 4/52. Now, suppose I tell you, "The card you drew is a heart." This new information changes the game. You no lo..."

**Qwen3.8-27B (D10 base) on a Fireworks on-demand H100, `question_count_valid` (20 failures):**

- `b_fixtures/teaching_plan_reflect` (REFLECT): "Let’s pause and consider what you’ve just done. You identified the possible results for a fair coin toss. Since the coin is fair, each outcome is equally likely. Take a moment to reflect: Does your list include every single result that could happen? If you were to flip the coin again, would any new outcome appear that ..."
- `mathdial_test/mathdial:train:0972:t10` (HINT): "You are getting closer, but let's look closely at the drying sequence. You have 3 coats: base, paint, and glitter. 1. Apply base (20 min) 2. Dry base (20 min) 3. Apply paint (20 min) 4. Dry paint (20 min) 5. Apply glitter (20 min) 6. Dry glitter (20 min) Does the problem state that the nails are "done" only when the fi..."

**Qwen3.8-27B (D10 base) on a Fireworks on-demand H100, `item_fidelity` (11 failures):**

- `b_fixtures/teaching_plan_check_mastery` (CHECK_MASTERY): "Let’s check your understanding of sample spaces. Consider a standard six-sided die. List the complete sample space for a single roll. Then, identify the subset of outcomes that represent rolling an even number. Please attempt this." Reasons: reply reuses 0 of 6 stem words; a CHECK_MASTERY must present the issued problem (at least 50%); reply introduces objects the plan does not contain: die, roll; reply introduces quantities the plan does not contain: 6 die.
- `b_fixtures/teaching_plan_challenge` (CHALLENGE): "Let’s test your understanding of sample spaces. Consider a standard six-sided die. The sample space is {1, 2, 3, 4, 5, 6}. Now, imagine a modified die where the face labeled "6" is replaced by a face labeled "7". The sample space becomes {1, 2, 3, 4, 5, 7}. If you roll this modified die once, what is the probability of..." Reasons: reply reuses 2 of 6 stem words; a CHALLENGE must present the issued problem (at least 50%); reply introduces objects the plan does not contain: die, roll; reply introduces quantities the plan does not contain: 6 die.

## Spend

| Ledger entry | Actual USD |
|---|---:|
| gpt-6.1-sol | 0.2953 |
| gpt-6.1-sol-temp0-probe | 0.0000 |
| qwen3.8-27b-fw-h100 | 0.9963 |
| **Total** | **1.2916** |

Cap: $25.00; token costs are computed from each response's `usage` times list prices; GPU costs are wall time from the create request to the verified deletion times the list price (an upper bound).

## Deployments

- `accounts/peli/deployments/tut-bench-qwen38-20261102-223910`: 1x NVIDIA_H100_80GB (BF16), shape `accounts/fireworks/deploymentShapes/sva2n3mh`, created 2026-11-02T22:39:10+00:00, ended 2026-11-02T22:42:48+00:00, final state DELETED, deletion verified True, estimated GPU cost $0.48.
- `accounts/peli/deployments/tut-bench-qwen38-20261102-213251`: 1x NVIDIA_H100_80GB (BF16), shape `accounts/fireworks/deploymentShapes/sva2n3mh`, created 2026-11-02T21:32:51+00:00, ended 2026-11-02T21:36:42+00:00, final state DELETED, deletion verified True, estimated GPU cost $0.51.

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

**Qwen3.8-27B (D10 base) on a Fireworks on-demand deployment.** `GET
/v1/accounts/fireworks/models/qwen3p8-27b` reports `supportsServerless: false`,
`supportsLora: true`, `supervisedLoraTunable: true` and `supportsServerlessTraining: false`,
so the model needs a dedicated deployment to benchmark and to serve (and D10's "serverless
training pool" claim deserves a check against that last flag). The validated deployment
shapes for it are 1x H100 80 GB BF16 (`accounts/fireworks/deploymentShapes/sva2n3mh`,
context 131072), 1x B200 BF16 and 2x B300 BF16; the entry uses the 1x H100 shape at the
list price of $8.00 per GPU hour with scale-to-zero after 5 minutes and a 75-minute hard
limit ($10 reserve). A first attempt on account `accounts/xueming-r9e8nrrxg6ml` stopped at
the free dry run with HTTP 400 `code 9: "payment method is required"` and created nothing.
With a key for `accounts/peli` the dry run passed and the harness created
`accounts/peli/deployments/tut-bench-qwen38-20261102-213251` (1x NVIDIA_H100_80GB, BF16,
shape version `k9aui8v2`, GLOBAL placement, min 0 / max 1 replicas, 300 s scale-to-zero) at
21:32:51Z; a replica served the warm-up request at 21:35:28Z; the 3-case pilot and the
other 104 cases ran at concurrency 4 (0.6 s per case, 0 errors); the deletion was requested
at 21:36:11Z (the first DELETE returned 400 while replicas were active, the retry with
`ignoreChecks` returned 200) and verified at 21:36:42Z by GET (`state: DELETED`,
`deleteTime 21:36:11Z`) and by a `showDeleted` listing. Billable upper bound: 3.8 minutes
of create-to-verified-deletion wall time, $0.51 at list price. Decoding was
`temperature 0`, `max_tokens 384`, `reasoning_effort: none`; no `<think>` text appeared
and no reasoning tokens were reported.

**What the automated checks missed (human read of the fixture cases).** On B's two
CHECK_MASTERY fixtures and the CHALLENGE fixture, Qwen3.8-27B replaced the verified coin
problem with a six-sided die of its own ("Consider a standard six-sided die. List the
complete sample space...") and every automated check passed, because the coin stem has no
digits for `action_faithful` or `math_supported` to anchor on. That is a protected-check
item substitution, which PRD 14.1 treats as critical, and it reached delivery mode
`model`. On the smoke CHALLENGE and WORKED_EXAMPLE it also invented marble problems, which
the number check did catch. gpt-6.1-sol kept the issued problem in all 17 fixture and smoke
cases. The validator needs a stem-overlap check that does not depend on digits before any
prompt-only model is trusted on a protected check; until then the review sheet is the
only measure of this failure.

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
`evaluation/response/probability_cases_v1.jsonl` did not exist during the first run; it
was added later and is covered by the follow-up below.

## Follow-up under validator_v2 with the original probability cases (2026-11-02, later)

**Re-scoring.** `benchmark rescore` re-checked every stored text of the three systems (107
cases each) under validator_v2 from the plans in the configured case sets, with no model
call; usage, latency and cost are carried over and `validator_version` is stamped on every
record. The digit-independent item-fidelity check now scores Qwen3.8-27B's three fixture
substitutions (both CHECK_MASTERY fixtures and the CHALLENGE fixture, coin replaced by a
die) as `action_faithful` failures, so its delivered-as-model count on the first 107 cases
drops from 41 to 38; gpt-6.1-sol stays at 40 and the fallback at 43. One caveat: main
changed the smoke CHALLENGE plan from difficulty medium (action 49) to hard (action 52)
after the texts were generated, so the three stored smoke-CHALLENGE responses were
produced under a prompt whose difficulty line differed; the checks do not depend on
difficulty and no model was re-called for the old case sets.

**Original probability cases (55; all five prototype concepts, all eight moves; synthetic,
reviewer status pending; one allows reveal).** Delivered as model: fallback 55/55,
gpt-6.1-sol 33/55, Qwen3.8-27B 19/55 (0 errors each). Item fidelity on the 40 applicable
cases: fallback 40/40, gpt-6.1-sol 37/40, Qwen3.8-27B 36/40. The per-check table in
"Results by case set" has the rest. On this set the strict checks are closer to their
intent than on MathDial, because the plans carry approved hints, facts and examples and
the learner context is short: gpt-6.1-sol's 22 fallbacks come from 14 action-fidelity, 10
math and 5 disclosure flags (overlapping), spread over HINT, PROBE, REFLECT and
WORKED_EXAMPLE plans; Qwen3.8-27B's 36 come from 25 action-fidelity, 16 math and 11
disclosure flags, and it also asked two questions where one was allowed in 8 cases and hit
the 384-token cap twice.

**Second deployment.** `accounts/peli/deployments/tut-bench-qwen38-20261102-223910`, the
same 1x H100 80 GB BF16 shape (`sva2n3mh`, version `k9aui8v2`), min 0 / max 1 replicas,
300 s scale-to-zero; the free `validateOnly` dry run passed first; created 22:39:10Z,
serving 22:41:47Z, pilot of 3 then 52 cases at concurrency 4, deletion requested 22:42:17Z
(first DELETE 400 while a replica was active, retry with `ignoreChecks` 200) and verified
22:42:48Z by GET (`state: DELETED`) and a `showDeleted` listing; no live deployment
remained. 3.6 minutes, $0.48 upper bound. Spend for this follow-up: gpt-6.1-sol $0.0833
(55 cases), Qwen GPU $0.4844; running total for the whole benchmark $1.29 of the $25 cap.

**Review.** The published review sheet and its supplement are unchanged; no rows were added
for the probability cases.

## Limitations

- Automated checks only. `math_supported` accepts a number only if it appears in the plan's approved content (or is a small integer); numbers the learner wrote, intermediate results and LaTeX-escaped percentages count as unsupported, so it over-reports math failures on dialogue data and never proves reasoning correct.
- `disclosure_allowed` matches answer fingerprints anywhere in the text, including values that the problem stem or the learner already stated.
- `action_faithful` is a heuristic: a PROBE must restate every stem number, a WORKED_EXAMPLE every example number, and a HINT must stay within 25 words of the permitted hint. Natural tutor replies fail it without being unfaithful.
- MathDial is GSM8K-style arithmetic with LLM-simulated students, not probability; its plans carry mapped (unaudited) moves and no approved hint, facts or example.
- Samples are small (see the case-set table); differences of a few cases are noise. Hosted models with a default temperature of 1 are not deterministic, so a rerun gives different texts.
- Latency is one client on a home connection with concurrency per entry as configured, not the PRD 14.3 serving measurement.

Review sheet: `review/review_sheet_blinded.csv` (40 cases, 102 responses including any supplement); key in `review/review_key_do_not_open_before_rating.csv`. The published sheet is frozen; `review/review_sheet_blinded_supplement.csv` holds responses from models that ran later, under the same items.
