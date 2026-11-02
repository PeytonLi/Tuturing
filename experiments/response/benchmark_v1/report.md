# Response benchmark v1: prompt-only models on the tutoring response task

Generated 2026-11-02T21:39:27+00:00 by `tools.response.benchmark report` (benchmark_v1, prompt_v1, validator_v1). Every number below is an automated check from `packages.response.validate` on one sampled response per case; no human rating is included. Read the limitations before using any table.

## Models

| Model | Provider | Model ID | Reasoning | Temp. | Price in/out ($/MTok) | Cases |
|---|---|---|---|---|---|---|
| Deterministic fallback (floor) | fallback | `-` | n/a | default | free | 107 |
| OpenAI GPT-6.1 Sol (frontier) | openai_compatible | `gpt-6.1-sol` | reasoning_effort=low | default | 2.0 / 10.0 | 107 |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 | openai_compatible | `accounts/fireworks/models/qwen3p8-27b` | reasoning_effort=none | 0 | GPU 8.0/h x1 | 107 |
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
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 | 107 | 41/107 (38%) | 41/107 (38%) | 66/107 (62%) | 107/107 (100%) | 18/47 (38%) | 61/107 (57%) | 101/107 (94%) | 106/107 (99%) | 95/107 (89%) |

| Model | Errors | Repair | Fallback | Unavailable | p50 ms | p95 ms | In tok | Out tok | Reasoning tok | Words | Over 384 visible | <think> leaks | Cost $ | $/case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic fallback (floor) | 64 | 0 | 0 | 64 | 0 | 0 | 0.0 | 0.0 | 0.0 | 43.5 | 0 | 0 | 0.0 | 0.0 |
| OpenAI GPT-6.1 Sol (frontier) | 0 | 0 | 67 | 0 | 2572 | 4578 | 583.4 | 81.5 | 3.3 | 49.8 | 0 | 0 | 0.212 | 0.001982 |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 | 0 | 5 | 61 | 0 | 1181 | 3017 | 657.3 | 88.2 | 0.0 | 59.9 | 0 | 0 | 0.5119 | 0.004784 |

Diagnostics on the strict checks (counts of failing cases): 

| Model | math failures where every flagged number came from the learner's message | disclosure flags whose fingerprint already appears in the stem or learner text |
|---|---:|---:|
| Deterministic fallback (floor) | 0 of 0 | 0 of 0 |
| OpenAI GPT-6.1 Sol (frontier) | 45 of 56 | 7 of 7 |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 | 41 of 46 | 6 of 6 |

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
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / CHALLENGE | 2 | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) | 2/2 (100%) | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / CHECK_MASTERY | 3 | 3/3 (100%) | 3/3 (100%) | 0/3 (0%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / EXPLAIN_CONCEPT | 32 | 21/32 (66%) | 21/32 (66%) | 11/32 (34%) | 32/32 (100%) | 2/2 (100%) | 22/32 (69%) | 32/32 (100%) | 32/32 (100%) | 31/32 (97%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / HINT | 32 | 9/32 (28%) | 9/32 (28%) | 23/32 (72%) | 32/32 (100%) | 0/2 (0%) | 13/32 (41%) | 28/32 (88%) | 32/32 (100%) | 28/32 (88%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / PRACTICE | 2 | 2/2 (100%) | 2/2 (100%) | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / PROBE | 32 | 3/32 (9%) | 3/32 (9%) | 29/32 (91%) | 32/32 (100%) | 7/32 (22%) | 15/32 (47%) | 30/32 (94%) | 32/32 (100%) | 27/32 (84%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / REFLECT | 2 | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 1/2 (50%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / WORKED_EXAMPLE | 2 | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) | 2/2 (100%) | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) | 1/2 (50%) | 1/2 (50%) |

## Results by case set

| Model / case set | N | All applicable pass | Delivered as model | Fallback needed | schema_valid | action_faithful | math_supported | disclosure_allowed | length_valid | question_count_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic fallback (floor) / b_fixtures | 9 | 9/9 (100%) | 9/9 (100%) | 0/9 (0%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) |
| Deterministic fallback (floor) / mathdial_test | 90 | 26/90 (29%) | 26/90 (29%) | 64/90 (71%) | 26/90 (29%) | 26/30 (87%) | 90/90 (100%) | 90/90 (100%) | 90/90 (100%) | 90/90 (100%) |
| Deterministic fallback (floor) / smoke | 8 | 8/8 (100%) | 8/8 (100%) | 0/8 (0%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / b_fixtures | 9 | 8/9 (89%) | 8/9 (89%) | 1/9 (11%) | 9/9 (100%) | 8/9 (89%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / mathdial_test | 90 | 26/90 (29%) | 26/90 (29%) | 64/90 (71%) | 90/90 (100%) | 8/30 (27%) | 34/90 (38%) | 83/90 (92%) | 90/90 (100%) | 90/90 (100%) |
| OpenAI GPT-6.1 Sol (frontier) / smoke | 8 | 6/8 (75%) | 6/8 (75%) | 2/8 (25%) | 8/8 (100%) | 6/8 (75%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / b_fixtures | 9 | 7/9 (78%) | 7/9 (78%) | 2/9 (22%) | 9/9 (100%) | 8/9 (89%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 8/9 (89%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / mathdial_test | 90 | 29/90 (32%) | 29/90 (32%) | 61/90 (68%) | 90/90 (100%) | 5/30 (17%) | 44/90 (49%) | 84/90 (93%) | 90/90 (100%) | 80/90 (89%) |
| Qwen3.8-27B (D10 base) on a Fireworks on-demand H100 / smoke | 8 | 5/8 (62%) | 5/8 (62%) | 3/8 (38%) | 8/8 (100%) | 5/8 (62%) | 8/8 (100%) | 8/8 (100%) | 7/8 (88%) | 7/8 (88%) |

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

**Qwen3.8-27B (D10 base) on a Fireworks on-demand H100, `action_faithful` (29 failures):**

- `b_fixtures/teaching_plan_hint` (HINT): "To find the probability of heads, we first need to define the sample space. A fair coin has two equally likely outcomes. List the two outcomes."
- `mathdial_test/mathdial:train:1760:t14` (PROBE): "Let’s check that math. If she adds 4 strawberries to the basket each time, and the basket holds 60, how many handfuls does she need to pick to reach 60?"

**Qwen3.8-27B (D10 base) on a Fireworks on-demand H100, `math_supported` (46 failures):**

- `mathdial_test/mathdial:train:1684:t08` (HINT): "That reasoning contains a logical error. Each hatched egg becomes one new dove, not three. You already calculated that 45 eggs hatched. Therefore, there are 45 new doves. What is the correct total number of doves Giselle has now?" Unsupported numbers: ['45', '45']; from the learner's own message: ['45', '45'].
- `mathdial_test/mathdial:train:1677:t08` (PROBE): "I see you still have 14 keys left, but you are still using 45 minutes for the cleaning time. How many minutes does it take to clean just those 14 remaining keys?" Unsupported numbers: ['45']; from the learner's own message: ['45'].

**Qwen3.8-27B (D10 base) on a Fireworks on-demand H100, `disclosure_allowed` (6 failures):**

- `mathdial_test/mathdial:train:2061:t10` (HINT): "That is correct. James got 700, leaving 300 for the third candidate. Now, compare the third candidate’s 300 votes to John’s 150 votes. How many more votes did the third candidate receive?" Fingerprints: ['150']; already in the stem or learner text: True.
- `mathdial_test/mathdial:train:1110:t04` (PROBE): "I notice a small inconsistency in your final step. You calculated 1700 divided by 25, which equals 68. However, you then stated the result is 13.6. Can you explain how you arrived at 13.6 from that division?" Fingerprints: ['68']; already in the stem or learner text: True.

**Qwen3.8-27B (D10 base) on a Fireworks on-demand H100, `length_valid` (1 failures):**

- `smoke/smoke_worked_example_v1` (WORKED_EXAMPLE): "Let’s look at a slightly more complex scenario to see how conditional probability works. Imagine you have a bag with 3 red balls and 2 blue balls. You draw one ball without looking. **Question:** What is the probability that the ball is red, *given that* it is not blue? **Worked Example:** 1. **Identify the condition:*..."

**Qwen3.8-27B (D10 base) on a Fireworks on-demand H100, `question_count_valid` (12 failures):**

- `b_fixtures/teaching_plan_reflect` (REFLECT): "Let’s pause and consider what you’ve just done. You identified the possible results for a fair coin toss. Since the coin is fair, each outcome is equally likely. Take a moment to reflect: Does your list include every single result that could happen? If you were to flip the coin again, would any new outcome appear that ..."
- `mathdial_test/mathdial:train:0972:t10` (HINT): "You are getting closer, but let's look closely at the drying sequence. You have 3 coats: base, paint, and glitter. 1. Apply base (20 min) 2. Dry base (20 min) 3. Apply paint (20 min) 4. Dry paint (20 min) 5. Apply glitter (20 min) 6. Dry glitter (20 min) Does the problem state that the nails are "done" only when the fi..."

## Spend

| Ledger entry | Actual USD |
|---|---:|
| gpt-6.1-sol | 0.2120 |
| gpt-6.1-sol-temp0-probe | 0.0000 |
| qwen3.8-27b-fw-h100 | 0.5119 |
| **Total** | **0.7239** |

Cap: $25.00; token costs are computed from each response's `usage` times list prices; GPU costs are wall time from the create request to the verified deletion times the list price (an upper bound).

## Deployments

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
`evaluation/response/probability_cases_v1.jsonl` did not exist when this ran; it is listed
in the config and will be included by a rerun without a code change.

## Limitations

- Automated checks only. `math_supported` accepts a number only if it appears in the plan's approved content (or is a small integer); numbers the learner wrote, intermediate results and LaTeX-escaped percentages count as unsupported, so it over-reports math failures on dialogue data and never proves reasoning correct.
- `disclosure_allowed` matches answer fingerprints anywhere in the text, including values that the problem stem or the learner already stated.
- `action_faithful` is a heuristic: a PROBE must restate every stem number, a WORKED_EXAMPLE every example number, and a HINT must stay within 25 words of the permitted hint. Natural tutor replies fail it without being unfaithful.
- MathDial is GSM8K-style arithmetic with LLM-simulated students, not probability; its plans carry mapped (unaudited) moves and no approved hint, facts or example.
- Samples are small (see the case-set table); differences of a few cases are noise. Hosted models with a default temperature of 1 are not deterministic, so a rerun gives different texts.
- Latency is one client on a home connection with concurrency per entry as configured, not the PRD 14.3 serving measurement.

Review sheet: `review/review_sheet_blinded.csv` (40 cases, 102 responses including any supplement); key in `review/review_key_do_not_open_before_rating.csv`. The published sheet is frozen; `review/review_sheet_blinded_supplement.csv` holds responses from models that ran later, under the same items.
