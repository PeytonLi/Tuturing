# Full-session LLM Strategists vs rules, BC and random (simulated)

Every number below is **simulated**. These are gains on simulator students, not human learning.

Setup: validation split, 5 scenarios × 20 episodes = 100 paired episodes (seed 20250115, `--paired-seeds`), simulator `simulator_proto_v2`, config `configs/learning/ppo_strategist.yaml`. Every policy teaches the same students with the same random events. The LLMs get the 30-case decision prompt (`evaluation/learning/llm_strategist_prompt_v1.txt`), filled with public tutor state only. Unusable replies fall back to `RulesStrategist`. BC is the frozen `bc_rules3_seed0` checkpoint (sha256 `38cca155…`). Rules, BC and random reproduced their final-report gains to 4 decimals in both runs.

| Strategist | Mean gain [95% CI] | Paired diff vs rules [95% CI] | Dropout | Mean turns | Fallback rate |
|---|---|---|---:|---:|---:|
| BC (frozen copy of rules v3) | 0.0886 | +0.0036 [−0.0006, +0.0091] | 26% | 14.39 | – |
| Rules v3 | 0.0850 | – | 21% | 14.58 | – |
| PPO, 5 seeds (from strategist_final_v1) | 0.083–0.087 | – | – | – | – |
| gpt-6.1-sol | 0.0745 [0.0612, 0.0880] | −0.0105 [−0.0165, −0.0037] | 27% | 13.88 | 0% |
| Base Qwen 3.8 27B | 0.0739 [0.0618, 0.0862] | −0.0111 [−0.0204, −0.0024] | 28% | 14.35 | 4.5% (all truncated) |
| Random | 0.0549 | −0.0301 [−0.0413, −0.0190] | 45% | 12.82 | – |

Move mix:
- **gpt-6.1-sol:** PROBE 35%, EXPLAIN_CONCEPT 31%, CHECK_MASTERY 17%, HINT 11%, WORKED_EXAMPLE 4%, PRACTICE 1%.
- **Qwen:** EXPLAIN_CONCEPT 37%, PROBE 23%, HINT 15%, PRACTICE 11%, CHECK_MASTERY 9%, WORKED_EXAMPLE 5%.
- **BC:** mostly WORKED_EXAMPLE and PRACTICE (see strategist_final_v1).

Profile gains (gpt-6.1-sol / Qwen): confident_misconception 0.142 / 0.129, novice_missing_prereqs 0.040 / 0.049, knowledgeable_concise 0.011 / 0.015.

## Reading

- Both LLMs beat random and lose to rules by about 0.011, and both paired CIs exclude zero. PPO (0.083–0.087) and BC already score at or above gpt-6.1-sol **in this simulator**.
- The LLMs mostly probe and explain, and they rarely give worked examples or practice. Simulator v2's hand-set effect sizes favor worked examples followed by independent practice. So the gap partly measures how well each policy fits the simulator's assumptions, not only teaching quality. This doesn't show that gpt-6.1-sol teaches real students worse.
- This matches the 30 one-step cases (rules 86.7%, gpt-6.1-sol 84.4%, Qwen 80.0% acceptable), which measure single decisions, not lessons.

## Usage

| Model | Calls | Cache hits | Prompt tokens | Completion tokens |
|---|---:|---:|---:|---:|
| gpt-6.1-sol | 977 | 411 | 942,683 | 48,502 |
| Qwen 3.8 27B (1× H100 on-demand, ~50 min, deleted) | 948 | 487 | 1,056,692 | 2,264,033 |

Reports: `evaluation/learning/results/strategist_llm_fullsession_{gpt61sol,qwen38base}_v1.{json,md}`. Re-run with `tools/learning/evaluate.py --policies rules,checkpoint:<bc.pt>,random,llm:<alias> --paired-seeds --split validation --episodes-per-scenario 20`.
