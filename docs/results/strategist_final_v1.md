# Teaching Strategist final evaluation v1: simulated results

Every number in this document is **simulated**. It comes from the prototype student simulator (`configs/learning/simulator_proto_v2.yaml`), which is uncalibrated, and no human data were used. Nothing here is a human learning outcome.

## Findings (simulated)

- **The chosen Strategist is the behavior-cloning (BC) checkpoint** `experiments/learning/runs/bc_rules3_seed0/bc.pt`. The rule, declared before the test split was run: highest validation mean gain, with ties going to rules. BC had 0.0886 vs 0.0850 for rules. On validation its edge over rules is inconclusive: the paired difference is +0.004 (-0.001, +0.009).
- **On the sealed test split (run once), BC beat rules by a small margin:** a paired difference of +0.005 (+0.001, +0.010). Random was clearly worse than rules (-0.025, CI -0.033 to -0.017). The effect is small, and the intervals are wide relative to it.
- **PPO did not beat its BC starting point.** The five BC-initialized PPO seeds scored 0.083 to 0.087 on validation, and every paired interval against rules crosses zero. PPO from scratch (0.046) was clearly worse. The one allowed change, entropy coefficient 0.01 to 0.05 on seed 0, did not help (0.073).
- **The worked-example collapse came mostly from a rules bug, not from the simulator.** An always-worked-example baseline loses to rules (paired -0.022 under the first rules fix and -0.033 under the final rules), so the simulator does not reward one move by construction, and no v3 simulator was made. The rules Strategist got stuck repeating worked examples. Once two hinted attempts had failed, it chose WORKED_EXAMPLE on every later turn, because the hint counter only resets after a hint is followed by a correct answer and rules never hinted again. In 66 of 100 validation episodes it gave 3 or more worked examples in a row, often 10. BC copied this, and PPO amplified it (73 to 88% WORKED_EXAMPLE). With the loop fixed, rules uses WORKED_EXAMPLE on 32% of turns, BC on 43%, and PPO seeds on 40 to 67%.
- **Two rules changes.** (1) A confident-misconception branch: PROBE when public evidence suggests a misconception, then WORKED_EXAMPLE or EXPLAIN_CONCEPT once it is confirmed. (2) After a worked example, the learner gets an independent PRACTICE attempt instead of another example. On the decision benchmark, change (1) raised confident_misconception from 0.2 to 1.0 and overall from 0.733 to 0.867. Change (2) moved no benchmark case.
- **The sanity checks now pass on v2 validation.** Always-hint is below rules on every profile. Random is clearly worse overall (-0.030, CI -0.041 to -0.019) and on novice_missing_prereqs (0.040 vs 0.060). On knowledgeable_concise, random and rules are a near tie (0.014 vs 0.016, overlapping intervals). Every policy gains little on that profile, so this check stays inconclusive there.
- **The ranking holds under the alternate simulator** (`simulator_proto_alt.yaml`: action effects halved, guess and slip raised by 0.05, initial mastery 0.10 lower). Gains fall by about a quarter for every policy. BC stays highest (+0.004 vs rules, CI +0.001 to +0.009), and the PPO seeds stay between rules and BC.
- Gains still come mostly from the confident_misconception profile (about 0.15 for good policies vs 0.06 for novices and under 0.02 for knowledgeable learners).

## What changed and why

### Simulator: no change
v1 and v2 are unchanged and no v3 was created. Step 1 rule: if always_worked_example matches or beats rules, fix the simulator (PRD 16.4). It did not (table below), so the existing repeat penalty (novelty factor 0.6 per consecutive repeat of a move) is treated as doing its job.

### Public misconception evidence (shared with the learning service)
In simulator episodes the public state previously always carried zero misconception evidence, so no public-only rule could react to a misconception. `packages/learning/episode.py::update_public_misconceptions` now fills the **existing** 25 misconception slots (M01 to M25) of the public state and of the 140-feature `features_v1` observation. No feature was added, and the count and order are unchanged. The signal uses only visible information. If an incorrect answer, normalized with the family's own answer normalizer, equals the answer a catalogued misconception would produce for that item, the slot is set to 0.6 (or 0.85 after a PROBE, or if it was already likely). Each correct answer on that family halves the slot, and values under 0.2 are cleared. `apps/learning_service/engine.py` calls the same helper, and `tests/integration/learning/test_engine.py` checks that the episode path and the service path produce identical 140-feature vectors for the same answer.

### Rules Strategist (`rules_strategist_v3`)
- Likely misconception (slot at 0.5 or more, not yet confirmed): PROBE.
- Confirmed (0.75 or more, or likely plus a meaningful error right after PROBE, REFLECT or HINT): WORKED_EXAMPLE if mastery is under 0.4, otherwise EXPLAIN_CONCEPT, unless the previous move was already one of those two.
- After two unsuccessful hinted attempts: WORKED_EXAMPLE, then an independent PRACTICE attempt if the previous move was a worked example. This pairs examples with problems instead of looping.
- All other branches are unchanged. The plan version string and one learning-service contract assertion were updated to the new version name.

## Setup

- Branch `devin/1793653846-strategist-eval-v2`. Code commit for the final training, validation, robustness and test runs: `08a77dbc8084170f0d46a127fcc18108125a8138`.
- Simulator: `configs/learning/simulator_proto_v2.yaml` for training, validation and test. `configs/learning/simulator_proto_alt.yaml` (from the parallel robustness branch) for robustness only.
- Navigator: rules in every run. Only the Teaching Strategist is learned.
- BC: `configs/learning/bc.yaml`, 300 train rules episodes, seed 0. Action agreement 0.631 train, 0.629 holdout. (The lower agreement than in v1 reflects a less repetitive teacher.)
- PPO: `configs/learning/ppo_strategist.yaml`, 2,500 train sessions per run (79 updates of 32), seeds 0 to 4 from BC, plus seed 0 from scratch, plus seed 0 from BC with `configs/learning/ppo_strategist_entropy05.yaml` (entropy coefficient 0.05). Seven CPU processes in parallel, about 6 to 7 minutes each.
- Evaluation: `tools/learning/evaluate.py --paired-seeds`, 20 episodes per scenario across 5 scenarios, so 100 episodes per policy. Every policy faces the same sampled students and the same episode seeds. The 95% intervals are percentile bootstraps over episodes (1,000 resamples), and paired differences are bootstrapped from per-episode differences against rules. Greedy checkpoints, eval seed 20250115.
- Sealed test command, run once after selection:
  `OMP_NUM_THREADS=1 uv run python -m tools.learning.evaluate --config configs/learning/ppo_strategist.yaml --split test --unseal-test --paired-seeds --episodes-per-scenario 20 --policies rules,random,checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt --name sealed_test_final`
  The chosen checkpoint is BC, so the requested set (random, rules, BC, chosen) is three policies. An earlier launch of this command was killed while running, before any output was written, when the rules diagnosis was reopened. No test numbers were seen before this single completed run.
- Robustness command:
  `OMP_NUM_THREADS=1 uv run python -m tools.learning.robustness --policies rules --checkpoint experiments/learning/runs/bc_rules3_seed0/bc.pt --checkpoint experiments/learning/runs/ppo_bc_r3_seed1/ppo.pt --checkpoint experiments/learning/runs/ppo_bc_r3_seed2/ppo.pt --output-prefix evaluation/learning/results/robustness_strategist_final`
  Best PPO seed by validation gain: 1. Median: 2.
- Result files: `evaluation/learning/results/strategist_final_validation_v2.json`, `strategist_final_test_v2.json`, `strategist_interim_validation_rules_v2.json`, `robustness_strategist_final.{json,md}`, `move_diagnostic_v2.json`, `decision_benchmark_v1.json`.

## Tables (all simulated)

### Step 1: is one move rewarded by construction? (validation, original rules v1)

| Policy | Mean gain (95% CI) | Paired diff vs rules (95% CI) |
|---|---|---|
| rules v1 | 0.064 (0.051, 0.076) | n/a |
| always_worked_example | 0.052 (0.040, 0.065) | -0.012 (-0.021, -0.004) |
| random | 0.055 (0.043, 0.067) | -0.009 (-0.021, 0.002) |
| always_hint | 0.015 (0.012, 0.018) | -0.049 (-0.063, -0.036) |

### Per-move diagnostic: expected one-step post-score change x 1,000, fresh state (v2)

| Move | confident_misconception | knowledgeable_concise | novice_missing_prereqs |
|---|---:|---:|---:|
| PROBE | 0.00 | 0.00 | 0.00 |
| CHECK_MASTERY | 0.00 | 0.00 | 0.00 |
| PRACTICE | 0.67 | 1.18 | 6.30 |
| HINT | 0.20 | 0.35 | 1.89 |
| EXPLAIN_CONCEPT | 14.63 | 0.49 | 8.83 |
| WORKED_EXAMPLE | 14.80 | 0.58 | 10.09 |
| CHALLENGE | 0.13 | 1.42 | 1.26 |
| REFLECT | 0.27 | 0.47 | 2.52 |

For confident_misconception, once a PROBE has surfaced the misconception, EXPLAIN_CONCEPT and WORKED_EXAMPLE are worth about 0.071 each, roughly five times the fresh-state value. Because PROBE earns nothing directly, a short-sighted policy skips it. Six worked examples in a row give a novice about 22.4 in total, versus 10.1 for the first one alone. The repeat penalty makes a mixed sequence worth more than a loop.

### Decision benchmark, rules move-OK rate (30 cases, no LLM calls)

| Category | rules v1 | rules v2 (misconception branch) | rules v3 (final) |
|---|---:|---:|---:|
| confident_misconception | 0.2 | 1.0 | 1.0 |
| faster_pace | 1.0 | 1.0 | 1.0 |
| known_topic_readiness | 1.0 | 1.0 | 1.0 |
| prerequisite_gap | 0.6 | 0.6 | 0.6 |
| return_after_remediation | 1.0 | 1.0 | 1.0 |
| uncertain_skip | 0.6 | 0.6 | 0.6 |
| Overall | 0.733 | 0.867 | 0.867 |

Cases that changed with v2: DC06, DC08 and DC10 HINT to PROBE; DC07 HINT to EXPLAIN_CONCEPT; DC09 HINT to WORKED_EXAMPLE. DC28 stays HINT because its evidence is only 0.4. Other policies' entries are unchanged.

### Interim validation with rules v2 (before the loop fix)
Rules v2 scored 0.074 (0.061, 0.088). BC 0.071. PPO seeds 0.062 to 0.067, all below rules with paired intervals at or under zero, and 73 to 88% WORKED_EXAMPLE. Scratch 0.044. The entropy variant 0.066. Full numbers are in `strategist_interim_validation_rules_v2.json`. This round is superseded by the final round below.

### Final validation (v2 simulator, rules v3, 100 paired episodes per policy)

| Policy | Mean gain (95% CI) | Paired diff vs rules (95% CI) | Dropout | Disclosures | Mean turns |
|---|---|---|---:|---:|---:|
| rules v3 | 0.085 (0.072, 0.099) | n/a | 21% | 0 | 14.6 |
| BC (chosen) | 0.089 (0.077, 0.102) | +0.004 (-0.001, +0.009) | 26% | 0 | 14.4 |
| PPO seed 0 | 0.083 (0.071, 0.094) | -0.002 (-0.006, +0.004) | 18% | 0 | 14.6 |
| PPO seed 1 | 0.087 (0.075, 0.099) | +0.002 (-0.001, +0.006) | 35% | 0 | 13.7 |
| PPO seed 2 | 0.086 (0.074, 0.099) | +0.001 (-0.005, +0.007) | 27% | 0 | 13.5 |
| PPO seed 3 | 0.087 (0.075, 0.099) | +0.002 (-0.002, +0.007) | 26% | 0 | 14.3 |
| PPO seed 4 | 0.084 (0.071, 0.096) | -0.001 (-0.009, +0.006) | 25% | 0 | 14.5 |
| PPO scratch seed 0 | 0.046 (0.034, 0.058) | -0.039 (-0.052, -0.029) | 40% | 0 | 12.8 |
| PPO entropy 0.05 seed 0 (separate change) | 0.073 (0.061, 0.086) | -0.012 (-0.019, -0.003) | 21% | 0 | 14.3 |
| random | 0.055 (0.043, 0.067) | -0.030 (-0.041, -0.019) | 45% | 0 | 12.8 |
| always_worked_example | 0.052 (0.040, 0.064) | -0.033 (-0.043, -0.023) | 13% | 0 | 14.9 |
| always_hint | 0.015 (0.012, 0.018) | -0.070 (-0.083, -0.057) | 17% | 0 | 14.7 |

Per-profile gain (95% CI):

| Policy | confident_misconception | knowledgeable_concise | novice_missing_prereqs |
|---|---|---|---|
| rules v3 | 0.144 (0.126, 0.160) | 0.016 (0.012, 0.020) | 0.060 (0.053, 0.068) |
| BC | 0.153 (0.138, 0.164) | 0.015 (0.011, 0.019) | 0.061 (0.054, 0.069) |
| PPO seed 0 | 0.142 (0.127, 0.153) | 0.012 (0.009, 0.016) | 0.060 (0.052, 0.068) |
| PPO seed 1 | 0.148 (0.131, 0.162) | 0.007 (0.005, 0.009) | 0.066 (0.059, 0.074) |
| PPO seed 2 | 0.148 (0.131, 0.162) | 0.005 (0.003, 0.008) | 0.065 (0.057, 0.072) |
| PPO seed 3 | 0.145 (0.131, 0.156) | 0.015 (0.010, 0.020) | 0.065 (0.058, 0.072) |
| PPO seed 4 | 0.130 (0.110, 0.149) | 0.017 (0.013, 0.021) | 0.071 (0.064, 0.080) |
| PPO scratch | 0.090 (0.067, 0.112) | 0.001 (0.001, 0.002) | 0.023 (0.021, 0.026) |
| PPO entropy 0.05 | 0.142 (0.131, 0.152) | 0.014 (0.010, 0.017) | 0.035 (0.029, 0.040) |
| random | 0.090 (0.069, 0.114) | 0.014 (0.011, 0.018) | 0.040 (0.034, 0.046) |
| always_worked_example | 0.106 (0.083, 0.125) | 0.002 (0.001, 0.003) | 0.022 (0.020, 0.024) |
| always_hint | 0.004 (0.004, 0.005) | 0.009 (0.006, 0.011) | 0.029 (0.024, 0.034) |

Move and pace frequency (share of Strategist turns):

| Policy | PROBE | CHECK_MASTERY | PRACTICE | HINT | EXPLAIN | WORKED_EX | CHALLENGE | REFLECT | concise | normal | detailed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rules v3 | 15.2% | 1.4% | 27.1% | 23.9% | 0.1% | 32.4% | 0% | 0% | 17.4% | 82.6% | 0% |
| BC | 9.9% | 0% | 34.7% | 12.5% | 0% | 42.9% | 0% | 0% | 17.7% | 82.3% | 0% |
| PPO seed 0 | 11.6% | 0% | 17.2% | 4.4% | 0% | 66.7% | 0% | 0% | 14.6% | 85.4% | 0% |
| PPO seed 1 | 7.3% | 0% | 52.6% | 0% | 0% | 40.1% | 0% | 0% | 2.2% | 97.8% | 0% |
| PPO seed 2 | 20.0% | 0% | 31.8% | 0% | 0% | 48.3% | 0% | 0% | 14.6% | 85.4% | 0% |
| PPO seed 3 | 8.5% | 0% | 37.7% | 1.5% | 0% | 52.3% | 0% | 0% | 15.8% | 84.2% | 0% |
| PPO seed 4 | 2.8% | 0% | 41.5% | 7.4% | 0% | 48.4% | 0% | 0% | 16.9% | 83.1% | 0% |
| PPO scratch | 0% | 0% | 0% | 0% | 0% | 100% | 0% | 0% | 100% | 0% | 0% |
| PPO entropy 0.05 | 7.0% | 0% | 16.0% | 0% | 0% | 77.0% | 0% | 0% | 0% | 93.6% | 6.4% |
| random | 16.9% | 18.3% | 18.2% | 3.8% | 19.2% | 18.3% | 0.8% | 4.6% | 34.9% | 32.8% | 32.4% |
| always_worked_example | 0% | 0% | 0% | 0% | 0% | 100% | 0% | 0% | 17.0% | 83.0% | 0% |
| always_hint | 25.5% | 0% | 0% | 74.5% | 0% | 0% | 0% | 0% | 17.3% | 82.7% | 0% |

### Robustness: alternate simulator (validation, no retraining)

| Simulator | Policy | Mean gain (95% CI) | Paired diff vs rules (95% CI) | Dropout | Mean turns |
|---|---|---|---|---:|---:|
| v2 | rules v3 | 0.085 (0.072, 0.099) | n/a | 21% | 14.6 |
| v2 | BC | 0.089 (0.075, 0.102) | +0.004 (-0.001, +0.009) | 26% | 14.4 |
| v2 | PPO seed 1 (best) | 0.087 (0.073, 0.101) | +0.002 (-0.001, +0.006) | 35% | 13.7 |
| v2 | PPO seed 2 (median) | 0.086 (0.072, 0.099) | +0.001 (-0.005, +0.007) | 27% | 13.5 |
| alt | rules v3 | 0.063 (0.052, 0.075) | n/a | 27% | 14.5 |
| alt | BC | 0.068 (0.056, 0.079) | +0.004 (+0.001, +0.009) | 29% | 14.3 |
| alt | PPO seed 1 (best) | 0.066 (0.054, 0.077) | +0.003 (-0.000, +0.006) | 44% | 13.3 |
| alt | PPO seed 2 (median) | 0.065 (0.054, 0.077) | +0.002 (-0.003, +0.007) | 35% | 13.4 |

Per profile under alt: confident_misconception rules 0.118, BC 0.126, PPO 0.122 / 0.122; knowledgeable_concise rules 0.009, BC 0.009, PPO 0.004 / 0.005; novice_missing_prereqs rules 0.036, BC 0.039, PPO 0.040 / 0.039. Move mixes barely change between simulators. Full tables: `evaluation/learning/results/robustness_strategist_final.md`.

### Sealed test (v2 simulator, run once, 100 paired episodes per policy)

| Policy | Mean gain (95% CI) | Paired diff vs rules (95% CI) | Dropout | Disclosures | Mean turns |
|---|---|---|---:|---:|---:|
| rules v3 | 0.061 (0.051, 0.072) | n/a | 30% | 0 | 13.9 |
| BC (chosen) | 0.066 (0.056, 0.076) | +0.005 (+0.001, +0.010) | 33% | 0 | 13.7 |
| random | 0.037 (0.028, 0.046) | -0.025 (-0.033, -0.017) | 59% | 0 | 11.2 |

| Policy | confident_misconception | knowledgeable_concise | novice_missing_prereqs |
|---|---|---|---|
| rules v3 | 0.151 (0.133, 0.164) | 0.020 (0.013, 0.028) | 0.045 (0.040, 0.051) |
| BC | 0.158 (0.152, 0.165) | 0.017 (0.011, 0.024) | 0.052 (0.045, 0.057) |
| random | 0.090 (0.059, 0.118) | 0.009 (0.006, 0.013) | 0.029 (0.025, 0.033) |

Test move mix: BC uses WORKED_EXAMPLE 47.9%, PRACTICE 38.9%, PROBE 13.2%. Rules uses WORKED_EXAMPLE 34.7%, PRACTICE 28.4%, HINT 21.5%, PROBE 15.1%.

## For Person A: what the chosen Strategist asks for most

The chosen Strategist is BC (`bc_rules3_seed0`). On validation (simulated) it uses only four moves:

| Move | Overall | confident_misconception | knowledgeable_concise | novice_missing_prereqs |
|---|---:|---:|---:|---:|
| WORKED_EXAMPLE | 42.9% | 51.2% | 31.5% | 40.6% |
| PRACTICE | 34.7% | 32.9% | 35.1% | 36.4% |
| HINT | 12.5% | 5.7% | 24.8% | 12.8% |
| PROBE | 9.9% | 10.2% | 8.6% | 10.2% |

Paces: normal 82.3% and concise 17.7%. Concise comes almost entirely from the knowledgeable_concise profile (84% concise there). It never picks detailed. It does not use EXPLAIN_CONCEPT, CHALLENGE, REFLECT or CHECK_MASTERY in these runs. SFT coverage should go first to worked examples and practice items at normal pace, then concise hints, then probes. Rules v3, the live default, adds about 24% HINT and 1.4% CHECK_MASTERY.

## Limitations

- The simulator is uncalibrated. Its learning rates, novelty penalty, guess, slip, dropout and profile ranges are assumptions, not measurements. No human data were used, and nothing here shows that any policy helps real learners.
- The intervals are wide compared with the differences that matter. BC's edge over rules is about 0.004 to 0.005, the same size as the interval half-widths. On validation it is not significant. On test it barely is, and on one split only. The bootstrap covers episode sampling, not simulator-model uncertainty.
- Five validation and five test scenarios across three profiles are a small set, and knowledgeable learners gain so little that the comparisons there are uninformative.
- PPO adds nothing over its BC start in 2,500 sessions. The only tuning tried was one entropy change on one seed.
- The rules policy was revised twice during this evaluation, before the test split was opened. The second revision was a bug fix found by diagnosing the sanity check, not a tune to the gain metric: it changed no decision-benchmark case, and it was chosen on teaching grounds (pair worked examples with practice).
- The chosen policy has 5 points higher dropout than rules on validation (26% vs 21%) and 3 points on test. The simulator's dropout model is itself an assumption.
