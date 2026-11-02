# Teaching Strategist PPO v1: simulated results

Every number in this document is **simulated**. It comes from the prototype student simulator, not from human learners, and does not support any claim about human learning outcomes.

## Findings (simulated)

- **PPO does not beat rules on validation mean gain.** Rules: 0.064 [0.051, 0.076]. The five BC-initialized PPO seeds average 0.060 (range 0.052 to 0.064); the best seed (ppo_bc_seed3, 0.064 [0.052, 0.078]) ties rules. Every PPO confidence interval overlaps the rules interval, so no difference in simulated gain is detectable at 100 episodes per policy.
- BC-initialized PPO has a slightly higher mean total reward than rules (-0.094 averaged over seeds vs -0.105), mostly from lower teaching cost of a narrower move mix. No confidence interval was computed for reward, so treat this as an observation, not a result.
- PPO collapsed toward a single move: BC-initialized seeds use WORKED_EXAMPLE 82% to 100% of turns and never call CHECK_MASTERY on validation, so they never certify a concept. Rules uses WORKED_EXAMPLE 57.5%, HINT 25.5%, PROBE 9.7%, PRACTICE 5.9%, CHECK_MASTERY 1.4%.
- Gains come almost entirely from the confident_misconception profile; on knowledgeable_concise every policy gains under 0.012 and PPO is below rules there.
- Scratch PPO (seed 0) is clearly worse than BC-initialized seed 0: gain 0.052 vs 0.063 and 48% dropout vs 17%, using EXPLAIN_CONCEPT on 95% of turns. Its gain interval still overlaps rules.
- Training curves are flat: the per-update train gain does not rise between the first and last update in any run, so 2,500 sessions did not produce measurable simulated improvement over the BC start.
- The v2 dropout fix roughly halves simulated rules dropout (31% to 15.5%), almost entirely from the base hazard (50 to 20 dropouts). Certification is now reachable within a split (the collector serves a fresh assessment family before reusing one), but simulated certification stays rare (13.5% of rules episodes certify any concept).

## Setup

- Code commit used for BC, PPO, and evaluation: `182ec8d` (branch `devin/1793651877-simulator-v2-ppo`)
- Simulator: `configs/learning/simulator_proto_v2.yaml` (v1 with `dropout.base_per_turn` lowered from 0.01 to 0.004, an uncalibrated assumption) plus a second assessment template family per concept per split
- Learned policy: Teaching Strategist only. Navigator: `RulesNavigator` in every run
- Behavior cloning: `configs/learning/bc.yaml`, 300 train rules episodes, seed 0; holdout action agreement 0.813 (438 transitions), train agreement 0.834
- PPO: `configs/learning/ppo_strategist.yaml` defaults, 2,500 train sessions per run, 32 sessions per update (79 updates). Seeds 0 to 4 initialized from the BC checkpoint, plus seed 0 from scratch. CPU, six runs in parallel
- Evaluation: validation split only (test stays sealed), 20 episodes per scenario x 5 validation scenarios = 100 episodes per policy, greedy checkpoints, eval seed 20250115. `tools/learning/evaluate.py` gives each policy its own episode seed offset, so policies do not face identical sampled students

## Simulator v1 vs v2 diagnostic (simulated, 200 train rules episodes)

| Metric (simulated) | v1 | v2 |
|---|---:|---:|
| Mean gain | 0.047 | 0.050 |
| Mean total reward | -0.194 | -0.170 |
| Dropout rate | 31.0% | 15.5% |
| Mean turns | 13.37 | 14.50 |
| Goal concept certified | 3.5% | 5.0% |
| Any concept certified | 12.0% | 13.5% |
| Assessment family fallbacks (total) | 0 | 48 |
| Dropouts from base | 50 | 20 |
| Dropouts from incorrect_streak | 4 | 4 |
| Dropouts from pace_mismatch | 8 | 7 |
| Mean reward component: terminal_gain | 0.189 | 0.198 |
| Mean reward component: teaching_cost | -0.267 | -0.290 |
| Mean reward component: disclosure_penalty | 0.000 | 0.000 |
| Mean reward component: repetition_penalty | -0.039 | -0.041 |
| Mean reward component: dropout_penalty | -0.077 | -0.039 |
| Mean reward component: invalid_action_penalty | 0.000 | 0.000 |
| Mean reward component: shaping | 0.000 | 0.000 |

The v1 column is the pre-change code (one assessment family per concept per split), so its fallback count was not tracked (shown as 0).

## Policies on validation (simulated)

| Policy (simulated) | Mean gain | 95% bootstrap CI | Disclosure | Dropout | Mean turns | Mean total reward |
|---|---:|---:|---:|---:|---:|---:|
| always_hint | 0.017 | [0.014, 0.020] | 0.0% | 16.0% | 14.85 | -0.353 |
| ppo_bc_seed0 | 0.063 | [0.050, 0.077] | 0.0% | 17.0% | 14.74 | -0.089 |
| ppo_bc_seed1 | 0.061 | [0.049, 0.073] | 0.0% | 19.0% | 14.58 | -0.096 |
| ppo_bc_seed2 | 0.061 | [0.049, 0.074] | 0.0% | 17.0% | 14.69 | -0.092 |
| ppo_bc_seed3 | 0.064 | [0.052, 0.078] | 0.0% | 19.0% | 14.27 | -0.077 |
| ppo_bc_seed4 | 0.052 | [0.039, 0.063] | 0.0% | 9.0% | 15.12 | -0.119 |
| ppo_scratch_seed0 | 0.052 | [0.041, 0.063] | 0.0% | 48.0% | 12.36 | -0.160 |
| random | 0.046 | [0.035, 0.057] | 0.0% | 51.0% | 11.43 | -0.177 |
| rules | 0.064 | [0.051, 0.076] | 0.0% | 14.0% | 14.80 | -0.105 |

## Spread across the 5 BC-initialized seeds (simulated)

| Metric (simulated) | Mean | Std dev | Min | Max |
|---|---:|---:|---:|---:|
| Mean gain | 0.060 | 0.005 | 0.052 | 0.064 |
| Mean total reward | -0.094 | 0.015 | -0.119 | -0.077 |
| Dropout rate | 0.162 | 0.041 | 0.090 | 0.190 |
| Mean turns | 14.680 | 0.306 | 14.270 | 15.120 |

## Scratch vs BC initialization, seed 0 (simulated)

| Run (simulated) | Mean gain | 95% CI | Dropout | Mean turns | Mean total reward |
|---|---:|---:|---:|---:|---:|
| ppo_bc_seed0 | 0.063 | [0.050, 0.077] | 17.0% | 14.74 | -0.089 |
| ppo_scratch_seed0 | 0.052 | [0.041, 0.063] | 48.0% | 12.36 | -0.160 |
| rules | 0.064 | [0.051, 0.076] | 14.0% | 14.80 | -0.105 |

## Per-profile mean gain on validation (simulated)

| Policy (simulated) | confident_misconception | knowledgeable_concise | novice_missing_prereqs |
|---|---:|---:|---:|
| always_hint | 0.005 [0.004, 0.006] | 0.009 [0.006, 0.012] | 0.032 [0.028, 0.037] |
| ppo_bc_seed0 | 0.130 [0.109, 0.146] | 0.006 [0.004, 0.007] | 0.025 [0.023, 0.028] |
| ppo_bc_seed1 | 0.125 [0.106, 0.142] | 0.006 [0.004, 0.008] | 0.024 [0.022, 0.027] |
| ppo_bc_seed2 | 0.129 [0.113, 0.146] | 0.002 [0.001, 0.002] | 0.023 [0.021, 0.025] |
| ppo_bc_seed3 | 0.132 [0.117, 0.145] | 0.006 [0.004, 0.008] | 0.026 [0.023, 0.028] |
| ppo_bc_seed4 | 0.103 [0.082, 0.123] | 0.001 [0.001, 0.002] | 0.025 [0.023, 0.028] |
| ppo_scratch_seed0 | 0.096 [0.076, 0.115] | 0.002 [0.001, 0.002] | 0.033 [0.027, 0.039] |
| random | 0.072 [0.050, 0.096] | 0.009 [0.006, 0.013] | 0.037 [0.031, 0.043] |
| rules | 0.121 [0.101, 0.138] | 0.011 [0.009, 0.014] | 0.033 [0.029, 0.037] |

Each profile cell is 20 episodes (one validation scenario per profile), so the per-profile intervals are wide.

## Learning curves, first vs last update (simulated, train split)

| Run (simulated) | Updates | Reward first | Reward last | Reward mean last 10 | Gain first | Gain last | Dropout first | Dropout last | Entropy first | Entropy last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_bc_seed0 | 79 | -0.162 | -0.186 | -0.137 | 0.051 | 0.039 | 9.4% | 21.9% | 0.786 | 0.570 |
| ppo_bc_seed1 | 79 | -0.183 | -0.161 | -0.158 | 0.047 | 0.043 | 25.0% | 25.0% | 0.803 | 1.199 |
| ppo_bc_seed2 | 79 | -0.213 | -0.150 | -0.164 | 0.046 | 0.048 | 28.1% | 18.8% | 0.780 | 0.767 |
| ppo_bc_seed3 | 79 | -0.225 | -0.173 | -0.160 | 0.041 | 0.036 | 31.2% | 25.0% | 0.828 | 0.491 |
| ppo_bc_seed4 | 79 | -0.140 | -0.195 | -0.165 | 0.053 | 0.038 | 28.1% | 28.1% | 0.842 | 0.463 |
| ppo_scratch_seed0 | 79 | -0.188 | -0.158 | -0.140 | 0.050 | 0.052 | 50.0% | 40.6% | 3.915 | 2.599 |

Each update averages 32 sampled train sessions, so single-update values are noisy; the last-10 mean is steadier.

## Move frequency, PPO vs rules on validation (simulated)

| Policy (simulated) | CHALLENGE | CHECK_MASTERY | EXPLAIN_CONCEPT | HINT | PRACTICE | PROBE | REFLECT | WORKED_EXAMPLE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | 0.0% | 1.4% | 0.0% | 25.5% | 5.9% | 9.7% | 0.0% | 57.5% |
| ppo_bc_seed0 | 0.0% | 0.0% | 0.0% | 12.4% | 0.8% | 4.7% | 0.0% | 82.1% |
| ppo_bc_seed1 | 0.0% | 0.0% | 0.0% | 2.1% | 5.3% | 2.7% | 0.0% | 89.8% |
| ppo_bc_seed2 | 0.0% | 0.0% | 0.0% | 2.7% | 1.4% | 2.7% | 0.0% | 93.3% |
| ppo_bc_seed3 | 0.0% | 0.0% | 0.0% | 3.4% | 0.7% | 6.6% | 0.0% | 89.3% |
| ppo_bc_seed4 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| ppo_scratch_seed0 | 0.0% | 0.0% | 95.3% | 0.0% | 0.0% | 0.0% | 0.0% | 4.7% |
