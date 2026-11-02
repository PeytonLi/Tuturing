# SIMULATED alternate-simulator robustness

Validation only; 100 episodes per policy/config, seed 20250115. Robustness only, never used for training or checkpoint selection.
All numbers are SIMULATED, not human learning evidence.

The same ordered scenario/episode-seed schedule is reused across every policy and config. 95% percentile bootstrap CIs use 1,000 episode resamples; differences resample paired policy-minus-rules gains. These CIs do not capture simulator-model uncertainty.
Disclosure count is the number of turns with a negative disclosure penalty. Rules is always included as the reference, even when omitted from --policies.
The alternate config halves per-move action effects (ell stays fixed), raises guess/slip by 0.05, and lowers configured initial mastery ranges by 0.10 (floor 0.05). The novice's hard-coded unrelated-concept initialization is unchanged.

| Config | Policy | Mean gain [95% CI] | Paired difference vs rules [95% CI] | Dropout | Disclosures | Mean turns |
|---|---|---:|---:|---:|---:|---:|
| simulator_proto_v2 | rules | 0.0850 [0.0719, 0.0986] | 0.0000 [0.0000, 0.0000] | 21.0% | 0 | 14.58 |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | 0.0886 [0.0752, 0.1021] | 0.0036 [-0.0007, 0.0091] | 26.0% | 0 | 14.39 |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/ppo_bc_r3_seed1/ppo.pt | 0.0871 [0.0729, 0.1007] | 0.0021 [-0.0008, 0.0059] | 35.0% | 0 | 13.71 |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/ppo_bc_r3_seed2/ppo.pt | 0.0860 [0.0720, 0.0993] | 0.0010 [-0.0051, 0.0072] | 27.0% | 0 | 13.51 |
| simulator_proto_alt | rules | 0.0633 [0.0520, 0.0750] | 0.0000 [0.0000, 0.0000] | 27.0% | 0 | 14.50 |
| simulator_proto_alt | checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | 0.0675 [0.0561, 0.0789] | 0.0042 [0.0008, 0.0086] | 29.0% | 0 | 14.27 |
| simulator_proto_alt | checkpoint:experiments/learning/runs/ppo_bc_r3_seed1/ppo.pt | 0.0657 [0.0540, 0.0773] | 0.0025 [-0.0001, 0.0059] | 44.0% | 0 | 13.32 |
| simulator_proto_alt | checkpoint:experiments/learning/runs/ppo_bc_r3_seed2/ppo.pt | 0.0652 [0.0535, 0.0766] | 0.0020 [-0.0031, 0.0072] | 35.0% | 0 | 13.39 |

## Per-profile gain (SIMULATED)

| Config | Policy | Profile | Episodes | Mean gain [95% CI] |
|---|---|---|---:|---:|
| simulator_proto_v2 | rules | confident_misconception | 40 | 0.1444 [0.1259, 0.1605] |
| simulator_proto_v2 | rules | knowledgeable_concise | 20 | 0.0160 [0.0124, 0.0198] |
| simulator_proto_v2 | rules | novice_missing_prereqs | 40 | 0.0601 [0.0519, 0.0683] |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | confident_misconception | 40 | 0.1526 [0.1366, 0.1648] |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | knowledgeable_concise | 20 | 0.0148 [0.0110, 0.0188] |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | novice_missing_prereqs | 40 | 0.0614 [0.0539, 0.0694] |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/ppo_bc_r3_seed1/ppo.pt | confident_misconception | 40 | 0.1484 [0.1319, 0.1622] |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/ppo_bc_r3_seed1/ppo.pt | knowledgeable_concise | 20 | 0.0070 [0.0051, 0.0091] |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/ppo_bc_r3_seed1/ppo.pt | novice_missing_prereqs | 40 | 0.0658 [0.0579, 0.0735] |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/ppo_bc_r3_seed2/ppo.pt | confident_misconception | 40 | 0.1477 [0.1308, 0.1614] |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/ppo_bc_r3_seed2/ppo.pt | knowledgeable_concise | 20 | 0.0054 [0.0034, 0.0075] |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/ppo_bc_r3_seed2/ppo.pt | novice_missing_prereqs | 40 | 0.0645 [0.0570, 0.0725] |
| simulator_proto_alt | rules | confident_misconception | 40 | 0.1176 [0.1024, 0.1303] |
| simulator_proto_alt | rules | knowledgeable_concise | 20 | 0.0093 [0.0075, 0.0112] |
| simulator_proto_alt | rules | novice_missing_prereqs | 40 | 0.0359 [0.0306, 0.0411] |
| simulator_proto_alt | checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | confident_misconception | 40 | 0.1255 [0.1123, 0.1352] |
| simulator_proto_alt | checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | knowledgeable_concise | 20 | 0.0094 [0.0071, 0.0118] |
| simulator_proto_alt | checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | novice_missing_prereqs | 40 | 0.0385 [0.0337, 0.0433] |
| simulator_proto_alt | checkpoint:experiments/learning/runs/ppo_bc_r3_seed1/ppo.pt | confident_misconception | 40 | 0.1220 [0.1085, 0.1333] |
| simulator_proto_alt | checkpoint:experiments/learning/runs/ppo_bc_r3_seed1/ppo.pt | knowledgeable_concise | 20 | 0.0043 [0.0030, 0.0061] |
| simulator_proto_alt | checkpoint:experiments/learning/runs/ppo_bc_r3_seed1/ppo.pt | novice_missing_prereqs | 40 | 0.0401 [0.0351, 0.0452] |
| simulator_proto_alt | checkpoint:experiments/learning/runs/ppo_bc_r3_seed2/ppo.pt | confident_misconception | 40 | 0.1215 [0.1073, 0.1328] |
| simulator_proto_alt | checkpoint:experiments/learning/runs/ppo_bc_r3_seed2/ppo.pt | knowledgeable_concise | 20 | 0.0045 [0.0034, 0.0055] |
| simulator_proto_alt | checkpoint:experiments/learning/runs/ppo_bc_r3_seed2/ppo.pt | novice_missing_prereqs | 40 | 0.0393 [0.0343, 0.0443] |

## Move frequencies (SIMULATED)

| Config | Policy | PROBE | HINT | EXPLAIN_CONCEPT | WORKED_EXAMPLE | PRACTICE | CHALLENGE | CHECK_MASTERY | REFLECT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| simulator_proto_v2 | rules | 15.2% | 23.9% | 0.1% | 32.4% | 27.1% | 0.0% | 1.4% | 0.0% |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | 9.9% | 12.5% | 0.0% | 42.9% | 34.7% | 0.0% | 0.0% | 0.0% |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/ppo_bc_r3_seed1/ppo.pt | 7.3% | 0.0% | 0.0% | 40.1% | 52.6% | 0.0% | 0.0% | 0.0% |
| simulator_proto_v2 | checkpoint:experiments/learning/runs/ppo_bc_r3_seed2/ppo.pt | 20.0% | 0.0% | 0.0% | 48.3% | 31.8% | 0.0% | 0.0% | 0.0% |
| simulator_proto_alt | rules | 14.7% | 22.6% | 0.1% | 34.5% | 26.7% | 0.0% | 1.4% | 0.0% |
| simulator_proto_alt | checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | 9.7% | 12.6% | 0.0% | 42.7% | 35.0% | 0.0% | 0.0% | 0.0% |
| simulator_proto_alt | checkpoint:experiments/learning/runs/ppo_bc_r3_seed1/ppo.pt | 7.5% | 0.0% | 0.0% | 40.5% | 52.0% | 0.0% | 0.0% | 0.0% |
| simulator_proto_alt | checkpoint:experiments/learning/runs/ppo_bc_r3_seed2/ppo.pt | 20.4% | 0.0% | 0.0% | 48.6% | 31.0% | 0.0% | 0.0% | 0.0% |
