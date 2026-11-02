# SIMULATED alternate-simulator robustness

Validation only; 100 episodes per policy/config, seed 20250115. Robustness only, never used for training or checkpoint selection.
All numbers are SIMULATED, not human learning evidence.

The same ordered scenario/episode-seed schedule is reused across every policy and config. 95% percentile bootstrap CIs use 1,000 episode resamples; differences resample paired policy-minus-rules gains. These CIs do not capture simulator-model uncertainty.
Disclosure count is the number of turns with a negative disclosure penalty. Rules is always included as the reference, even when omitted from --policies.
The alternate config halves per-move action effects (ell stays fixed), raises guess/slip by 0.05, and lowers configured initial mastery ranges by 0.10 (floor 0.05). The novice's hard-coded unrelated-concept initialization is unchanged.

| Config | Policy | Mean gain [95% CI] | Paired difference vs rules [95% CI] | Dropout | Disclosures | Mean turns |
|---|---|---:|---:|---:|---:|---:|
| simulator_proto_v2 | rules | 0.0637 [0.0511, 0.0762] | 0.0000 [0.0000, 0.0000] | 14.0% | 0 | 14.80 |
| simulator_proto_v2 | random | 0.0549 [0.0434, 0.0668] | -0.0089 [-0.0211, 0.0023] | 45.0% | 0 | 12.82 |
| simulator_proto_v2 | always_hint | 0.0149 [0.0122, 0.0179] | -0.0488 [-0.0623, -0.0349] | 17.0% | 0 | 14.65 |
| simulator_proto_alt | rules | 0.0478 [0.0370, 0.0585] | 0.0000 [0.0000, 0.0000] | 15.0% | 0 | 14.79 |
| simulator_proto_alt | random | 0.0398 [0.0298, 0.0499] | -0.0080 [-0.0182, 0.0012] | 48.0% | 0 | 12.75 |
| simulator_proto_alt | always_hint | 0.0076 [0.0062, 0.0091] | -0.0403 [-0.0516, -0.0287] | 18.0% | 0 | 14.62 |

## Per-profile gain (SIMULATED)

| Config | Policy | Profile | Episodes | Mean gain [95% CI] |
|---|---|---|---:|---:|
| simulator_proto_v2 | rules | confident_misconception | 40 | 0.1208 [0.1012, 0.1382] |
| simulator_proto_v2 | rules | knowledgeable_concise | 20 | 0.0113 [0.0086, 0.0144] |
| simulator_proto_v2 | rules | novice_missing_prereqs | 40 | 0.0329 [0.0283, 0.0372] |
| simulator_proto_v2 | random | confident_misconception | 40 | 0.0903 [0.0688, 0.1129] |
| simulator_proto_v2 | random | knowledgeable_concise | 20 | 0.0140 [0.0103, 0.0175] |
| simulator_proto_v2 | random | novice_missing_prereqs | 40 | 0.0398 [0.0340, 0.0460] |
| simulator_proto_v2 | always_hint | confident_misconception | 40 | 0.0043 [0.0037, 0.0050] |
| simulator_proto_v2 | always_hint | knowledgeable_concise | 20 | 0.0086 [0.0065, 0.0107] |
| simulator_proto_v2 | always_hint | novice_missing_prereqs | 40 | 0.0286 [0.0235, 0.0341] |
| simulator_proto_alt | rules | confident_misconception | 40 | 0.0998 [0.0836, 0.1143] |
| simulator_proto_alt | rules | knowledgeable_concise | 20 | 0.0055 [0.0043, 0.0071] |
| simulator_proto_alt | rules | novice_missing_prereqs | 40 | 0.0170 [0.0146, 0.0194] |
| simulator_proto_alt | random | confident_misconception | 40 | 0.0742 [0.0562, 0.0927] |
| simulator_proto_alt | random | knowledgeable_concise | 20 | 0.0071 [0.0054, 0.0088] |
| simulator_proto_alt | random | novice_missing_prereqs | 40 | 0.0219 [0.0187, 0.0251] |
| simulator_proto_alt | always_hint | confident_misconception | 40 | 0.0020 [0.0017, 0.0023] |
| simulator_proto_alt | always_hint | knowledgeable_concise | 20 | 0.0044 [0.0035, 0.0054] |
| simulator_proto_alt | always_hint | novice_missing_prereqs | 40 | 0.0147 [0.0121, 0.0175] |

## Move frequencies (SIMULATED)

| Config | Policy | PROBE | HINT | EXPLAIN_CONCEPT | WORKED_EXAMPLE | PRACTICE | CHALLENGE | CHECK_MASTERY | REFLECT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| simulator_proto_v2 | rules | 9.7% | 25.5% | 0.0% | 57.5% | 5.9% | 0.0% | 1.4% | 0.0% |
| simulator_proto_v2 | random | 16.9% | 3.8% | 19.2% | 18.3% | 18.2% | 0.8% | 18.3% | 4.6% |
| simulator_proto_v2 | always_hint | 25.5% | 74.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| simulator_proto_alt | rules | 8.9% | 24.1% | 0.0% | 61.3% | 4.4% | 0.0% | 1.4% | 0.0% |
| simulator_proto_alt | random | 16.9% | 3.8% | 19.2% | 18.3% | 18.2% | 0.9% | 18.2% | 4.5% |
| simulator_proto_alt | always_hint | 25.5% | 74.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
