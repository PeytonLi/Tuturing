# Simulated learning evaluation

Split: **validation**. All results are **simulated**.

| Policy | Profile | Mean gain (95% bootstrap CI) | Disclosure | Dropout | Mean turns | Mean reward | Pace frequency |
|---|---|---:|---:|---:|---:|---:|---|
| rules | overall | 0.0850 (0.0719, 0.0986) | 0.000 | 0.210 | 14.58 | -0.0322 | concise=0.174, normal=0.826 |
| rules | confident_misconception | 0.1444 (0.1259, 0.1605) | 0.000 | 0.250 | 14.40 | 0.2183 | normal=1.000 |
| rules | knowledgeable_concise | 0.0160 (0.0122, 0.0197) | 0.000 | 0.100 | 15.10 | -0.2856 | concise=0.841, normal=0.159 |
| rules | novice_missing_prereqs | 0.0601 (0.0525, 0.0681) | 0.000 | 0.225 | 14.50 | -0.1560 | normal=1.000 |
| checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | overall | 0.0886 (0.0765, 0.1012) | 0.000 | 0.260 | 14.39 | -0.0155 | concise=0.177, normal=0.823 |
| checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | confident_misconception | 0.1526 (0.1373, 0.1650) | 0.000 | 0.325 | 14.15 | 0.2438 | normal=1.000 |
| checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | knowledgeable_concise | 0.0148 (0.0113, 0.0185) | 0.000 | 0.100 | 15.10 | -0.2828 | concise=0.841, normal=0.159 |
| checkpoint:experiments/learning/runs/bc_rules3_seed0/bc.pt | novice_missing_prereqs | 0.0614 (0.0538, 0.0693) | 0.000 | 0.275 | 14.28 | -0.1410 | normal=1.000 |
| random | overall | 0.0549 (0.0439, 0.0658) | 0.000 | 0.450 | 12.82 | -0.1514 | concise=0.349, detailed=0.324, normal=0.328 |
| random | confident_misconception | 0.0903 (0.0689, 0.1114) | 0.000 | 0.600 | 11.47 | -0.0184 | concise=0.355, detailed=0.331, normal=0.314 |
| random | knowledgeable_concise | 0.0140 (0.0105, 0.0176) | 0.000 | 0.250 | 15.00 | -0.3063 | concise=0.340, detailed=0.313, normal=0.347 |
| random | novice_missing_prereqs | 0.0398 (0.0343, 0.0456) | 0.000 | 0.400 | 13.07 | -0.2071 | concise=0.348, detailed=0.323, normal=0.329 |
| llm:gpt_6_1_sol | overall | 0.0745 (0.0612, 0.0880) | 0.000 | 0.270 | 13.88 | -0.0571 | concise=0.161, detailed=0.001, normal=0.838 |
| llm:gpt_6_1_sol | confident_misconception | 0.1415 (0.1255, 0.1554) | 0.000 | 0.225 | 14.07 | 0.2269 | concise=0.004, detailed=0.002, normal=0.995 |
| llm:gpt_6_1_sol | knowledgeable_concise | 0.0106 (0.0076, 0.0136) | 0.000 | 0.150 | 13.50 | -0.2726 | concise=0.822, normal=0.178 |
| llm:gpt_6_1_sol | novice_missing_prereqs | 0.0395 (0.0331, 0.0465) | 0.000 | 0.375 | 13.88 | -0.2334 | normal=1.000 |

## LLM usage

| Policy | Model | Calls | Cache hits | Fallback rate | Prompt tokens | Completion tokens |
|---|---|---:|---:|---:|---:|---:|
| llm:gpt_6_1_sol | gpt-6.1-sol | 977 | 411 | 0.0000 | 942683 | 48502 |
