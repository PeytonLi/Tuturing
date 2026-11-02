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
| llm:qwen3p8_27b_base | overall | 0.0739 (0.0618, 0.0862) | 0.000 | 0.280 | 14.35 | -0.0803 | concise=0.182, detailed=0.004, normal=0.814 |
| llm:qwen3p8_27b_base | confident_misconception | 0.1289 (0.1106, 0.1459) | 0.000 | 0.250 | 14.43 | 0.1635 | concise=0.014, detailed=0.007, normal=0.979 |
| llm:qwen3p8_27b_base | knowledgeable_concise | 0.0147 (0.0111, 0.0181) | 0.000 | 0.250 | 15.05 | -0.3224 | concise=0.841, normal=0.159 |
| llm:qwen3p8_27b_base | novice_missing_prereqs | 0.0486 (0.0430, 0.0543) | 0.000 | 0.325 | 13.93 | -0.2030 | detailed=0.004, normal=0.996 |

## LLM usage

| Policy | Model | Calls | Cache hits | Fallback rate | Prompt tokens | Completion tokens |
|---|---|---:|---:|---:|---:|---:|
| llm:qwen3p8_27b_base | accounts/fireworks/models/qwen3p8-27b#accounts/peli/deployments/tutoring-qwen38-fullsess-1793660139 | 948 | 487 | 0.0453 | 1056692 | 2264033 |
