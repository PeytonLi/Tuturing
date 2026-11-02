# SIMULATED text bridge

These are not measured human learning outcomes.

| Run | Pairs | Structured gain (95% CI) | Text gain (95% CI) | Text minus structured (paired 95% CI) | Outcome >= -0.02 | G8 |
|---|---:|---:|---:|---:|---|---|
| stub | 300 | 0.083136 [0.076164, 0.090795] | 0.083136 [0.076164, 0.090795] | 0.000000 [0.000000, 0.000000] | True | FAIL |
| degraded | 300 | 0.083136 [0.076164, 0.090795] | 0.075163 [0.068295, 0.082636] | -0.007973 [-0.010311, -0.005863] | True | FAIL |

PRD line 895: content_fidelity=0 for unsupported or contradictory delivered content. Here any failed independent A check or item-identity check sets content_realized=False; otherwise the simulator's configured stub_realized coefficient (at most 1) and seeded failure process apply. Extraction disagreement is separate from content fidelity.

## Flag counts (turn denominator)

| Run | Path | Turns | answer_disclosed | draft_validator_failure | extractor_disagreement | fallback_used | item_changed | validator_failure | verbalizer_rejected | wrong_move |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stub | structured | 4307 | 0 | 665 | 0 | 4307 | 0 | 0 | 0 | 0 |
| stub | text | 4307 | 0 | 665 | 143 | 4307 | 0 | 0 | 0 | 0 |
| degraded | structured | 4307 | 0 | 665 | 0 | 4307 | 0 | 0 | 0 | 0 |
| degraded | text | 4303 | 0 | 660 | 141 | 4303 | 585 | 878 | 0 | 878 |

stub calls: `{"actual_model_calls": 0, "max_calls_budget": 0, "maximum_primary_calls": 0, "maximum_repair_calls": 0, "maximum_turns_per_path": 4800, "maximum_verbalizer_calls": 0, "note": "Nominal count follows stub decisions; text may change decisions and length. Upper bounds include every turn up to each scenario horizon, including repairs.", "planned_primary_calls_at_stub_trajectory": 0, "planned_verbalizer_calls_at_stub_trajectory": 0, "structured_rehearsal_turns": 4307}`

stub G8 detail: `{"automated_critical_candidates": 143, "eligible_test_budget": false, "flagged_traces": 143, "human_critical_failures": 0, "outcome_threshold_met": true, "paired_uncertainty_published": true, "real_text_run": false, "review_complete": false, "reviewed_traces": 0, "rule": "test pairs>=300 AND real replies AND all critical-flagged traces reviewed with no critical failures AND text-minus-structured mean>=-0.02; publish paired bootstrap 95% CI (CI crossing -0.02 is reported, not an extra PRD gate)", "status": "FAIL"}`


degraded calls: `{"actual_model_calls": 0, "max_calls_budget": 0, "maximum_primary_calls": 0, "maximum_repair_calls": 0, "maximum_turns_per_path": 4800, "maximum_verbalizer_calls": 0, "note": "Nominal count follows stub decisions; text may change decisions and length. Upper bounds include every turn up to each scenario horizon, including repairs.", "planned_primary_calls_at_stub_trajectory": 0, "planned_verbalizer_calls_at_stub_trajectory": 0, "structured_rehearsal_turns": 4307}`

degraded G8 detail: `{"automated_critical_candidates": 988, "eligible_test_budget": false, "flagged_traces": 988, "human_critical_failures": 0, "outcome_threshold_met": true, "paired_uncertainty_published": true, "real_text_run": false, "review_complete": false, "reviewed_traces": 0, "rule": "test pairs>=300 AND real replies AND all critical-flagged traces reviewed with no critical failures AND text-minus-structured mean>=-0.02; publish paired bootstrap 95% CI (CI crossing -0.02 is reported, not an extra PRD gate)", "status": "FAIL"}`

Validation and stub/fake runs are engineering checks, not G8 clearance. Only critical flags require review. Fallback and rejected-draft counts are informational; alone they never require review. Human reviews are absent unless supplied.
Bootstrap resamples paired episode differences (1,000 resamples); repeated seeds within the same scenario are not new scenario families. Intervals are conditional on this scenario bank, simulator and policy, not intervals for human effects.
Item identity uses A's public item_fidelity check (validator_v2); these checks are heuristics, not a semantic/math oracle. Review remains mandatory.
