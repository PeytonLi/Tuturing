# D09: Draft MathDial teacher-act mapping

- **Status:** proposed
- **Owner:** Person A
- **Reviewer:** Person B

## Draft mapping

| MathDial act | Draft response move | Rationale |
|---|---|---|
| `probing` | `PROBE` | Usually invites the learner to state or extend a reasoning step. |
| `focus` | `HINT` | Often narrows attention to a relevant part of the problem. |
| `telling` | `EXPLAIN_CONCEPT` | Often explains a calculation or gives substantive reasoning. |
| `generic` | `EXCLUDE` (`generic_act`) | Frequently provides praise, acknowledgement, or conversational scaffolding without a stable response action. |

The mapping is unaudited and deliberately provisional. Ambiguous examples
selected for review are in
`evaluation/response/mathdial_mapping_audit_200.csv`; the exact source text is
kept there rather than duplicated in this decision record. Representative
row references include:

- `probing`: `mathdial:test:0035:t10`, `mathdial:test:0118:t14`,
  `mathdial:test:0282:t16`
- `focus`: `mathdial:test:0049:t04`, `mathdial:test:0462:t02`,
  `mathdial:test:0504:t16`
- `telling`: `mathdial:test:0012:t18`, `mathdial:test:0226:t12`,
  `mathdial:test:0247:t06`
- `generic`: `mathdial:test:0048:t00`, `mathdial:test:0265:t00`,
  `mathdial:test:0324:t00`

## Limitations

- MathDial is primarily GSM8K-style arithmetic, not the probability domain of
  the prototype curriculum.
- The dialogues use LLM-simulated students and do not establish response
  quality with real learners.
- A mapped `HINT` plan has no source-provided `permitted_hint`, and a mapped
  `EXPLAIN_CONCEPT` plan has no source-provided `approved_facts`; those plans
  therefore do not satisfy the corresponding discriminated requirements in
  PRD 9.3.
- The external MathDial concept is not in the prototype concept catalogue, so
  its navigation action uses `action_id: -1`.
- Mapping an act does not establish that each turn is faithful to that move;
  the audit is intended to reveal these mismatches before any behavior-cloning
  decision.

## Audit procedure and decision rule

Person A and Person B independently fill `label_person_a` and
`label_person_b` in the 200-row audit CSV using one of the eight response moves
or `EXCLUDE`. Reviewers should label the teacher turn itself in context, not
merely copy its source act. After both columns are complete, report raw
agreement and Cohen's kappa over all 200 rows.

If raw agreement is below 80% (fewer than 160 agreements), drop MathDial
behavior cloning. Retain it, at most, as response-style data only after this
decision is rewritten and approved. No mapped labels are considered audited
before the independent review is complete.
