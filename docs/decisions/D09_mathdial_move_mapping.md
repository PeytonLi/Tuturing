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

The mapping is unaudited and deliberately provisional. The following short
verbatim excerpts are selected from the 200-row audit; their qids belong to
the regrouped training split. Each illustrates why a source act is not a
reliable response-move label by itself.

- **`probing`**
  - `mathdial:test:0035:t10` — “wonderful. now answer this. Ron has red, blue and black pens. together there are 60 pens…” — poses a new problem and asks for its answer, closer to practice than a diagnostic probe.
  - `mathdial:test:0118:t14` — “no 3 boxes hold 24 pencils so 24/3=” — supplies the operation setup, making this partly a hint rather than an open probe.
  - `mathdial:test:0282:t16` — “No, we already calculated that Jack's total is 10*30 = 300” — gives a numeric result outright, which is telling rather than probing.
- **`focus`**
  - `mathdial:test:0049:t04` — “brillaint, so with the 15 cups and 25 cups what do we need to do with these numbers?” — asks the learner to choose an operation, so it functions as a probe.
  - `mathdial:test:0462:t02` — “How many visitors were in December?” — asks for a specific answer instead of narrowing attention with a hint.
  - `mathdial:test:0504:t16` — “And don't forget to add the percentage of blue eggs that have 5 candies.” — gives a direct procedural cue, which can function as a hint.
- **`telling`**
  - `mathdial:test:0012:t18` — “so 14 guests got a second hotdog. hence 40-14=26 guests did not get a second hotdog” — states the final count, going beyond a concept explanation.
  - `mathdial:test:0247:t06` — “The pants and coat costs ($100 - X/5) + X = $244.” — supplies a worked equation and result, making answer disclosure a concern.
  - `mathdial:test:0293:t12` — “120 students / 2 in a couple = 60 couples.” — directly states the computed quantity, rather than only explaining a method.
- **`generic`**
  - `mathdial:test:0048:t00` — “Hi Brenda, could you have a read through the question again and attempt the answer in a differnt way?” — asks for a new attempt, so it has a probing function.
  - `mathdial:test:0324:t00` — “Hi Claire, please walk me through your solution.” — requests reasoning and could be labeled as a probe.
  - `mathdial:test:0538:t00` — “You understood the new price is £240.  The second tank would also be £240…” — gives substantive correction and answer content, not just generic scaffolding.

Attribution for the quoted excerpts: Macina et al. (2023), “MathDial: A
Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math
Reasoning Problems,” Findings of EMNLP 2023, arXiv:2305.14536. Dataset
licensed CC BY 4.0; source revision
`acc3878459e0bd8c04ab840056572f0b8b1abe1f`.

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
