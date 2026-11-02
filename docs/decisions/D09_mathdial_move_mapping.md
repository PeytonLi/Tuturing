# D09: Draft MathDial teacher-act mapping

- **Status:** superseded for the hackathon by Xueming's decision of 2026-11-02 (see below); the human audit procedure stays desirable
- **Owner:** Person A
- **Reviewer:** Person B

## 2026-11-02: human audit waived; model-made turn-level labels instead

**Decision (Xueming, Lark message in the Tuturing topic, 2026-11-02):** skip the human labeling of the 200-turn audit for the hackathon; have models produce the move mapping and merge MathDial into the SFT data. Peyton's reviewer role is unchanged: Person B's Strategist may not imitate these labels without Peyton's acceptance, and the labels are never presented as human labels (`action_label_source: mapped`, labeler recorded).

**What was done instead.**

1. **Calibration set.** The 200-row audit sheet carries model-drafted labels (`evaluation/response/mathdial_mapping_audit_200_model_draft.csv`, written by claude-fable-5-1 under `audit_draft_v1`, all from the regrouped train split; the drafter flagged 178 of 200 as ambiguous). The draft per-act mapping below agrees with those turn-level drafts on **74 of 200 turns (37 percent)**: probing to PROBE 17/50, focus to HINT 21/50, telling to EXPLAIN_CONCEPT 21/50, generic to EXCLUDE 15/50. The act is not a usable move label.
2. **Turn-level labeling.** `tools/response/label_mathdial_moves.py` (config `configs/response/mathdial_moves_v1.yaml`) labels each non-empty teacher turn within the word cap (18,541 of 18,609 turns) through the OpenAI Chat Completions API with one of the eight moves or EXCLUDE. Each request sees only the student's last message and the teacher turn, the reply is a strict two-line format (one retry, then EXCLUDE), temperature is 0 where the model accepts it, and reasoning is off. The prompt (`configs/response/mathdial_moves_prompt_v3.txt`) gives the PRD move definitions, disambiguation rules and 18 drafted rows as examples (3 per label, seed 20261102), which are held out of every agreement figure.
3. **Model choice.** Pilots on the 182 held-out rows with prompt v3. Standard prices (USD per 1M input/output tokens) were read from developers.openai.com/api/docs/pricing on 2026-11-02:

   | Model | Price | Agreement |
   |---|---|---|
   | gpt-5-nano | $0.05/$0.40 | 63.7 percent |
   | gpt-5-mini | $0.25/$2.00 | 58.8 percent |
   | gpt-5.4-nano | $0.20/$1.25 | 70.9 percent (72.0 with low reasoning) |
   | **gpt-5.4-mini** | **$0.75/$4.50** | **80.2 percent (146/182; Wilson 95% 73.8 to 85.4)** |

   Per act, gpt-5.4-mini agreed on probing 82, focus 74, telling 73 and generic 91 percent. Six examples per label did not help (79.3 percent on 164 rows), and earlier prompt versions scored 38 to 69 percent. **No candidate reached the 85 percent target.** gpt-5.4-mini was used as the closest, and the shortfall is recorded in `data/manifests/response/mathdial_moves_v1.json`. The pilot and the full run gave the same label on 85 of the 88 calibration rows both labeled.
4. **Budget and coverage.** The task's hard cap was $5. Labeling all 18,541 turns with gpt-5.4-mini was estimated at $10.81 from the pilot's token usage, and would cost about $7.7 at the rate later observed. So every test-split turn was labeled, then train and validation turns in a per-act round robin (four train picks per validation pick) until the budget was used, in two passes. Labeled: 9,575 turns. That is all 2,127 test turns, 1,489 of 1,576 validation turns and 5,959 of 14,838 train turns, with no API errors and no malformed replies. Counts per label: PROBE 2,763, HINT 2,810, REFLECT 1,656, EXCLUDE 1,153, EXPLAIN_CONCEPT 858, PRACTICE 286, CHALLENGE 24, WORKED_EXAMPLE 20, CHECK_MASTERY 5. Spend was $4.57 of $5.00 ($0.55 for probes and pilots, $4.01 for the full passes), recorded in the ignored ledger under `experiments/response/runs/mathdial_moves_v1/`. `data/response/mathdial_moves_v1/labels.jsonl` holds ids and labels only.
5. **Use in SFT.** `configs/response/mathdial_move_map_v2.yaml` (status `model_labeled`) points `prepare_sft` at the labels file. EXCLUDE-labeled and unlabeled turns are rejected, and the leak review's rejections of protected examples stay in force: 270 targets state the protected answer, and 6 prompts hold the reference solution. 8,146 MathDial examples remain (train 5,030, validation 1,271, test 1,845), labeled `action_label_source: mapped` with the labeler in `extensions.move_label`.
6. **Admission to the mix.** `tools/response/build_sft_mix.py` writes `data/sft/mix_v1` at the PRD 13.2 mix (60 percent original, 40 percent MathDial, capped by what passes, sampled per move). A MathDial target enters only if it passes these rules (`configs/response/sft_mix_v1.yaml`):
   - its numbers are supported by the plan, it states no protected answer, and it respects the length and one-question limits;
   - under PRACTICE, CHALLENGE or CHECK_MASTERY it keeps the issued problem (validator_v2 item fidelity), because most such MathDial turns pose a new problem and would teach item substitution;
   - it is not WORKED_EXAMPLE, because a MathDial plan has no approved example;
   - it does not address the learner by a profile name the prompt never shows (864 targets; leak review open point 2).

   5,224 of 8,146 pass. The mix holds 619 examples: train 261 original + 174 MathDial, validation 55 + 37, test 55 + 37, with no repeated exposure. `action_faithful` is not required, because MathDial plans lack the hint, facts, previous step and example its move rules need (see the limitations below).

**Still desirable.** A human review of a stratified sample of the model labels would turn the 80 percent model-to-model agreement into an audited figure, as would Xueming's and Peyton's independent labels on the 200-row sheet. Until then the labels serve response wording only, which is what the original decision rule prescribed for anything under 80 percent human agreement. The calibration labels are themselves model drafts, so 80 percent measures agreement between two models, not accuracy.

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
