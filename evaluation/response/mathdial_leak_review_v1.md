# MathDial SFT leak review v1

- Date: 2026-11-02. Reviewer: Claude Opus 5.5 (model ID `claude-opus-5-5`) running as a Claude Code agent for Person A. This is a model-written review: it is pending Xueming's confirmation and is not a human sign-off.
- Source: MathDial revision `acc3878459e0bd8c04ab840056572f0b8b1abe1f` (CC BY 4.0), downloaded with `uv run python -m tools.response.prepare_sft --config configs/response/mathdial_v1.yaml`. The raw file hashes equal the ones already recorded in `data/manifests/response/mathdial_v1.json`, so the pinned download reproduced.
- Question: does any prepared example show (1) a future turn, (2) the answer key in a protected prompt, or (3) the success label? "Protected" here means `constraints.reveal_allowed = false`. MathDial plans never use `assessment_mode = independent`, so none of them is a protected assessment in the PRD 9.3 sense.

## Verdict

1. **Future turns: none found.** In all 14,111 examples of the original run, the context equals the opening student solution plus the turns before the target, the prompt equals `compile_prompt(plan)`, and the completion equals the target turn. No later turn's text appears in a prompt unless the past dialogue already contains the same text.
2. **Answer key in protected prompts: found in 5 examples, now fixed.** The pipeline never copies `ground_truth` into a protected plan. In 5 examples, though, an earlier teacher turn had pasted the whole reference solution into the dialogue, so the protected prompt contained it verbatim. A related problem: 410 protected targets state the forbidden final answer, which trains the model to break the prompt's own "Do not reveal the final answer" rule (PRD 13.2 step 6). `tools/response/prepare_sft.py` now rejects both kinds of example, and `tests/unit/response/test_prepare_sft_protected.py` covers them.
3. **Success label: none found.** No prompt or plan contains the `self-correctness` value or any other private field (`teacher_described_confusion`, `student_profile`, `self-typical-*`), and no record contains those field names. `teacher_described_confusion` text shows up in 34 prompts (32 after the fix) only because a teacher said the same words in an earlier turn.

## Method

**Automated audit of every record.** `uv run python -m tools.response.audit_mathdial_sft --config configs/response/mathdial_v1.yaml` (new tool) compares each prepared record with the raw dialogue it came from. Text matches are made on case- and whitespace-normalized strings of at least 25 characters. A match counts as a leak only if no earlier turn, the opening solution, the problem stem, or (when reveal is allowed) the permitted solution contains the same text.

| Finding (count of examples) | Before fix (14,111) | After fix (13,696) |
|---|---:|---:|
| Context differs from the preceding turns | 0 | 0 |
| Prompt differs from the compiled plan | 0 | 0 |
| Completion differs from the target turn | 0 | 0 |
| Later-turn text in prompt, not explained by earlier text | 0 | 0 |
| Private field text in prompt, not explained by dialogue | 0 | 0 |
| Private field name in the record | 0 | 0 |
| Reveal-forbidden examples | 11,025 | 10,610 |
| Protected prompt contains the full reference solution | 5 | 0 |
| Protected target fails the disclosure check (states the final answer) | 410 | 0 |
| Protected prompt where an earlier tutor turn already stated the final answer | 648 | 511 |
| Protected prompt containing one reference-solution line, through dialogue | 168 | 158 |
| Target uses the student's profile name, which is absent from the prompt | 407 | 396 |

The "before" column comes from running the committed `prepare_sft.py` (HEAD `782f948`) into a scratch directory and auditing that output with the same tool.

**Split check.** In the accepted output, 0 of 1,108 qids and 0 exact problem texts appear in more than one regrouped split.

**Reading.** I read 28 examples end to end: the full system and user prompt, the target, and the raw turns after the target, checked against the raw record's `ground_truth` and `self-correctness`. Three examples were drawn per split and mapped move with `random.Random(20261102)`; one more is a flagged case. Splits are the regrouped ones (by qid), not MathDial's official split.

| # | Example | Split | Move (act) | Reveal | What I saw |
|---|---|---|---|---|---|
| 1 | `mathdial:train:1775:t17` | test | EXPLAIN_CONCEPT (telling) | yes | 15 prior turns. Target pastes the reference solution, which is allowed. A later student turn is absent from the prompt. A prior tutor turn has a typo artifact ("worth.4"). |
| 2 | `mathdial:test:0270:t02` | test | EXPLAIN_CONCEPT (telling) | yes | Target tells one step ("today is 35"), but the prompt carries the full solution and final answer, because every telling turn gets `reveal_solution`. |
| 3 | `mathdial:train:1018:t08` | test | EXPLAIN_CONCEPT (telling) | yes | Target is the praise "Yes correct well done", so the label is noise (generic-like). No leak. |
| 4 | `mathdial:train:1760:t12` | test | HINT (focus) | no | "But the basket can only hold 60". Clean. |
| 5 | `mathdial:train:1834:t04` | test | HINT (focus) | no | The target is a question ("how many hits?") that functions as a probe. Clean. |
| 6 | `mathdial:train:0054:t14` | test | HINT (focus) | no | The learner had already computed 33 - 31 = 2, the final answer, which is learner-produced and not a leak. The target asks for a recheck. |
| 7 | `mathdial:test:0280:t04` | test | PROBE (probing) | no | Clean diagnostic question. |
| 8 | `mathdial:train:0664:t04` | test | PROBE (probing) | no | Same qid as #2. Both sit in regrouped test, which confirms the qid grouping. Clean. |
| 9 | `mathdial:test:0116:t00` | test | PROBE (probing) | no | First turn: the context is only the opening solution. The target greets the learner as "Claire", a name that appears nowhere in the prompt. |
| 10 | `mathdial:train:0355:t12` | train | EXPLAIN_CONCEPT (telling) | yes | Target explains the misconception. The insulting `teacher_described_confusion` text is not in the prompt. |
| 11 | `mathdial:train:1336:t04` | train | EXPLAIN_CONCEPT (telling) | yes | Target is "yes", so the label is noise. A prior tutor turn had pasted part of the solution. |
| 12 | `mathdial:train:1847:t12` | train | EXPLAIN_CONCEPT (telling) | yes | Target "so 66 feet minus 7 feet?" is a probe. The later "= 59" turn is absent from the prompt. |
| 13 | `mathdial:train:2095:t04` | train | HINT (focus) | no | Directive hint. Clean. |
| 14 | `mathdial:train:1480:t06` | train | HINT (focus) | no | Probe-like question. Clean. |
| 15 | `mathdial:train:0344:t04` | train | HINT (focus) | no | Clean. |
| 16 | `mathdial:train:2096:t16` | train | PROBE (probing) | no | 17-turn context. The target poses an analogous problem. Clean. |
| 17 | `mathdial:train:1383:t02` | train | PROBE (probing) | no | "Let's think about how much Rene had" is closer to a hint. Clean. |
| 18 | `mathdial:test:0593:t08` | train | PROBE (probing) | no | "What is 5 divided by 5?" stops one step short of the answer, 1. A one-digit fingerprint makes the disclosure check brittle. |
| 19 | `mathdial:train:1667:t14` | validation | EXPLAIN_CONCEPT (telling) | yes | States the final answer (12), which is allowed. |
| 20 | `mathdial:train:1098:t20` | validation | EXPLAIN_CONCEPT (telling) | yes | States the final answer (2), which is allowed. |
| 21 | `mathdial:train:0747:t10` | validation | EXPLAIN_CONCEPT (telling) | yes | States 27201, which is allowed. An earlier tutor turn had already said 27200. |
| 22 | `mathdial:train:2034:t08` | validation | HINT (focus) | no | Analogy question. Clean. |
| 23 | `mathdial:train:1432:t06` | validation | HINT (focus) | no | Clean. |
| 24 | `mathdial:train:2176:t07` | validation | HINT (focus) | no | The source repeats a tutor turn ("Talk me through your solution?" twice). Clean. |
| 25 | `mathdial:train:0778:t01` | validation | PROBE (probing) | no | Turn t00 was the bare greeting "Emily,". `compile_prompt` shows that greeting before the learner's opening solution, reversing the true order. This is a rendering quirk, not a leak. |
| 26 | `mathdial:train:0178:t08` | validation | PROBE (probing) | no | Clean. |
| 27 | `mathdial:train:1817:t04` | validation | PROBE (probing) | no | Analogy question. Clean. |
| 28 | `mathdial:test:0041:t16` | train | HINT (focus) | no | Flagged case. Tutor turn t14 had pasted the full reference solution, so this protected prompt contained it. Now rejected as `reference_solution_in_protected_context`. |

Totals: 28 examples read, covering 9 per split plus one extra train case and all three mapped moves. All 28 were free of future turns and of the success label. One (#28) showed the answer key in a protected prompt.

## Fix

The `prepare_sft.py` generator is now `v1.1`. After building a plan, it rejects a reveal-forbidden example in either of two cases:

- `reference_solution_in_protected_context` (5): the normalized full `ground_truth` occurs in the compiled prompt.
- `target_states_protected_answer` (410): `packages.response.validate.run_checks` reports `disclosure_allowed = false` for the target against the plan's forbidden final-answer fingerprint. Many of these targets restate or confirm a number the learner had already given, as in "That is right, 26". The serving validator has no context, so it would reject the same reply at run time, and the model should not learn it under a no-reveal plan.

Accepted examples went from 14,111 to 13,696: train 10,921, validation 1,192, test 1,583. The 200-row mapping audit sheet is unchanged, because it samples candidates before rejection; I regenerated it and compared byte for byte. The manifest was regenerated.

## Open points for Xueming (not fixed here)

1. **Answer already shown, plan still protected (511 examples).** An earlier tutor turn stated the final answer, but the plan still forbids reveal and does not mark `assistance = answer_shown`. I kept these because the answer was visible to the learner and the target does not restate it. Dropping them is a one-line rule if you prefer.
2. **Invented names (396 targets).** Teachers address the student by a name that comes from the private `student_profile`. The model cannot see the name in the prompt, so it would learn to make names up. Options: strip leading vocatives, replace them with a neutral address, or reject these targets.
3. **Telling turns get the full solution.** Every `telling` turn becomes EXPLAIN_CONCEPT with `reveal_allowed = true` and the whole `ground_truth` in the prompt, even when the teacher tells one step (#2) or only praises (#3, #11). That conflicts with the mapping audit, which may relabel many of them. Revisit after the audit.
4. **Schema conformance.** Prepared MathDial records keep `checks` as nulls and add a top-level `extensions` key, so they do not validate against `SFTExample` in `contracts/response_v1.schema.json` (booleans required, no extra keys). The trainer must not schema-validate them as-is, or the prep should compute the checks and move extensions into the manifest.
5. **Prompt ordering quirk (#25).** `compile_prompt` drops any student turn whose text equals the last learner message and shows the message after the dialogue. This can reorder an opening solution after a tutor greeting, and it can drop an earlier verbatim repeat. Small, but worth a fix in the prompt compiler.
6. **Label noise.** Of the 28 examples read, 4 mapped labels look wrong for the turn: #3 and #11 are praise labeled telling, #12 is a probe labeled telling, and #17 is a hint labeled probing. Two more (#5, #14) are questions labeled focus. This matches D09's caution. The model draft labels for the 200-row sheet are in `evaluation/response/mathdial_mapping_audit_200_model_draft.csv`.

Attribution: Macina et al. (2023), "MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems", Findings of EMNLP 2023, arXiv:2305.14536. Licensed CC BY 4.0. Short excerpts above are quoted from revision `acc3878459e0bd8c04ab840056572f0b8b1abe1f`.
