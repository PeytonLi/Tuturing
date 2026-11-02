# Decisions — Tuturing

Workspace-level decisions and their rationale, with dates and sources. Product and architecture decisions that belong to the PRD's owners go in `docs/decisions/` as the PRD's section 11.1 prescribes; this file points to them when they matter for the workspace.

## 2026-11-02 — Register the shared repository as a workspace project

**User direction:** initialize `https://github.com/PeytonLi/Tuturing` as a project; it is a hackathon project and the team needs to execute fast. Source: Personal Workspace capture `cap_4d37b006-5c9d-4541-b5dc-f82fceec5f89` (linked copy in this project's local journal).

**Implementation choices (assistant):**

- The repository belongs to Peyton Li and Xueming has write access, so the project directory is a clone of it rather than a new repository under `xueming-Felix`. The workspace record (`repoUrl`) points at `PeytonLi/Tuturing`; the Workspace's automatic `gh repo create` path was bypassed on purpose.
- Workspace files added at the root: `AGENTS.md` (with `CLAUDE.md` linking to it), `brief.md`, `decisions.md`, `.gitignore` additions and a README section. They do not collide with the PRD's layout in section 11.1.
- `journal/` (Lark conversation source records exported by the Workspace) is gitignored because the repository is shared with a teammate; the records stay on the Mac mini and in the Workspace database. Reversible by removing the `/journal/` line. `tasks/` (generated task records) is committed so the teammate can see what the agents are working on.
- `.gitignore` also gets the PRD's ignored directories (`artifacts/`, `data/private_raw/`, `data/sft/`) and standard Python caches.

**Status:** see brief.md for the Lark group, focus and next steps. No application code was written during initialization.

## 2026-11-02 — Make Felix-Claude an admin for Tuturing

**User direction:** “help me put felix-claude as admin for this project.” Source: capture `cap_110977ab-8071-4187-9e12-6a71a0af9e4d` in the Tuturing Lark topic.

**Implementation (assistant):** Granted Felix-Claude (`cli_aa52cde0b578c08b`) the Lark group-admin role in Tuturing (`oc_454768961d023702c1b8895f7238be31`). Lark returned success, and a separate group-info read confirmed the app in `bot_manager_id_list`. Added its missing Workspace wildcard binding to project `tuturing` and its Botmux group-directory binding to this checkout through the live daemon API; the saved configuration was verified without restarting services. The assignment is scoped to this project group; the Workspace-wide default bot and the PRD's Person A/Person B ownership remain unchanged.

## 2026-11-02 — Xueming's side is Person A

**User direction:** “we will be person A and my teammate is pushing setup one sec.” Source: capture `cap_f7092820-6b99-45d1-9140-729c3649ca5d` in the Tuturing Lark topic.

**Consequences (PRD sections 2 and 20):** workspace agents change only Person A's paths, as listed in the PRD and in the setup branch's `OWNERS.md`, and build against Person B's frozen fixtures and mocks. Person B's side is Peyton's.

**Assistant notes:** Peyton's agent had already pushed the setup on two branches (see brief.md, "Our side: Person A"). The response lane on one of them is Person A code, so Xueming's side maintains it once merged. Decisions recorded there, such as D10's base-model switch, are Person A decisions under PRD 2.3 and still need Xueming's confirmation; this file does not record one.

## 2026-11-02 — Run all Person A items in parallel; build datasets and benchmark frontier and open models

**User direction:** “go ahead, do them in parallel. whatever needs me. i think we should create the datasets and benchmark it with frontier models and open-sourced models too.” Source: capture `cap_5db17ad4-fe00-4f06-b3b7-1b975ea4fe64`. Xueming then supplied OpenAI and Fireworks API keys in the Lark topic; their values are never recorded in the repository.

**Implementation choices (assistant):**

- Five agents run in parallel from the Tuturing topic, each in its own git worktree and branch: `a/public-api-v1`, `a/response-service-v1`, `a/datasets-v1`, `a/benchmark-v1` and `a/extractor-v1`. Workspace tasks were not used because the Workspace dispatches one managed task per project at a time (platform `docs/operations.md`), which would serialize work Xueming asked to run in parallel. The coordinating session merges the branches into `main` after the CPU smoke passes.
- Agents change only Person A paths, leave the SFT config, trainer and D10 files to Peyton's agent, flag model-written data as synthetic and pending review, and never fill the human audit labels.
- Keys are stored in `~/.config/tuturing/llm.env` (mode 600), outside Git. Benchmark API spend is capped at $25 total across OpenAI and Fireworks; the cap is an assistant default awaiting Xueming's confirmation. Small open models also run locally through llama.cpp on the Mac mini at no cost.
- Fireworks control-plane metadata shows Qwen3.8-27B (D10's choice) supports LoRA but not serverless inference, so benchmarking or serving it needs a paid dedicated deployment. No deployment was created.

## 2026-11-02 — Benchmark models: gpt-6.1-sol and Qwen3.8-27B on a Fireworks deployment

**User direction:** “for frontier model benchmark use gpt6.1sol, for opensourced model use Qwen3.8-27B. we do deloy is fine.” Source: Lark message in the Tuturing topic at about 20:45 UTC, sent while the previous turn was running; its capture had not been exported when this was recorded.

**Implementation (assistant):** The benchmark's required models are now OpenAI `gpt-6.1-sol` and `accounts/fireworks/models/qwen3p8-27b` on an on-demand Fireworks deployment. The agent creates the deployment only once the harness is ready, deletes it after the run, and records proof of deletion. Other paid models were dropped. The $25 total cap now also covers the deployment's GPU hours and still awaits Xueming's confirmation.

## 2026-11-02 — Person A owns SFT

**User direction:** “send me what to review, when it's ready. 4. we own the sft” — the second part answers the open question on who edits Person A's files. Source: capture `cap_7c410346-63da-4a59-9c26-bfb410856672`.

**Consequences:** SFT is Person A's, as PRD section 2 and `OWNERS.md` already assign: the SFT configs, `tools/response/train_sft.py`, SFT data and decision records D09 and D10. Xueming's agents maintain these from now on, and the earlier hands-off rule for the files Peyton's agent was editing no longer applies to new work. D10's base-model choice is still unconfirmed; the benchmark of `gpt-6.1-sol` and Qwen3.8-27B is meant to inform it. Peyton's agent is outside this workspace, so Peyton has to relay the change to it.

## 2026-11-02 — Person B transaction engine implementation

**Source:** Person B learning-service handoff in this agent session; PRD sections 8 and 9.

**Implementation choices:** SQLite stores the full session snapshot as JSON and pending intervention
snapshots separately, so cancellation retains evidence but never commits option/item assignments.
Observe returns either `{kind: "plan", teaching_plan: ...}` or
`{kind: "session_end", session_end: ...}`. Live sessions sample only the original `train` families;
held-out validation/test families remain reserved. Items persist their family and sample seed.
Plan counters, rather than delivered turn counts, identify reservations and sampled instances.
The rules runtime defaults to `work/learning.sqlite3` under the existing ignored `work/` directory.
The store creates its parent directory automatically, including when used without the HTTP server.

`PROBE` and `CHECK_MASTERY` are independent assessments in this runtime; requested answers cannot
reveal an independent assessment. Both extractor assistance and committed-plan assistance gate BKT.
Explicit skip targets become the goal, matching the episode initializer's request semantics;
live goal changes update the goal and trigger the scheduler without inventing a navigation action.
Navigator streaks reset at option boundaries, as in `run_episode`. Policy failures retain already
accepted evidence, but unsuccessful operations are not cached. Exact successful idempotent retries
take precedence over the ended-session guard; fresh mutations on ended sessions are rejected.
Idempotency conflicts precede lookup of a different, potentially unknown session in a reused key's
body. No HTTP or Person A implementation is included.

## 2026-11-02 — Internal learning HTTP hand-off

**Source:** Follow-up Person B learning-service handoff in this agent session.

The stdlib HTTP server authenticates before reading any body. Missing or invalid trace IDs are
authentication failures (401), as requested. Error correlation IDs use `unknown` when absent or
invalid so even rejected requests receive contract-valid Error records. Unknown GET/POST routes
return 422 `INVALID_SCHEMA`. Successful requests return 200, including session creation.
POSTs require a nonempty ASCII Idempotency-Key of at most 128 characters. Chunked requests are not
supported; POSTs require Content-Length and at most 65,536 bytes of finite JSON. The server closes
each response connection, so an unread rejected body cannot become the next request.
`docs/learning_service.md` records a real local transport run with a development-only token.

## 2026-11-02 — Episode helper compatibility after simulator v2 merge

**Source:** Follow-up request to adapt the learning service after merging origin/main.

`_next_item` now consumes assessment-family usage and per-concept reuse counters, not an
assessment-family set. The engine persists both counters in the proposed plan snapshot and
promotes them only on commit. Cancellation and expiry still consume no teaching turn or family
usage. The existing `used_assessment` field now holds a count mapping. Legacy lists in sessions
and pending plan snapshots are accepted as one recorded use per listed family; earlier unrecorded
PROBE usage cannot be inferred from that list. Missing legacy reuse counters start empty.
Other imported episode helpers are unchanged. New train assessment families enter the existing
train-only allowlist automatically. Simulator v2 student/reward settings and training configs are
not used by the live engine; its rules config and checkpoint loading remain unchanged.

## 2026-11-02 — Benchmark v1 merged; Qwen3.8-27B blocked on a payment method

**Source:** the benchmark agent's final report in the Tuturing topic session; results in `experiments/response/benchmark_v1/`.

**Facts:** `gpt-6.1-sol` ran on 107 cases at `reasoning_effort: low` for $0.212; the model rejects `temperature: 0`, so the run is not deterministic. Qwen3.8-27B needs a dedicated Fireworks deployment (1x H100, $8.00 per GPU hour list) and the free dry run returned "payment method is required", so nothing was created and GPU spend is $0. Fireworks reports the model as `supportsServerlessTraining: false`, which contradicts D10's serverless training claim and needs checking before any SFT plan relies on it.

**Assistant note:** the automated checks over-report on dialogue data (numbers from the learner's own message count as unsupported math; answer fingerprints already in the stem count as disclosure). Whether to relax the validator is a Person A decision for Xueming; the blinded human review measures the gap meanwhile.

## 2026-11-02 — Fireworks key replaced; Qwen3.8-27B run started on Peyton's account

**User direction:** “use this for fireworks key instead …” (Lark message in the Tuturing topic on 2026-11-02; the key value is stored only in `~/.config/tuturing/llm.env`).

**Facts:** the new key resolves to Fireworks account `accounts/peli`, Peyton's. The free deployment dry run passed there, and the live deployment list was empty. The list also shows two deleted deployments of the same model created from that account earlier the same day, before this session used it. The benchmark agent was told to run the Qwen3.8-27B entry with the same safeguards (pilot first, $25 total cap including GPU hours, deletion verified afterward) and to add the Qwen replies as a supplement to the review sheet without reshuffling the 62 replies Xueming is already rating.

**Assistant note:** the response service merge (`afdd0e3`) left one open labeling choice: template-stub output is reported as `delivery_mode: model`. Xueming was asked; the assistant proposed `fallback` as the more honest label and will apply it unless Xueming objects.

## 2026-11-02 — Template-stub output is labeled `fallback`

**User direction:** “pick "fallback"” (Lark message in the Tuturing topic), answering the assistant's question from the response service merge.

**Implementation (commit `e497294`):** responders declare their own delivery label. The template stub says `fallback`, including when its CHECK_MASTERY template goes through the one permitted repair, because a template is a verified fallback realization and never model text (PRD 8.4). Only a language-model responder yields `model` or `repaired`. The runbook and tests were updated; 237 tests pass.

## 2026-11-02 — Qwen3.8-27B benchmark result and the item-substitution gap

**Source:** the benchmark agent's final report; results in `experiments/response/benchmark_v1/` (merge `374758a`).

**Facts:** Qwen3.8-27B ran on all 107 cases from a one-H100 Fireworks deployment on Peyton's account, created at 21:32:51 UTC and verified deleted at 21:36:42 UTC, for at most $0.51 of GPU time. Automated results: 41 of 107 delivered as model, 5 repaired, 61 fallback, 1.2 s median latency. A human read found that it substituted a six-sided die for the issued coin problem on both CHECK_MASTERY fixtures and the CHALLENGE fixture; the automated checks passed because the coin stem contains no digits to anchor the fidelity and math checks. gpt-6.1-sol kept the issued problem in all 17 fixture and smoke cases.

**Consequences (assistant):** an item substitution on a protected check is a critical failure under PRD 14.1, so the validator gets a digit-independent item-fidelity check before any model output is trusted; the response-service agent was tasked with it. The finding weighs on D10, which Xueming has not yet confirmed. The review sheet Xueming is rating was left byte-for-byte unchanged; the 40 Qwen replies were added as a supplement under new labels, which weakens blinding between Qwen and the earlier two systems but not between those two.

## 2026-11-02 — Public API merged; ten-session smoke passes against B's service

**Source:** the public API agent's final report; merge `428bda3`.

**Facts:** the public contract, fixtures, `apps/public_api` and the ten-session smoke workflow are on `main`. The smoke test starts Peyton's real learning service in-process and completed 10 of 10 sessions with byte-identical retries and hashes matching B's records (gate G4, A's side). The agent mapped PRD gaps to existing codes: `request_id` reuse with different content and an in-flight competing request both return `409 STATE_CONFLICT`; unexpected failures return `503 LEARNING_UNAVAILABLE`.

**Open for the owners:** D11 proposes keeping `Error` in A's contract and referencing B's `PublicState` and `SessionEnd`; the public session ID is B's allocated ID because B rejects a client-supplied one, a deviation from PRD 8.1; and four findings about B's service are recorded in D11 for the first integration review rather than patched.

## 2026-11-02 — Evidence extractor v2 merged

**Source:** the extractor agent's final report; merge `e1e0183` (462 tests pass).

**Facts:** the extractor was rewritten as `extractor_rules_v2`, tuned on the development split only, with the test split frozen by hash before tuning. It never emits `correct` or `incorrect`; `confidence` now means answer-commitment confidence (0.9, 0.85, 0.6 or 0), and B's engine drops grades below 0.80, so a learner message with several numbers does not update mastery. Both semantics are to be confirmed with Peyton at the first integration review (PRD 21.3). Two known test failures remain for a later version.


## 2026-11-02 — Strategist final evaluation: no v3 simulator, rules v3, BC chosen (all simulated)

**Source:** docs/results/strategist_final_v1.md (Person B Strategist evaluation session).

**Facts:** an always-worked-example baseline loses to rules on validation, so simulator v2 stays (no v3). The worked-example collapse came from a rules loop: after two unsuccessful hints, rules repeated WORKED_EXAMPLE indefinitely. Rules v3 adds a public-evidence misconception branch (PROBE, then WORKED_EXAMPLE or EXPLAIN_CONCEPT) and follows a worked example with independent PRACTICE. Public misconception evidence fills the existing 25 slots of features_v1 (no contract change), and the learning service uses the same helper. Checkpoint rule, declared before test: highest validation mean gain, with ties going to rules. BC (`bc_rules3_seed0`) was chosen. Its edge over rules is +0.004 on validation (CI crosses zero) and +0.005 on the single test run. Simulator uncalibrated, no human data.

## 2026-11-02 — validator_v2 item-fidelity check merged

**Source:** the response-service agent's final report; merge `bd32b5b` (480 tests pass).

**Facts:** `action_faithful` now includes a digit-independent item-fidelity check for item-presenting and item-referring moves: the reply must reuse enough of the issued stem's content words (or the item's numbers, or the permitted hint) and must not introduce a competing scenario or quantity that the plan does not contain. It is not repairable, so a substituted item always falls back. Calibration on the review-sheet texts: 0 of 11 fallback, 0 of 11 gpt-6.1-sol fixture and 0 of 30 MathDial replies flagged; 4 of 11 Qwen fixture replies flagged, all genuine substitutions. Known limits: the lexicon covers probability devices rather than arbitrary story objects, WORKED_EXAMPLE is not yet covered, and the benchmark report still shows the pre-v2 numbers.

## 2026-11-02 — Skip the human MathDial move audit; model-made mapping, merged into the SFT data

**User direction:** “we skip human labeling for MathDial move audit, claude code you help do the mapping and merge into our datasets.” (Lark message in the Tuturing topic.)

**What changes:** D09's decision rule required Xueming and Peyton to label the 200-turn audit independently and reach 80 percent agreement before MathDial's mapped moves could be used. Xueming waives that for the hackathon. The mapping is now produced by models: a turn-level move label for every usable MathDial teacher turn, calibrated against the 200 model-drafted labels already in `evaluation/response/mathdial_mapping_audit_200_model_draft.csv`, with the agreement between the draft act mapping and the turn-level labels reported. MathDial examples then enter the SFT mix at the PRD's proposed 40 percent, labeled `action_label_source: mapped` with the labeling model recorded, never as human labels.

**Assistant note:** this keeps MathDial out of Peyton's Strategist behavior cloning unless Peyton separately accepts model labels; D09's reviewer line stays with Peyton. Labeling API spend comes out of the $25 benchmark cap.

## 2026-11-02 — Item-fidelity false positives fixed; main pushed at 512 tests

**Source:** the response-service agent's follow-up report; merges `9457fa4` and `3120ad3`.

**Facts:** the four rejections under `validator_v2` were false positives: "in your head" matched the coin family's `head`, and "in three moves. A marbles experiment" let a number cross a sentence boundary into the stem's noun. The quantity rule now requires a scenario object noun in the same sentence, and `head`/`tail` count only alongside a coin anchor. The four Qwen substitutions still fail. `build_original_sft` accepts 371 of 371 under `validator_v2`. Main was pushed after the full CPU smoke (512 tests, both validators, content validation, a ten-episode simulate).

## 2026-11-02 — Benchmark re-scored under validator_v2 with the probability set

**Source:** the benchmark agent's final report; merge `ab53f8e`; report in `experiments/response/benchmark_v1/report.md`.

**Facts:** stored outputs were re-scored without new calls; the new `rescore` subcommand recomputes checks from stored texts. Qwen's three fixture substitutions now fail `action_faithful`. On the 55 synthetic probability cases (eight moves, five concepts): fallback 55/55, gpt-6.1-sol 33/55, Qwen3.8-27B 19/55 delivered as model; Qwen asked two questions in 8 cases and hit the token cap twice. The second Qwen deployment ran 22:39 to 22:42 UTC and was verified deleted; this run cost $0.57, all time $1.29. The three smoke CHALLENGE texts predate the fixture's change to hard difficulty; the checks do not depend on it.

**Reading for D10 (assistant):** prompt-only, the frontier model passes the strict checks far more often than Qwen3.8-27B on our own probability items, and the gap is driven by action fidelity and math support, which SFT targets. This is evidence for the decision, not the decision.

## 2026-11-02 — Run a smoke SFT of Qwen3.8-27B on Fireworks

**User direction:** “smoke SFT on Qwen3.8-27B via Fireworks” (Lark message in the Tuturing topic), answering the assistant's question on whether to fine-tune.

**Scope and budget:** a smoke run in the PRD 13.5 sense on the 261 training examples of `original_v1` (MathDial is not included while its mapping is in progress), evaluated on the 55 validation cases against prompt-only Qwen3.8-27B; the test split stays frozen. Budget: $20 for training plus evaluation deployments, the figure the assistant proposed in the same exchange and Xueming did not change. Fireworks reports the model as not serverless-trainable, so the agent first checks feasibility with free calls and stops if training is impossible on the account or would exceed the cap.

**What this is not:** D10 is not yet confirmed. The smoke result is evidence for that decision, alongside the benchmark.


## 2026-11-02 — Pace adaptation: one-step requests, a pace estimator, and reused pace slots (Person B)

**User direction:** Peyton asked that the tutor notice pace without being told, and selected "Bug fix plus the full plan, steps 1 to 6".

**What changes:** "faster" and "slower" now move pace one step along detailed, normal, concise instead of jumping to an extreme, so a learner can get back to normal. A per-session pace estimator (`packages/learning/pace.py`) keeps a belief over the three paces, treats a typed request as strong evidence, adds weak behavioral evidence (fast correct streaks lean concise; repeated wrong answers, hint and answer requests lean detailed; optional reply-time and message-length cues from `evidence.extensions.pace_cues` when present), and switches only after two agreeing turns. features_v1 keeps its 140 slots and order; slots 136 and 137 (`explicit_faster`, `explicit_slower`) change meaning from on/off request flags to 0 to 1 "leaning faster" and "leaning slower" scores, which equal 1.0 on a turn with a typed request. The simulator gains observable pace cues and a detail-preferring profile. Simulated results are uncalibrated and are not human evidence.


## 2026-11-02 — Simulator pace cues and the careful_detailed learner (Person B, simulator lane)

**What changes:** `packages/simulator/pace_cues.py` adds `sample_pace_cues`, which returns the shared per-turn cue dict (reply_seconds, message_words, skim, confusion, why_question) from a new `pace_cues` random stream, so no existing draw changes. A latent drive, strength × ((preferred − 1) + mismatch_weight × (preferred − tutor)) on the concise/normal/detailed index, shifts lognormal reply times, Poisson word counts and Bernoulli flags; noise keeps the classes overlapping. Parameters live in each simulator config's `pace_cues` block (strength 1.0 in v1 and v2, 0.5 with wider noise in alt) and fall back to code defaults. A `careful_detailed` profile (preferred_pace detailed, mastery 0.35 to 0.6, help-seeking 0.15) is in all three configs, and `help_seeking.detailed_too_concise_bonus` (0.10) raises help-seeking only when a caller passes `tutor_pace` to `sample_student_outcome`.

**Why a separate scenario file:** six careful_detailed scenarios (ids 31 to 36; 4 train, 1 validation, 1 test) live in `data/scenarios/proto_v1_pace.yaml`, generated by `tools/dev/generate_pace_scenarios.py`, and load only with `load_scenarios(include_pace_scenarios=True)`. Appending them to `proto_v1.yaml` would change the per-split cycling in `simulate_episodes` (index modulo split size) and the scenario hash recorded in `data/manifests/response/original_v1.json`, so earlier simulated results would no longer reproduce. These cue dynamics are assumptions, not calibrated effects.

## 2026-11-02 — MathDial move labels and the SFT mix merged

**Source:** the mapping agent's final report; merge `9bc3a1e`; D09 updated to superseded for the hackathon.

**Facts:** the per-act draft mapping agreed with the 200 model-drafted turn labels on 74 of 200 (37 percent), so it was replaced by per-turn labels from `gpt-5.4-mini`, the cheapest tested model with the highest agreement: 80.2 percent on 182 held-out rows (95 percent interval 73.8 to 85.4), short of the 85 percent target. 9,575 turns were labeled for $4.57. The agent added admission rules beyond its brief that keep 5,224 of 8,146 labeled MathDial examples; the mix uses 248 of them alongside all 371 original examples. When merging, the generated `original_v1` files were regenerated on `main` with the narrowed validator and the MathDial comparison (371 of 371); the only change to `probability_cases_v1.jsonl` is the strategist version stamp from Peyton's rules v3 and one reworded reference reply.

**Also recorded:** another Claude session on this Mac committed a note to `brief.md` that A's OpenAI-compatible responder sends `temperature: 0` and `max_tokens`, which `gpt-6.1-sol` rejects, so frontier-model turns through the real pipeline fall back silently. The response-service agent is fixing it.

## 2026-11-02 — Full SFT run requested

**User direction:** “do the full SFT run” (Lark message in the Tuturing topic), sent while the smoke run was still in progress.

**Implementation (assistant):** the full run is queued on the smoke-run agent and starts only if the smoke run shows the Fireworks training path works (PRD 13.5 requires the smoke checks before a long run). Data: the 60/40 `mix_v1` (435 training examples). At most 3 epochs. Checkpoints are chosen on validation by the PRD 13.5 order, with no human score yet. The probability test split is then used once, for the selected checkpoint only, compared with the prompt-only results already measured. Budget: $40 for the full run on top of the smoke's $20, an assistant figure Xueming can change. Every deployment is deleted and verified.

**On D10:** Xueming directed both the smoke and the full run on Qwen3.8-27B. The assistant treats that as confirming D10's base model for the hackathon run unless Xueming says otherwise; D10's file is left unchanged until the results are in.

