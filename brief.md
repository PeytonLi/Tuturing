# Tuturing — current understanding

## What this is

A hackathon project by Peyton Li and Xueming Xu: an adaptive probability tutor (English text, one learner per session, prototype concepts C01/C02/C03/C06/C08) defined by Peyton Li's PRD v1.0 in [docs/adaptive_ai_tutor_prd.md](docs/adaptive_ai_tutor_prd.md) (commit `ea8c388`, dated 2026-11-02, "proposed design"; nothing trained or measured yet). Three trained components:

- **SFT response model** (Person A): LoRA on `Qwen/Qwen2.5-7B-Instruct`; renders the issued `TeachingPlan` and cannot alter curriculum, mastery, reward or episode end.
- **Curriculum Navigator** (Person B): picks a concept option (76 actions) at session start and at deterministic boundaries (4-turn option, two independent successes/failures, skip or goal request, content exhausted).
- **Teaching Strategist** (Person B): picks move/difficulty/pace (72 actions) every teaching turn; horizon 16; PPO trains each policy with the other frozen.

Non-goals (PRD 1.3): token-level RL of the response model, LM-chosen curriculum, learned termination, multi-subject, audio/vision, inferred prerequisites, live learning from real students, high-stakes grading, school integrations.

## Ownership (PRD 2 and 20)

- **Person A** owns prompts, fallbacks, validators, evidence extraction, data preparation, SFT and serving: `contracts/response_v1.schema.json`, `contracts/public_v1.openapi.yaml`, the root bundle manifest and root tooling, `apps/public_api`, `apps/response_service`, `packages/response`, `packages/evidence_extractor`, and every `*/response/` directory.
- **Person B** owns content, grading, state, scheduler, simulator, rewards, PPO and evaluation: `contracts/learning_v1.schema.json`, `apps/learning_service`, `packages/learning`, `packages/simulator`, `packages/policies`, `data/original_content`, `data/scenarios`, and every `*/learning/` directory.
- A never owns simulator calibration, student state, knowledge graph, either policy or RL evaluation; B never owns SFT, response generation, inference infrastructure or the public response API. The owner of a contract changes it and ships fixtures after cross-review; semantic changes need a major version. Disputes are recorded in `docs/decisions/`.
- Each side builds against the other's frozen fixtures and stubs: A on B's `TeachingPlan` fixtures and mock observe/commit/status/conflict service; B on A's CPU response stub through the contract adapter. Handoffs publish immutable artifact and schema versions, passing fixtures, limitations and one reproducible command (PRD 20.4).

## Layout, commands and resources

- Layout is PRD 11.1 (`contracts/`, `apps/`, `packages/`, `data/`, `configs/`, `experiments/`, `evaluation/`, `tests/`, `tools/`, `artifacts/`, `docs/decisions/`, `docs/runbooks/`, plus `OWNERS.md`, `pyproject.toml`, `dependency-lock/`). The workspace files at the root (AGENTS.md, CLAUDE.md, brief.md, decisions.md) do not collide with it.
- Entrypoints are `python -m tools.{response,learning}.<command>` (PRD 11.2): `validate_contracts`, `validate_content`, `prepare_sft`, `train_sft`, `serve`, `simulate`, `train`, `evaluate`. No task runner. Python with a committed lock; no Python version or package manager is named.
- A CPU-only smoke workflow must start both stubs, run ten scripted sessions, exercise a retry and produce a report with no GPU, cloud credential, gated dataset or internet dependency. GPU (one rented 48 GB) is assumed only from week three; B has no GPU dependency through M2.
- External data is optional and authored content comes first. FoundationalASSIST needs gated-access approval; TutorBench terms are unverified; Qwen weights are pinned to a revision.

## Milestones (PRD 20.1, illustrative, "not a schedule commitment")

M0 days 1-2 contract and fixture freeze (gate G0) → M1 week 1 prompt compiler and fallbacks, graph/solver/BKT/rules baseline, CPU ten-session demo (G1, G2, G4) → M2 week 2 SFT data pilot and 30-scenario simulator (G3) → M3 weeks 3-4 first frozen SFT model and prototype policies → M4 weeks 5-6 600-case suite, five-seed training → M5 week 7 final evaluation (G5-G9) → M6 week 8 release bundle. The PRD records no hackathon deadline or demo date.

## First implementation action (PRD 21 and 26.3)

The contract and fixture freeze comes first. Day-one lists:

- **A (21.1):** read sections 2, 8, 9, 13, 14, 19; create response directories and OWNERS entries; GenerateRequest/GenerateResponse/PublicState/Error/endpoint schemas plus fixtures (eight moves, protected check, rejected response); CPU stub that renders approved content and rejects non-null reveal solutions on independent plans; prompt projection hiding IDs, fingerprints, private state and solutions; durable request keys and staging on B's mock with an identical response after a dropped connection; 30 original response examples plus source/split manifest; publish the stub artifact with loading instructions and an SFT masking smoke config.
- **B (21.2):** read sections 2, 5-10, 15-19; create learning directories, fixtures, feature/action encoding specs; five-concept graph with 15 stable indices, validation and eight misconceptions; five verified problems with solver fixtures, then 15 teaching templates, with assessment families reserved; belief updates with separate assisted/independent counts; rules navigator and strategist, scheduler, masks, goal stack; simulator reset/step with private state, seed streams and a leakage test; export `TeachingPlan` fixtures, the mock observe/commit/status/conflict service and a ten-session scenario script.
- **Day-one acceptance:** B loads A's stub, gets a faithful response, sees an invalid plan rejected and reproduces a response on retry; A uses B's fixtures without hidden state; B replays a seeded session with correct-duration records. First integration review (21.3): skip, novice misconception, answer shown, immediate stop, duplicated request and generation failure, side by side; fix contract disagreements before any training.

## Workspace setup (2026-11-02)

- Xueming asked to initialize the repository as a project that must execute fast (Personal Workspace capture `cap_4d37b006-5c9d-4541-b5dc-f82fceec5f89`, linked here as `cap_09da8dc3-bd37-47c4-bfc6-4168a3469f66`; setup receipt `cap_ce17386a-d4be-47bb-a271-115e8c20aede`).
- Project `tuturing` is registered in the Workspace with this shared repository (`PeytonLi/Tuturing`, Xueming has write access); checkout at `/Users/xuemingxu/workspace/projects/tuturing`. `journal/` stays local; see decisions.md.
- Lark topic group **Tuturing** (`oc_454768961d023702c1b8895f7238be31`): Felix (Codex) and Felix-Claude are members. Xueming requested Felix-Claude as admin (capture `cap_110977ab-8071-4187-9e12-6a71a0af9e4d`); its group-admin role was granted and verified through Lark, and its Workspace project binding was added on 2026-11-02. See decisions.md for setup details.
- `tuturing` is in the Workspace execution focus, so authorized tasks run unattended.

## Our side: Person A (2026-11-02)

- Xueming: "we will be person A and my teammate is pushing setup" (capture `cap_f7092820-6b99-45d1-9140-729c3649ca5d`). Workspace agents build only Person A's paths; Person B's side is Peyton's.
- Peyton's agent pushed the setup as two branches based on `ea8c388`, then merged both into `main` through PR #1 (`2e7c6a3`, 2026-11-02 20:27 UTC) together with later additions: behavior cloning and PPO code, decision scoring, and the task guides `docs/person_a_tasks.md` and `docs/person_b_tasks.md`. The branches contain:
  - `devin/1793649093-scaffold-a-b-workstreams`: Python 3.12 with uv and `uv.lock`, `OWNERS.md`, both contracts with valid and invalid fixtures, action and feature specs, B's content, belief, rules policies, simulator and episode collector, and A's CPU response stub with the independent-reveal rejection.
  - `devin/1793649184-person-a-response-lane`: A's prompt compiler with snapshots for all eight moves, fallbacks, validators, content hashing, evidence extractor, MathDial SFT preparation with a 200-row mapping audit, SFT training and evaluation utilities, and decision records D09 (MathDial move mapping, proposed) and D10 (base model, see open questions).
- CPU smoke on `main` after the merge, with the workspace record commits rebased on top: `uv sync --extra learning`, `pytest` (69 passed), `ruff check`, both `validate_contracts` commands, `validate_content` (15 concepts, 35 families, 30 scenarios) and a 10-episode `simulate` all passed on 2026-11-02.
- `docs/person_a_tasks.md`, written by Peyton's agent, is the working checklist for A and assumes D10. `docs/person_b_tasks.md` says B's current plan learns only the Strategist while the Navigator stays rule-based.
- Still missing for A's day-one list (PRD 21.1) and gates G0/G4: `contracts/public_v1.openapi.yaml`; the public API in `apps/public_api` (durable request keys, response staging, a retry returning the same response ID and hash); `apps/response_service` and `tools.response.serve` (`configs/response/serve_stub.yaml` is a placeholder); 30 original response examples with a source and split manifest (only eight synthetic smoke cases exist); the published stub artifact with loading instructions for B; the CPU ten-session smoke workflow.
- B's mock observe/commit/status/conflict service is not on `main` yet, and `apps/learning_service` is empty. A's retry demonstration and the ten-session smoke depend on it.
- Contract placement differs from PRD 2.2 and 21.1: `Error` and `PublicState` are defined in B's `learning_v1.schema.json`, not in A's contracts. `response_v1.schema.json` embeds identical generated copies of `TeachingPlan`, `TeachingAction` and `NavigationAction` from the shared generator `tools/dev/generate_contract_assets.py`. Both points belong to the contract freeze (G0).

## In flight (2026-11-02)

Xueming authorized running every Person A item in parallel and asked for datasets plus a benchmark of frontier and open-source models (capture `cap_5db17ad4-fe00-4f06-b3b7-1b975ea4fe64`). Five agents run from the Tuturing topic, each on its own branch in its own worktree. The coordinating session merges them into `main` after the CPU smoke passes.

- `a/public-api-v1` (merged to `main` on 2026-11-02, commit `428bda3`): `contracts/public_v1.openapi.yaml` with 42 public fixtures, `docs/decisions/D11_public_contract_placement.md` (proposed; Peyton reviews), `apps/public_api` with the durable request cache, staging and recovery, and `tools/response/smoke_sessions.py`, the PRD 11.2 CPU smoke. Its test runs ten scripted sessions against Peyton's real learning service in-process: 10 of 10 completed, 10 of 10 retries byte-identical, 9 of 9 checkable hashes equal to B's. That is the G4 evidence for A's side. Four findings for Peyton are recorded in D11: plan status returns no state version, a stop before the first committed response cannot be expressed, a skip to a disabled concept yields 422, and the handoff defines `content_hash` over text only while PRD 9.4 includes the action.
- `a/response-service-v1` (merged to `main` on 2026-11-02, commit `afdd0e3`): the generation pipeline with one repair and a verified fallback, an OpenAI-compatible responder, `apps/response_service`, `tools.response.serve`, and the stub artifact manifest and runbook for B. Follow-up merged on 2026-11-02 (`bd32b5b`): `validator_v2` adds a digit-independent item-fidelity check inside `action_faithful`, so a reply that swaps the issued problem for another scenario always falls back; it catches all four of Qwen's substitutions and flags none of the available gpt-6.1-sol or fallback texts.
- `a/datasets-v1` (merged to `main` on 2026-11-02, commit `e26381a`): the MathDial leak review and prepare_sft fix, frozen family-level splits, model-drafted audit labels, and 371 original probability SFT examples (train 261, validation 55, test 55; all 8 moves, 5 concepts, 3 paces, 52 adversarial records) with a rejection report, manifest, `evaluation/response/probability_cases_v1.jsonl` (55 test cases) and a 40-example review packet. All synthetic, pending Xueming's review. After the item-fidelity rule was narrowed (merge `9457fa4`), all 371 build under `validator_v2`; the manifest records it.
- MathDial mapping (in progress): Xueming skipped the human audit on 2026-11-02; a model labels every usable MathDial teacher turn with one of the 8 moves or EXCLUDE, calibrated on the 200 drafted labels, and MathDial joins the SFT mix at 40 percent with `action_label_source: mapped`.
- `a/benchmark-v1` (merged to `main` on 2026-11-02, commit `c01e4d0`): the harness `tools/response/benchmark.py`, the deployment helper, the report at `experiments/response/benchmark_v1/report.md` and a blinded review sheet. `gpt-6.1-sol` ran on all 107 cases for $0.21; 40 of 107 passed every automated check, against 43 for the deterministic fallback, but the diagnostics show the math and disclosure checks over-flag numbers the learner already said. Qwen3.8-27B then ran on 2026-11-02 on a Fireworks H100 deployment on Peyton's account (created 21:32 UTC, verified deleted 21:36 UTC, at most $0.51 of GPU time; merge `374758a`): 41 of 107 passed every check at 1.2 s median, but a human read found it replaced the issued coin problem with a die on both CHECK_MASTERY fixtures and the CHALLENGE fixture, and the checks missed it because the coin stem has no digits. gpt-6.1-sol kept the issued problem in every fixture case. Benchmark spend so far: $0.72 of the $25 cap. Fireworks also reports the model as not serverless-trainable, which contradicts D10.
- `a/extractor-v1` (merged to `main` on 2026-11-02, commit `e1e0183`): `extractor_rules_v2` in `packages/evidence_extractor`, the eval set `evaluation/response/extraction_cases_v1.jsonl` (376 synthetic cases, 190 frozen test, labels pending Xueming's review) and `experiments/response/extraction_report_v1.md` (on the test split, explicit requests 154 to 190 of 190, answers 156 to 190 of 190, misconception recall 0 to 16 of 17; these are agreement rates with model-authored labels, not human-measured). The extractor no longer emits a grade; B's grader decides, as PRD 2.1 says.

Keys live in `~/.config/tuturing/llm.env` (mode 600), outside Git. Both keys answered free model-list calls with HTTP 200.

Peyton merged the internal learning service to `main` on 2026-11-02 (`apps/learning_service`, `tools/learning/serve.py`, handoff in `docs/learning_service.md`), which unblocks the public API integration and the ten-session smoke (G4). Peyton's agent also appends its own decision entries to this repository's `decisions.md`; the PRD places owner decisions in `docs/decisions/`.

Xueming decided that Person A owns SFT (capture `cap_7c410346-63da-4a59-9c26-bfb410856672`): the SFT configs, trainer, SFT data and D09/D10 are maintained by Xueming's agents from now on. Xueming asked to be sent review items as each workstream finishes.

## Open questions

- D10 on `main` switches the SFT base model from the PRD's `Qwen/Qwen2.5-7B-Instruct` to `Qwen/Qwen3.8-27B` on Fireworks and is marked "accepted by lead". PRD 2.3 makes SFT selection a Person A decision, so Xueming should confirm it. Fireworks lists the model as LoRA-tunable but not serverless, so benchmarking and serving it need a paid dedicated deployment. Xueming approved a deployment so the benchmark can include it; the results are meant to inform this decision.
- The $25 benchmark spend cap is an assistant default that Xueming has not confirmed; $0.72 has been spent.
- `validator_v2`'s thresholds were calibrated on about 50 texts about one coin problem; extending the scenario rule to WORKED_EXAMPLE is an open follow-up. The benchmark was re-scored under v2 and extended with the 55 probability test cases on 2026-11-02 (merge `ab53f8e`): delivered as model on the probability set, fallback 55 of 55 (it renders approved content by construction), gpt-6.1-sol 33 of 55, Qwen3.8-27B 19 of 55; item-fidelity failures across all texts, gpt-6.1-sol 5 of 111, Qwen 11 of 111. Benchmark spend to date $1.29 of the $25 cap; both Fireworks deployments verified deleted.
- Peyton's agent appends its decisions to this repository's `decisions.md` (a merge marker had to be removed upstream on 2026-11-02); both sides appending to one file will keep conflicting. Proposal: Peyton's entries move to `docs/decisions/` per PRD 11.1.
- Anthropic and Google keys are absent, so Claude and Gemini are not in the benchmark yet.
- Hackathon deadline and demo format are not recorded anywhere in the repository.
- D11 and the four public-API findings need Peyton's review; the public session ID is now B's allocated ID rather than A's (PRD 8.1), which Xueming should accept or reject.

## Next (assistant proposal, not a decision)

1. Merge the five branches into `main` as each passes the CPU smoke, add the new A-owned `data/response/` to `OWNERS.md`, and rerun the combined smoke.
2. Xueming reviews the dataset and benchmark review packets and fills `label_person_a` in the mapping audit. Peyton fills `label_person_b` and reviews D11.
3. Decide D10 with the benchmark results, then run the SFT smoke.
4. Hold the first integration review (PRD 21.3) with Peyton before either side trains anything.
