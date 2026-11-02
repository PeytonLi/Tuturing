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
- Lark topic group **Tuturing** (`oc_454768961d023702c1b8895f7238be31`): Felix (Codex) is a member and bound to this directory; Felix-Claude must be added by the owner in group settings → Bots, then `node scripts/add-bot.mjs cli_aa52cde0b578c08b claude-code` in the platform repository mirrors its binding.
- `tuturing` is in the Workspace execution focus, so authorized tasks run unattended.

## Open questions

- Which PRD owner is Xueming (Person A or Person B)? This decides which workstream the workspace agents execute; the ownership rule forbids building the other side.
- Hackathon deadline and demo format are not recorded anywhere in the repository.

## Proposed next steps (assistant proposal, not a decision)

1. Once the owner side is confirmed, run the matching day-one list (21.1 or 21.2) as one bounded workspace task whose completion condition is that side's day-one acceptance plus a passing `validate_contracts` run; include the root skeleton (`OWNERS.md`, `pyproject.toml`, lock, `docs/decisions/`) if Xueming is A, since A owns root tooling.
2. Follow with the M1 week-1 task for that side (A: prompt compiler, validators, fallbacks, evidence extractor, CPU ten-session demo against B's mock; B: rules baseline end to end with BKT and the G1/G2/G4 checks against A's stub or a local stand-in).
3. Hold the first integration review (21.3) with Peyton before either side trains anything.
