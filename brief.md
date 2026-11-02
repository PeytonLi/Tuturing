# Tuturing — current understanding

## What this is

A hackathon project by Peyton Li and Xueming Xu: an adaptive AI tutor with two hierarchical reinforcement-learning policies (Curriculum Navigator, Teaching Strategist) and a separately supervised (SFT) response model. The product requirements and system design are Peyton Li's PRD in [docs/adaptive_ai_tutor_prd.md](docs/adaptive_ai_tutor_prd.md) (commit `ea8c388`, 2026-11-02). The PRD assigns strict ownership: Person A owns SFT, response generation, model serving, evidence extraction and the public API; Person B owns the Curriculum Navigator, Teaching Strategist, simulator, student state, knowledge graph, grading, rewards and RL evaluation. The README says to start with the contract and fixture freeze, then build each workstream against the other owner's mocks.

## Workspace setup (2026-11-02)

- Xueming asked to initialize the repository as a project that must execute fast (Personal Workspace capture `cap_4d37b006-5c9d-4541-b5dc-f82fceec5f89`, linked here as `cap_09da8dc3-bd37-47c4-bfc6-4168a3469f66`).
- Project `tuturing` is registered in the Workspace with this shared repository (`PeytonLi/Tuturing`, Xueming has write access); checkout at `/Users/xuemingxu/workspace/projects/tuturing`. `journal/` stays local; see decisions.md.
- Lark topic group **Tuturing** (`oc_454768961d023702c1b8895f7238be31`): Felix (Codex) is a member and bound to this directory; Felix-Claude must be added by the owner in group settings → Bots, then `node scripts/add-bot.mjs cli_aa52cde0b578c08b claude-code` in the platform repository mirrors its binding.
- `tuturing` is in the Workspace execution focus, so authorized tasks run unattended.

## Open questions

- Which PRD owner is Xueming (Person A or Person B)? This decides which workstream the workspace agents execute.
- Hackathon deadline and demo format are not recorded in the repository.

## Next steps

To be filled from the PRD's immediate tasks (section 21) once the owner split is confirmed; nothing has been started.
