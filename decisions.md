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

