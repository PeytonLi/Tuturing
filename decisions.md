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
