# Tuturing

Implementation plan for an adaptive AI tutor with two hierarchical reinforcement learning policies and a separately supervised response model.

## Documentation

Read the [complete product requirements and system design](docs/adaptive_ai_tutor_prd.md). It contains the architecture, interface schemas, repository layout, training protocols, simulator assumptions, evaluation splits, milestones, acceptance criteria, and immediate implementation tasks.

## Ownership

| Owner | Responsibilities |
|---|---|
| Person A | SFT, response generation, model serving, evidence extraction, and public API response functionality |
| Person B | Curriculum Navigator, Teaching Strategist, simulator, student state, knowledge graph, grading, rewards, and RL evaluation |

Start with the contract and fixture freeze described in the PRD, then build each workstream against the other owner's mocks.

## Status

This repository currently contains design documentation. The training budgets, metrics, and milestones are proposed targets; no project models have been trained and no learning outcomes have been measured. Dataset and method references are linked in the PRD.
