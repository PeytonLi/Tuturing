# Tuturing

Implementation plan and prototype foundation for an adaptive AI tutor with two hierarchical reinforcement learning policies and a separately supervised response model.

## Documentation

Read the [complete product requirements and system design](docs/adaptive_ai_tutor_prd.md). It contains the architecture, interface schemas, repository layout, training protocols, simulator assumptions, evaluation splits, milestones, acceptance criteria, and immediate implementation tasks.

## Ownership

| Owner | Responsibilities |
|---|---|
| Person A | SFT, response generation, model serving, evidence extraction, and public API response functionality |
| Person B | Curriculum Navigator, Teaching Strategist, simulator, student state, knowledge graph, grading, rewards, and RL evaluation |

Start with the contract and fixture freeze described in the PRD, then build each workstream against the other owner's mocks.

## Local development

The repository targets Python 3.12 and uses `uv`. From the repository root:

```bash
uv sync --extra learning
uv run pytest -q
uv run ruff check .
```

Run module entrypoints from the repository root, for example:

```bash
uv run python -m tools.learning.validate_contracts
uv run python -m tools.response.validate_contracts
uv run python -m tools.learning.validate_content --config configs/learning/prototype.yaml
uv run python -m tools.learning.simulate --config configs/learning/smoke.yaml --episodes 100
```

## Status

This repository currently contains design documentation. The training budgets, metrics, and milestones are proposed targets; no project models have been trained and no learning outcomes have been measured. Dataset and method references are linked in the PRD.
