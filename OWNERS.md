# Ownership map

Person A owns response generation, SFT, API orchestration, and response contracts. Person B owns learning, simulator, policies, and learning contracts. Shared project-level files are coordinated jointly; Person A maintains the root tooling and Person B maintains its learning-specific configuration.

| Top-level path | Owner |
|---|---|
| `README.md` | Shared (A coordinates) |
| `OWNERS.md` | Shared |
| `.gitignore` | Shared |
| `pyproject.toml`, `uv.lock` | Shared (A: base/dev; B: learning extra) |
| `apps/public_api/`, `apps/response_service/`, `apps/demo/`, `packages/response/`, `packages/evidence_extractor/`, `tools/response/`, `configs/response/`, `data/manifests/response/`, `experiments/response/`, `evaluation/response/`, `tests/contracts/response/`, `tests/integration/response/` | Person A |
| `apps/learning_service/`, `packages/learning/`, `packages/simulator/`, `packages/policies/`, `tools/learning/`, `configs/learning/`, `data/original_content/`, `data/scenarios/`, `data/manifests/learning/`, `experiments/learning/`, `evaluation/learning/`, `tests/contracts/learning/`, `tests/integration/learning/` | Person B |
| `contracts/learning_v1.schema.json`, `contracts/feature_spec_v1.json`, `contracts/action_spec_v1.json`, `contracts/fixtures/learning/` | Person B |
| `contracts/response_v1.schema.json`, `contracts/fixtures/response/` | Person A |
| `contracts/` shared files | Shared |
| `tests/` shared tests | Shared, following the tested component owner |
| `docs/` | Shared; each owner maintains their own decision records and runbooks |
| `data/raw/`, `data/private_raw/`, `data/sft/`, `artifacts/` | Respectively source owner; generated/private content is not committed |
| `experiments/` | Each owner owns their own reports; run directories are ignored |

Review follows this map. No CODEOWNERS file is defined until Person A's GitHub handle is known.
