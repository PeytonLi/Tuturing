# Adaptive AI Tutor Product Requirements and System Design

**Version:** 1.0  
**Date:** November 2, 2026  
**Audience:** Person A, Person B, and a new implementation agent joining either workstream  
**Delivery:** This Markdown document is the editable source of truth and the sole deliverable.  
**Implementation status:** Proposed design. No models have been trained, no experiments have been run, and no learning gains, accuracy, latency, cost, or safety rates have been measured for this system.

Build a probability tutor with two discrete reinforcement learning policies and a separately supervised language model. Person B's Curriculum Navigator chooses the concept to study at decision boundaries. Person B's Teaching Strategist chooses an instructional action each teaching turn. Person A's response model expresses that action in accurate, constrained language. The response model cannot change the curriculum decision, update mastery, award rewards, or terminate an episode.

The first release is a research prototype with a deterministic baseline, an explicitly parameterized student simulator, and a small probability curriculum. Progress to a larger research system only after the simulator, state estimator, contracts, and baseline pass their gates. A result inside a simulator supports a claim about that simulator. Demonstrating human learning requires a separate prospective study.

All numbers labeled **target**, **budget**, **default**, **assumption**, or **illustrative** are engineering proposals, not measured outcomes or scientifically established minimums. Published dataset facts are attributed in Section 25. A prior conversation informs the product direction; its factual claims are rechecked where used here.

## Contents

1. [Product scope and requirements](#1-product-scope-and-requirements)
2. [Strict ownership and decision rights](#2-strict-ownership-and-decision-rights)
3. [Learning system concepts](#3-learning-system-concepts)
4. [Architecture and service boundaries](#4-architecture-and-service-boundaries)
5. [Curriculum and educational content](#5-curriculum-and-educational-content)
6. [Student state and evidence](#6-student-state-and-evidence)
7. [Policy actions and scheduling](#7-policy-actions-and-scheduling)
8. [Runtime flow and transactional behavior](#8-runtime-flow-and-transactional-behavior)
9. [Normative interface schemas](#9-normative-interface-schemas)
10. [Worked integration example](#10-worked-integration-example)
11. [Repository layout and local workflow](#11-repository-layout-and-local-workflow)
12. [Data inventory and provenance](#12-data-inventory-and-provenance)
13. [SFT data and training protocol](#13-sft-data-and-training-protocol)
14. [Response serving and validation](#14-response-serving-and-validation)
15. [Simulator design and assumptions](#15-simulator-design-and-assumptions)
16. [Rewards and anti reward hacking controls](#16-rewards-and-anti-reward-hacking-controls)
17. [Hierarchical RL training protocol](#17-hierarchical-rl-training-protocol)
18. [Evaluation and data splits](#18-evaluation-and-data-splits)
19. [Acceptance criteria and release gates](#19-acceptance-criteria-and-release-gates)
20. [Independent workstreams and milestones](#20-independent-workstreams-and-milestones)
21. [Immediate tasks for each owner](#21-immediate-tasks-for-each-owner)
22. [Reproducibility and experiment records](#22-reproducibility-and-experiment-records)
23. [Operations privacy and security](#23-operations-privacy-and-security)
24. [Risks and decision log](#24-risks-and-decision-log)
25. [Sources and factual reference notes](#25-sources-and-factual-reference-notes)
26. [Implementation checklists](#26-implementation-checklists)

## 1 Product scope and requirements

### 1.1 Intended experience

A learner can start at a goal such as Bayes' theorem, ask to move faster, request a hint, challenge an explanation, ask to skip a topic, or return to a prerequisite. The tutor adapts the learning path and teaching move independently. A request for shorter explanations changes pacing; it is not evidence that the learner knows more. A correct answer after receiving the solution is assisted performance; it is not an independent demonstration of mastery.

The initial curriculum is probability because the preceding design used it as the worked domain. The prototype supports English text, typed numerical or short reasoning answers, original parameterized problems, and one learner per session. The model must acknowledge uncertainty and use a verified problem or fallback when it cannot safely explain an item.

**Primary research question:** With the response generator held fixed, does hierarchical action selection improve independent post assessment performance over a strong rules baseline under equal teaching budgets across multiple held out simulator configurations?

**Secondary questions:** Does navigation add value beyond the strategist alone? Does the strategist add value beyond navigation alone? Does the complete text system preserve gains observed with structured responses? Which simulated learner conditions cause harm or failure?

### 1.2 Required behavior

| ID | Requirement | Accountable owner |
|---|---|---|
| FR01 | Select and revisit concepts using a validated prerequisite graph | B |
| FR02 | Select a teaching move every instructional turn | B |
| FR03 | Distinguish faster pacing, higher difficulty, and topic skipping | B |
| FR04 | Estimate knowledge from observable evidence with uncertainty | B |
| FR05 | Generate correct responses that implement the issued action | A |
| FR06 | Prevent unauthorized answer disclosure on independent assessments | A |
| FR07 | Support hints, worked examples, practice, diagnosis, and checks | B selects; A realizes |
| FR08 | Honor explicit stop requests without another teaching decision | A transports; B closes state |
| FR09 | Resume the original goal after temporary prerequisite remediation | B |
| FR10 | Return stable, idempotent public API responses | A |
| FR11 | Record replayable state transitions and separate policy trajectories | B |
| FR12 | Report response quality separately from simulated learning outcomes | A response results; B learning results |
| FR13 | Operate with mocks before the other owner's service exists | Each owner for their own service |
| FR14 | Reject stale plans, incompatible artifacts, and malformed payloads | A at response boundary; B at learning boundary |

### 1.3 Scope boundaries

In scope: the two policies, structured student simulation, student belief state, curriculum graph, problem bank, SFT pipeline, response service, minimal session API, experiment harness, and an internal demonstration client. A owns a simple transcript viewer if needed to inspect responses. A production frontend is not a milestone dependency.

Out of scope for version 1: token level RL of the response model; an unconstrained language model deciding curriculum; a learned third policy for termination; multi subject coverage; audio and vision; automatically inferred prerequisite edges; real time policy learning from production students; high stakes grading; school identity integrations; and claims that archetypes diagnose real students.

The external datasets are optional inputs with task specific uses. The prototype must run entirely on original curriculum content and synthetic scenarios without gated data or paid model APIs.

### 1.4 Proposed operating budgets

Prototype defaults are five concepts, eight misconception types, three simulator profiles, 15 teaching templates plus a separate assessment bank, 30 starting scenarios, and 500 to 1,000 training episodes per experiment. Research scale targets are 15 concepts, 25 misconception types, 10 profiles, 120 teaching template families plus 45 assessment families, 400 scenarios, and 10,000 training episodes per final training seed. Section 18 reserves held out families before any rollout is generated. These counts distinguish teaching coverage from the additional independent assessment content needed to test it.

Target teaching length is 8 to 12 turns, with a hard finite horizon of 16. A teaching turn is one issued instructional plan, one accepted tutor response, and the learner's next answer or message. A chat message is not necessarily a teaching turn. Pre assessment and post assessment are separate fixed evaluations and are excluded from this teaching budget. All systems being compared receive the same evaluation overhead.

The initial research run uses five independent training seeds. At 10,000 episodes per seed, this is 50,000 training episodes for one configuration, before tuning, baselines, ablations, or evaluation. Do not describe 10,000 episodes as the total project compute budget.

## 2 Strict ownership and decision rights

### 2.1 Ownership rule

Each executable behavior and dataset has one accountable owner. Cross review is encouraged; implementation ownership stays as assigned below. In particular, Person A never takes ownership of simulator calibration, student state, knowledge graph, either policy, or RL evaluation. Person B never takes ownership of SFT, response generation, inference infrastructure, or the public response API.

| Component or artifact | Owner A | Owner B |
|---|---|---|
| SFT source adapters and tutoring response labels | Owns | Supplies action definitions and content identifiers |
| Response model, prompts, tokenizer, adapter, serving | Owns | Consumes a versioned response service |
| Response correctness and action fidelity checks | Owns | Supplies gold mathematical item specifications |
| Answer disclosure detector and response fallback | Owns | Defines assessment and reveal permissions |
| Public session and turn HTTP API, auth, transport, retry cache | Owns | Supplies authoritative learning operations |
| Parsing a message into observable evidence | Owns extractor and its quality tests | Owns evidence semantics and acceptance rules |
| Deterministic numeric and symbolic grading | Calls grader | Owns grader and test bank |
| Learner belief state, mastery update, uncertainty | Reads approved projection only | Owns |
| Raw event storage for public request metadata | Owns | Consumes redacted correlation metadata |
| Authoritative learning event log and snapshots | Reads permitted projection | Owns |
| Curriculum graph, concepts, misconceptions, problem bank | Consumes approved content | Owns |
| Simulator, latent student dynamics, response behavior | Supplies response realization service | Owns |
| Simulator utterance generation if used | No simulator code ownership | Owns with a separate frozen model or templates |
| Navigator, strategist, masks, scheduler, reward, trainer | No policy ownership | Owns |
| RL datasets, scenarios, rollout logging and evaluation | Supplies response quality metrics | Owns |
| Response benchmark evaluation and human response review | Owns | Provides educational cross review |
| End to end outcome evaluation | Runs response service and traces | Owns harness, metrics, statistical report |
| Deployment of public API and model server | Owns | Publishes compatible learning artifacts |
| Deployment of learning service and CPU policy runtime | Calls service | Owns |

### 2.2 Contract ownership without shared implementation ambiguity

B owns `contracts/learning_v1.schema.json`: evidence, student belief, graph, actions, plans, outcomes, and trajectories. A owns `contracts/response_v1.schema.json` and `contracts/public_v1.openapi.yaml`: generation requests, responses, public endpoints, and errors. A owns the root release bundle manifest format; B owns the learning artifact compatibility declaration inside it. Both must review changes affecting the other's consumer. The schema file owner makes the change and ships fixtures; review does not transfer ownership.

Additive optional fields require a minor contract version and an explicit extension container. Removing a field, changing an enum, changing action meaning, changing feature order, or altering reward semantics requires a major schema or artifact revision. Strict version 1 objects reject unknown top level fields. A producer cannot silently add fields and assume old consumers will accept them.

### 2.3 Decision process

A decides response architecture, inference implementation, SFT selection, and response quality remediation. B decides curriculum semantics, state estimation, simulator dynamics, policy algorithms, reward changes, and outcome evaluation. Cross boundary disputes are recorded in `docs/decisions/` with the proposed behavior, fixtures, and impact on old artifacts. Keep the last compatible contract while a change is unresolved.

For release, A signs the response report and B signs the learning report. A assembles the deployment bundle only after both component reports pass. Neither owner may substitute the other's metric with their own. A high quality hint is not evidence of learning; a high simulator reward is not evidence that the rendered hint was correct.

## 3 Learning system concepts

### 3.1 The three trained components

**Supervised fine tuning or SFT:** Train the language model on examples of a context, an issued teaching plan, and a good response. It learns how to express a hint, question, or explanation. It does not learn the curriculum reward function from these examples.

**Curriculum Navigator:** A small policy that chooses a concept level option at boundaries. An option persists for one or more teaching turns. Its transition contains the rewards accumulated until the next boundary and the actual number of turns elapsed.

**Teaching Strategist:** A small policy that selects a teaching move each instructional turn, conditioned on the current option and observable student belief. It is not the language generator. Reinforcement learning updates the action selection parameters, not the response model weights.

### 3.2 RL vocabulary used here

A policy maps an observation to a probability distribution over legal actions. The environment applies the selected action, changes its hidden student state, emits an observable answer, and computes reward. A trajectory records these interactions. A value function predicts future discounted return and helps estimate whether an action performed better or worse than expected. PPO updates a policy from freshly collected trajectories while limiting the size of each update through a clipped objective [S7].

The student's actual understanding is hidden. Policies receive a belief estimate computed from history, not a perfect mastery vector. This is a partially observed control problem. The finite belief representation is an engineering approximation and may fail to summarize everything needed for optimal decisions.

The hierarchy follows temporal abstraction: a curriculum option can span multiple teaching actions [S6]. There are exactly two learned action policies in version 1. Boundary detection and termination rules are deterministic, versioned code. Each policy has its own value head; value heads are training estimators, not additional action policies.

### 3.3 Why decision counts differ

If an episode contains ten teaching turns and navigator boundaries at turns 0, 3, 6, and 8, it contributes ten strategist transitions and four navigator transitions. The navigator durations are 3, 3, 2, and 2. Both are valid views of the same interaction. Do not invent repeated navigator actions on intervening turns, and do not discard strategist turns to equalize counts.

Illustratively, 10,000 episodes averaging ten turns and four boundaries produce 100,000 strategist transitions and 40,000 navigator transitions. These are planning arithmetic, not observed counts. Buffers are batched by valid transition counts separately for each policy. Section 17 gives duration aware return and advantage equations.

## 4 Architecture and service boundaries

### 4.1 Runtime architecture

```mermaid
flowchart TD
    L[Learner or demo client] --> A[Person A public API and orchestrator]
    A --> E[Person A evidence extractor]
    E --> B[Person B learning service]
    B --> S[Person B belief state and event log]
    B --> G[Person B graph problem bank and grader]
    B --> N[Person B Curriculum Navigator]
    N --> T[Person B Teaching Strategist]
    T --> P[Versioned TeachingPlan]
    P --> R[Person A response generator]
    R --> V[Person A validation and fallback]
    V --> A
    A --> L
    A --> C[Person B plan commit and recovery]
```

The orchestrator sequences calls and handles transport failures. It does not implement learning decisions. The learning service owns all writes that affect a learner's knowledge estimate, option, teaching history, problem assignment, or episode end. A's extractor produces evidence proposals from raw text; B verifies deterministic grades and uses confidence thresholds before changing belief.

The original problem bank supplies verified stems, solutions, misconception mappings, hint ladders, and allowed response facts. The generator may change wording but cannot invent new assessment problems, change numeric values, or grade itself. Dynamic original problem creation is deferred until separately validated.

### 4.2 Training architecture

```mermaid
flowchart LR
    DA[Permitted tutoring examples] --> SA[Person A SFT and response evaluation]
    SA --> FA[Frozen response artifact]
    DB[Original scenarios and permitted student logs] --> SB[Person B simulator and calibration]
    SB --> RB[Person B rollout collection]
    FA --> RB
    RB --> PB[Person B PPO training]
    PB --> EB[Person B frozen evaluation harness]
    EB --> BA[Compatible release bundle]
    FA --> BA
```

Fast RL training initially uses structured response realizations, avoiding a language model call for every turn. A publishes a fixed structured response stub implementing the same contract as the model service. B owns the simulator's interpretation of the rendered content. Before release, rerun integration evaluation with the frozen SFT service and measured response checks. Results from the stub and the text system must be reported separately.

### 4.3 Deployment and resource assumptions

Default deployment is a public API process, a learning service with CPU policies, a single response inference worker, a relational event store, and a local or object store artifact registry. For development, all except inference can run on one machine. Use a single relational database with separate A and B tables and credentials initially; a graph database and distributed queue are unnecessary for 15 concepts.

A candidate response base model is `Qwen/Qwen2.5-7B-Instruct`, whose publisher card lists Apache 2.0 [S10]. This is a concrete starting candidate, not a claim it is best for tutoring. A pins a model revision and validates its math and pedagogical quality before choosing it. Planning assumption: one rented GPU with 48 GB memory for LoRA SFT and inference experiments, and an eight core CPU with 32 GB RAM for structured simulation. Actual fit, throughput, and cost must be benchmarked. A smaller approved model is a documented fallback if the resource gate fails.

Initial response implementation can use the pinned model directly through a Python inference worker. Keep the HTTP response contract independent of inference engine choice. The policies are small categorical neural networks implemented by B using a duration aware PPO trainer; no language model RL framework is required for the policies.

## 5 Curriculum and educational content

### 5.1 Concept catalogue

Concept IDs and indices below are fixed for version 1 feature and action encoding. Prototype support includes C01, C02, C03, C06, and C08 with a reduced graph; expand only after the prototype gate. Unavailable nodes remain masked, not renumbered.

| Index | ID | Concept | Direct prerequisites |
|---|---|---|---|
| 0 | C01 | Outcomes events and sample spaces | None |
| 1 | C02 | Probability bounds and equally likely outcomes | C01 |
| 2 | C03 | Complements | C02 |
| 3 | C04 | Unions intersections and set notation | C01 |
| 4 | C05 | Addition rule | C02 C04 |
| 5 | C06 | Conditional probability | C02 C04 |
| 6 | C07 | Independence | C06 |
| 7 | C08 | Multiplication rule and probability trees | C06 |
| 8 | C09 | Total probability | C05 C08 |
| 9 | C10 | Bayes theorem | C06 C09 |
| 10 | C11 | Counting with order | C01 |
| 11 | C12 | Counting without order | C11 |
| 12 | C13 | Bernoulli trials | C02 C03 C07 |
| 13 | C14 | Binomial probability | C12 C13 |
| 14 | C15 | Expected value of a discrete variable | C02 C05 |

These edges are proposed pedagogical dependencies, not a validated causal graph. B and an educational reviewer must check them. Prototype graph uses explicit simplified dependencies C01 to C02, C02 to C03, C02 to C06, and C06 to C08; omitted C04 is introduced within C06 content. This reduced graph has a separate graph version and is never confused with the full prerequisite graph.

### 5.2 Graph records and invariants

Each concept record includes ID, stable index, name, measurable objective, prerequisites, diagnostic template IDs, practice template IDs, allowed misconception IDs, and content provenance. A prerequisite edge includes source concept, dependent concept, rationale, reviewer, review date, and confidence category `proposed` or `reviewed`. Misconception association is a different edge type and never converted automatically into a prerequisite.

B validates acyclicity, uniqueness of IDs and indices, existence of every reference, at least one reachable diagnostic and practice family for every enabled concept, and complete answer keys. Remediation uses ancestors of the current concept; return uses a goal stack. Traversal uses sorted IDs to make tied choices reproducible. Failed graph validation prevents startup.

### 5.3 Misconceptions and templates

The first eight misconceptions are: treating unequal outcomes as equally likely; confusing an event with a single outcome; assigning probability above one; subtracting the wrong complement; reversing conditional probabilities; using the wrong conditional denominator; assuming independence without evidence; and multiplying along incompatible tree paths. Expand to 25 only after writing distinct observable signatures and diagnostic counterexamples for each.

Misconception IDs describe hypothesis types. A learner belief can assign probability to multiple misconceptions. A simulator profile does not become a learner label. Do not infer a misconception merely from a demographic description or from one ambiguous wrong number.

Every template has a `template_family_id` identifying its reasoning structure, a parameter sampler with valid ranges, a deterministic solver, answer normalization rules and numeric tolerance, concept mapping, difficulty bin, misconception distractors or error generators, hint ladder, separate worked example, disclosure policy, and split provenance. Numeric substitutions are new instances of the same family. Template families, including paraphrases, must remain within one split.

For probability calculations, reject undefined conditional denominators, impossible parameter combinations, ambiguous rounding, and results outside [0,1]. Store exact rational reference values where feasible. A's response checks compare to this verified content; model agreement is not the answer key.

## 6 Student state and evidence

### 6.1 Separate three representations

**Private simulator state:** Latent mastery, true misconception activation, learning rates, slip, guess, fatigue, and dropout parameters. B's environment can use these to generate outcomes and rewards. They never enter actor or critic observations, evidence extractor prompts, or A's response requests. A privileged critic is also prohibited in version 1 to simplify leakage auditing.

**Authoritative learner belief:** Per concept mastery estimate, uncertainty and evidence count; misconception probabilities; current and target concepts; active option; goal stack; supported learner preferences; assistance history; and independent assessment evidence. B alone updates it. It is an estimate, not a diagnosis or a measured percentage of knowledge.

**Response projection:** Only the current and goal concept names, relevant recent evidence, current problem, permitted misconception hypotheses with qualified wording, pacing preference, action constraints, and approved teaching content. A does not need the complete student vector. The public client receives an even smaller projection without hidden solutions, training rewards, or private simulator attributes.

### 6.2 Initial state estimator

Use a transparent per concept Bayesian knowledge tracing baseline with a binary latent skill approximation, documented guess `g`, slip `s`, and learning probability `l`. These parameters are configurable assumptions until fit to eligible training data. Default initialization is `p=0.35`, `g=0.20`, `s=0.10`, `l=0.08`; they are not empirically calibrated values.

For an independent correct attempt, posterior knowledge is `p*(1-s) / (p*(1-s)+(1-p)*g)`. For an independent incorrect attempt, it is `p*s / (p*s+(1-p)*(1-g))`. After an eligible teaching opportunity, apply `p_next = posterior + (1-posterior)*l`. Eligibility is a versioned estimator rule: PRACTICE, approved EXPLAIN_CONCEPT, and approved WORKED_EXAMPLE can qualify; PROBE and CHECK_MASTERY do not in the primary configuration. Item difficulty enters a later calibrated observation model; the initial BKT estimator must not pretend all items have calibrated difficulty.

Assisted answers, copied answers, and non answers do not run the independent correctness update. They may produce an explicitly smaller learning transition if a validated estimator specifies one, but the default is no positive mastery update from an assisted answer. A hint request updates assistance count and provides evidence of support need, not a direct knowledge decrement. Uncertain grading causes no correctness update.

For illustration, `p=0.35`, `g=0.20`, `s=0.10` and one unassisted correct response give a posterior of approximately 0.708 before any learning transition. This is a calculation from the assumed model, not a measured student's mastery.

Maintain `independent_correct`, `independent_incorrect`, and `assisted_attempts` separately. Define a conservative v1 uncertainty proxy as `1/sqrt(1+n_independent)` normalized to [0,1], not as a Bayesian credible interval. Production calibration may replace it; changes require a feature version. A mastery threshold of 0.80 alone cannot certify progress: require two recent independent correct attempts from different families and no contradictory recent independent evidence.

### 6.3 Evidence pipeline

A's extractor proposes an `Evidence` object from the learner message and the last committed response. B's deterministic grader independently verifies objective answers where possible. B resolves contradictory numeric grades in favor of the verified grader and logs the extractor error. Free form reasoning with confidence below 0.80 is `unknown` for mastery updates; B can request a diagnostic teaching plan.

The extractor may identify `skip`, `faster`, `slower`, `hint`, `answer`, `stop`, or `none` as an explicit request. A raw request must be distinguished from a model inference. For example, "I know this" is self report and is not `correct=true`. Misconception hypotheses require message span references and confidence. A cannot insert an authoritative mastery field into evidence.

Cold starts use the prior and an optional short diagnostic. Returning sessions replay the B event log. Reset and simulated reset are separate operations; a learner cannot reset a turn to erase mistakes while preserving learned policy rewards.

### 6.4 Policy observation vector

B owns `feature_spec_v1.json`. Encode 15 values each for mastery, uncertainty, and evidence count clipped at 10 and divided by 10; 25 misconception probabilities; 15 value one hot vectors each for current concept, goal concept, and active option target; six one hot option types; eight one hot previous teaching moves plus a none flag; and the following ten scalars: turn/16, turns in option/4 clipped at one, hint count/4 clipped at one, independent success streak/3 clipped at one, remaining budget/16, explicit skip flag, explicit faster flag, explicit slower flag, response failure flag, and stack depth/4. Total: 140 features. Assert this count and order in tests.

Correction to the arithmetic is forbidden by silent implementation choice: 45 + 25 + 45 + 6 + 9 + 10 = 140. Unavailable concepts use the same indices and an availability mask in the action builder. Navigator observations omit the active option type and target by zeroing those fields before scoring a new option; strategist observations include them. Actor and critic each use only their appropriate observable vector. No textual student ID, scenario ID, seed, post test answer, or private profile is a feature.

## 7 Policy actions and scheduling

### 7.1 Navigator action space

The navigator selects one operation and its target concept. Use a fixed categorical action space of 76 entries: five operations times 15 concept indices, followed by `END_SESSION` at index 75. The first five operation indices follow table order below, starting at zero. Encode `action_id = operation_index*15 + concept_index`. An action can be valid only when its operation and target pair is legal. `END_SESSION` has `target_concept_id=null`.

| Operation | Meaning | Valid target and constraints |
|---|---|---|
| STAY | Continue the current concept for another option | Current enabled concept only |
| ADVANCE_ONE | Move toward the goal by one graph edge | Enabled direct successor on an admissible goal path; prerequisite evidence rule applies |
| SKIP_TO_TARGET | Jump to an explicitly requested or selected session goal | Goal concept only; allowed under uncertainty as a monitored jump, not a mastery claim |
| REVIEW_PREREQUISITE | Study an unmet prerequisite | Enabled ancestor with weakest supported mastery; push original destination on stack |
| RETURN_TO_GOAL | Return after remediation | Top of goal stack; pop exactly one frame on commit |
| END_SESSION | End teaching and run fixed evaluation if simulated | Available at a boundary; reason must be voluntary stop, demonstrated completion, or policy stop |

`SKIP_TO_TARGET` triggers an independent diagnostic within its first two teaching turns. A skip request never manufactures high mastery. If the learner explicitly repeats a skip request after being offered a diagnostic, permit the monitored jump unless the concept is unavailable. Explain the transition briefly; do not block progress with an endless prerequisite loop. Keep the original goal during remediation.

A maximum stack depth of four and a two visit cooldown for the same concept pair prevent oscillation. If no progression action is legal, `STAY` remains valid on the current enabled concept. If the state is invalid or no current concept exists, use deterministic recovery or end with a configuration error; never sample an all masked distribution.

### 7.2 Strategist action space

Select a joint action over eight moves, three difficulty bins, and three pacing styles: 72 categorical entries. Encode `action_id = move_index*9 + difficulty_index*3 + pace_index`. Move order is table order; difficulty is `easy`, `medium`, `hard`; pace is `concise`, `normal`, `detailed`. A plan contains the decoded fields and a fixed problem or content reference chosen by B's deterministic content resolver. Template selection is not a third learned policy.

| Move | Educational purpose | Content and disclosure rule |
|---|---|---|
| PROBE | Diagnose reasoning or a misconception | One diagnostic item or question; no solution |
| HINT | Give the next permitted hint on an active item | Requires active item and remaining hint level; mark assistance |
| EXPLAIN_CONCEPT | Explain the relevant concept | Approved concept facts; no hidden active assessment answer |
| WORKED_EXAMPLE | Demonstrate a separate solved example | Example must differ from the active independent assessment family instance |
| PRACTICE | Offer a fresh practice item | Original enabled template at chosen difficulty |
| CHALLENGE | Test a harder application | Requires adequate independent evidence; harder difficulty only |
| CHECK_MASTERY | Gather independent evidence | Fresh family relative to recent checks; no hints or solution in the same response |
| REFLECT | Ask the learner to explain a previous step | Prior step reference; cannot certify mastery by itself |

Difficulty for HINT and REFLECT is fixed to the active item's bin, reducing redundant actions through masks. CHALLENGE is masked below mastery estimate 0.65 or without one recent independent correct attempt. WORKED_EXAMPLE uses a different instance with a different answer; sharing the same numeric answer with an active assessment is rejected where it creates answer disclosure. Pace changes length and presentation, not mastery, item correctness, or difficulty.

An explicit request for an answer is a response mode handled by a deterministic B rule. Outside a protected independent assessment, B may issue `EXPLAIN_CONCEPT` with `reveal_allowed=true`, solution content, and `assistance=answer_shown`. During an independent check, offer to exit the check first; accepting exits independent mode, and all subsequent answers to that item are assisted. These externally forced actions are logged with `actor_eligible=false` so PPO does not treat them as sampled policy actions.

### 7.3 Boundary scheduler

The navigator runs at session start and after the previous turn's evidence if any of the following hold: the option lasted four teaching turns; two independent successes demonstrate readiness; two independent failures suggest a prerequisite gap; the learner requests a skip or a different goal; or an enabled content family is exhausted. Multiple triggers in the same turn produce exactly one boundary with a sorted reason list.

The strategist runs once for each teaching turn after any navigator action. A boundary can occur after one turn, so an option duration is between one and four instructional turns. `END_SESSION` is the explicit exception: it executes no strategist action and closes a navigator transition with duration zero. It is terminal and cannot loop. B tests the zero duration case separately.

An explicit stop request terminates immediately through deterministic session control, closes the active option at its observed duration, and does not call either policy again. Stop is not masked or delayed by the policy. The hard horizon of 16 is a true terminal condition of this finite horizon task. Collection fragments inside an unfinished episode are truncations for value bootstrapping; they are not learner session ends [S9].

### 7.4 Action masks and inference behavior

B constructs and versions masks from the observable belief, graph, budget, prerequisites, assistance state, and content availability. Masks and content selection cannot use private simulator mastery, true misconception flags, hidden profiles, or assessment outcomes. Apply masks before categorical normalization in rollout collection, PPO updates, and inference. Record the exact behavior mask and log probability. Use identical masks when computing old and new action probabilities for a stored transition.

For training, sample actions using the per episode action RNG. For deterministic evaluation, choose the highest probability valid action, resolving ties by action ID. Report evaluation mode. No action probability floor is allowed to assign mass to an invalid action. Masks are constraints, not a substitute for a meaningful baseline: report mask frequency and the fraction of decisions with only one valid action.

## 8 Runtime flow and transactional behavior

### 8.1 New session

1. A authenticates the caller, validates the requested goal and contract version, allocates public correlation IDs, and calls B's session creation operation.
2. B initializes belief, graph and policy versions, goal stack, teaching budget, and event sequence. It returns an approved public state projection.
3. B optionally supplies a fixed diagnostic outside the teaching budget or starts the first instructional option. Diagnostic settings are identical across experimental arms.
4. A requests generation for B's plan, validates the result, durably stages the exact response, commits delivery with B, and then returns the response to the client.

### 8.2 Learner turn

1. The client sends the last `response_id`, the expected learning `state_version`, a unique request idempotency key, and raw learner text.
2. A checks the durable request cache before extracting anything. Reusing a key with different content returns a conflict.
3. A's extractor emits evidence referring to the last committed problem and response. B's `observe-and-plan` transaction validates the response reference, verifies objective grading, applies evidence once, checks ending rules, runs the boundary scheduler and policies, reserves a plan, and returns it. Previous learner evidence is authoritative even if the next response cannot be generated.
4. A generates text from the reserved plan. Failed validation allows at most one repair attempt. A then uses a verified fallback realizing the same plan if available. If neither is valid, return a retryable service error and leave the plan pending. Do not invent a different teaching move.
5. A stages the accepted response and its hash in durable storage. B's `commit-plan` transaction checks the pending plan, hash, effective move, and assistance flags, commits the option and item assignment, appends the teaching event, and increments `state_version`. This is the sole point at which a delivered teaching action consumes a teaching turn.
6. A marks its request complete and returns the committed response and public state. When the next learner message arrives, B attributes it to this response and updates belief. Rewards in simulation are computed only after the simulator supplies the corresponding student outcome.

### 8.3 Recovery and consistency

One pending plan is allowed per session. Competing requests with the same expected version produce one reservation and a `409 STATE_CONFLICT` for the other. A request retry reuses the plan and exact staged response. Regeneration after a response is committed is prohibited.

If A crashes after B commits but before the client receives the response, recovery reads the staged response and queries B's plan status, then returns the exact committed response. If B crashes before commit, its transaction either committed fully or can be safely retried. B's learning state and event log share the transaction boundary. A's request cache is not the authoritative learning state.

A pending plan expires after a proposed ten minute idle timeout. B can cancel it without consuming a teaching turn, applying a mastery change from an undelivered intervention, or generating a policy transition. Earlier valid learner evidence is retained. A new plan is based on the current state. A response that was exposed to a learner must never be silently canceled; the client receives only committed responses in version 1.

Public response streaming is deferred. Internally streamed model tokens are buffered until validation succeeds. This prevents the API from disclosing an answer before its final validation decision.

### 8.4 Non instructional events

Transport retries, rate limits, authorization failures, invalid messages, and generation infrastructure failures are not student learning failures and do not earn educational reward. B may record operational events, but they are excluded from RL actor updates. A validated pedagogical fallback is a real delivered action; log it as `delivery_mode=fallback` and evaluate the system using the effective content. An apology without instructional content is an operational response and does not masquerade as PRACTICE or HINT.

## 9 Normative interface schemas

### 9.1 Conventions

The following language independent record definitions are normative. Implement them as JSON Schema 2020 12 and OpenAPI in the owned files before feature implementation. `?` means an optional field; otherwise every field is required, including explicit nullable values. `ID` is a nonempty ASCII string of at most 128 characters. `Probability` is a finite number in [0,1]. `Int` is a JSON integer. `Timestamp` is an RFC 3339 UTC string. Arrays have unique IDs where they identify records. JSON objects reject unknown fields except within an explicitly defined `extensions` object. Examples use fictional IDs and invented values.

All internal calls use service authentication and a `trace_id`. All schema references are versioned; reject unsupported major versions. Optional extension keys are namespaced by producer. Numeric NaN and Infinity, negative durations, unknown concepts, invalid action IDs, and inconsistent duplicate enum or index fields fail validation.

### 9.2 Observable evidence and student belief

```text
Evidence {
  schema_version: "1.0",
  event_id: ID, session_id: ID, trace_id: ID,
  source_response_id: ID, problem_instance_id: ID | null,
  observed_at: Timestamp,
  normalized_answer: string | null,
  correctness: "correct" | "incorrect" | "unknown" | "not_applicable",
  grade_source: "deterministic" | "extractor" | "none",
  confidence: Probability,
  assistance: "none" | "hint" | "example" | "answer_shown" | "unknown",
  explicit_request: "none" | "skip" | "faster" | "slower" |
                    "hint" | "answer" | "stop" | "change_goal",
  requested_concept_id: ID | null,
  misconception_hypotheses: [{
    misconception_id: ID, probability: Probability,
    evidence_spans: [{start: Int, end: Int}]
  }],
  raw_message_ref: ID,
  extractor_version: ID,
  extensions: object
}

ConceptBelief {
  concept_id: ID, mastery: Probability, uncertainty: Probability,
  independent_correct: Int, independent_incorrect: Int,
  assisted_attempts: Int, last_evidence_event_id: ID | null
}

StudentBelief {
  schema_version: "1.0", session_id: ID, state_version: Int,
  learner_pseudonym: ID,
  graph_version: ID, feature_version: "features_v1",
  current_concept_id: ID, goal_concept_id: ID,
  concepts: ConceptBelief[15],
  misconceptions: [{misconception_id: ID, probability: Probability}][25],
  pace_preference: "concise" | "normal" | "detailed",
  active_option: NavigationAction | null,
  option_id: ID | null, option_turns: Int,
  goal_stack: ID[], teaching_turns: Int,
  current_problem_instance_id: ID | null,
  hint_level: Int, recent_evidence_event_ids: ID[],
  session_status: "active" | "pending_plan" | "ended",
  end_reason: EndReason | null,
  updated_at: Timestamp
}
```

Use zero based Unicode code point offsets into the stored original message for spans. B rejects spans outside message length and does not assume that a quoted span proves the proposed misconception. Counts are nonnegative; goal stack length is at most four; teaching turns are at most 16. The catalogue contains 25 entries even when only eight are enabled; inactive entries are zero and excluded from inference claims.

### 9.3 Actions and teaching plans

```text
NavigationAction {
  action_id: Int,
  operation: "STAY" | "ADVANCE_ONE" | "SKIP_TO_TARGET" |
             "REVIEW_PREREQUISITE" | "RETURN_TO_GOAL" | "END_SESSION",
  target_concept_id: ID | null
}

TeachingAction {
  action_id: Int,
  move: "PROBE" | "HINT" | "EXPLAIN_CONCEPT" | "WORKED_EXAMPLE" |
        "PRACTICE" | "CHALLENGE" | "CHECK_MASTERY" | "REFLECT",
  difficulty: "easy" | "medium" | "hard",
  pace: "concise" | "normal" | "detailed"
}

TeachingPlan {
  schema_version: "1.0", plan_id: ID, session_id: ID, trace_id: ID,
  based_on_state_version: Int, option_id: ID,
  navigation_action: NavigationAction,
  teaching_action: TeachingAction,
  actor_eligible: boolean,
  concept_id: ID, goal_concept_id: ID,
  problem_instance_id: ID | null,
  template_family_id: ID | null,
  assessment_mode: "none" | "practice" | "independent",
  assistance: "none" | "hint" | "example" | "answer_shown",
  content_bundle_id: ID,
  content: {
    concept_name: string, goal_name: string,
    problem_stem: string | null,
    approved_facts: string[],
    permitted_hint: string | null,
    worked_example: string | null,
    reveal_solution: string | null,
    previous_step: string | null
  },
  constraints: {
    reveal_allowed: boolean,
    max_words: Int,
    max_questions: Int,
    must_request_student_attempt: boolean,
    forbidden_answer_fingerprints: string[]
  },
  response_context: {
    last_student_message: string,
    recent_dialogue: [{role: "student" | "tutor", text: string}],
    qualified_misconception_summary: string | null
  },
  versions: {
    graph: ID, content: ID, state_estimator: ID,
    navigator: ID, strategist: ID, mask: ID, feature: ID
  },
  expires_at: Timestamp
}
```

A plan's `navigation_action` is the active option even when the navigator did not act this turn. The internal trajectory separately marks whether a new navigator decision occurred. `reveal_solution` is null unless reveal permission is true. Independent assessments always set `reveal_allowed=false`, `assistance=none`, and omit the reference solution. Forbidden fingerprints are detector inputs and must never be copied into the prompt or returned to the client. A receives them in a separate validation context, outside model input.

For PRACTICE, CHALLENGE, PROBE with an item, and CHECK_MASTERY, a problem instance and family are required. HINT requires a permitted hint and an active item. WORKED_EXAMPLE requires an approved example. EXPLAIN_CONCEPT requires approved facts. REFLECT requires a previous step. The content resolver validates these discriminated constraints before issuing a plan. End of session is a separate `SessionEnd` response from B and never a teaching plan with an absent action.

Default word caps are concise 80, normal 150, detailed 250, with one question maximum. These are proposed response constraints; B may issue a lower cap. A cannot increase it. Length counting uses whitespace separated words after Markdown rendering; retain a separate tokenizer count for serving budgets.

### 9.4 Generation and delivery

```text
GenerateRequest {
  schema_version: "1.0", request_id: ID, trace_id: ID,
  plan: TeachingPlan,
  response_artifact_id: ID,
  generation_seed: Int,
  deadline_ms: Int
}

GenerateResponse {
  schema_version: "1.0", request_id: ID, plan_id: ID,
  response_id: ID, text: string,
  effective_move: TeachingAction,
  delivery_mode: "model" | "repaired" | "fallback",
  content_hash: string,
  checks: {
    schema_valid: boolean, action_faithful: boolean,
    math_supported: boolean, disclosure_allowed: boolean,
    length_valid: boolean, question_count_valid: boolean
  },
  usage: {input_tokens: Int, output_tokens: Int, elapsed_ms: Int},
  versions: {response_artifact: ID, prompt: ID, validator: ID}
}

CommitPlanRequest {
  schema_version: "1.0", session_id: ID, plan_id: ID,
  based_on_state_version: Int, response_id: ID,
  content_hash: string, effective_move: TeachingAction,
  delivery_mode: "model" | "repaired" | "fallback"
}

CommitPlanResponse {
  schema_version: "1.0", plan_id: ID,
  status: "committed", state_version: Int,
  public_state: PublicState
}

PublicState {
  session_id: ID, state_version: Int,
  current_concept_id: ID, goal_concept_id: ID,
  teaching_turns: Int, status: "active" | "ended",
  end_reason: EndReason | null
}
```

All six response checks must pass to return a successful GenerateResponse. Failed checks produce an internal error or repaired result; they never produce a successful unchecked response. `content_hash` is SHA 256 of the exact UTF 8 response text plus canonical serialized effective action. A owns canonical serialization fixtures; B checks the hash format and same hash on retries. B trusts the authenticated response service's checks while recording its version; the checks are not an adversarial cryptographic proof of pedagogical quality.

### 9.5 Endpoints and errors

| Surface and owner | Endpoint | Request and successful result |
|---|---|---|
| Public A | `POST /v1/sessions` | `{request_id, goal_concept_id, pace_preference}` to `{session_id, public_state}` |
| Public A | `POST /v1/sessions/{id}/turns` | `{request_id, expected_state_version, source_response_id, message}` to committed response or SessionEnd |
| Public A | `GET /v1/sessions/{id}` | Permitted public state and committed transcript |
| Public A | `POST /v1/sessions/{id}/stop` | `{request_id, expected_state_version}` to SessionEnd |
| Internal B | `POST /internal/v1/sessions` | Validated session specification to initial state |
| Internal B | `POST /internal/v1/observe-and-plan` | Session ID, expected version, request ID, Evidence or null on first turn to TeachingPlan or SessionEnd |
| Internal B | `POST /internal/v1/commit-plan` | CommitPlanRequest to CommitPlanResponse |
| Internal B | `GET /internal/v1/plans/{plan_id}` | Pending, committed or canceled status and hash if committed |
| Internal B | `POST /internal/v1/cancel-plan` | Session ID and pending plan ID to canceled status |
| Internal A | `POST /internal/v1/generate` | GenerateRequest to GenerateResponse |
| Internal B | `POST /internal/v1/grade` | Problem instance ID and normalized answer to verified grade or unknown |

First turn has `source_response_id=null` and no Evidence; subsequent turns require a committed source response. Public messages have a proposed maximum of 8,000 characters and requests a maximum of 64 KB. Internal generation requests have a maximum of 256 KB. All POST requests require an idempotency key equal to `request_id`. The path session ID must agree with the payload where both exist.

`SessionEnd` contains `schema_version`, `session_id`, `state_version`, `reason`, `summary_content_id`, and `public_state`. End reasons are `learner_stop`, `demonstrated_completion`, `policy_stop`, `teaching_budget`, `simulated_dropout`, or `configuration_error`. A generates a brief summary from approved content; a failure to generate the summary cannot reopen the session.

`Error` contains `schema_version`, `request_id`, `trace_id`, `code`, `message`, `retryable`, and `current_state_version` or null. Codes and HTTP statuses: `INVALID_SCHEMA` 422, `UNSUPPORTED_VERSION` 400, `UNAUTHORIZED` 401, `FORBIDDEN` 403, `STATE_CONFLICT` 409, `PLAN_EXPIRED` 409, `SESSION_ENDED` 409, `RATE_LIMITED` 429, `GENERATION_UNAVAILABLE` 503, `LEARNING_UNAVAILABLE` 503, and `DEADLINE_EXCEEDED` 504. Error messages contain no answer keys or model prompt internals.

The full public turn result is `{request_id, response, public_state}` or `{request_id, session_end}`. Schema validation enforces exactly one variant. Public response is `{response_id, plan_id, text, delivery_mode}`; internal checks, fingerprints, hidden content, and private versions do not leak through the public result.

### 9.6 Simulator and training records

```text
Scenario {
  scenario_id: ID, scenario_family_id: ID,
  split: "train" | "validation" | "test",
  goal_concept_id: ID, start_concept_id: ID,
  profile_id: ID, simulator_config_id: ID,
  private_initial_state_ref: ID,
  public_history: Evidence[],
  template_family_allowlist: ID[],
  teaching_horizon: Int,
  provenance: {author: ID, reviewed: boolean, revision: ID}
}

StrategistTransition {
  run_id: ID, episode_id: ID, transition_id: ID,
  turn_index: Int, option_id: ID,
  observation: number[140], action_mask: boolean[72],
  action_id: Int, old_log_prob: number, old_value: number,
  reward_components: RewardComponents, reward: number,
  next_observation: number[140],
  terminated: boolean, truncated: boolean,
  actor_eligible: boolean, collector_cut: boolean,
  policy_version: ID, environment_bundle_id: ID
}

NavigatorTransition {
  run_id: ID, episode_id: ID, transition_id: ID,
  option_id: ID, start_turn: Int, end_turn: Int,
  observation: number[140], action_mask: boolean[76],
  action_id: Int, old_log_prob: number, old_value: number,
  duration: Int, discounted_option_reward: number,
  next_observation: number[140],
  terminated: boolean, truncated: boolean,
  actor_eligible: boolean, boundary_reasons: string[],
  policy_version: ID, environment_bundle_id: ID
}

RewardComponents {
  terminal_gain: number, teaching_cost: number,
  disclosure_penalty: number, repetition_penalty: number,
  dropout_penalty: number, invalid_action_penalty: number, shaping: number
}
```

Training records additionally reference one episode manifest containing graph, content, simulator, estimator, response, reward, feature, mask, and both policy versions; seeds; profile; split; end reason; pre and post scores; and event log hash. Private state is stored in a separately permissioned diagnostics file keyed by transition ID. The training loader only loads whitelisted feature vectors and rewards; diagnostics are never joined into observations.

### 9.7 SFT example schema

```text
SFTExample {
  example_id: ID, dialogue_id: ID, original_problem_id: ID,
  template_family_id: ID, split: "train" | "validation" | "test",
  source_id: ID, source_revision: ID, license_ref: ID,
  synthetic: boolean, reviewer_status: "approved" | "rejected" | "pending",
  action_label_source: "human" | "mapped" | "synthetic",
  plan: TeachingPlan,
  prompt: [{role: "system" | "user" | "assistant", content: string}],
  completion: [{role: "assistant", content: string}],
  checks: {math_correct: boolean, faithful: boolean, no_early_answer: boolean}
}
```

Do not use a real production plan ID as a unique training feature. The SFT prompt compiler excludes IDs, hashes, private state, and validation fingerprints. `original_problem_id` and family IDs are for splitting and provenance only. A and B publish valid and invalid fixtures for every discriminated schema branch, with at least one retry and stale state fixture.

## 10 Worked integration example

### 10.1 Skip request with uncertain conditional probability

All values in this example are fictional. A learner's goal is C10 Bayes' theorem; current concept is C06 conditional probability. Estimated mastery of C06 is 0.45 with high uncertainty. The learner says, "I already know this. Let's skip to Bayes." A extracts `explicit_request=skip`, `requested_concept_id=C10`, `correctness=not_applicable`, and `assistance=none`. It does not extract a correct answer or raise mastery.

B reserves a `SKIP_TO_TARGET` option to C10 and a `PROBE` teaching action with concise pace, using a verified diagnostic. A returns: "Let's try Bayes. Before we use the formula, what does the condition in P(A given B) tell us about which outcomes to count?" The public response does not mention a numeric mastery estimate.

If the learner reverses the condition, B records a misconception hypothesis supported by the response and may schedule prerequisite remediation. If the learner repeats the request to move on, B allows a monitored attempt at C10 rather than forcing repeated checks. If remediation is accepted, B pushes C10, selects C06, and later returns to C10 after independent evidence improves. A only expresses those plans.

### 10.2 Numeric problem and retry

A practice item asks: "A bag contains 3 red and 7 blue counters. What is the probability of drawing red?" B's private solver stores 3/10. The learner answers "0.3" without assistance. A's extractor proposes correct; B's grader verifies the rational equivalence and applies the belief update exactly once.

B issues the next plan at version 12. A generates and stages response R13, B commits it at version 13, and A's connection drops. A retry with the same request key returns R13 and version 13. It does not apply the 0.3 answer again, consume a second teaching turn, or regenerate a different question. A new request with expected version 12 receives `STATE_CONFLICT`.

### 10.3 Hand calculation for duration aware credit

An option spans three teaching turns with rewards 0.10, 0.00, and 0.20 and per turn discount 0.99. Its stored option reward is `0.10 + 0.99*0.00 + 0.99^2*0.20 = 0.29602`. Its bootstrap discount is `0.99^3 = 0.970299`. The navigator TD target is `0.29602 + 0.970299*V(next_boundary)` unless terminal. Applying discount 0.99 once at this boundary would overvalue long options relative to short ones.

The strategist stores all three individual transitions. Option termination alone does not end the strategist return; it continues across option boundaries when the session continues. Neither policy receives a duplicated terminal bonus for the same episode event.

## 11 Repository layout and local workflow

### 11.1 Directory layout

```text
adaptive-tutor/
  README.md                         A assembles onboarding; B owns learning section
  OWNERS.md                         strict component and contract ownership
  pyproject.toml                    A owns root tooling; B supplies learning extras
  dependency-lock/                  exact versions and platform locks by owner
  contracts/
    learning_v1.schema.json         B
    response_v1.schema.json         A
    public_v1.openapi.yaml          A
    fixtures/learning/              B
    fixtures/response/              A
  apps/
    public_api/                     A orchestration auth retry cache
    response_service/               A model worker prompts validators fallbacks
    learning_service/               B state transactions scheduler policy inference
    demo/                           A minimal transcript client if needed
  packages/
    response/                       A prompt compiler and content checks
    evidence_extractor/             A observable extraction only
    learning/                       B belief model graph grader masks and rewards
    simulator/                      B hidden dynamics and student renderer
    policies/                       B actor critics PPO and rollout buffers
  data/
    manifests/response/             A sources licenses hashes and splits
    manifests/learning/             B sources licenses hashes and splits
    original_content/               B concepts misconceptions templates and hints
    sft/                            A ignored generated examples by manifest
    scenarios/                      B versioned scenario definitions
    private_raw/                    ignored access controlled never committed
  configs/
    response/                       A SFT inference and validator settings
    learning/                       B simulator rewards policies and evaluation
  experiments/
    response/                       A response reports and run manifests
    learning/                       B outcome reports and run manifests
  evaluation/
    response/                       A frozen response tests and review rubrics
    learning/                       B frozen scenarios baselines and statistics
  tests/
    contracts/response/             A producer and consumer fixtures
    contracts/learning/             B producer and consumer fixtures
    integration/response/           A public API recovery and response boundaries
    integration/learning/           B full episode and outcome regressions
  tools/
    response/                       A data train serve and evaluation entrypoints
    learning/                       B content simulate train and evaluate entrypoints
  artifacts/                        ignored immutable local model and policy bundles
  docs/decisions/                   owner authored decision records
  docs/runbooks/                    each owner documents their own service
```

Use CODEOWNERS or equivalent review rules matching OWNERS.md. Generated clients are generated from schema and never manually edited. Every service imports the versioned contract package rather than copying definitions. Tests can reside together but remain owned by the producer or consumer indicated above.

### 11.2 Required project commands

These are entrypoints to implement, not commands that already exist. Use the standard `python -m` interface so the workflow is portable across operating systems and does not require a task runner. The repository README must document environment creation from the committed lock and the appropriate Python executable.

```bash
python -m tools.response.validate_contracts
python -m tools.learning.validate_contracts
python -m tools.learning.validate_content --config configs/learning/prototype.yaml
python -m tools.response.prepare_sft --manifest data/manifests/response/train.json
python -m tools.response.train_sft --config configs/response/sft.yaml
python -m tools.response.serve --config configs/response/serve_stub.yaml
python -m tools.learning.serve --config configs/learning/rules.yaml
python -m tools.learning.simulate --config configs/learning/smoke.yaml --episodes 100
python -m tools.learning.train --config configs/learning/ppo_strategist.yaml
python -m tools.learning.train --config configs/learning/ppo_navigator.yaml
python -m tools.response.evaluate --config configs/response/eval_frozen.yaml
python -m tools.learning.evaluate --bundle artifacts/release.json --split validation
```

One CPU only smoke workflow must start both stubs, run ten scripted sessions, exercise a retry, and produce a report without any GPU, cloud credential, gated dataset, or internet dependency. The RL structured simulator must run independently of A's HTTP service by loading A's frozen stub artifact through the contract adapter. A's response evaluation must run independently of B's process using B's plan fixtures.

## 12 Data inventory and provenance

### 12.1 Source selection

The factual descriptions below are based on publisher pages or original papers checked for this design. They do not imply the files were downloaded, their contents audited, or permission obtained for this project. Each import must record its exact revision, download date, row counts, license snapshot, filtering decisions, and permitted uses.

| Source | Verified fact and intended use | Owner and limitation |
|---|---|---|
| Original probability content | Proposed 120 teaching families, 45 assessment families, 25 misconception definitions, and reviewed hint ladders | B; the core runnable dataset, not an existing public resource |
| Original response examples | Proposed plan conditioned responses in the probability domain | A; review math and action fidelity before training |
| MathDial | Publisher lists 2,861 dialogue records and CC BY 4.0; paper describes human teachers interacting with simulated students [S1] [S2] | A SFT and response behavior; B may use audited move mappings for imitation. It is not a dataset of longitudinal real student learning |
| FoundationalASSIST | Publisher describes 5,000 students, 3,395 problems, and 1,722,169 interactions; gated access and CC BY NC 4.0 [S3] | B optional student response probability and state estimator calibration; noncommercial research path only unless additional rights are obtained |
| Eedi Misconceptions Graph version 1 | Publisher lists 2,258 constructs, 8,117 misconceptions, and 19,294 misconception to construct associations under CC BY 4.0 [S4] | B taxonomy reference and associations. Contains neither student response records nor prerequisite edges |
| ASSISTments 2009 to 2010 | Official page distinguishes skill builder data and warns that different skills for one record may occur in multiple rows [S5] | B optional fallback for knowledge tracing research; pin file version, deduplicate attempts, and verify terms before use |
| Eedi NeurIPS 2020 challenge | Publisher lists two task specific datasets with different question and student counts and restrictive reuse conditions [S12] | B optional offline calibration only; not a mandatory content source or a source of unrestricted question images |
| Eedi learning paths research | Publisher describes experimental data on learning paths and constructs [S13] | B optional external validation of causal assumptions. Do not translate its treatment effects directly to this tutor's actions |
| TutorBench | Publisher describes adaptive explanations, actionable feedback and hint generation; the card lists 1,473 rows [S11] | A optional response benchmark. Verify usable modality and terms; exclude image dependent records for this text prototype |
| BEA 2025 pedagogical assessment | Official task describes mistake identification, mistake location, guidance and actionability; development set has 300 dialogues and over 2,480 tutor responses drawn from MathDial and Bridge [S14] | A response rubric validation. Audit overlap; this is not an independent navigation or learning outcome benchmark |

Do not combine the two Eedi 2020 task datasets into a single claimed population without checking overlap and source definitions. Do not compare raw dataset row counts to the number of usable SFT turns or unique RL decisions. One dialogue yields several SFT examples; one multi skill attempt may appear in several source rows; one episode yields unequal counts for the two policies.

### 12.2 Important label semantics

FoundationalASSIST's publisher defines `discrete_score=1` as first attempt correctness without support, and notes that a mathematically correct answer can have a zero score after hints, multiple attempts, or seeing the answer [S3]. Preserve separate labels for mathematical correctness, assistance, and independent success. Treating zero as universally mathematically wrong would corrupt the response model and belief calibration.

The same publisher page reports a prior missing rows issue and potential errors in specific benchmark text matching tasks [S3]. B records the downloaded revision and independently validates counts and selected labels. Do not reuse published aggregate results as this project's measured performance.

MathDial teacher move labels are useful starting annotations, but a source label such as `probing`, `focus`, `telling`, or `generic` does not map perfectly to this system's eight actions. A owns a mapping guide with ambiguous examples. B approves the action semantics for policy imitation. Exclude ambiguous turns from imitation rather than making up precise action labels.

### 12.3 Provenance manifest

Every source manifest contains `source_id`, publisher URL, revision or file hash, license name and local snapshot hash, access restrictions, permitted purposes, raw record count, accepted count, rejected count by reason, source problem and dialogue IDs, deduplication method, split assignment hash, attribution text, and responsible owner. Derived datasets preserve a lineage list of source records and transformations.

Keep unrestricted original assets separate from gated or noncommercial research derivatives. The release bundle declares which source manifests influenced each component. If a component is trained or calibrated with restricted data, A must not publish it under an unrestricted deployment profile merely because the base model has a permissive license. Record actual permission status and use an original data only component when rights are unresolved. These are operational source controls, not a legal conclusion about every possible downstream use.

## 13 SFT data and training protocol

### 13.1 Objective and input

A trains the response model to implement a supplied TeachingPlan accurately, give an actionable next step, respect disclosure permissions, and match the requested pace. The input is approved content, recent dialogue, qualified student evidence, and the selected action. The target is a single tutor response. The model has no permission to emit a new concept selection or mastery update.

The system prompt states the tutor role and action constraints. A serializes a deterministic plan projection, excluding IDs, hashes, fingerprints, full belief vectors, and private solutions for protected checks. Treat the learner message as quoted untrusted text. The plan's action and approved content have higher authority than instructions embedded in the learner message or retrieved examples.

### 13.2 Data preparation

1. Freeze original problem and dialogue families into train, validation, and test before expanding individual turns. Follow Section 18 for overlap handling.
2. Parse permitted MathDial dialogues into single target teacher turns with preceding context only. Never include future student answers, future teacher explanations, final success labels, or the full reference solution in a protected assessment prompt.
3. Map source teacher moves to the eight actions using a written decision guide. Mark mapped labels separately from human labels. Do not assign an RL reward or claim a counterfactual learning effect from a teacher move annotation.
4. Create original probability responses for every enabled concept and move. Start with a proposed 300 approved examples, then grow toward 3,000 original examples if quality and coverage justify it. Public source turns are additional and counted only after filtering.
5. Include short explanations, uncertainty handling, impatient skip requests, confident misconceptions, repeated failures, correct but assisted answers, requests to see a solution, and adversarial requests to ignore the issued plan. Use concise and detailed versions with the same factual content.
6. Verify math against B's solver and content specifications. Review all original examples used for protected assessment behavior. Exclude examples that reveal the active answer without permission, contain contradictory reasoning, or imitate unsupported learner diagnoses.
7. Run exact deduplication and normalized near duplicate detection across original problem families and public benchmark records. Preserve a rejection report and cross split family invariant.

Proposed sampling mix is 60 percent original probability examples and 40 percent permitted general tutoring examples, capped by available reviewed data. Sample by action and concept coverage, not just source size. Repeated sampling does not create new examples and must be reported as repeated exposure. If a move lacks enough valid examples, use a deterministic fallback for that move until data is ready.

### 13.3 Annotation and review

Score examples on mathematical correctness, action fidelity, disclosure permission, relevance to the learner's last message, and whether the learner has a clear next step. Each is pass or fail for admission. A records the reviewer's rationale for rejected math and disclosure cases. A and B independently review a proposed 100 example calibration set, then adjudicate disagreements against the verified item and action guide. They are technical reviewers; external educator review is a separate desirable resource, not assumed to have happened.

Synthetic examples are explicitly flagged. Their generating model and prompt versions are recorded. Do not use synthetic approval by the same generating model as the only correctness check. Keep adversarial prompts in a separate group so paraphrases of one attack do not cross splits.

### 13.4 Concrete initial training configuration

The values below are proposed defaults. A creates a resolved run configuration containing exact installed package versions, base model revision and tokenizer revision. Do not infer compatible versions by combining documentation snippets from different library releases.

```yaml
run_type: response_sft
base_model: Qwen/Qwen2.5-7B-Instruct
base_model_revision: REQUIRED_RESOLVED_COMMIT
tokenizer_revision: REQUIRED_RESOLVED_COMMIT
method: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
precision: bfloat16
max_sequence_tokens: 4096
micro_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 0.0001
epochs_max: 3
warmup_ratio: 0.05
weight_decay: 0.01
gradient_clip_norm: 1.0
packing: false
loss_scope: target_completion_only
evaluation_interval_optimizer_steps: 100
checkpoint_interval_optimizer_steps: 100
seed: 17
```

`REQUIRED_RESOLVED_COMMIT` is a configuration validation failure until filled; it is not an acceptable artifact revision. These keys are this project's configuration vocabulary, mapped by A to the pinned trainer APIs. The four projection module names must be checked against the selected checkpoint before training. If bfloat16 is unsupported, A uses a separately recorded compatible precision path and reruns the memory smoke test.

Use the prompt completion data format and explicitly configure completion only loss. TRL documents this masking for prompt completion inputs [S8]. If A instead chooses conversational assistant only loss, first verify the chat template exposes generation boundaries; that mode requires compatible template markers [S8]. Do not assume the base model's default template supplies them.

Before a long run, inspect the token level loss mask on 20 diverse examples. Require positive target tokens, zero loss on the plan and learner text, correct end of turn markers, and no answer key in the prompt when reveal is forbidden. Reject overlength examples or truncate older context deterministically; never truncate away constraints, the active problem, the learner's latest message, or the target response. Keep packing disabled initially to make masking and sequence boundary audits straightforward.

### 13.5 Training sequence and model choice

Run a 50 to 100 optimizer step smoke experiment and verify finite loss, usable output, memory headroom, and correct masks. Compare the untouched instruction model, a prompt only constrained model, and the SFT model on validation. Train at most three epochs initially and stop early if validation math errors, action violations, or premature answers increase, even if token loss decreases.

Select a checkpoint by a preregistered validation ordering: zero critical math and protected disclosure failures first; action fidelity second; human pedagogical score third; response length and cost fourth. A lower cross entropy alone cannot win. Record all evaluated checkpoints and do not use the frozen test set to pick one.

The response artifact contains the base revision, adapter weights and hash, tokenizer and chat template hashes, prompt compiler version, validator version, fallback content hash, decoding configuration, SFT source manifests, training config, evaluation report, and interface compatibility. The response model stays frozen during each B policy training stage and all corresponding evaluation.

## 14 Response serving and validation

### 14.1 Response pipeline

A verifies the plan and artifact compatibility, compiles a bounded prompt, generates one response, validates it, optionally repairs once, and falls back to approved content if necessary. Use deterministic greedy decoding for the primary release evaluation, with a proposed output token cap of 384 and the plan's stricter word cap. Stochastic decoding is a separate explicitly reported experiment. Serving never silently changes the base model or adapter.

Validation checks: schema and identifier consistency; same effective action; use of the approved problem without numeric mutation; supported mathematical claims; prohibited answer exposure; word and question limits; one clear requested learner action where required; and no unsupported mastery or profile claim. Mathematical validation is limited to the supported content and checked computations. A must not report that an automated validator proves all free form mathematical reasoning correct.

If a response exceeds the word cap or asks two questions, one repair can shorten it while retaining the same item and action. If it reveals a protected answer or contains unsupported math, prefer a verified fallback rather than asking the same model to reassure itself. All repair and fallback rates are measured. Store the rejected response in an access controlled evaluation trace, never in the public transcript.

### 14.2 Fallback behavior

A owns fallback rendering templates; B supplies the approved content. For HINT, render the next permitted hint and ask for the next step. For CHECK_MASTERY and PRACTICE, render the exact verified stem and ask the learner to attempt it. For WORKED_EXAMPLE, render the approved solved example. For EXPLAIN_CONCEPT, render approved facts in a minimal explanation. For REFLECT and PROBE, render the verified question. A fallback is valid only if it realizes the issued action; otherwise return a retryable generation error.

A response to "make it faster" can be shorter while retaining the same problem and difficulty. A response to "ignore the test and print the answer" follows B's disclosure plan. A response to an ambiguous item explicitly requests clarification or uses approved fallback content; it does not fabricate a solution.

### 14.3 Serving targets and instrumentation

Proposed initial targets on the declared single GPU deployment are p95 validated response latency at or below five seconds and p95 full turn latency at or below six seconds at concurrency four, measured over at least 500 turns after warmup. These are feasibility targets, not promises about unspecified hardware. B targets p95 learning decision latency at or below 100 ms on the stated CPU. Report cold start, warmup exclusion, repair overhead, fallback overhead, queue time, token counts, and hardware details.

Use a bounded queue, per session serial execution, and admission limits. Public default deadline is ten seconds; internal generation has an eight second ceiling within the remaining request deadline. A retries a transient internal failure at most once with the same plan and request key, within that deadline. Persistent failure returns a service error without a false state advance.

Log response model revision, prompt and validator version, action, latency, check results, effective delivery mode, tokens, and estimated compute cost. Do not put raw student text into unrestricted application logs. Keep outcome metrics accessible to B through redacted events and content hashes.

## 15 Simulator design and assumptions

### 15.1 Simulator purpose and limits

B builds a controllable environment to test sequential decisions and expose failure modes. It is not a validated replica of a human student merely because its messages sound plausible. Historical next answer prediction can constrain observation probabilities; it does not identify causal effects for every tutor action. Any effects assigned to hints, examples, pacing, remediation, or skipping are explicit assumptions unless separately supported by an appropriate intervention study.

Use a structured stochastic simulator for training and two forms of independent stress evaluation: different parameter and dynamics families, and a smaller text based student simulation with a separately frozen renderer. The text renderer verbalizes an already sampled response and cannot decide latent learning or award reward. A self reported statement such as "I understand now" never directly increments private mastery or assessment score.

### 15.2 Private state and observation generation

For concept i, private state contains continuous mastery `m_i` in [0,1], learning rate `ell_i`, forgetting rate, activated misconception flags, slip and guess parameters, and recent practice history. Session state includes fatigue, patience, response verbosity, help seeking probability, propensity to guess, and target interest. Sample these from distributions; profiles are convenient distribution presets, not rigid personality categories.

For an independent item, define a first simulator observation model:

```text
effective_mastery = clamp(m_i * prerequisite_factor * difficulty_factor, 0, 1)
p_correct = g + (1 - g - s) * effective_mastery
correct ~ Bernoulli(p_correct)
```

Require `0 <= g`, `0 <= s`, and `g+s < 1`. For free response, initial guess is sampled from [0.00,0.05]; for four choice items from [0.10,0.25]; slip from [0.02,0.20]. These are proposed sensitivity ranges. Difficulty factors default to easy 1.00, medium 0.85, hard 0.65. The prerequisite factor is `0.5 + 0.5*min(m_j)` across relevant direct prerequisites, or 1.0 with no prerequisites. Held out dynamics may instead use a noisy AND or additive prerequisite model. These model choices are assumptions to challenge, not educational facts.

On an incorrect response, choose an active misconception consistent with the item, otherwise a random or arithmetic error. Generate a corresponding wrong answer with the template's error function. On a correct response, sometimes generate weak reasoning or a guessed answer, so correctness does not perfectly identify mastery. Assistance changes the response distribution and observable flags; an answer shown can produce a correct submitted number with no independent evidence.

### 15.3 Teaching effects

Use the following initial bounded update, implemented independently of the belief estimator:

```text
learning_opportunity = ell_i * action_effect * readiness * novelty * content_fidelity
delta = (1 - m_i) * learning_opportunity
m_i_next = clamp(m_i + delta - forgetting_loss, 0, 1)
```

`ell_i` starts in [0.02,0.12]. `action_effect` is profile and state dependent, initially in [0,1]. `readiness` incorporates prerequisite support and difficulty fit; `novelty` declines with repeated identical interventions; `content_fidelity` is zero for unsupported or contradictory rendered content and at most one for verified content. Forgetting is zero by default in short sessions, with nonzero rates reserved for sensitivity and delayed retention tests. No unbounded additive noise may create mastery outside [0,1].

CHECK_MASTERY and PROBE have zero direct learning effect in the primary simulator to distinguish assessment from teaching. PRACTICE can learn through retrieval; its effect depends on effort and feedback rather than correctness alone. A HINT can help a stuck learner but has little effect if repeated or irrelevant. A WORKED_EXAMPLE can help a novice and can be redundant for a knowledgeable learner. No action is globally best by construction. Preference matched pace can affect fatigue or engagement; it cannot directly add knowledge points.

In the structured training mode, `content_fidelity` comes from a frozen response stub with a declared fidelity assumption, plus a configurable failure process. In text evaluation, use independent content checks and audited samples; do not let the generator's self reported pass flag alone control learning. Record the uncertainty in this bridge between language quality and simulated learning.

### 15.4 Profiles and scenario generation

The three prototype presets are: novice missing prerequisites; knowledgeable learner seeking concise progress; and confident learner with a conditional reversal misconception. Research scale adds slow acquisition, high slip, frequent guessing, high help seeking, low patience, mixed strengths, and misconception persistence. Ten profiles are distribution presets, and parameters vary within each.

A scenario fixes initial concept and goal, a coherent prior history, latent initial state distribution, allowed content families, profile parameter distribution, horizon, and expected failure cases. Reusing it with different seeds creates episodes, not new scenario families. B validates that the public history is plausible given private initialization; do not initialize a supposedly novice student with a long perfect independent history without marking the inconsistency as an adversarial test.

Sample a balanced training mixture: proposed 40 percent prerequisite gaps, 20 percent misconceptions, 20 percent near mastery or already knowledgeable, and 20 percent mixed or preference driven cases. Log actual proportions. Draw values within each group and oversample known failures only in training. Do not use test failures to change this mixture without retiring the test set.

### 15.5 Calibration and validity gates

If eligible student logs are available, fit next answer probabilities and estimator parameters on training learners only. Evaluate log loss, Brier score, calibration by probability bin, item difficulty, assistance, and missingness on held out learners. Historical hints are not randomized; a correlation between hint use and failure does not identify the effect of giving a hint.

Before PPO, require reproducible response distributions, action sensitivity, no privileged feature leakage, valid probability ranges, and at least one realistic scenario in which each supported teaching move is useful. Compare distributions of error patterns, assistance, successful independent answers, turn count, and dropout to available logs or a separately reviewed specification. If no real logs match the probability domain, label calibration as structural and expert review only. Do not transfer middle school arithmetic calibration to Bayes instruction without testing that domain shift.

An invalid simulator blocks learning claims even if PPO reward rises. A simulator that guarantees progress after any hint or allows answer copying to certify mastery must be repaired before more training is purchased.

## 16 Rewards and anti reward hacking controls

### 16.1 Primary reward definition

Both policies optimize the same educational base reward. The hierarchy changes decision duration and credit assignment, not the definition of success. Do not give separate rewards merely for advancing a concept, selecting a preferred move, generating a long response, or increasing the belief estimator's mastery number.

At reset, the evaluator fixes a target concept weighting and a hidden assessment specification independent of the policies. Pre and post scores are on the same scale, use balanced equivalent forms, and are unassisted. In the primary structured simulator, use expected independent correctness over those fixed probes to reduce sampling noise. The policy sees neither probe identities nor answers. A secondary analysis uses sampled answers to expose variance and reliance on the analytic simulator.

```text
r_t = -0.02                                      teaching turn cost
      -0.10 * unauthorized_disclosure
      -0.05 * unproductive_repetition
      -0.25 * terminal_simulated_dropout
      +4.00 * (post_score - pre_score) * terminal
      +optional_potential_shaping
```

All coefficients are proposed defaults, versioned before training. `unauthorized_disclosure` is a verified exposure on a protected check; a permitted solution demonstration is not this violation. `unproductive_repetition` requires repeating the same item or same hint level without new learner evidence, not merely repeating a beneficial teaching move on a new problem. A's validator should prevent disclosure in the full system; fault injection ensures reward and evaluation do not assume prevention is perfect.

Set `invalid_action_penalty=0` in all valid rollouts. An invalid action is an implementation failure that aborts collection for that episode and blocks the release gate; it is not something PPO should learn to work around. Fault injection tests may assign -1 for diagnostics, but those transitions are excluded from training. Require stored total reward to equal the sum of all components within numerical tolerance.

Terminal reward occurs once, on the final primitive teaching transition or, for immediate policy stop, on the zero duration terminal navigator transition. The navigator's final option reward contains that terminal event exactly once. A forced learner stop finalizes the active option but does not create another sampled navigator action. For a simulated dropout, evaluate the private student state at withdrawal with the fixed non teaching probes; do not drop the episode or report only completers.

If an episode starts with a knowledgeable learner, gain may be near zero. Report unnecessary teaching burden and absolute post score for that subgroup. Do not add artificial knowledge gain for reviewing a mastered skill simply to create positive rewards.

### 16.2 Optional dense shaping

The primary baseline uses no dense shaping. If sparse learning makes optimization fail, B can preregister a shaping experiment using `F_t = gamma*Phi(x_next) - Phi(x_t)` with `Phi` a bounded training only expected target assessment score and terminal potential set to zero. Record shaping separately from base reward and select models using unshaped outcomes. For an option of duration k, the corresponding shaping sum is `gamma^k*Phi(x_end)-Phi(x_start)`.

Potential based shaping has policy invariance conditions in the underlying Markov formulation [S15]. This does not prove convergence, human learning, or invariance for an incorrectly implemented belief approximation, changing simulator, or inconsistent terminal boundary. Include a no shaping ablation and a test that reward cannot be farmed by cycling back and forth.

### 16.3 Assessment independence

Training reward probes come only from a reserved training assessment bank, distinct from training teaching instances and preferably distinct template families. Validation and test probes are never called by the training environment. Within each split, B labels families as `teaching` or `assessment`; different numeric instances of a teaching family do not become supposedly independent assessment evidence by changing IDs.

For evaluation, fixed pre and post forms cover the original goal and necessary prerequisite transfer, not just the final concept the policy happens to choose. The navigator cannot switch the target weighting to an easier topic, choose the post test, set its difficulty, lower its passing threshold, or stop the assessment after a lucky correct answer. Post assessment runs at the fixed teaching horizon, policy stop, or simulated dropout; its complete evaluation budget is identical across arms.

### 16.4 Required adversarial controls

| Exploit | Preventive control | Required test |
|---|---|---|
| Repeat hints to collect mastery points | No reward for belief increase; saturation and novelty effects | Always HINT policy cannot dominate across all profiles |
| Copy answer then certify knowledge | Assistance flags and independent transfer assessment | Correct copied answers do not raise independent success count |
| Avoid hard goals by changing curriculum | Fixed original goal scoring | Jump to an easy unrelated concept gives no target gain |
| Inflate mastery through the response model | B owns estimator and grader; A emits evidence only | Inject fake mastery fields and reject them |
| Read latent student attributes | Separate diagnostics; allowlisted actor and critic features | Replace latent diagnostics while holding observation fixed; policy logits unchanged |
| Memorize scenario or seed identifiers | No IDs or RNG state in features | Rename IDs and preserve decisions |
| Farm navigation or assessment bonuses | No bonus for switching or passing a self chosen test | Cycle options and recheck identical items; no positive base gain |
| Hide failed episodes | Include dropout and early stopping in intent to treat metrics | Report all seeded scenarios including failures |
| Exploit one preferred action model | Held out effect reversals and alternate dynamics | Test hints with negligible effect and examples with delayed effect |
| Reward pleasing learner messages | Score solver verified independent responses | "I understand" alone changes no knowledge score |
| Exploit response length or retries | Fixed teaching cost; operational events excluded | Retry response 20 times; zero extra teaching transitions |
| Learn evaluation content | Split by source and reasoning family before rollout | Automated intersection check on every run manifest |

The always HINT test is a falsification test, not a theorem that HINT must lose every individual case. A useful simulator must support action tradeoffs. Publish an action frequency report, repeated content rate, option switching rate, dropout, answer exposure, and reward component breakdown alongside return curves.

## 17 Hierarchical RL training protocol

### 17.1 Model architecture and baselines

B implements two independent actor critic networks. Each receives a 140 dimensional observation with its role appropriate fields. Start with two hidden layers of 128 units and tanh activations, a categorical action head of 76 or 72 logits, and a scalar value head. Separate networks simplify staged training and versioning. Inputs are bounded by the feature specification; fitted normalizers, if added, use training data only and become part of the artifact.

The navigator selects a concept option; its environment includes the frozen or currently fixed strategist. The strategist selects a teaching action; its environment includes the fixed navigator for that training stage. During evaluation, both are fixed. Record the complete pair of policy versions on every episode because the other policy changes the environment seen by the one being trained.

Required baselines are: legal uniform random actions; a strong rules navigator and rules strategist; rules navigator with learned strategist; learned navigator with rules strategist; learned navigator with learned strategist; and a flat learned policy or a scope documented flat approximation for comparison of hierarchy. A prompt only generator comparison is a response ablation with policies held fixed. Equalize teaching budgets, content, response artifact, evaluator, and training resource accounting.

The rules baseline diagnoses before unsupported advancement, offers a hint after a meaningful error, uses a worked example after repeated unsuccessful hints, checks with a fresh family, reviews the weakest relevant prerequisite after repeated independent failures, and returns to the prior goal after remediation. It honors pace and skip requests using the same deterministic agency rules as learned policies. Implement and freeze it before measuring learned policy improvement.

### 17.2 Transition mathematics

Let t index primitive teaching turns, k the navigator option duration, gamma the per turn discount, and r the shared base reward. Default `gamma=0.99` and `lambda=0.95`. For a navigator option starting at t:

```text
R_option = sum(j=0..k-1, gamma^j * r_(t+j))
delta_N = R_option + gamma^k * bootstrap_N - V_N(obs_t)
bootstrap_N = 0 if true terminal else V_N(obs_(t+k))
A_N = delta_N + (gamma*lambda)^k * continuation * A_N_next
```

The duration based lambda convention above is this project's choice; keep it consistent and test it. `continuation` is one only when the next finalized navigator transition is in the same collected episode segment. At a collection cut, bootstrap from the actual next observation but do not chain an advantage into a new reset state. A zero duration terminal END_SESSION action uses its terminal event reward and bootstrap zero; it must not create a recursive zero duration continuation.

For the strategist, use the same equations with k=1 and its own value function. Its next observation includes the next active option if the navigator changes options before the next teaching decision. Do not end strategist returns at every option boundary. For true finite horizon termination or simulated dropout, bootstrap is zero. For an artificial collector fragment, bootstrap from the final pre reset observation [S9].

Do not divide option reward by duration unless intentionally changing the objective. An average per option reward would prefer different behaviors from discounted per turn return. Do not use the next episode's initial observation as the terminal bootstrap. Unit tests must cover duration 0, 1, 3, and 4; early stop; collection cut; option change; and terminal reward exactly once.

### 17.3 PPO objective and initial hyperparameters

For eligible sampled actions, compute `ratio = exp(new_log_prob - old_log_prob)` under the stored legal action mask. Maximize the mean of `min(ratio*A, clip(ratio,1-epsilon,1+epsilon)*A)` plus entropy regularization, with a separate value loss [S7]. Normalize advantages separately within each policy's current rollout batch. Pad variable sequences only for storage; mask every padded row from policy, value, entropy, and statistics calculations.

| Setting | Navigator default | Strategist default |
|---|---|---|
| Discount per teaching turn | 0.99 raised to option duration | 0.99 |
| GAE lambda | 0.95 with duration convention | 0.95 |
| Learning rate | 0.0003 | 0.0003 |
| PPO clip epsilon | 0.20 | 0.20 |
| Value coefficient | 0.5 | 0.5 |
| Entropy coefficient | 0.01 | 0.01 |
| Max gradient norm | 0.5 | 0.5 |
| Target rollout batch | 1,024 finalized eligible decisions | 2,048 eligible decisions |
| Minibatch | 128 | 256 |
| Epochs per fresh batch | 4 maximum | 4 maximum |
| Approximate KL early stop | 0.02 | 0.02 |
| Optimizer | Adam | Adam |

These are tuning defaults, not a claim of optimality. Use complete episodes in v1 collection so navigator options are never artificially cut to fill a buffer. Collect until the target is met or the stage episode budget is exhausted. Train on a final partial batch if it has at least one minibatch; otherwise record discarded samples. Report exact valid sample counts, total episodes, and updates. Do not generate more unreported episodes to make a batch boundary convenient.

Forced learner agency actions and fault recovery actions have `actor_eligible=false`. They may contribute rewards to a trajectory and valid value targets, but they have no actor likelihood ratio loss. B documents their effect on return propagation. Rule generated actions are not sampled PPO behavior and cannot be relabeled as learned policy samples.

### 17.4 Training stages

**Stage zero, outside final research training budgets:** Validate content and contracts; run 100 deterministic smoke episodes; compare random and rules behavior; test simulator assumptions and reward exploits. No performance claims are based on these episodes. Optional behavior cloning uses original reviewed action examples or audited source move mappings; its own dataset, budget, and ablation are reported.

**Stage one:** Freeze the rules navigator and train the strategist for 2,500 episodes. Freeze simulator, content, estimator, reward, and A's response stub. Choose a strategist checkpoint on validation only. The navigator may be a rule component at this point, not an untrained neural network.

**Stage two:** Freeze the chosen strategist and train the navigator for 2,500 episodes. Rollouts must be newly collected under this navigator and the frozen strategist pair. Discard actor rollout buffers when changing a counterpart policy, response artifact, reward, or simulator.

**Stage three:** Run two alternating cycles. In each cycle, train the strategist for 1,250 fresh episodes with the navigator frozen, then train the navigator for 1,250 fresh episodes with the strategist frozen. Total stage three budget is 5,000 episodes. Stages one through three total 10,000 episodes per training seed. This is a proposed allocation; any revised allocation is a separate preregistered configuration.

**Stage four:** Freeze the pair and evaluate with A's real SFT responses. If the structured to text gap violates the outcome or response gate, fix the response contract bridge or retrain against a versioned response distribution using training scenarios only. That adds a new named experiment budget. Never quietly claim the text system achieved the structured simulator score.

Alternating optimization is a practical response to the nonstationarity created when both policies change. It offers no guarantee of a global optimum. Simultaneous updates may be explored later as an ablation with synchronized rollout versions and clearly defined behavior distributions.

### 17.5 Collector pseudocode

```text
for each training stage:
    freeze simulator, estimator, response artifact, reward, and counterpart policy
    initialize fresh rollout buffer for the policy being trained
    for each seeded episode in the stage budget:
        reset private simulator and observable belief separately
        compute fixed pre assessment without learning or revealing items
        while session remains active:
            apply previous student's observable evidence to belief
            if a deterministic ending rule applies:
                close active option and episode; break
            if navigator boundary:
                finalize prior navigator transition if present
                sample a valid navigator action or apply a forced control rule
                if END_SESSION: evaluate and close zero duration terminal; break
            sample valid strategist action or apply a forced agency action
            resolve verified content and realize response
            apply effective intervention to simulator
            sample student answer, assistance flags, and engagement events
            update private simulator state using declared teaching dynamics
            compute next belief from observable evidence only
            record strategist transition and accumulate active option reward
            if terminal: run fixed post assessment; attach terminal reward once
        finalize the last option; store episode manifest and diagnostics
        if fresh eligible buffer reaches target:
            update active policy with PPO for the configured epochs
            clear consumed on policy rows before collecting the next batch
    flush valid partial batch, log unused rows, freeze candidate checkpoint
    evaluate validation cases; record selection; clear all actor buffers
```

Implementation must define observation ordering exactly. Sample the learner answer from the state at the time of the requested attempt, apply the learning effect of that attempt afterward, and score post assessment on the resulting state. A worked example can change private state before the next practice attempt; a check's own update is zero. Do not both apply an answer at the end of one loop and again at the start of the next. The pseudocode's previous evidence application is a logical step, implemented once through event IDs.

### 17.6 Monitoring and stop conditions

Track base return, terminal target gain, absolute post score, failure and dropout rates, actor entropy, KL, clip fraction, value error, action mask size, action frequencies, option durations, switches, and response fallback. Stop a run on nonfinite values, version mismatch, invalid action, privileged data leakage, assessment contamination, or inconsistent reward accounting. A flat validation curve triggers a simulator and credit assignment diagnosis before increasing episode count.

Use validation checks every proposed 500 episodes and at stage boundaries. The final test remains sealed until a final pair and all thresholds are frozen. A training curve with rising reward and flat independent transfer is a failure of the educational objective or simulator, not evidence of successful tutoring.

## 18 Evaluation and data splits

### 18.1 Split assets before generating episodes

| Asset | Proposed allocation | Leakage rule |
|---|---|---|
| 120 original teaching template families | 60 train, 30 validation, 30 test | Parameter variants and paraphrases stay with their family |
| 45 additional independent assessment families | 15 train reward, 15 validation assessment, 15 sealed test assessment | Never used as instructional items or SFT targets |
| 400 curated scenario families | 280 train, 60 validation, 60 test | Same initial history, goal structure and profile combination cannot cross splits as paraphrases |
| Original SFT examples | 80 percent train, 10 validation, 10 test by original problem and dialogue group | Split before target turns or synthetic rewrites are expanded |
| Public dialogue SFT | Preserve official splits where possible; otherwise family grouped 80 10 10 | Audit original problem overlap with response benchmarks |
| Real student calibration logs | Learner grouped 70 15 15, with chronological order within learner | Fit transforms on train only; never random split interaction rows |
| Ten simulator profile presets | Seven train, one validation, two stress test only | Also reserve new parameter ranges and alternate dynamics, not only new seeds |
| Human reviewed navigation cases | Separate sealed 120 cases, 20 in each of six categories | Used for final evaluation, never imitation labels or training scenarios |
| Response quality cases | Separate 600 case frozen suite | Distinct original problem families or explicitly labeled overlap |

These are construction targets. Asset counts after review may differ and must be reported. Teaching splits contain at least four families per concept in training and two per concept in validation and test so the independent mastery rule can be exercised. The additional assessment bank has at least one distinct family per concept per split, with balanced pre and post forms generated within that family. This remains a small assessment sample; document coverage and expand before broad claims. The total content budget is 165 families. If valid distinct families cannot be authored for a concept, reduce the enabled scope rather than reusing teaching families as independent transfer tests.

Prototype splitting follows the same rules with smaller counts and yields engineering evidence only. Use 20 training scenarios, five validation, and five test among the first 30; reserve additional assessment content outside the 15 teaching templates. Report that scope explicitly. A 30 scenario prototype cannot establish broad generalization.

### 18.2 Evaluation suites

**Contract and integration suite:** All schema branches, mask/action encoding, state updates, assistance semantics, retry races, abandoned plan recovery, double submission, event replay, response validation, timeout behavior, and reward aggregation. These are pass or fail engineering checks.

**Response suite owned by A:** Proposed 600 cases stratified across eight moves, five prototype concepts initially, three pace settings, misconceptions, protected checks, and explicit user requests. At least 100 target protected disclosure, 100 target wrong math or mutated item values, and 100 target instruction conflict; categories can overlap but report overlaps. A and B review all severe cases, and a proposed 200 sampled responses receive independent dual ratings for correctness, fidelity, actionability, and tone. An external educator review is desirable before a human pilot.

**Sealed decision suite owned by B:** 120 cases covering known topic readiness, confident misconception, uncertain skip request, prerequisite gap, faster pace without increased difficulty, and return after remediation. Define a set of acceptable navigation and teaching actions plus forbidden behaviors for each case. Multiple moves can be defensible. Score rubric compliance rather than matching one arbitrary action label.

**Full simulated episodes owned by B:** For each final trained seed and baseline, evaluate on 60 test scenario families with 20 stochastic episode seeds each, for 1,200 episodes per policy pair. Cross these with the declared primary simulator configuration; run separate OOD profile, parameter, and alternate dynamics suites at the same budget where feasible. Do not pool different simulator families into a single unqualified number. The five trained policy seeds produce five independent learned artifacts, not 6,000 independent human learners.

**Text system bridge:** Run at least 300 paired test episodes with structured responses and real SFT responses under the same policy pair, scenarios, and random event bank. Review where actual text fails the action, changes the item, discloses an answer, or changes simulator interpretation. This does not replace the full structured evaluation; it measures a separate integration gap.

**Optional external benchmarks:** Use a verified text only TutorBench subset for response quality and BEA dimensions as rubric references. Report exclusions, overlap status, task relevance, judge versions, and actual denominators. Do not claim a full benchmark score after selecting a small convenient subset or converting image questions without verified transcriptions.

### 18.3 Metrics and estimands

Primary simulated metric is mean `post_score - pre_score` on fixed independent goal aligned assessments, in [-1,1], with intention to treat inclusion of every seeded episode. Report absolute pre and post scores, episode completion, dropout, independent success, assisted success, and teaching turns. A gain of 0.05 means five percentage points on this assessment probability or accuracy scale, not a five percent relative improvement in human learning.

The primary comparison is the learned hierarchy versus the frozen rules pair with the same SFT response artifact or the same structured stub, depending on the named experiment. Secondary comparisons isolate each policy, hierarchy, estimator, shaping, and response generator. Report independent retention after a declared simulated delay separately from immediate transfer. A modeled forgetting test is not observed long term human retention.

Efficiency metrics are teaching turns to predefined independent target success, area under the success versus turn curve, and cost per completed episode. Include censored failures and fixed horizon failures; do not compute time to success only among completers. Report critical response failures, unauthorized disclosures, misconception diagnostic precision and recall where labels exist, intervention frequencies, option cycles, and learner request compliance.

### 18.4 Statistical analysis and checkpoint discipline

Freeze hypotheses, final checkpoints, metrics, seed lists, evaluation files, and analysis code before the first final test run. For paired policy comparisons, use a common event random bank keyed by scenario, episode seed, turn, and event type. This preserves correct marginal draws when policies take different actions and prevents a differing number of RNG calls from destroying pairing. Never key test outcomes by the selected policy's identity.

Report each trained seed and the aggregate mean with a 95 percent confidence interval. Use a hierarchical paired bootstrap over trained policy seeds and scenario families, preserving repeated stochastic episodes within each cluster. Do not treat every teaching turn or repeated rollout as an independent sample. Show per profile and per simulator family estimates with denominators; label small subgroup results exploratory.

The proposed improvement gate is a point estimate of at least +0.05 absolute gain versus rules with the paired 95 percent confidence interval lower bound above zero in the primary simulator. This is a target, not an outcome. Require no material degradation in prespecified stress groups; Section 19 defines the engineering threshold. Passing only one simulator cannot support a claim of robustness across all learners.

At most one final test exposure is allowed for the selected release candidate. A changed model, reward, simulator or policy after test failures requires a new version and either a fresh sealed test set or an explicit statement that the old test became development data. All outcomes, including no improvement, are reportable. Pretraining contamination in public models cannot generally be ruled out from dataset deduplication alone; qualify benchmark results accordingly.

### 18.5 Prospective human evaluation

A human study is a separate release gate after response and simulation quality pass. Start with supervised usability sessions and educator review. For a comparative study, define population, consent and privacy requirements, randomization unit, baseline, fixed teaching time, equivalent pre and post forms, retention delay, primary endpoint, adverse experience reporting, and an analysis plan before recruiting. Any requirements for minors and institutional review are resolved with the responsible organization.

Determine sample size from a justified minimum detectable effect, assessment variance, and the chosen design; do not invent a convincing participant count from simulation results. Evaluate independent transfer and retention, not satisfaction alone. Until that study exists, use the phrase "simulated assessment improvement" and do not claim measured student learning gains.

## 19 Acceptance criteria and release gates

All numerical thresholds in this section are proposed acceptance targets. Report observed counts and confidence intervals when the implementation exists. A gate can pass for an internal prototype while remaining unqualified for a human pilot or unrestricted deployment.

| Gate | Pass criteria | Owner and evidence |
|---|---|---|
| G0 Contract freeze | Every record variant has valid and invalid fixtures; action index round trips; mocks pass both consumer suites | A response contracts; B learning contracts; signed fixture manifest |
| G1 Content and state | Enabled graph is acyclic; all references resolve; solver checks pass; assisted answers cannot certify mastery; deterministic replay matches | B content report and event replay tests |
| G2 Response prototype | Each supported move has an accurate fallback; zero observed protected disclosure or critical math failures in initial 100 reviewed fixtures | A response report; scope limited to the reviewed content |
| G3 Simulator validity | Reproducible 100 episode smoke suite; no hidden features; action tradeoffs and counterexamples; reward accounting and terminal tests pass | B simulator validity report with assumptions |
| G4 Prototype integration | Ten end to end sessions with stubs; retry and crash recovery cases pass; no duplicate state update; independent workstream tests pass | A API recovery report; B episode consistency report |
| G5 Final response artifact | On 600 frozen cases: at least 95 percent action fidelity, at least 95 percent supported math correctness, zero observed critical math or protected disclosure failures; at least 90 percent actionable on dual reviewed applicable cases | A frozen response report with actual denominators and reviewer disagreement |
| G6 Final policy artifact | Zero invalid actions, leaked private features, or post test contamination; every seeded episode included; all required baselines and policy ablations reported | B evaluation report and manifests |
| G7 Simulated improvement | At least +0.05 absolute gain over rules in the primary simulator with paired 95 percent interval lower bound above zero; no prespecified stress group point estimate below rules by more than 0.02 | B outcome report; inconclusive or failed stress tests block robustness claims |
| G8 Text integration | No critical failures in reviewed bridge traces; text system gain no more than 0.02 below structured system point estimate; publish paired uncertainty | B bridge outcome report plus A response checks |
| G9 Serving feasibility | Meet stated warm latency targets at declared hardware and concurrency; failure rate below 1 percent excluding deliberate fault injection; durable retry recovery passes | A load report and B decision latency report |
| G10 Human pilot readiness | Educator review, privacy and consent process, participant protection, monitoring and approved study plan | A service readiness; B study and outcome protocol; organization approval where applicable |

Zero observed severe failures is not proof of zero risk. With zero failures among 600 independent applicable cases, a rough rule of three upper 95 percent bound is about 0.5 percent; a smaller protected subset has a wider bound. Report the applicable denominator for each failure type rather than using all 600 cases to imply a stronger bound.

If G7 fails, the team may still deliver a technically complete prototype and a negative research result. Do not label the learned policy production superior. Use the rules policy as the default if it is more reliable. If G5 fails, keep responses on verified fallbacks or block the affected move until corrected. Threshold exceptions are explicit decision records and cannot convert an unmeasured result into a pass.

## 20 Independent workstreams and milestones

### 20.1 Milestone plan

The following is an illustrative eight week plan for two full time contributors. It assumes access to a suitable GPU by week three and time for original content review. It is not a schedule commitment. Progress is gate based; a late simulator or failed response gate changes dates rather than lowering the bar.

| Milestone | Person A deliverable | Person B deliverable | Integration checkpoint |
|---|---|---|---|
| M0 Days 1 to 2 | Public and generation schemas, response stub, API idempotency design | Learning schemas, action encoding, belief and plan fixtures | Freeze v1 fields and fixture hashes; G0 |
| M1 Week 1 | Prompt compiler, deterministic fallbacks, response fixture tests | Five concept graph, solver, eight misconceptions, BKT estimator, rules baseline | CPU only ten session demo; G1 G2 G4 |
| M2 Week 2 | Original SFT data pilot, source manifests, extraction evaluation, API recovery | Structured simulator, 30 scenarios, private state boundary, reward and end tests | No GPU dependency for B; simulator audit; G3 |
| M3 Weeks 3 to 4 | SFT smoke and first frozen model; response quality comparisons | Prototype strategist and navigator stages; 500 to 1,000 episode experiments | Evaluate actions with stub, then real text; inspect gap |
| M4 Weeks 5 to 6 | Expanded data and 600 case suite; latency measurement; frozen release candidate | 15 concepts, 120 teaching families, separate assessment bank, 400 scenarios; five seed research training and ablations | Freeze all candidate artifacts and final evaluation plan |
| M5 Week 7 | Final response evaluation and serving report | Final test, OOD tests, sealed decisions and statistical analysis | G5 through G9; preserve negative findings |
| M6 Week 8 | Release bundle, API runbook, demonstration and response limitations | Reproducibility package, outcome report, simulator limitations, human study plan | Handoff rehearsal and conditional pilot readiness review |

### 20.2 A can work independently

A begins with B's plan fixtures and a mock learning service. A can implement prompts, fallbacks, validators, evidence extraction, data preparation, SFT, and serving without a learned policy. The mock supplies valid known plans and error responses; it never implements a competing student state estimator. A tests public state conflicts and retries against the mock transaction contract, then repeats those tests against B's real service.

Required A outputs at each handoff: response artifact manifest; generation and public API schema versions; deterministic generation fixtures; validation failure fixtures; fallback coverage by action; extraction confusion matrix; response benchmark report; and model resource measurements. B must not need A's raw SFT data or GPU training environment to consume the response artifact.

### 20.3 B can work independently

B begins with A's response stub and locally loaded response fixtures. B can implement content, grading, state, scheduler, simulator, rewards, PPO, and evaluation without an SFT checkpoint or public API. The simulator's student renderer is B's code and can remain templated. Fast rollouts operate on structured actions and frozen response realizations, with assumptions stated.

Required B outputs at each handoff: learning contract and valid plans; graph and content manifest; legal action and end fixtures; authoritative service behavior; state estimator version; simulator validity report; policy bundle; evaluation config; and outcome report. A must not need private simulator profiles or training checkpoints to generate a response to a plan.

### 20.4 Daily handoff contract

Each owner publishes an artifact version, schema version, compatibility list, passing fixtures, known limitations, and one reproducible command for their component. Integration uses immutable versions, not a branch name such as `latest`. Each owner can advance independently while the integration bundle stays on the last compatible pair. A breaking contract change must include a migration example and both old and new consumer tests.

## 21 Immediate tasks for each owner

### 21.1 Person A first working day

1. Read Sections 2, 8, 9, 13, 14, and 19. Create the response owned directories and OWNERS entries without implementing student state or policies.
2. Implement GenerateRequest, GenerateResponse, PublicState projections, Error, and public endpoint schemas. Add one fixture for each of the eight teaching moves, a protected check, and a response that must be rejected.
3. Implement a CPU response stub that renders approved content and checks required plan fields. Add a hard rejection when an independent plan supplies a nonnull reveal solution.
4. Implement prompt projection so IDs, fingerprints, private state, and hidden solutions cannot enter the model prompt. Snapshot the compiled prompt from representative fixtures.
5. Implement durable request keys and response staging using B's mock plan service. Demonstrate the same request returning the same response ID and content hash after an injected dropped connection.
6. Write the first 30 original response examples across moves, with math checked against B's fixture keys. Create the source and split manifest before expanding them.
7. Publish the stub artifact and its loading instructions for B, then prepare the SFT memory and masking smoke configuration. A GPU run is not the first dependency.

**Day one acceptance:** B can load A's stub, generate a faithful response to a valid plan, observe rejection of an invalid plan, and reproduce the same committed response under a retry. No training metric is required to claim this engineering task complete.

### 21.2 Person B first working day

1. Read Sections 2, 5 through 10, and 15 through 19. Create learning owned directories, schema fixtures, and feature and action encoding specifications.
2. Implement the five concept reduced graph with stable 15 concept indices. Add acyclicity and reference validation. Define the first eight misconceptions with distinct diagnostic signatures.
3. Build five verified parameterized problems and deterministic solver fixtures, then grow to the first 15 teaching templates. Reserve independent assessment families before reuse across episodes.
4. Implement observable belief updates with separate assisted and independent counts. Test a correct copied answer, an unknown grade, and a contradictory numeric extractor proposal.
5. Implement the rules navigator and strategist, scheduler, masks, and goal stack. Test an uncertain skip, a pace request, remediation, and return to the goal.
6. Implement the structured simulator reset and one step with separate private state. Add deterministic seed streams and a feature leakage test before collecting episodes.
7. Export valid TeachingPlan fixtures and the mock observe, commit, status, and conflict behavior for A. Publish the ten session scenario script.

**Day one acceptance:** A can consume B's fixtures without accessing hidden student state. B can replay a short seeded session and produce valid distinct navigator and strategist records with the correct durations. A neural policy is not required for this first handoff.

### 21.3 First integration review

Review a known student skip, a novice misconception, an answer shown event, an immediate stop, a duplicated request, and a generation failure. Inspect raw events, public transcript, authoritative belief, response checks, and policy records side by side. Fix semantic disagreements in the owned contract file before training. Record which assumptions remain unvalidated; do not call an integrated demonstration an evaluation result.

## 22 Reproducibility and experiment records

### 22.1 Required run manifest

Every SFT, RL, calibration, and evaluation run includes run ID, UTC timestamps, owner, source commit and dirty diff hash, exact dependency lock hash, operating system, CPU/GPU and driver details, numeric precision, model and tokenizer revisions, all data manifest hashes, split assignment hash, resolved configuration, seed tree, artifact input hashes, total samples and updates, checkpoint selection rule, and output hashes.

For RL also include both policy versions, simulator dynamics and parameter distribution version, graph and content hashes, state estimator and feature version, mask implementation, response artifact and failure model, reward version, teaching horizon, assessment bank hash, end reason counts, actual action counts, and actor ineligible rows. Failed and aborted runs keep a status and reason rather than disappearing from the experiment table.

### 22.2 Randomness and determinism

Use independent seeded streams for scenario sampling, initial private state, item parameters, action sampling, answer correctness, misconceptions, engagement or dropout, response generation, and model initialization. Derive child seeds from a recorded master seed with a stable method. Reordering parallel workers must not change which episode seed is assigned to a scenario.

For CPU structured simulation, replaying a seed and action sequence must reproduce events, hidden states, and rewards exactly under the same dependency lock. For GPU model training, document nondeterministic kernels and provide statistical reproduction tolerance rather than promising bit identical weights. For evaluation, cache and hash actual rendered responses so retries and audit replay do not depend on GPU reproducibility.

### 22.3 Artifact promotion and rollback

Immutable release bundles reference A's response artifact and B's graph, content, state estimator, feature, mask, navigator and strategist artifacts. A root bundle also declares supported schema major versions and allowed data use profile. Promotion fails on a missing artifact, hash mismatch, incompatible feature width, action mapping mismatch, or unsupported graph version.

Roll back a complete compatible bundle, not one policy independently unless the compatibility declaration permits it. Maintain the rules bundle as a known fallback. Session artifacts are pinned at session creation; existing sessions finish on their pinned bundle unless an explicit migration or emergency safe stop is recorded.

### 22.4 Handoff package

The final repository must include onboarding commands, CPU smoke fixtures, contract schemas, content and provenance manifests, configuration examples with unresolved values rejected, run summaries, evaluation code, checkpoint hashes, replay examples, and limitations. Raw gated data and raw learner messages are not copied into the public package. Include a tiny original synthetic fixture dataset that reproduces the pipeline without special access.

## 23 Operations privacy and security

### 23.1 Storage and access

A owns public request metadata, staged responses, redacted response quality logs, and auth. B owns learning snapshots, plan reservations, committed teaching events, assessment results, and simulator diagnostics. Use separate database roles, session authorization on every public request, and internal service credentials. A learner supplied session ID does not establish access.

Raw text can contain personal information. Use pseudonymous learner IDs, encryption in transit and at rest where deployed, and access controlled transcript storage. Proposed default retention is 30 days for raw prototype transcripts and 90 days for redacted operational aggregates, subject to the actual consent and organizational policy. This is a proposed product retention policy, not an assertion about applicable law. Delete raw text and derived learner specific records when required by the approved policy; preserve only genuinely non identifying aggregates.

Do not include real learner text in SFT or simulator calibration by default. A dataset admission process must verify consent or other applicable authorization, permitted purposes, retention, and deletion propagation. Original synthetic fixtures remain the default for public demonstrations.

### 23.2 Input trust and model permissions

The response model has no database credentials, state write tool, shell, policy loading tool, or reward update tool. Learner text, uploaded content, retrieved examples, and simulator utterances are untrusted inputs. The model receives bounded approved content and cannot promote a request inside a student message into a contract change.

Escape transcript content in the demo client. Do not render arbitrary model HTML. Validate IDs and schemas before database access. Load only approved hashed policy and model artifacts; prefer safe weight serialization and prohibit loading arbitrary executable checkpoints from user input.

### 23.3 Monitoring and incident behavior

Dashboards show response failures, disclosure checks, state conflicts, latency, fallback use, session ending reasons, and artifact versions. B's research dashboards additionally show simulator and policy metrics; keep those separate from real learner outcomes. Alert on repeated unauthorized disclosure, impossible state versions, graph validation failure, or an unexpected artifact change.

A disables a failing response artifact or affected move and uses verified fallbacks. B disables a failing policy bundle and supplies rules plans or ends safely if state integrity is uncertain. Preserve redacted traces and record the action taken. After an incident fix, replay the affected scenario and relevant regression cases before promotion.

## 24 Risks and decision log

### 24.1 Risk register

| Risk | Observable sign | Mitigation and accountable owner |
|---|---|---|
| Simulator teaches the desired answer by construction | A single move dominates every profile | B changes assumptions, tests alternate dynamics, and reports limitations |
| Hidden state or assessment leakage | Unrealistically perfect decisions or IDs predicting actions | B allowlists features and tests counterfactual diagnostics |
| SFT memorizes answers or reveals them early | Low loss but protected check failures | A filters data, verifies loss masks, and keeps deterministic disclosure checks |
| Language realization changes policy semantics | Structured results exceed text results | A fixes fidelity; B owns the measured bridge and any retraining |
| Two policy nonstationarity | Value errors spike after counterpart updates | B freezes counterpart per stage and discards stale actor buffers |
| Dataset rights or access block progress | Gated approval delayed or restricted derivative detected | Each owner uses original data path and records source permissions |
| Historical data does not identify treatment effects | Simulator action coefficients are justified only by correlation | B labels assumptions and uses intervention evidence only within its scope |
| Knowledge belief is overconfident | Correct guesses or assisted answers trigger premature advancement | B tracks uncertainty and independent diverse evidence |
| Topic switching sacrifices learner agency | Repeated remedial loops after explicit skip | B applies monitored jump rules and cycle limits; A expresses them clearly |
| Problem solving fluency hides wrong explanation | Correct final number with invalid intermediate claims | A checks approved facts and audits reasoning, not only answer equality |
| Small held out families yield unstable estimates | Large seed or subgroup variation | B expands families and reports clustered uncertainty |
| Eight week plan exceeds content review capacity | Review backlog grows before training | Keep prototype scope; delay research expansion; A and B own their assets |
| Public benchmark contamination | SFT source problem overlaps test dialogue | A deduplicates and qualifies residual pretraining uncertainty |
| No real learning improvement | Strong simulator scores fail prospective assessment | B treats this as a research finding and revises assumptions |

### 24.2 Initial decisions to record

**D01 Two policies and one response model:** B owns both policies; A owns SFT and expression. Chosen to isolate pedagogical decisions from language quality and permit independent experiments.

**D02 Probability first:** Narrow curriculum supports verified solvers and inspectable prerequisites. Public arithmetic tutoring data is supplementary; domain transfer is explicitly evaluated.

**D03 Discrete actions and fixed boundary rules:** Avoid token level RL and learned termination for the first system. Change only when a failure case and evaluation justify added complexity.

**D04 Belief based observations:** No hidden student attributes in actor or critic. This improves auditability and keeps simulator success closer to the information available in deployment.

**D05 Duration aware navigator PPO:** Options accumulate per turn discounted rewards and bootstrap by elapsed turns. Fixed gamma per navigator decision is rejected because it changes the meaning of time across options.

**D06 Original data runnable path:** Gated sources, restricted datasets, and paid services cannot block the first demo. Data permissions remain attached to any later calibrated or trained component.

**D07 Transactional plan delivery:** Apply learner evidence once, reserve one plan, stage a response, commit with B, and return it. This makes recovery and duplicate suppression concrete.

**D08 Outcome claims require independent tests:** Keep response scores, simulated gains, and human learning evidence separate. A release with no demonstrated RL advantage remains a valid prototype, with rules as the default.

## 25 Sources and factual reference notes

Sources were checked on November 2, 2026. Dataset counts describe publisher reported records, not this project's imported or usable examples. Publisher pages can change; each actual run must pin revisions and preserve the relevant license and data dictionary snapshots. This section supports factual dataset and method descriptions. Curriculum edges, resource estimates, numerical targets, thresholds, simulator effects, timelines, and architecture choices are this document's proposals.

- **S1 MathDial dataset card.** [eth nlped MathDial](https://huggingface.co/datasets/eth-nlped/mathdial). Publisher card lists 2,861 rows, fields including teacher described confusion and dialogue acts, and CC BY 4.0. Source for schema and licensing description, not evidence of human learning gains.
- **S2 MathDial original paper.** [Macina and colleagues 2023](https://arxiv.org/abs/2305.14536). Describes collecting tutoring dialogues with human teachers and simulated students. Supports the distinction between tutoring behavior examples and real longitudinal student trajectories.
- **S3 FoundationalASSIST publisher card.** [ASSISTments FoundationalASSIST](https://huggingface.co/datasets/ASSISTments/FoundationalASSIST). Source for record counts, grade range, support aware discrete score semantics, gated access, noncommercial license marker, and revision warnings.
- **S4 Eedi Misconceptions Graph publisher card.** [Eedi Misconceptions Graph](https://huggingface.co/datasets/Eedi/Eedi-Misconceptions-Graph). Source for version 1 catalogue and association counts, CC BY 4.0, graph weight meaning, and exclusions. Associations are not prerequisite ordering or student response data.
- **S5 ASSISTments official data documentation.** [2009 to 2010 ASSISTment Data](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data). Describes skill builder files, multi skill record layout, and limitations of hint and attempt sequence information. This design intentionally does not assert one universal row count across releases and preprocessing conventions.
- **S6 Options framework.** [Sutton Precup and Singh 1999](https://www.sciencedirect.com/science/article/pii/S0004370299000521). Defines temporal abstraction with options and the resulting semi Markov formulation. The tutor's exact option schedule is proposed here.
- **S7 PPO original paper.** [Schulman and colleagues 2017](https://arxiv.org/abs/1707.06347). Describes alternating fresh interaction and clipped surrogate optimization. Does not establish that this tutor or any particular training budget will improve learning.
- **S8 TRL official SFT documentation.** [SFT trainer](https://huggingface.co/docs/trl/en/sft_trainer). Current documentation retrieved through Context7 from the project's official source. Supports prompt completion loss masking and the chat template requirements for assistant only loss. Actual implementation must pin a compatible release.
- **S9 Gymnasium official API semantics.** [Time limits and bootstrapping](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) and [environment API](https://gymnasium.farama.org/api/env/). Official documentation retrieved through Context7 supports the terminated versus truncated distinction and seeded resets. This design's 16 turn cap is explicitly a finite task horizon.
- **S10 Candidate response model publisher card.** [Qwen2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). Source for the candidate checkpoint and Apache 2.0 license marker. No tutoring suitability or hardware fit is assumed from the card alone.
- **S11 TutorBench publisher card.** [ScaleAI TutorBench](https://huggingface.co/datasets/ScaleAI/TutorBench). Describes three tutoring task types and lists 1,473 rows. The observed card does not by itself establish unrestricted redistribution rights; verify applicable terms before import or publication.
- **S12 Eedi NeurIPS 2020 publisher page.** [NeurIPS Education Challenge](https://www.eedischool.com/projects/neurips-education-challenge). Distinguishes task specific datasets and states noncommercial and other content restrictions. This design does not rely on its question images for production content.
- **S13 Eedi research and data overview.** [Data and competitions](https://www.eedi.com/data-and-competitions). Describes learning paths and experimental construct relationships. It does not validate this tutor's action level causal model.
- **S14 BEA 2025 official shared task.** [Pedagogical Ability Assessment](https://sig-edu.org/sharedtask/2025). Source for rubric dimensions, development dialogue and response counts, and MathDial and Bridge origins. Supports the overlap warning and the distinction between response evaluation and navigation evaluation.
- **S15 Potential based reward shaping.** [Ng Harada and Russell 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf). Source for the potential difference shaping formulation and its conditions in the Markov setting. No guarantee for this approximate learner model is inferred.

## 26 Implementation checklists

### 26.1 Before either owner trains a model

- [ ] Ownership map and contract file owners are committed.
- [ ] Schema fixtures cover first turn, all eight moves, six navigation operations, end conditions, and errors.
- [ ] Feature width is exactly 140; action encodings and masks round trip.
- [ ] Original content has verified solvers and a separate assessment bank.
- [ ] Source, problem family, dialogue, and scenario splits are frozen and hashed.
- [ ] Private simulator data is absent from actor, critic, extractor, and response prompts.
- [ ] Assisted answers cannot count as independent evidence.
- [ ] Duration aware returns pass hand calculations and terminal tests.
- [ ] Rules baseline and response stub are versioned and runnable independently.
- [ ] CPU smoke run completes without a cloud account or gated data.

### 26.2 Before final evaluation

- [ ] Final response artifact and both policy checkpoints are frozen.
- [ ] Data use profile and artifact lineage are declared.
- [ ] Simulator assumptions, calibration status, and OOD configurations are documented.
- [ ] Final metrics, thresholds, seeds, and analysis code are preregistered.
- [ ] Evaluation sources are deduplicated against SFT and imitation inputs.
- [ ] Every failure, dropout and early stop remains in the outcome denominator.
- [ ] Structured and real text results are reported separately.
- [ ] Response quality, policy decisions, simulated outcomes, and human claims remain distinct.

### 26.3 Definition of the final handoff

The handoff is complete when a fresh contributor can install from a lock, validate the contracts, run the CPU smoke demo, replay one seeded episode, generate a constrained response from a plan fixture, inspect one SFT example and its loss mask, train a small policy experiment, reproduce the evaluation report from saved artifacts, and identify every component's accountable owner without referring to an earlier chat.

The first implementation action is the contract and fixture freeze. A then builds response realization against B's plans; B builds learning behavior against A's response stub. Training follows validated semantics, not the other way around.

[S1]: https://huggingface.co/datasets/eth-nlped/mathdial
[S2]: https://arxiv.org/abs/2305.14536
[S3]: https://huggingface.co/datasets/ASSISTments/FoundationalASSIST
[S4]: https://huggingface.co/datasets/Eedi/Eedi-Misconceptions-Graph
[S5]: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
[S6]: https://www.sciencedirect.com/science/article/pii/S0004370299000521
[S7]: https://arxiv.org/abs/1707.06347
[S8]: https://huggingface.co/docs/trl/en/sft_trainer
[S9]: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
[S10]: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
[S11]: https://huggingface.co/datasets/ScaleAI/TutorBench
[S12]: https://www.eedischool.com/projects/neurips-education-challenge
[S13]: https://www.eedi.com/data-and-competitions
[S14]: https://sig-edu.org/sharedtask/2025
[S15]: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
