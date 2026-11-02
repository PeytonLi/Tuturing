# Person B: what to do

You own deciding what to teach. That covers the content, the student belief, the rules Navigator, the learned Teaching Strategist, the simulator, the reward and the RL evaluation. The full spec is the PRD. This page is the working checklist for the current plan, which is to learn the Strategist only while the Navigator stays as rules.

## How your training data works

You don't download a dataset for RL. The Strategist learns by running thousands of simulated sessions against the student simulator in packages/simulator. Each session creates one training record per teaching turn. PPO learns from fresh sessions, then throws them away and collects more. One seed of 2,500 sessions gives roughly 25,000 to 40,000 Strategist decisions, all on a CPU.

MathDial is not part of your RL. Its only possible role is a Strategist head start, and only if the move mapping audit agrees on at least 80 percent of turns (see docs/decisions/D09_mathdial_move_mapping.md).

## What is already set up in this repo

- Contracts, plus valid and invalid examples, in contracts/.
- Five prototype concepts (C01, C02, C03, C06, C08), 8 misconceptions, and 35 problem families with exact solvers. Each concept has 4 teaching families and 3 assessment families.
- The BKT belief tracker, the 140-number observation, action encodings and masks.
- The rules Navigator, the rules Strategist and the boundary scheduler.
- The simulator, with 3 student profiles and 30 scenarios split 20 train, 5 validation and 5 test. All its parameters are labeled assumptions in configs/learning/simulator_proto_v1.yaml.
- The reward from PRD 16.1 in configs/learning/reward_v1.yaml.
- The collector, which turns sessions into transition records.
- PPO, behavior cloning from the rules Strategist, the evaluation tool, and the decision case scorer.

## Your checklist

1. Run the smoke commands in the README and read one episode's transitions end to end.
2. Review the simulator assumptions. Each one is a guess. Change the ones you disagree with before training anything, and write down why.
3. Review the 30 decision cases in evaluation/learning/decision_cases_v1.jsonl. Fix any acceptable or forbidden list you disagree with. This is your first benchmark.
4. Score random, the rules Strategist, the small model (Qwen 3.8 27B, accounts/fireworks/models/qwen3p8-27b on Fireworks) and a frontier model on the decision cases with tools.learning.score_decisions. Point TUTOR_LLM_BASE_URL, TUTOR_LLM_API_KEY and TUTOR_LLM_MODEL at an OpenAI-compatible endpoint. For Fireworks, the base URL is https://api.fireworks.ai/inference/v1.
5. Run the sanity checks: always-HINT must not beat the rules Strategist on every profile, and random should be clearly worse. If either fails, the simulator gives away the answer by construction. Fix the simulator, not the policy.
6. Train: behavior cloning from the rules Strategist first, then PPO for 2,500 sessions per seed with 5 seeds. Also train one from scratch.
7. Evaluate random, rules, behavior cloning only, PPO, and the LLM Strategists on the validation split. Keep the test split sealed until everything is frozen. Then run it once.
8. Run a second simulator setting, with different learning dynamics or profile ranges, to check the result isn't a quirk of one simulator.
9. Write the outcome report: gain with confidence intervals, disclosures, dropout, and turns used, plus the simulator's limitations. Label every result as simulated.

## What you owe Person A

- Done: the internal learning service hand-off, with startup and endpoint examples in [the learning service guide](learning_service.md).
- Valid teaching plan examples for every move, with content A can render.
- Problem templates, hint ladders, worked examples and approved facts for every enabled concept. A's original SFT examples and math checks depend on these.
- Once PPO runs, which moves and paces the Strategist picks most, so A can make sure those have enough SFT examples.
- Hand-labels for the 200-turn MathDial mapping audit, and your approval of what each mapped move means. A owns and maintains the mapping guide. You don't write it, but nothing mapped from MathDial is used to imitate teaching decisions until you approve it.

## Later, not now

The learned Navigator (PRD 17.4, stage two) and the full 15-concept curriculum.
