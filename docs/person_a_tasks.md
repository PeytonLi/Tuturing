# Person A: what to do

You own how the tutor says things. That covers the response model and its SFT, the prompt compiler, the checks and fallbacks, the evidence extractor, serving, and the public API. You never decide what to teach. You realize the teaching plan Person B sends. The full spec is the PRD. This page is the working checklist.

## How your training data works

SFT learns from fixed examples. Each one is the conversation so far plus a teaching plan, followed by one good tutor reply, with the loss on the reply only. You have two sources:

1. Original probability examples you write from Person B's content. The PRD's target is 300 to start, growing toward 3,000. Aim for 60 percent of the mix.
2. MathDial teacher turns, for 40 percent of the mix. The teachers are real people, but the students are language models, so student messages there are cleaner than real ones.

## The MathDial move mapping is yours

You own the guide that maps MathDial's 4 teacher labels to our 8 moves (PRD 12.2). Person B approves what each mapped move means before it is used to imitate teaching decisions. The draft mapping is in configs/response/mathdial_move_map_v1.yaml, and the reasoning is in docs/decisions/D09_mathdial_move_mapping.md:

- probing becomes PROBE
- focus becomes HINT
- telling becomes EXPLAIN_CONCEPT
- generic is dropped, with the reason generic_act

This mapping is a draft until the 200-turn audit is done. Drop any turn that doesn't clearly fit a move instead of forcing a label. Keep mapped labels separate from hand labels. Never use a mapped label as a reward or as proof that a student learned something.

## What is already set up in this repo

- Contracts, plus valid and invalid examples, in contracts/.
- The response stub that B's RL uses (packages/response/stub.py), and the content hash.
- The prompt compiler, which leaves out IDs, hashes, fingerprints and hidden solutions.
- The six response checks and a verified fallback reply for every move.
- A rule-based evidence extractor stub.
- MathDial prep. It is pinned to one revision, regrouped by problem because the official split leaks 318 of 394 test problems into train, and split into one example per teacher turn with past context only. It writes a provenance manifest.
- A 200-turn mapping audit sheet at evaluation/response/mathdial_mapping_audit_200.csv.
- The SFT config from PRD 13.4, a trainer script with a config check and a loss-mask inspector, and a response benchmark tool.

## Your checklist

1. Run the MathDial prep and read 20 examples end to end. Make sure no example shows the future, the answer key or the success label.
2. Keep the mapping guide current, with ambiguous examples. Fill in label_person_a in the 200-turn audit, and have B fill label_person_b. Compute agreement. If it's under 80 percent, MathDial is only used for SFT wording and never for the Strategist. B signs off on the move meanings either way.
3. Write the first 30 original probability examples across all 8 moves, with B's solver checking the math. Then grow toward 300. Include messy student messages: "idk", bare numbers, typos, skip requests, asking for the answer, and attempts to override the plan.
4. Benchmark before training: run tools.response.evaluate with the small model (Qwen 3.8 27B) and a frontier model, prompt only, on B's plan examples, the MathDial test split, and your probability cases. If the frontier model is much better, you can use it to draft examples, flagged as synthetic and reviewed.
5. The SFT base model is Qwen 3.8 27B: Qwen/Qwen3.8-27B on Hugging Face, accounts/fireworks/models/qwen3p8-27b on Fireworks. It's in Fireworks' serverless training pool, so LoRA SFT bills per token with no GPU to rent. Qwen3.5 9B isn't. Pin the base model revision in configs/response/sft.yaml. The config check fails until you do. Run the loss-mask inspector on 20 examples.
6. Run a 50 to 100 step smoke SFT first, then the real run with at most 3 epochs. Pick the checkpoint by zero math and disclosure failures first, then action fidelity.
7. Build the evidence extractor beyond the stub, and measure it on real-looking messy messages.
8. Freeze and serve the response model. Then run the text bridge test with B: the same sessions with the stub and with the real model, compared.

## What you owe Person B

- A frozen response stub, already done, and later the frozen SFT artifact with its manifest.
- Evidence objects from student messages. B's grader double-checks any number.
- Notes on any plan field you can't render faithfully. Fix those in the contract, never by having the model improvise.

## Shared, with B

The 200-turn mapping audit, the first integration review (PRD 21.3), and the text bridge test.
