# Where the training data comes from, and whether MathDial fits

This note is for the whole team. It explains how training data for SFT and for RL is produced under the PRD, checks the idea of "MathDial plus a Curriculum Navigator", and lays out the options with their tradeoffs. Numbers about MathDial below were counted from the published files (revision acc3878) on November 2, 2026. Everything else is a proposal.

## 1. The short version

SFT and RL get their data in completely different ways.

SFT (Person A) learns from a fixed dataset. Each example says "here is the conversation so far and the teaching plan, write this tutor reply". MathDial is a good source for this, which is already the PRD's plan.

RL (Person B) does not learn from a dataset. The Navigator and Strategist learn by running thousands of practice sessions against a simulated student that Person B builds. The "training data" is generated on the fly: every simulated session produces navigator and strategist transitions, and PPO learns from them, then throws them away and collects fresh ones. What Person B actually has to build is the content (probability problems, solvers, misconceptions, a concept graph), the student simulator, and the reward.

So "use MathDial and add the curriculum navigator" only half makes sense. MathDial can help the language model and can give the Strategist a head start, but it cannot train the Navigator, because nothing in MathDial is a navigation decision.

## 2. How each piece is trained

| Component | Owner | What it learns | Where its data comes from | Method |
|---|---|---|---|---|
| Response model | A | How to say a given teaching move well | Fixed examples: original probability replies plus filtered MathDial teacher turns | LoRA SFT on Qwen2.5 7B, loss only on the tutor reply |
| Teaching Strategist | B | Which of 8 moves, 3 difficulties and 3 paces to use each turn (72 actions) | Simulated sessions, optionally a warm start from mapped MathDial moves | PPO, optional behavior cloning first |
| Curriculum Navigator | B | Which concept to work on next: stay, advance, skip, review a prerequisite, return, or end (76 actions) | Simulated sessions only | PPO with duration aware returns |
| Student simulator | B | Not trained. Hand written rules for hidden mastery, misconceptions, guessing, slipping, fatigue | PRD section 15 assumptions, optionally calibrated on real student logs | Code plus calibration |

### How one RL training session is generated

1. Pick a scenario: a start concept, a goal concept such as Bayes' theorem, and a student profile such as "novice missing prerequisites". Sample the hidden student state from that profile.
2. Run a fixed pre test on the hidden state.
3. Navigator picks a concept option. Strategist picks a move, difficulty and pace. Content code picks a verified problem.
4. The simulator decides whether the student answers correctly, updates hidden mastery using the teaching effect rules, and produces an observable answer.
5. The belief estimator (BKT) updates what the tutor thinks the student knows, using only the observable answer.
6. Repeat until 16 turns, a stop, a dropout, or END_SESSION. Run the post test. Reward is mostly 4 times the post minus pre gain, minus small costs per turn, for disclosure, for repetition, and for dropout.
7. Store one strategist transition per turn and one navigator transition per option (each option lasts 1 to 4 turns).

The PRD budget is 10,000 sessions per training seed, which is roughly 100,000 strategist and 40,000 navigator decisions. These come from the simulator, not from any dataset. It all runs on CPU with a templated response stub, so RL training doesn't need the GPU.

### How one SFT example is generated from MathDial

1. Split by problem (qid) first, so the same word problem never ends up in both train and test.
2. Split each dialogue at every teacher turn. The input is everything before that turn plus a teaching plan, and the target is that single teacher reply. Never include later turns or the final outcome.
3. Map the teacher's label to one of our 8 moves using the written mapping guide. Mark these labels as "mapped". Drop turns that are ambiguous.
4. Check the math, check that the reply doesn't reveal the answer when it isn't allowed, and deduplicate against benchmarks.

## 3. What MathDial actually contains

| Fact | Value |
|---|---|
| Dialogues | 2,861 (2,262 train, 599 test) |
| Unique word problems | 1,111, from GSM8K, 1 to 8 dialogues each |
| Problems that appear in both the official train and test split | 318 of the 394 test problems |
| Teacher turns | 18,609, about 6.5 per dialogue |
| Teacher move labels | focus 6,808, generic 4,456, probing 4,252, telling 3,091 |
| Final outcome label | solved 2,129, solved after answer revealed 397, not solved 322, missing 13 |
| Problems that mention probability or chance | about 10 |
| Who the students are | Large language models role playing 7th graders with a given confusion, tutored by real teachers |
| License | CC BY 4.0 |

What this means for us:

1. It is grade school arithmetic word problems, not probability. Our curriculum is probability with Bayes' theorem as a goal.
2. Each dialogue is one problem. There is no sequence of concepts, no prerequisite graph, and no point where a teacher chooses what topic to study next. That is exactly what the Navigator needs, and it is missing.
3. The students are language models, not real learners, and there is no pre and post test. The outcome label says whether this one problem got solved in this conversation, not whether the student learned anything.
4. It has only 4 teacher move labels, and "generic" is mostly praise and filler. Our Strategist has 8 moves plus difficulty and pace. Mapping is lossy: probing roughly becomes PROBE or REFLECT, focus roughly becomes HINT, telling roughly becomes EXPLAIN_CONCEPT or WORKED_EXAMPLE, and there is no clean source for PRACTICE, CHALLENGE or CHECK_MASTERY.
5. The official split leaks problems between train and test. We must regroup by qid before using it.

## 4. The options

### Option 1. Follow the PRD as written (recommended)

MathDial goes into A's SFT mix. B writes original probability content and trains both policies with PPO in the structured simulator.

Good: runnable without any gated data or GPU, fully under our control, answers are verifiable, matches the probability domain, and the PRD is already written around it.

Bad: results only say something about our simulator, whose teaching effects are our assumptions. Writing 165 problem families with solvers and hint ladders is the real bottleneck for B. The five concept prototype with 15 templates is the realistic first target.

### Option 2. Option 1, plus a Strategist warm start from MathDial

Before PPO, pretrain the Strategist by imitating mapped MathDial teacher moves (behavior cloning), then fine tune with PPO in the simulator.

Good: the Strategist starts with human-looking habits such as probing before telling, which may make early PPO less random. It reuses mapping work A already has to do for SFT.

Bad: the observation the Strategist sees (140 numbers from our belief state) doesn't exist in MathDial, so we'd have to reconstruct it roughly from the transcript. No difficulty or pace labels exist. The domain is different. It must be reported as a separate ablation, and it may not help at all compared with PPO from scratch. The Navigator still gets nothing.

### Option 3. Switch the whole project to arithmetic so MathDial is in domain

Build a skill graph over GSM8K style skills (fractions, percents, ratios, multi step) and run the Navigator over that.

Good: SFT data would match the domain, and there would be much more of it.

Bad: GSM8K problems are fixed word problems that mix several skills, not parameterized templates with clean concepts, so we'd still have to author templates and solvers. There is still no navigation data. It reverses PRD decision D02 and rewrites sections 5, 15 and 18. A lot of rework for little gain on the RL side.

### Option 4. Offline RL directly from MathDial outcomes

Treat the solved or not solved label as reward and learn a policy from the logged dialogues without a simulator.

Good: no simulator assumptions.

Bad: only about 2,800 short episodes, each with a single problem. The students are language models. Teachers chose moves based on what they saw, so the data mixes cause and effect. Revealing the answer often leads to "solved", which teaches the wrong lesson. The PRD explicitly forbids reading learning effects off teacher move labels. Not recommended.

### Option 5. Language model students instead of the structured simulator

Copy MathDial's own trick: let an LLM play the student during RL.

Good: more realistic text and more varied confusions.

Bad: an LLM call per turn for 100,000 or more turns per seed is slow and costly. Hidden mastery is undefined, so reward becomes "the LLM said it got it". The PRD says the text renderer may not decide learning. Better used as a small stress test after training, and MathDial's teacher_described_confusion and student_profile fields are good seed material for writing those student personas.

### Comparison

| Option | Helps SFT | Helps Strategist | Helps Navigator | Extra work | Risk |
|---|---|---|---|---|---|
| 1. PRD as written | Yes | Through simulator | Through simulator | Content authoring | Simulator assumptions |
| 2. Plus BC warm start | Yes | Some head start | No | Mapping and an ablation | Lossy labels, domain shift |
| 3. Switch to arithmetic | Strongly | Through simulator | Through simulator | Large rewrite | Messy skills, no gain for RL |
| 4. Offline RL on MathDial | No | Weak and biased | No | Medium | Confounded, misleading |
| 5. LLM student simulator | No | Realistic text | Realistic text | High compute | Reward is not grounded |

## 5. If the team wants real data for the Navigator

The right kind of data is long sequences of real students answering problems across many skills, which is knowledge tracing data, not tutoring dialogue. FoundationalASSIST (gated, non commercial) and ASSISTments 2009 to 2010 are listed in the PRD for this. They can calibrate the simulator's guess, slip, learning rates and prerequisite effects, which is what makes Navigator training meaningful. They are also not probability, so the calibration has to be labeled as a domain transfer, and it stays optional.

## 6. What to do now

Person B:
1. Build the five concept probability graph, eight misconceptions, five verified problem templates with solvers, and the BKT belief update (PRD 21.2).
2. Build the rules Navigator and Strategist first. Every learned policy is judged against them.
3. Build the structured simulator, the reward, and the duration aware return tests, then run 100 smoke sessions.
4. Train the Strategist with PPO against the rules Navigator, then the Navigator against the frozen Strategist.

Person A:
1. Regroup MathDial by qid into train, validation and test before splitting into turns.
2. Write the four to eight move mapping guide with B, label about 200 turns by hand, and measure how often A and B agree. Low agreement means drop behavior cloning.
3. Build SFT examples from MathDial plus the original probability replies, at the PRD's 60 original to 40 public mix.

Decision for the team: do we try the behavior cloning warm start (option 2)? I'd decide after the 200 turn mapping check, since that tells us whether the labels are clean enough to imitate.
