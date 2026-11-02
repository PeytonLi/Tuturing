## Notes from the 2026-11-02 run (hand-written; included verbatim by `benchmark report`)

**Scope.** Xueming chose `gpt-6.1-sol` as the frontier model and `Qwen3.8-27B` as the open
model; the cheaper OpenAI tier, other Fireworks serverless models and new local downloads
were dropped. The deterministic fallback is the floor. Anthropic and Google entries are
configured with verified prices and wait for keys.

**gpt-6.1-sol.** Ran on all 107 cases (0 errors) through the OpenAI Chat Completions API
with `reasoning_effort: low` (the lowest level its model page lists) and
`max_completion_tokens: 1024`. A probe with `temperature: 0` returned HTTP 400
("Unsupported value: 'temperature' does not support 0 with this model. Only the default (1)
value is supported."), so sampling ran at the default temperature and is not deterministic.
Reasoning tokens are recorded separately (mean 3.7 per case, maximum 102); no visible reply
exceeded the PRD 14.1 cap of 384 tokens and no length or question-count check failed.

**Qwen3.8-27B (D10 base) on a Fireworks on-demand deployment.** `GET
/v1/accounts/fireworks/models/qwen3p8-27b` reports `supportsServerless: false`,
`supportsLora: true`, `supervisedLoraTunable: true` and `supportsServerlessTraining: false`,
so the model needs a dedicated deployment to benchmark and to serve (and D10's "serverless
training pool" claim deserves a check against that last flag). The validated deployment
shapes for it are 1x H100 80 GB BF16 (`accounts/fireworks/deploymentShapes/sva2n3mh`,
context 131072), 1x B200 BF16 and 2x B300 BF16; the entry uses the 1x H100 shape at the
list price of $8.00 per GPU hour with scale-to-zero after 5 minutes and a 75-minute hard
limit ($10 reserve). A first attempt on account `accounts/xueming-r9e8nrrxg6ml` stopped at
the free dry run with HTTP 400 `code 9: "payment method is required"` and created nothing.
With a key for `accounts/peli` the dry run passed and the harness created
`accounts/peli/deployments/tut-bench-qwen38-20261102-213251` (1x NVIDIA_H100_80GB, BF16,
shape version `k9aui8v2`, GLOBAL placement, min 0 / max 1 replicas, 300 s scale-to-zero) at
21:32:51Z; a replica served the warm-up request at 21:35:28Z; the 3-case pilot and the
other 104 cases ran at concurrency 4 (0.6 s per case, 0 errors); the deletion was requested
at 21:36:11Z (the first DELETE returned 400 while replicas were active, the retry with
`ignoreChecks` returned 200) and verified at 21:36:42Z by GET (`state: DELETED`,
`deleteTime 21:36:11Z`) and by a `showDeleted` listing. Billable upper bound: 3.8 minutes
of create-to-verified-deletion wall time, $0.51 at list price. Decoding was
`temperature 0`, `max_tokens 384`, `reasoning_effort: none`; no `<think>` text appeared
and no reasoning tokens were reported.

**What the automated checks missed (human read of the fixture cases).** On B's two
CHECK_MASTERY fixtures and the CHALLENGE fixture, Qwen3.8-27B replaced the verified coin
problem with a six-sided die of its own ("Consider a standard six-sided die. List the
complete sample space...") and every automated check passed, because the coin stem has no
digits for `action_faithful` or `math_supported` to anchor on. That is a protected-check
item substitution, which PRD 14.1 treats as critical, and it reached delivery mode
`model`. On the smoke CHALLENGE and WORKED_EXAMPLE it also invented marble problems, which
the number check did catch. gpt-6.1-sol kept the issued problem in all 17 fixture and smoke
cases. The validator needs a stem-overlap check that does not depend on digits before any
prompt-only model is trusted on a protected check; until then the review sheet is the
only measure of this failure.

**Reading the automated results.** Two of the three strict checks over-report on dialogue
data: `math_supported` accepts only numbers from the plan's approved content, so a tutor
who echoes the learner's own intermediate value fails it, and `disclosure_allowed` flags an
answer fingerprint even when the problem stem or the learner already stated that number.
The per-model diagnostics table counts both situations. `action_faithful` requires a PROBE
to restate every number in the stem, which diagnostic questions about one step rarely do.
Treat "delivered as model" as a lower bound on acceptable responses and the review sheet as
the way to measure the gap; a human read of the quoted failures is in the final report.

**Case sets.** B's 9 valid plan fixtures and the 8 smoke cases are tiny and all about one
coin problem. The MathDial sample is 30 turns per mapped move (90 of 1,618 test turns, at
most one turn per problem and move); its plans carry no approved hint, facts or example.
`evaluation/response/probability_cases_v1.jsonl` did not exist when this ran; it is listed
in the config and will be included by a rerun without a code change.
