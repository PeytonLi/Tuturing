# Blinded response review

`review_sheet_blinded.csv` holds 40 cases and 62 responses. Within each item the responses are shuffled and labelled R1, R2, ...; the mapping to models is in `review_key_do_not_open_before_rating.csv` and nowhere else in the sheet. Fill the `*_yn`, `tone_1to5` and `notes` columns per response (PRD 13.3 criteria: mathematical correctness, action fidelity, disclosure permission, relevance and a clear next step), then join on item and response_label.

Responses per model (seen only after unblinding): 22, 40 across 2 models.

## Supplement added 2026-11-02T21:37:54+00:00

`review_sheet_blinded_supplement.csv` adds 40 responses from 1 model(s) that ran after the sheet above was published, for the same items and columns, each under the next unused label of its item (R2 or R3); the mappings were appended to the key. A rater can tell these rows were added later, which weakens the blinding between the added model(s) and the earlier ones but not among the earlier ones.
