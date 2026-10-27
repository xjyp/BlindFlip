# Exact processed inputs

Only the four fields consumed by the evaluator are included: `question_id`, `instruction`, `response_a`, and `response_b`. Task and answer text is preserved verbatim. Unused metadata and duplicate copies of answer text were removed.

| File | Items | Role |
|---|---:|---|
| llmbar_calibration_224_seed43.json | 224 | Exact seed-43 sample from the project's 896-row training pool |
| llmbar_test.json | 224 | Historical test split; content overlaps calibration |
| mtbench_test.json | 224 | Historical test split |
| alpaca_eval_test.json | 161 | Historical test split |
| arena_hard_test.json | 150 | Historical test split |

`manifest.json` records exported SHA-256 values and original input-file hashes. `split_audit.json` reports duplicates and the exact IDs of calibration/test content overlaps, treating A/B swaps as the same comparison. The names describe the original project's processed datasets, not certified official benchmark partitions.

For LLMBar, 50 of 224 test items repeat a calibration comparison: 1 in the same A/B order and 49 in reversed order. Original IDs alone did not reveal this. No content overlaps were found between the bundled calibration set and the other three test sets. Historical results should explicitly disclose this limitation.

Third-party texts retain their original rights. Raw source provenance and complete dataset license notices were not present beside these processed inputs; this export does not establish a new license for them. Verify applicable redistribution terms before publishing these data files, or distribute only the code and require users to obtain the inputs separately.
