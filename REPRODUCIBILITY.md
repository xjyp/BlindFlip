# Reproduction protocol

## Preserved from the research implementation

- All 11 direction definitions and all 6 wrapper strings, including punctuation, Unicode and line breaks.
- The original wrapper renderer's whitespace handling and replacement order. In particular, the historical JSON wrapper can contain literal newlines inside its rendered finding; this release preserves that behavior instead of silently changing the prompt.
- Candidate binding to the opposite of each target judge's clean preference.
- The exact system prompt and source-note placement after the comparison question.
- The historical permissive verdict parser (JSON choice, then first/last-token matching). It can accept text beyond the requested label; a stricter parser would be a different evaluation protocol.
- One champion per family, then descending individual reliability; ties use descending action-tag lexicographic order.
- Fixed prefixes and union success; all selected actions are evaluated.
- Dataset sampling: sample sorted question IDs with Python Random(43), then retain original file order.
- 256 generated tokens, greedy temperature 0, top_p 1, o200k_base note cap 128. The original 132 rendered notes have at most 82 tokens under that encoding.
- Qwen3-named local checkpoints disable thinking via chat-template options. Other local models retain their checkpoint defaults, as in the original factory.

## Explicit changes for a portable release

- There are no imports from old DA-KAS/DGV runners or unrelated ablation code.
- User-supplied checkpoint paths replace machine-specific paths. The optional `--revision` pins an HF revision; local model metadata hashes are recorded, but weight files are not exhaustively hashed.
- `--trust-remote-code` is explicit. Original local runs enabled it; use the flag to match those runs when needed.
- vLLM seed is explicitly 0 (the historical factory default), separate from the dataset seed of 43. Backend/hardware/version changes can still change judgments.
- Resumption checks the data, action library, Python source, backend options and dependency versions. Do not move a live run between different checkpoint paths or installations and expect its manifest to match.
- API retries are recorded. Local inference exceptions stop the run instead of silently falling back to single-item calls with different execution conditions.
- Content-overlap checks ignore candidate order. Historical LLMBar replay requires explicit opt-in. Other bundled test splits have no comparison-content overlap with the bundled calibration set.
- Completed invalid API outputs remain cached as failures; rerunning the same command does not secretly replace them with another stochastic sample.

## Data and inferential limitations

The original filenames are experiment labels, not proof of an official benchmark split. The package preserves exactly the fields read by the evaluator. It does not reconstruct raw benchmark provenance, authorship, generation history or source licenses that were absent from the working project.

The 224 calibration examples contain 199 unique comparisons ignoring candidate order. The LLMBar test set contains 201 unique comparisons; MTBench also contains 201. Their ASR denominators remain the original item counts after clean-valid filtering. Uncertainty estimates that assume independent rows can be misleading; group related comparisons if performing inference.

LLMBar has 50 content overlaps with calibration. A strict new evaluation must select disjoint underlying comparisons before calibration, freeze the protocol, and rerun it. Merely deleting failed or duplicated test rows after inspecting outcomes would create a different protocol. Historical numbers cannot be relabeled as results on a corrected split.

Calibration costs are not test-time costs. With all 224 calibration examples clean-valid, the source attack matrix needs 14,784 logical attack judgments plus 224 clean judgments. Transfer to another target does not recalibrate that target, but still needs target clean judgments for opposite-preference binding. Budget curves share calls: evaluating through K=10 once supplies @2/@3/@5 as nested prefixes, not four independent runs.

## Validation performed for this release

The standalone tests cover family selection, descending-tag ties, invalid matrices, candidate binding, original-style parsing, union ASR, zero-success costs, content leakage checks, deterministic sampling, resume behavior, truncated-tail recovery, a full mock workflow, and HTTP retry accounting against a loopback server.

Export validation additionally compares all 132 rendered notes to the original renderer, checks the original selector against the six cached target-adaptive calibration matrices, and checks original versus extracted prompts/parsing. Archive checks scan for credentials/local infrastructure paths, reject unexpected files, test ZIP integrity and run tests from a separate extracted directory. These checks do not rerun expensive model experiments and cannot certify bitwise-identical future GPU/API outputs.
