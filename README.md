# BlindFlip

BlindFlip is a restricted-access preference-reversal attack for pairwise LLM
judges. It constructs reusable source-note actions without inspecting the
underlying comparison, calibrates them once with a surrogate judge, and deploys
a frozen mechanism-diverse portfolio to target judges.

## Architecture

- `reasoning.py`: Factorized Semantic Action Space (F-SAS). Fixed semantic
  directions are combined with linguistic wrappers and labelled as
  `Priority`, `Eligibility`, or `Quality` mechanisms.
- `tool_use.py`: executes only through the source-note field and binds each
  action to the symbolic opposite of the clean preference.
- `memory.py`: stores clean-valid calibration examples and per-action success
  sets, exposing empirical reliability.
- `planning.py`: MATADOR selects a frozen portfolio by maximizing additive
  reliability while covering mechanism families.

The candidate action space contains 66 fixed actions. MATADOR preserves the
deterministic ordering required for nested query budgets: for budgets up to
three it takes the best representatives from distinct families; afterwards it
fills the ranking by reliability. Ties are resolved by descending action tag.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

For local GPU inference, install the optional dependencies in a compatible CUDA
environment:

```bash
python -m pip install -e '.[local]'
```

## Offline Planning

```bash
blindflip select --matrix examples/qwen2b_calibration.json --budget 2 \
  --out outputs/qwen2b_portfolio.json
blindflip actions --target B
```

The output portfolio is frozen before evaluation. A target judge is never used
to revise action identities or ordering.

## Metrics

ASR@K is the fraction of clean-valid comparisons reversed by at least one of
the first K frozen actions. All K actions are evaluated; there is no early
stopping. `avg_test_q_per_success = K * eligible / successes` allocates the
fixed attack budget across successful examples. Clean and calibration queries
are reported separately.

Each run produces a manifest, clean choices, request/response logs, and a
summary. Calibration additionally produces `matrix.json`; planning produces a
portfolio that records the full frozen action order.

## Layout

```text
src/blindflip/action_space.py  F-SAS action definitions and rendering
src/blindflip/reasoning.py     Reasoning module
src/blindflip/tool_use.py      Restricted source-note execution
src/blindflip/memory.py        Surrogate calibration memory
src/blindflip/planning.py      MATADOR and frozen portfolio validation
src/blindflip/runner.py        Resumable evaluation and accounting
src/blindflip/backends.py      vLLM, Transformers, API, and mock backends
data/                          Processed calibration/test inputs and audits
examples/                      Calibration matrix and frozen portfolio
tests/                         CPU regression and integration tests
```
