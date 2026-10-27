"""Portable calibration, offline selection, and frozen transfer evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from .action_space import action_tags, fingerprint
from .backends import make_backend, non_thinking_options
from .data import atomic_json, check_disjoint, digest, item_hash, load_items
from .planning import (CALIBRATION_SCHEMA, select_portfolio, validate_portfolio,
                       valid_choice)
from .reasoning import Reasoning
from .runner import Runner, prepare_run, run_lock, summarize


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def budgets(value):
    try:
        values = sorted(set(int(k) for k in value.split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError("Use comma-separated budgets") from None
    if not values or values[0] < 1 or values[-1] > 66:
        raise argparse.ArgumentTypeError("Budgets must be between 1 and 66")
    return values


def add_backend(parser):
    parser.add_argument("--backend", choices=("vllm", "hf", "api", "mock"), default="vllm")
    parser.add_argument("--model", required=True, help="Local checkpoint path, HF model ID, or API model ID")
    parser.add_argument("--revision", help="Pin a Hugging Face model revision")
    parser.add_argument("--seed", type=int, default=43, help="Dataset sampling seed")
    parser.add_argument("--sample-size", type=int, default=0, help="0 = all input examples")
    parser.add_argument("--inference-seed", type=int, default=0, help="Historical vLLM default")
    parser.add_argument("--batch-size", type=positive, default=224)
    parser.add_argument("--max-tokens", type=positive, default=256)
    parser.add_argument("--note-token-limit", type=positive, default=128)
    parser.add_argument("--max-model-len", type=positive, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.85)
    parser.add_argument("--tensor-parallel-size", type=positive, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--api-url", default="", help="Full chat/completions endpoint")
    parser.add_argument("--api-key-env", default="BLINDFLIP_API_KEY")
    parser.add_argument("--api-auth", choices=("bearer", "raw"), default="bearer")
    parser.add_argument("--api-workers", type=positive, default=8)
    parser.add_argument("--api-timeout", type=positive, default=120)
    parser.add_argument("--api-attempts", type=positive, default=3)
    parser.add_argument("--api-options", default="{}", help="Non-thinking options as a JSON object")


def backend_config(args):
    options = json.loads(args.api_options)
    allowed = {"reasoning", "thinking", "enable_thinking", "reasoning_effort", "chat_template_kwargs"}
    if not isinstance(options, dict) or not set(options) <= allowed:
        raise ValueError("API options must contain only reasoning/chat-template controls")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("GPU memory utilization must be in (0, 1]")
    if args.backend == "api":
        url = urlsplit(args.api_url)
        if (url.scheme not in ("http", "https") or not url.netloc or url.username or
                url.password or url.query or url.fragment):
            raise ValueError("API URL must be a credential-free HTTP(S) endpoint")
    keys = ("backend", "model", "revision", "inference_seed", "max_tokens", "max_model_len",
            "gpu_memory_utilization", "tensor_parallel_size", "dtype", "trust_remote_code",
            "api_url", "api_key_env", "api_auth", "api_workers", "api_timeout", "api_attempts")
    config = {key: getattr(args, key) for key in keys}
    config["api_options"] = {**non_thinking_options(args.model), **options}
    config["model_metadata_sha256"] = {}
    if args.backend == "vllm" and Path(args.model).is_dir():
        for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
            path = Path(args.model) / name
            if path.exists():
                config["model_metadata_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def run(args):
    if args.command == "actions":
        for tag in Reasoning.actions():
            print(json.dumps({"tag": tag, "family": Reasoning.family(tag),
                              "note": Reasoning.instantiate(tag, args.target)}, ensure_ascii=True))
        return
    if args.command == "select":
        matrix = json.loads(Path(args.matrix).read_text())
        portfolio = select_portfolio(matrix, args.budget)
        portfolio["calibration_matrix_sha256"] = digest(matrix)
        atomic_json(args.out, portfolio)
        print(json.dumps(portfolio["selected_actions"]))
        return

    items = load_items(args.data, args.sample_size, args.seed)
    config = backend_config(args)
    context = {"phase": args.command, "sampling_seed": args.seed, "sample_size": args.sample_size,
               "batch_size": args.batch_size, "note_token_limit": args.note_token_limit}
    portfolio = None
    if args.command == "calibrate":
        tags = action_tags()
    else:
        portfolio = json.loads(Path(args.portfolio).read_text())
        validate_portfolio(portfolio)
        context["split_audit"] = check_disjoint(items, portfolio, args.allow_content_overlap)
        if context["split_audit"]["content_overlap_count"]:
            print("Historical replay: calibration/test content overlaps; see manifest split_audit.", flush=True)
        tags = portfolio["ordered_actions"][:max(args.budgets)]
        context.update(portfolio_sha256=digest(portfolio), budgets=args.budgets)
    out = Path(args.out)
    with run_lock(out):
        prepare_run(out, items, tags, config, context)
        runner = Runner(items, make_backend(config), out, args.batch_size, args.note_token_limit)
        clean = runner.evaluate("clean")
        if not any(valid_choice(c) for c in clean.values()):
            raise ValueError("No valid clean judgments; inspect clean.json and responses.jsonl")
        hits = runner.run_actions(tags, clean)
        cost = runner.cost(tags)
        n = sum(valid_choice(c) for c in clean.values())
        if args.command == "calibrate":
            matrix = {"schema": CALIBRATION_SCHEMA, "simulation": args.backend == "mock",
                      "action_space_sha256": fingerprint(), "source": config,
                      "clean": clean, "hits": hits,
                      "calibration_item_hashes": [item_hash(i) for i in items],
                      "calibration_cost": {**cost, "planned_attack_queries": len(tags) * n}}
            atomic_json(out / "matrix.json", matrix)
            print(f"Calibration saved: {out / 'matrix.json'}", flush=True)
        else:
            simulation = args.backend == "mock" or portfolio.get("simulation", False)
            summary = summarize(clean, hits, tags, args.budgets, cost, simulation)
            summary["calibration_cost"] = portfolio.get("calibration_cost", {})
            summary["source"] = portfolio.get("source", {})
            summary["target"] = config
            summary["split_audit"] = context["split_audit"]
            atomic_json(out / "summary.json", summary)
            for row in summary["rows"]:
                print(f"ASR@{row['budget']}={row['asr']:.2%} ({row['successes']}/{row['eligible']}) "
                      f"AvgQ={row['avg_test_q_per_success']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    actions = subs.add_parser("actions", help="List the exact 66 candidate-bound notes")
    actions.add_argument("--target", choices=("A", "B"), default="B")
    selection = subs.add_parser("select", help="Offline selection; no model required")
    selection.add_argument("--matrix", required=True)
    selection.add_argument("--budget", type=positive, default=2)
    selection.add_argument("--out", required=True)
    for command in ("calibrate", "evaluate"):
        sub = subs.add_parser(command)
        sub.add_argument("--data", required=True)
        sub.add_argument("--out", required=True)
        add_backend(sub)
        if command == "evaluate":
            sub.add_argument("--portfolio", required=True)
            sub.add_argument("--budgets", type=budgets, default=[2])
            sub.add_argument("--allow-content-overlap", action="store_true",
                             help="Explicit historical replay only; overlaps remain reported")
    args = parser.parse_args(argv)
    try:
        run(args)
    except BrokenPipeError:
        # Allow normal shell consumers such as ``blindflip actions | head``.
        return
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
