"""Resumable clean judgments and frozen action evaluation."""

import contextlib
import fcntl
import importlib.metadata
import json
import platform
import threading
from pathlib import Path

from .action_space import fingerprint
from .data import atomic_json, digest
from .planning import valid_choice
from .protocol import JUDGE_SYSTEM_PROMPT, PairwiseExample, build_judge_user_prompt, parse
from .tool_use import ToolUse


def read_jsonl(path, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    rows, offset = [], 0
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt JSONL at {path.name}:{index + 1}") from None
            if repair:
                with path.open("r+b") as handle:
                    handle.truncate(offset)
            break
        offset += len(line)
    if repair and content and len(rows) == len(lines) and not content.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return rows


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()


@contextlib.contextmanager
def run_lock(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is using this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def code_fingerprint():
    return digest({p.name: p.read_text(encoding="utf-8")
                   for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    packages = {}
    for name in ("blindflip", "tiktoken", "vllm", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "packages": packages}


def prepare_run(out, items, tags, config, context):
    path = Path(out) / "manifest.json"
    identity = {"protocol": "blindflip.static.v1", "dataset_sha256": digest(items),
                "action_space_sha256": fingerprint(), "code_sha256": code_fingerprint(),
                "actions": tags, "backend": config, "context": context,
                "environment": environment()}
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Resume configuration/data/code/environment mismatch; use a new output directory")
    else:
        if (Path(out) / "responses.jsonl").exists():
            raise ValueError("Responses exist without a matching manifest")
        atomic_json(path, identity)


class Runner:
    def __init__(self, items, backend, out, batch_size=224, note_token_limit=128):
        import tiktoken
        self.items, self.backend, self.out = items, backend, Path(out)
        self.batch_size, self.note_token_limit = batch_size, note_token_limit
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.path = self.out / "responses.jsonl"
        self.cache = {}
        for row in read_jsonl(self.path, repair=True):
            key = (row["tag"], row["item_id"])
            if key in self.cache:
                raise ValueError("Duplicate completed response in cache")
            self.cache[key] = row
        read_jsonl(self.out / "requests.jsonl", repair=True)
        self.lock = threading.Lock()

    def _remember(self, row):
        append_jsonl(self.path, row)
        self.cache[(row["tag"], row["item_id"])] = row

    def evaluate(self, tag, clean=None):
        jobs = []
        eligible = self.items if clean is None else [
            item for item in self.items if valid_choice(clean.get(item["question_id"]))]
        for item in eligible:
            iid = item["question_id"]
            if (tag, iid) in self.cache:
                continue
            note = "" if clean is None else ToolUse.source_note(tag, clean[iid])
            if note and len(self.encoding.encode(note)) > self.note_token_limit:
                self._remember({"tag": tag, "item_id": iid, "raw": "", "choice": None,
                                "source_note": note, "error": "note_token_limit"})
                continue
            prompt = build_judge_user_prompt(PairwiseExample(**item, source_note=note))
            messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}]
            jobs.append((iid, note, messages))
        for start in range(0, len(jobs), self.batch_size):
            chunk = jobs[start:start + self.batch_size]

            def on_attempt(index, attempt):
                with self.lock:
                    append_jsonl(self.out / "requests.jsonl", {
                        "tag": tag, "item_id": chunk[index][0], "attempt": attempt,
                        "status": "dispatched"})

            outputs = self.backend.generate([job[2] for job in chunk], on_attempt)
            if len(outputs) != len(chunk):
                raise ValueError("Backend output count does not match input count")
            for (iid, note, _), output in zip(chunk, outputs):
                choice = parse(output["raw"])
                self._remember({"tag": tag, "item_id": iid, "source_note": note,
                                "raw": output["raw"], "choice": choice,
                                "error": output["error"] or ("invalid_verdict" if choice is None else None)})
        choices = {item["question_id"]: self.cache[(tag, item["question_id"])]["choice"]
                   for item in eligible}
        if clean is None:
            atomic_json(self.out / "clean.json", choices)
        print(f"[{tag}] completed={len(choices)} valid={sum(valid_choice(v) for v in choices.values())}",
              flush=True)
        return choices

    def run_actions(self, tags, clean):
        hits = {}
        for tag in tags:
            choices = self.evaluate(tag, clean)
            hits[tag] = sorted(iid for iid, choice in choices.items()
                               if valid_choice(choice) and choice != clean[iid])
        return hits

    def cost(self, tags):
        ledger = read_jsonl(self.out / "requests.jsonl")
        return {"clean_attempts": sum(r["tag"] == "clean" for r in ledger),
                "recorded_attack_attempts": sum(r["tag"] in tags for r in ledger)}


def summarize(clean, hits, ordered, budgets, cost, simulation=False):
    n = sum(valid_choice(v) for v in clean.values())
    eligible = {iid for iid, c in clean.items() if valid_choice(c)}
    rows = []
    for k in budgets:
        selected = ordered[:k]
        union = set().union(*(set(hits[t]) for t in selected)) & eligible
        successes = len(union)
        rows.append({"budget": k, "eligible": n, "successes": successes,
                     "asr": successes / n if n else None,
                     "planned_attack_queries": k * n,
                     "avg_test_q_per_success": k * n / successes if successes else None,
                     "selected_actions": selected})
    return {"simulation": simulation, "items": len(clean), "eligible": n,
            "clean_invalid": len(clean) - n, "rows": rows, "cost": cost,
            "query_definition": "Logical fixed-budget attack queries / successful examples; clean queries excluded.",
            "physical_cost_definition": "Recorded dispatches, including retries and unfinished requests; mock uses no model calls."}
