import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from blindflip.action_space import action_tags, family, fingerprint, render
from blindflip.backends import APIJudge, MockJudge
from blindflip.memory import Memory
from blindflip.planning import CALIBRATION_SCHEMA, MATADOR, select_portfolio
from blindflip.reasoning import Reasoning
from blindflip.tool_use import ToolUse
from blindflip.cli import main
from blindflip.data import atomic_json, check_disjoint, item_hash, load_items
from blindflip.protocol import PairwiseExample, build_judge_user_prompt, parse
from blindflip.runner import Runner, prepare_run, read_jsonl, summarize

ROOT = Path(__file__).resolve().parents[1]


def matrix():
    return {"schema": CALIBRATION_SCHEMA, "action_space_sha256": fingerprint(),
            "clean": {"x": 0, "y": 1, "invalid": None},
            "hits": {tag: [] for tag in action_tags()}}


def example(iid, text="Sample instruction"):
    return {"question_id": iid, "instruction": text, "response_a": "First answer",
            "response_b": "Second answer"}


class CoreTests(unittest.TestCase):
    def test_complete_space_and_binding(self):
        self.assertEqual(len(set(action_tags())), 66)
        self.assertEqual({family(t) for t in action_tags()}, {"quality", "priority", "eligibility"})
        for tag in action_tags():
            for target in ("A", "B"):
                note = render(tag, target)
                self.assertNotIn("{t}", note)
                self.assertNotIn("{o}", note)
                self.assertNotIn("{criterion}", note)
        with self.assertRaises(ValueError):
            render(action_tags()[0], "C")

    def test_family_champions_and_descending_ties(self):
        data = matrix()
        p = "T5_direct|P4_formatonly"
        p2 = "T4_json|P3_reframe"
        q = "T1_review|Q1_correctness"
        data["hits"][p] = data["hits"][p2] = ["x", "y"]
        data["hits"][q] = ["x"]
        result = select_portfolio(data, 4)
        self.assertEqual(result["selected_actions"][:2], [p, q])
        self.assertEqual(family(result["selected_actions"][2]), "eligibility")
        self.assertEqual(result["selected_actions"][3], p2)
        self.assertEqual(len(set(result["ordered_actions"])), 66)

    def test_reference_calibration(self):
        data = json.loads((ROOT / "examples/qwen2b_calibration.json").read_text())
        result = select_portfolio(data, 3)
        self.assertEqual(result["selected_actions"], ["T1_review|P4_formatonly",
                         "T5_direct|Q4_completeness", "T2_bullets|E2_calibration"])

    def test_blindflip_modules_preserve_reference_ordering(self):
        data = json.loads((ROOT / "examples/qwen2b_calibration.json").read_text())
        reference = json.loads((ROOT / "examples/qwen2b_portfolio.json").read_text())
        plan = MATADOR.plan(Memory.from_feedback(data["clean"], data["hits"]),
                            reference["budget"])
        selected = select_portfolio(data, reference["budget"])
        self.assertEqual(list(plan.ordered_actions), reference["ordered_actions"])
        self.assertEqual(selected["selected_actions"], reference["selected_actions"])
        self.assertEqual(selected["family_champions"], reference["family_champions"])
        self.assertEqual({Reasoning.family(tag) for tag in action_tags()},
                         {"Priority", "Eligibility", "Quality"})
        self.assertEqual(ToolUse.target_from_clean(0), "B")
        self.assertEqual(ToolUse.target_from_clean(1), "A")

    def test_reject_bad_matrices(self):
        for edit in (lambda m: m["hits"].pop(action_tags()[0]),
                     lambda m: m["hits"].update({action_tags()[0]: ["invalid"]}),
                     lambda m: m["clean"].update(x=True),
                     lambda m: m.update(action_space_sha256="changed")):
            data = matrix()
            edit(data)
            with self.assertRaises(ValueError):
                select_portfolio(data)
        for k in (0, 67):
            with self.assertRaises(ValueError):
                select_portfolio(matrix(), k)

    def test_legacy_parser(self):
        for raw, expected in [("Output (a)", 0), ("Output (b)", 1), ('{"choice":"B"}', 1),
                              ("No preference", None), ("", None), ("A explanation", 0)]:
            self.assertEqual(parse(raw), expected)

    def test_prompt_keeps_inputs(self):
        item = example("test", "Literal {instruction}")
        prompt = build_judge_user_prompt(PairwiseExample(**item, source_note="NOTE"))
        self.assertIn("# Instruction:\nLiteral {instruction}\n", prompt)
        self.assertTrue(prompt.endswith("[Source Note]\nNOTE"))

    def test_union_denominator_and_cost(self):
        result = summarize({"x": 0, "y": 1, "z": None},
                           {"a": ["x"], "b": ["x"]}, ["a", "b"], [1, 2], {})
        row = result["rows"][1]
        self.assertEqual((row["eligible"], row["successes"], row["asr"]), (2, 1, .5))
        self.assertEqual(row["avg_test_q_per_success"], 4)
        zero = summarize({"x": 0}, {"a": []}, ["a"], [1], {})
        self.assertIsNone(zero["rows"][0]["avg_test_q_per_success"])

    def test_disjoint_content_and_ids(self):
        item = example("calibration")
        portfolio = {"calibration_item_ids": [item["question_id"]],
                     "calibration_item_hashes": [item_hash(item)]}
        changed = {**item, "question_id": "renamed", "response_a": item["response_b"],
                   "response_b": item["response_a"]}
        with self.assertRaises(ValueError):
            check_disjoint([changed], portfolio)
        audit = check_disjoint([changed], portfolio, allow_content_overlap=True)
        self.assertEqual(audit["content_overlap_count"], 1)
        with self.assertRaises(ValueError):
            check_disjoint([item], portfolio, allow_content_overlap=True)

    def test_sampling_preserves_file_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            values = [example(str(i)) for i in range(8)]
            atomic_json(path, values)
            sampled = load_items(path, 3, 43)
            self.assertEqual(sampled, load_items(path, 3, 43))
            ids = [i["question_id"] for i in sampled]
            self.assertEqual(ids, sorted(ids))
            atomic_json(path, [values[0], values[0]])
            with self.assertRaises(ValueError):
                load_items(path)


class RunnerTests(unittest.TestCase):
    def test_resume_makes_no_new_calls(self):
        class Judge(MockJudge):
            calls = 0

            def generate(self, messages, on_attempt):
                self.calls += len(messages)
                return super().generate(messages, on_attempt)

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            items = [example("a"), example("b", "Another question")]
            tags = action_tags()[:2]
            judge = Judge()
            prepare_run(tmp, items, tags, {"backend": "mock"}, {})
            first = Runner(items, judge, tmp)
            clean = first.evaluate("clean")
            hits = first.run_actions(tags, clean)
            count = judge.calls
            resumed = Runner(items, judge, tmp)
            self.assertEqual(resumed.evaluate("clean"), clean)
            self.assertEqual(resumed.run_actions(tags, clean), hits)
            self.assertEqual(judge.calls, count)
            with self.assertRaises(ValueError):
                prepare_run(tmp, items, tags, {"backend": "different"}, {})

    def test_truncated_jsonl_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            path.write_bytes(b'{"ok":1}\n{"partial":')
            self.assertEqual(read_jsonl(path, repair=True), [{"ok": 1}])
            self.assertEqual(path.read_bytes(), b'{"ok":1}\n')
            path.write_bytes(b'bad\n{"ok":1}\n')
            with self.assertRaises(ValueError):
                read_jsonl(path, repair=True)

    def test_full_mock_cli(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            atomic_json(root / "train.json", [example("cal", "Calibration example")])
            atomic_json(root / "test.json", [example("test", "Evaluation example")])
            main(["calibrate", "--backend", "mock", "--model", "mock",
                  "--data", str(root / "train.json"), "--out", str(root / "cal")])
            main(["select", "--matrix", str(root / "cal/matrix.json"), "--budget", "10",
                  "--out", str(root / "portfolio.json")])
            args = ["evaluate", "--backend", "mock", "--model", "mock",
                    "--data", str(root / "test.json"), "--portfolio", str(root / "portfolio.json"),
                    "--budgets", "2,3,5,10", "--out", str(root / "test")]
            main(args)
            before = (root / "test/responses.jsonl").read_bytes()
            main(args)
            self.assertEqual(before, (root / "test/responses.jsonl").read_bytes())
            result = json.loads((root / "test/summary.json").read_text())
            self.assertTrue(result["simulation"])
            self.assertEqual([r["budget"] for r in result["rows"]], [2, 3, 5, 10])
            self.assertEqual(result["cost"]["recorded_attack_attempts"], 0)

    def test_api_retry_accounting_without_external_network(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.headers.get("Authorization"), value))
                if len(seen) == 1:
                    self.send_response(429)
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"choices":[{"message":{"content":"Output (b)"}}]}')

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = {"api_url": f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                      "api_key_env": "BLINDFLIP_TEST_KEY", "api_auth": "bearer", "model": "test",
                      "max_tokens": 256, "api_options": {"enable_thinking": False},
                      "api_attempts": 2, "api_timeout": 3, "api_workers": 1}
            attempts = []
            with patch.dict(os.environ, {"BLINDFLIP_TEST_KEY": "unit-test-token"}), \
                    patch("blindflip.backends.time.sleep"):
                outputs = APIJudge(config).generate([[{"role": "user", "content": "test"}]],
                                                    lambda i, a: attempts.append((i, a)))
            self.assertEqual(attempts, [(0, 1), (0, 2)])
            self.assertEqual(outputs[0]["raw"], "Output (b)")
            self.assertEqual(seen[0][0], "Bearer unit-test-token")
            self.assertFalse(seen[0][1]["enable_thinking"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
