import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import requests

from ai_requests import AnalysisRequests, estimated_cost


def body(moment="now"):
    return {"model": "deepseek-v4-pro", "messages": [{"role": "system", "content": "test"}, {"role": "user", "content": json.dumps({"分析时间": moment, "股票": "600519", "采集时间": "batch1"})}]}


def response():
    return SimpleNamespace(status_code=200, json=lambda: {"id": "provider-test-id", "choices": [{"message": {"content": "test report"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 153392, "prompt_cache_hit_tokens": 50304, "prompt_cache_miss_tokens": 103088, "completion_tokens": 70641, "total_tokens": 224033}})


class RequestTests(unittest.TestCase):
    def test_repeated_same_batch_calls_once_and_logs_no_secrets(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"audit.jsonl"
            client = AnalysisRequests(path)
            post = Mock(return_value=response())
            first = client.run("session1", "https://api.deepseek.com/chat/completions", body("first"), "test-private-key", (10, 300), post)
            second = client.run("session1", "https://api.deepseek.com/chat/completions", body("later"), "test-private-key", (10, 300), post)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(first["answer"], second["answer"])
            self.assertTrue(second["reused"])
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)  # Started and finished, NOT two HTTP requests.
            self.assertNotIn("test-private-key", path.read_text())
            self.assertNotIn("test report", path.read_text())
            self.assertEqual(second["audit"]["provider_request_id"], "provider-test-id")
            self.assertAlmostEqual(second["audit"]["cost_estimate"]["高峰时段估算元"], 2.850190, places=6)

    def test_timeout_is_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as d:
            client = AnalysisRequests(Path(d)/"audit.jsonl")
            post = Mock(side_effect=requests.ReadTimeout())
            a = client.run("session1", "https://api.deepseek.com/chat/completions", body(), "test-private-key", (10, 300), post)
            b = client.run("session1", "https://api.deepseek.com/chat/completions", body(), "test-private-key", (10, 300), post)
            self.assertEqual(a["status"], "uncertain")
            self.assertTrue(b["reused"])
            self.assertEqual(post.call_count, 1)

    def test_inflight_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            client = AnalysisRequests(Path(d)/"audit.jsonl")
            started, release = threading.Event(), threading.Event()
            calls = []
            def post(*args, **kwargs):
                calls.append(1)
                started.set()
                release.wait(5)
                return response()
            worker = threading.Thread(target=client.run, args=("s", "https://api.deepseek.com/chat/completions", body(), "test-key", (10, 300), post))
            worker.start()
            try:
                self.assertTrue(started.wait(5))
                duplicate = client.run("s", "https://api.deepseek.com/chat/completions", body(), "test-key", (10, 300), post)
                self.assertEqual(duplicate["status"], "pending")
                self.assertTrue(duplicate["reused"])
                self.assertEqual(len(calls), 1)
            finally:
                release.set()
                worker.join(5)

    def test_unknown_usage_has_no_fabricated_cost(self):
        self.assertIsNone(estimated_cost({}, "deepseek-v4-pro"))


if __name__ == "__main__":
    unittest.main()
