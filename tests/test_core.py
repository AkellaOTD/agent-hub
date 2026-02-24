"""Тесты ядра Agent Hub."""

import unittest
import tempfile
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import parse_response, _compute_start_round
from session import Session, Message
from context import _truncate_text, build_messages


class TestParseResponse(unittest.TestCase):

    def test_standard_format(self):
        response = "[THINKING]\nрассуждения\n[CONCLUSION]\nвыводы"
        thinking, conclusion, artifacts, questions = parse_response(response)
        self.assertEqual(thinking, "рассуждения")
        self.assertEqual(conclusion, "выводы")

    def test_no_tags_fallback(self):
        response = "просто текст без тегов"
        thinking, conclusion, artifacts, questions = parse_response(response)
        self.assertEqual(thinking, "")
        self.assertEqual(conclusion, "просто текст без тегов")

    def test_artifacts_extracted(self):
        response = (
            "[THINKING]\nдумаю\n[CONCLUSION]\nтекст\n"
            "[ARTIFACT:plan.md]\n# План\n[/ARTIFACT]"
        )
        _, conclusion, artifacts, _ = parse_response(response)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0][0], "plan.md")
        self.assertEqual(artifacts[0][1], "# План")
        self.assertNotIn("[ARTIFACT", conclusion)

    def test_questions_extracted(self):
        response = (
            "[THINKING]\nдумаю\n[CONCLUSION]\nтекст\n"
            "[QUESTION]\nКакой стек?\n[/QUESTION]"
        )
        _, conclusion, _, questions = parse_response(response)
        self.assertEqual(len(questions), 1)
        self.assertIn("Какой стек?", questions[0])
        self.assertNotIn("[QUESTION", conclusion)

    def test_multiple_artifacts(self):
        response = (
            "[THINKING]\nок\n[CONCLUSION]\n"
            "[ARTIFACT:a.md]\ncontentA\n[/ARTIFACT]\n"
            "[ARTIFACT:b.md]\ncontentB\n[/ARTIFACT]"
        )
        _, _, artifacts, _ = parse_response(response)
        self.assertEqual(len(artifacts), 2)
        self.assertEqual(artifacts[0][0], "a.md")
        self.assertEqual(artifacts[1][0], "b.md")


class TestPathTraversal(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.session = Session(self.tmpdir, ["agent1"], "artifacts")
        (Path(self.tmpdir) / "artifacts").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_traversal_dot_dot_slash(self):
        with self.assertRaises(ValueError):
            self.session.save_artifact("../../etc/passwd", "malicious")

    def test_traversal_parent(self):
        with self.assertRaises(ValueError):
            self.session.save_artifact("../secret.txt", "malicious")

    def test_normal_name_works(self):
        path = self.session.save_artifact("plan.md", "# Plan")
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "plan.md")

    def test_nested_path_rejected(self):
        with self.assertRaises(ValueError):
            self.session.save_artifact("subdir/file.md", "content")

    def test_read_traversal_returns_none(self):
        result = self.session.read_artifact("../../etc/passwd")
        self.assertIsNone(result)

    def test_read_normal_works(self):
        self.session.save_artifact("test.md", "hello")
        result = self.session.read_artifact("test.md")
        self.assertEqual(result, "hello")

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.session.save_artifact("", "content")

    def test_dot_name_raises(self):
        with self.assertRaises(ValueError):
            self.session.save_artifact(".", "content")


class TestComputeStartRound(unittest.TestCase):

    def _msg(self, agent: str, round_num: int, type_: str = "conclusion") -> Message:
        return Message(
            id=f"msg-{agent}-{round_num}",
            timestamp="2026-01-01T00:00:00+00:00",
            agent=agent, type=type_, content="text", round=round_num,
        )

    def test_empty_shared(self):
        start, answered = _compute_start_round([], ["gpt", "claude", "gemini"])
        self.assertEqual(start, 1)
        self.assertEqual(answered, set())

    def test_complete_round(self):
        shared = [
            self._msg("human", 0, "human"),
            self._msg("gpt", 1), self._msg("claude", 1), self._msg("gemini", 1),
        ]
        start, answered = _compute_start_round(shared, ["gpt", "claude", "gemini"])
        self.assertEqual(start, 2)
        self.assertEqual(answered, set())

    def test_incomplete_round(self):
        shared = [
            self._msg("human", 0, "human"),
            self._msg("gpt", 1), self._msg("claude", 1),
        ]
        start, answered = _compute_start_round(shared, ["gpt", "claude", "gemini"])
        self.assertEqual(start, 1)
        self.assertEqual(answered, {"gpt", "claude"})

    def test_only_human_messages(self):
        shared = [self._msg("human", 0, "human")]
        start, answered = _compute_start_round(shared, ["gpt", "claude"])
        self.assertEqual(start, 1)
        self.assertEqual(answered, set())

    def test_two_complete_rounds(self):
        shared = [
            self._msg("human", 0, "human"),
            self._msg("gpt", 1), self._msg("claude", 1),
            self._msg("gpt", 2), self._msg("claude", 2),
        ]
        start, answered = _compute_start_round(shared, ["gpt", "claude"])
        self.assertEqual(start, 3)
        self.assertEqual(answered, set())

    def test_second_round_incomplete(self):
        shared = [
            self._msg("human", 0, "human"),
            self._msg("gpt", 1), self._msg("claude", 1),
            self._msg("gpt", 2),
        ]
        start, answered = _compute_start_round(shared, ["gpt", "claude"])
        self.assertEqual(start, 2)
        self.assertEqual(answered, {"gpt"})


class TestPlanFinalExclusion(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.session = Session(self.tmpdir, ["gpt", "claude"], "artifacts")
        self.session.create("test task")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_plan_final_and_errors_filtered(self):
        self.session.save_artifact("plan-final.md", "old plan")
        self.session.save_artifact("plan-gpt.md", "gpt plan")
        self.session.save_artifact("error-gpt-round1.log", "stderr")

        artifacts = [
            name for name in self.session.list_artifacts()
            if name != "plan-final.md" and not name.startswith("error-")
        ]
        self.assertNotIn("plan-final.md", artifacts)
        self.assertNotIn("error-gpt-round1.log", artifacts)
        self.assertIn("plan-gpt.md", artifacts)

    def test_normal_artifacts_preserved(self):
        self.session.save_artifact("plan-gpt.md", "plan")
        self.session.save_artifact("schema.md", "schema")

        artifacts = [
            name for name in self.session.list_artifacts()
            if name != "plan-final.md" and not name.startswith("error-")
        ]
        self.assertIn("plan-gpt.md", artifacts)
        self.assertIn("schema.md", artifacts)


class TestTruncation(unittest.TestCase):

    def test_no_truncation_when_under_limit(self):
        text = "short text"
        self.assertEqual(_truncate_text(text, 100), text)

    def test_truncation_with_marker(self):
        text = "a" * 200
        result = _truncate_text(text, 50)
        self.assertTrue(result.startswith("a" * 50))
        self.assertTrue(result.endswith("[TRUNCATED]"))
        self.assertEqual(len(result.split("[TRUNCATED]")[0].strip()), 50)

    def test_zero_means_no_limit(self):
        text = "a" * 10000
        self.assertEqual(_truncate_text(text, 0), text)

    def test_negative_means_no_limit(self):
        text = "a" * 100
        self.assertEqual(_truncate_text(text, -5), text)

    def test_exact_limit_no_truncation(self):
        text = "a" * 50
        self.assertEqual(_truncate_text(text, 50), text)


class TestSlidingWindow(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.session = Session(self.tmpdir, ["gpt", "claude"], "artifacts")
        self.session.create("test task")
        for i in range(1, 6):
            for agent in ["gpt", "claude"]:
                self.session.append_shared(Message(
                    id=f"msg-{i}-{agent}",
                    timestamp="2026-01-01T00:00:00+00:00",
                    agent=agent, type="conclusion",
                    content=f"conclusion {agent} round {i}",
                    round=i,
                ))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_window_limits_messages(self):
        messages = build_messages(
            self.session, "gpt", 6,
            review_peers=False,
            max_shared_messages=3,
        )
        shared_msg = [m for m in messages if "Общий чат" in m["content"]]
        self.assertEqual(len(shared_msg), 1)
        # Последние 3 из 10 shared сообщений
        self.assertNotIn("round 1", shared_msg[0]["content"])
        self.assertNotIn("round 2", shared_msg[0]["content"])
        self.assertNotIn("round 3", shared_msg[0]["content"])
        self.assertIn("round 5", shared_msg[0]["content"])

    def test_no_limit_includes_all(self):
        messages = build_messages(
            self.session, "gpt", 6,
            review_peers=False,
            max_shared_messages=0,
        )
        shared_msg = [m for m in messages if "Общий чат" in m["content"]]
        self.assertEqual(len(shared_msg), 1)
        self.assertIn("round 1", shared_msg[0]["content"])
        self.assertIn("round 5", shared_msg[0]["content"])


if __name__ == '__main__':
    unittest.main()
