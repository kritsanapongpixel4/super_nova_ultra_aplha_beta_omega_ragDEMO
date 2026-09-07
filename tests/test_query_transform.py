"""Rewriting a follow-up into a standalone question.

Only the offline half is covered here: no key, no network, no quota.  What
matters is that every failure path hands back the original query — retrieval
from a slightly worse question beats a chat that raises in the user's face.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import query_transform  # noqa: E402

HISTORY = [
    {"role": "user", "content": "วิชา 04-620-201 คืออะไร"},
    {"role": "assistant", "content": "ปฏิบัติการควบคุมเวอร์ชัน"},
]


class Rewriter:
    """Stands in for the Gen AI client, without the network."""

    def __init__(self, reply=None, raises=None):
        self.reply, self.raises = reply, raises
        self.models = self
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return type("Response", (), {"text": self.reply})()


class RewriteQueryTests(unittest.TestCase):
    def test_the_first_question_is_never_rewritten(self):
        # Nothing to resolve against, so calling the API would spend quota to
        # learn nothing.
        client = Rewriter(reply="ไม่ควรถูกเรียก")
        self.assertEqual(
            query_transform.rewrite_query("ยื่นคำร้องที่ไหน", None, client=client),
            "ยื่นคำร้องที่ไหน",
        )
        self.assertEqual(client.calls, [])

    def test_a_follow_up_is_replaced_by_the_rewrite(self):
        client = Rewriter(reply="CLO ของวิชา 04-620-201 มีอะไรบ้าง")
        self.assertEqual(
            query_transform.rewrite_query("แล้ว CLO ล่ะ", HISTORY, client=client),
            "CLO ของวิชา 04-620-201 มีอะไรบ้าง",
        )

    def test_surrounding_quotes_are_stripped(self):
        client = Rewriter(reply='"คำถามที่สมบูรณ์"')
        self.assertEqual(
            query_transform.rewrite_query("แล้ว CLO ล่ะ", HISTORY, client=client),
            "คำถามที่สมบูรณ์",
        )

    def test_an_api_failure_falls_back_to_the_original(self):
        client = Rewriter(raises=RuntimeError("429 quota"))
        with self.assertLogs("src.query_transform", level="WARNING"):
            self.assertEqual(
                query_transform.rewrite_query("แล้ว CLO ล่ะ", HISTORY, client=client),
                "แล้ว CLO ล่ะ",
            )

    def test_an_empty_reply_falls_back_to_the_original(self):
        for reply in ("", "   ", None):
            with self.subTest(reply=reply):
                client = Rewriter(reply=reply)
                self.assertEqual(
                    query_transform.rewrite_query("แล้ว CLO ล่ะ", HISTORY, client=client),
                    "แล้ว CLO ล่ะ",
                )


class ModelChoiceTests(unittest.TestCase):
    def test_rewriting_does_not_spend_the_answering_model_quota(self):
        # The free tier counts 20 requests per model per day, so a rewrite on
        # the answering model is one answer the user cannot have later.
        self.assertNotEqual(query_transform.DEFAULT_REWRITE_MODEL, config.LLM_MODEL)

    def test_the_rewrite_model_is_one_the_project_knows(self):
        self.assertIn(
            query_transform.DEFAULT_REWRITE_MODEL,
            (config.LLM_MODEL, *config.LLM_FALLBACK_MODELS),
        )


if __name__ == "__main__":
    unittest.main()
