"""The answer cache — the second place a positional key was hiding.

Grading costs quota, so answers and verdicts are cached and a run resumes.
The key used to be ``query_id``, which is an index into the golden set, and
the golden set grows: adding the curriculum questions renumbered everything
from 129 on, and seven cached rows would have been served as the answer to a
question that never produced them.  No error, no warning — just a wrong
number in the thesis.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import eval_generation  # noqa: E402

CORPUS = "abc123"


class CacheKeyTests(unittest.TestCase):
    def test_the_same_question_keys_the_same_slot(self):
        self.assertEqual(
            eval_generation.cache_key("ยื่นคำร้องที่ไหน", CORPUS),
            eval_generation.cache_key("ยื่นคำร้องที่ไหน", CORPUS),
        )

    def test_a_different_question_keys_a_different_slot(self):
        self.assertNotEqual(
            eval_generation.cache_key("ยื่นคำร้องที่ไหน", CORPUS),
            eval_generation.cache_key("ค่าธรรมเนียมเท่าไร", CORPUS),
        )

    def test_a_rechunked_corpus_invalidates_the_slot(self):
        # The answer was produced from chunks that no longer exist.
        self.assertNotEqual(
            eval_generation.cache_key("ยื่นคำร้องที่ไหน", CORPUS),
            eval_generation.cache_key("ยื่นคำร้องที่ไหน", "different"),
        )


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        self.file = self.tmp / "eval_cache.json"
        patch = mock.patch.object(eval_generation, "CACHE_FILE", self.file)
        patch.start()
        self.addCleanup(patch.stop)

    def write(self, payload):
        self.file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_a_graded_row_is_rekeyed_by_its_question(self):
        question = "ยื่นคำร้องที่ไหน"
        self.write({
            f"7:{CORPUS}": {
                "row": {"question": question, "category": "faq"},
                "judge_model": "gemini-3.5-flash",
                "answer_model": "gemini-3.5-flash",
            }
        })
        cache = eval_generation.load_cache(False)
        self.assertIn(eval_generation.cache_key(question, CORPUS), cache)
        self.assertNotIn(f"7:{CORPUS}", cache)

    def test_a_slot_that_cannot_be_identified_is_dropped_not_guessed(self):
        # Answered but never graded, so no question was recorded.  Keeping it
        # under a positional key is how the wrong answer gets reused.
        self.write({
            f"9:{CORPUS}": {
                "result": {"answer": "...", "sources": []},
                "answer_model": "gemini-3.5-flash",
            }
        })
        self.assertEqual(eval_generation.load_cache(False), {})

    def test_an_already_migrated_cache_is_left_alone(self):
        key = eval_generation.cache_key("ยื่นคำร้องที่ไหน", CORPUS)
        self.write({key: {"row": {"question": "ยื่นคำร้องที่ไหน"}}})
        self.assertEqual(list(eval_generation.load_cache(False)), [key])

    def test_fresh_ignores_whatever_is_on_disk(self):
        self.write({f"1:{CORPUS}": {"row": {"question": "x"}}})
        self.assertEqual(eval_generation.load_cache(True), {})


class StratifiedTests(unittest.TestCase):
    entries = (
        [{"query_id": n, "category": "clo"} for n in range(1, 129)]
        + [{"query_id": n, "category": "curriculum"} for n in range(129, 417)]
        + [{"query_id": n, "category": "faq"} for n in range(417, 472)]
        + [{"query_id": n, "category": "unanswerable"} for n in range(472, 494)]
    )

    def test_every_category_is_represented(self):
        picked = eval_generation.stratified(list(self.entries), 60)
        seen = {e["category"] for e in picked}
        self.assertEqual(
            seen, {"clo", "curriculum", "faq", "unanswerable"}
        )

    def test_the_sample_is_not_just_the_front_of_the_list(self):
        # The golden set is ordered by category, so the first 60 are all CLO —
        # the category the system already answers well.
        head = {e["category"] for e in self.entries[:60]}
        self.assertEqual(head, {"clo"})
        picked = eval_generation.stratified(list(self.entries), 60)
        self.assertNotEqual({e["category"] for e in picked}, {"clo"})

    def test_shares_follow_the_category_sizes(self):
        picked = eval_generation.stratified(list(self.entries), 60)
        counts = {c: 0 for c in ("clo", "curriculum", "faq", "unanswerable")}
        for entry in picked:
            counts[entry["category"]] += 1
        # curriculum is 288 of 493, so it should dominate the sample too.
        self.assertGreater(counts["curriculum"], counts["clo"])
        self.assertGreater(counts["clo"], counts["faq"])

    def test_a_limit_of_zero_means_everything(self):
        self.assertEqual(len(eval_generation.stratified(list(self.entries), 0)), 493)

    def test_a_limit_larger_than_the_set_means_everything(self):
        self.assertEqual(len(eval_generation.stratified(list(self.entries), 900)), 493)

    def test_the_limit_is_never_exceeded(self):
        for n in (1, 7, 30, 60, 100, 492):
            with self.subTest(limit=n):
                self.assertLessEqual(
                    len(eval_generation.stratified(list(self.entries), n)), n
                )


if __name__ == "__main__":
    unittest.main()
