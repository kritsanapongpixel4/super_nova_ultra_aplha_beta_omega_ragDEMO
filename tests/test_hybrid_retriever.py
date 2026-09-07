"""Course-code pinning and fusion — the parts that decide what the LLM reads.

Pinning exists because a Thai word segmenter shreds "04-620-201" into three
meaningless pieces and RRF then buries whatever BM25 did find.  It was also,
until 2026-09-07, matching a metadata field that only 64 of 2,202 chunks
carry, so it silently did nothing for the two curriculum books.  The tiering
and the cap below are what fixed that, and they are easy to undo by accident.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hybrid_retriever import (  # noqa: E402
    exact_code_matches,
    reciprocal_rank_fusion,
    tokenize,
)

CODE = "04-620-201"

CARD = {"chunk_id": "card", "course_code": CODE, "text": f"รายวิชา {CODE} ปฏิบัติการ"}
DEFINITION = {"chunk_id": "def", "text": f"{CODE} ปฏิบัติการควบคุมเวอร์ชัน\nVersion Control\n3(2-2)"}
MENTION = {"chunk_id": "mention", "text": f"ต้องผ่าน {CODE} มาก่อนจึงจะลงวิชานี้ได้"}
UNRELATED = {"chunk_id": "other", "text": "04-620-999 วิชาอื่น"}


class TokenizeTests(unittest.TestCase):
    def test_a_course_code_survives_word_segmentation(self):
        # newmm splits the code into "04", "620", "201"; the whole code is
        # emitted as an extra token so BM25 still has something exact to hit.
        self.assertIn(CODE, tokenize(f"CLO ของวิชา {CODE} มีอะไรบ้าง"))

    def test_punctuation_is_not_a_token(self):
        self.assertNotIn(" ", tokenize("ยื่น คำร้อง ที่ไหน"))


class PinningTests(unittest.TestCase):
    chunks = [UNRELATED, MENTION, DEFINITION, CARD]

    def test_no_code_in_the_query_pins_nothing(self):
        self.assertEqual(exact_code_matches("ยื่นคำร้องที่ไหน", self.chunks), [])

    def test_another_course_is_not_a_match(self):
        pinned = exact_code_matches(f"วิชา {CODE} คืออะไร", self.chunks)
        self.assertNotIn(UNRELATED, pinned)

    def test_best_evidence_first(self):
        # The card that declares the code, then the entry that opens a line
        # with it, then a passing mention in a prerequisite list.
        pinned = exact_code_matches(f"วิชา {CODE} ชื่ออะไร", self.chunks)
        self.assertEqual(
            [c["chunk_id"] for c in pinned], ["card", "def", "mention"]
        )

    def test_the_text_of_a_curriculum_entry_counts(self):
        # The regression that mattered: no course_code field anywhere, which
        # is every chunk of the two หลักสูตร books.
        pinned = exact_code_matches(f"วิชา {CODE} กี่หน่วยกิต", [DEFINITION, MENTION])
        self.assertEqual([c["chunk_id"] for c in pinned], ["def", "mention"])

    def test_the_cap_holds(self):
        many = [
            {"chunk_id": f"c{n}", "text": f"{CODE} รายวิชาที่ {n}\nCourse\n3(3-0)"}
            for n in range(20)
        ]
        # Uncapped, a code written into twenty chunks would take every slot
        # the generator has and push out everything fusion found.
        self.assertEqual(len(exact_code_matches(f"{CODE} คืออะไร", many)), 5)
        self.assertEqual(
            len(exact_code_matches(f"{CODE} คืออะไร", many, limit=None)), 20
        )
        self.assertEqual(
            len(exact_code_matches(f"{CODE} คืออะไร", many, limit=2)), 2
        )


class FusionTests(unittest.TestCase):
    def test_a_chunk_both_retrievers_rank_beats_one_that_only_leads_a_list(self):
        dense = [{"chunk_id": "top"}, {"chunk_id": "both"}]
        sparse = [{"chunk_id": "other"}, {"chunk_id": "both"}]
        fused = reciprocal_rank_fusion([dense, sparse], k=3)
        self.assertEqual(fused[0]["chunk_id"], "both")

    def test_the_same_chunk_is_not_returned_twice(self):
        one = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        fused = reciprocal_rank_fusion([one, one], k=10)
        self.assertEqual([c["chunk_id"] for c in fused], ["a", "b"])

    def test_k_limits_the_result(self):
        ranking = [{"chunk_id": str(n)} for n in range(10)]
        self.assertEqual(len(reciprocal_rank_fusion([ranking], k=3)), 3)


if __name__ == "__main__":
    unittest.main()
