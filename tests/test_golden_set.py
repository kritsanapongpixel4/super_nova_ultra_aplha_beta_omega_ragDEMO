"""The golden set loader — the guard that was not there when it mattered.

Every question used to name its answer by ``chunk_id``, which is a position
in a list.  Adding documents renumbered that list, so the answers went on
pointing at whatever had moved into those slots, and the scorer did not
error — it just returned zeros, and zeros look like a bad retriever rather
than a broken test.

So the cases below are the ones that produced silence before: the corpus
grew, a chunk's text changed, the file predates content keys.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.golden_set import (  # noqa: E402
    GoldenSetError,
    chunk_key,
    fingerprint,
    load,
)


def chunk(chunk_id, source, text):
    return {"chunk_id": chunk_id, "source": source, "text": text}


CHUNKS = [
    chunk(0, "a.pdf", "ยื่นคำร้องที่สำนักส่งเสริมวิชาการ"),
    chunk(1, "a.pdf", "ค่าธรรมเนียม 500 บาท"),
    chunk(2, "b.pdf", "ยื่นคำร้องที่สำนักส่งเสริมวิชาการ"),
]


class ChunkKeyTests(unittest.TestCase):
    def test_position_does_not_change_the_key(self):
        moved = dict(CHUNKS[0], chunk_id=99)
        self.assertEqual(chunk_key(CHUNKS[0]), chunk_key(moved))

    def test_same_text_in_two_documents_is_two_keys(self):
        # Boilerplate repeats across the registrar documents; an answer that
        # could be either of them is not an answer.
        self.assertNotEqual(chunk_key(CHUNKS[0]), chunk_key(CHUNKS[2]))

    def test_changed_text_changes_the_key(self):
        edited = dict(CHUNKS[1], text="ค่าธรรมเนียม 600 บาท")
        self.assertNotEqual(chunk_key(CHUNKS[1]), chunk_key(edited))


class FingerprintTests(unittest.TestCase):
    def test_reordering_the_corpus_is_the_same_corpus(self):
        forward = fingerprint(CHUNKS)
        backward = fingerprint(list(reversed(CHUNKS)))
        self.assertEqual(forward["chunks_sha1"], backward["chunks_sha1"])

    def test_adding_a_chunk_is_a_different_corpus(self):
        grown = [*CHUNKS, chunk(3, "c.pdf", "ใหม่")]
        self.assertNotEqual(
            fingerprint(CHUNKS)["chunks_sha1"], fingerprint(grown)["chunks_sha1"]
        )


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "golden_set.json"

    def write(self, queries, meta=None):
        payload = {"meta": meta or {}, "queries": queries}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def test_ids_are_rebound_when_the_corpus_is_renumbered(self):
        self.write([
            {
                "query_id": 1,
                "question": "ยื่นที่ไหน",
                "relevant_chunk_ids": ["0"],
                "relevant_chunk_keys": [chunk_key(CHUNKS[0])],
                "category": "faq",
            }
        ])
        # Two new documents land in front, pushing every id along by two.
        renumbered = [
            chunk(0, "new.pdf", "แทรกหน้า"),
            chunk(1, "new.pdf", "แทรกอีกหน้า"),
            *(dict(c, chunk_id=c["chunk_id"] + 2) for c in CHUNKS),
        ]
        entries, report = load(renumbered, self.path, quiet=True)
        self.assertEqual(entries[0]["relevant_chunk_ids"], ["2"])
        self.assertEqual(report["rebound"], 1)

    def test_edited_text_raises_rather_than_scoring_zero(self):
        self.write([
            {
                "query_id": 1,
                "question": "ค่าธรรมเนียมเท่าไร",
                "relevant_chunk_ids": ["1"],
                "relevant_chunk_keys": ["0" * 16],
                "category": "faq",
            }
        ])
        with self.assertRaises(GoldenSetError):
            load(CHUNKS, self.path, quiet=True)

    def test_a_set_without_content_keys_is_refused(self):
        self.write([
            {
                "query_id": 1,
                "question": "ยื่นที่ไหน",
                "relevant_chunk_ids": ["0"],
                "category": "faq",
            }
        ])
        with self.assertRaises(GoldenSetError):
            load(CHUNKS, self.path, quiet=True)

    def test_missing_file_is_refused(self):
        with self.assertRaises(GoldenSetError):
            load(CHUNKS, Path(self.dir.name) / "nope.json", quiet=True)

    def test_unanswerable_questions_are_left_out_unless_asked_for(self):
        self.write([
            {
                "query_id": 1,
                "question": "ยื่นที่ไหน",
                "relevant_chunk_ids": ["0"],
                "relevant_chunk_keys": [chunk_key(CHUNKS[0])],
                "category": "faq",
            },
            {
                "query_id": 2,
                "question": "ค่าหอพักเท่าไร",
                "relevant_chunk_ids": [],
                "relevant_chunk_keys": [],
                "category": "unanswerable",
            },
        ])
        # Retrieval metrics score these 0 by construction, so counting them
        # would punish a retriever for a question with no right answer.
        entries, report = load(CHUNKS, self.path, quiet=True)
        self.assertEqual(len(entries), 1)
        self.assertEqual(report["n_unanswerable"], 1)

        kept, _ = load(CHUNKS, self.path, quiet=True, include_unanswerable=True)
        self.assertEqual(len(kept), 2)

    def test_a_grown_corpus_is_reported_as_changed(self):
        key = chunk_key(CHUNKS[0])
        self.write(
            [
                {
                    "query_id": 1,
                    "question": "ยื่นที่ไหน",
                    "relevant_chunk_ids": ["0"],
                    "relevant_chunk_keys": [key],
                    "category": "faq",
                }
            ],
            meta={"corpus": fingerprint(CHUNKS)},
        )
        grown = [*CHUNKS, chunk(3, "c.pdf", "เอกสารใหม่")]
        _, report = load(grown, self.path, quiet=True)
        self.assertTrue(report["corpus_changed"])


if __name__ == "__main__":
    unittest.main()
