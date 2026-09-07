"""Retrieval metrics.

Worth pinning down because the numbers in the thesis are these functions'
output, and two of them are easy to write plausibly and wrongly: precision
divided by the number returned instead of k, and recall over a multi-chunk
answer that cannot fit in the top slot.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import metrics  # noqa: E402


class HitTests(unittest.TestCase):
    def test_one_relevant_chunk_in_the_top_k_is_a_hit(self):
        self.assertEqual(metrics.hit_at_k(["a", "b", "c"], {"c"}, 3), 1.0)

    def test_just_outside_the_cut_is_not(self):
        self.assertEqual(metrics.hit_at_k(["a", "b", "c"], {"c"}, 2), 0.0)


class RecallTests(unittest.TestCase):
    def test_recall_counts_all_the_right_answers(self):
        self.assertEqual(metrics.recall_at_k(["a", "b"], {"a", "b"}, 2), 1.0)

    def test_several_gold_chunks_cannot_fit_in_one_slot(self):
        # Curriculum questions have up to four gold chunks, so recall@1 is
        # capped at 0.25 no matter how good the retriever is.  Read hit@k
        # there, not recall — this is the arithmetic that says why.
        self.assertEqual(metrics.recall_at_k(["a"], {"a", "b", "c", "d"}, 1), 0.25)

    def test_no_right_answer_scores_zero_rather_than_dividing_by_zero(self):
        self.assertEqual(metrics.recall_at_k(["a"], set(), 1), 0.0)


class PrecisionTests(unittest.TestCase):
    def test_precision_is_over_k_not_over_what_came_back(self):
        # Three good chunks out of three returned is not the same as three
        # out of the ten that were asked for.
        self.assertEqual(metrics.precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 10), 0.3)


class RankingTests(unittest.TestCase):
    def test_mrr_is_the_reciprocal_of_the_first_hit(self):
        self.assertEqual(metrics.mrr(["a", "b", "c"], {"c"}), 1 / 3)

    def test_mrr_is_zero_when_nothing_was_found(self):
        self.assertEqual(metrics.mrr(["a"], {"z"}), 0.0)

    def test_ndcg_rewards_the_higher_ranking(self):
        good = metrics.ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        worse = metrics.ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        self.assertEqual(good, 1.0)
        self.assertLess(worse, good)


class ScoreOneTests(unittest.TestCase):
    def test_doc_hit_only_appears_when_sources_are_given(self):
        plain = metrics.score_one(["a"], {"a"}, (1,))
        self.assertNotIn("doc_hit@1", plain)

        with_docs = metrics.score_one(
            ["a"], {"a"}, (1,), retrieved_sources=["x.pdf"], relevant_source="x.pdf"
        )
        self.assertEqual(with_docs["doc_hit@1"], 1.0)

    def test_the_right_document_can_hit_when_the_named_chunk_misses(self):
        # Several registrar documents repeat the same sentence, so the answer
        # can be genuinely present in a chunk the golden set does not name.
        # The gap between these two is how much of a miss is labelling.
        scores = metrics.score_one(
            ["wrong-chunk"], {"gold-chunk"}, (1,),
            retrieved_sources=["x.pdf"], relevant_source="x.pdf",
        )
        self.assertEqual(scores["hit@1"], 0.0)
        self.assertEqual(scores["doc_hit@1"], 1.0)


class AggregateTests(unittest.TestCase):
    def test_aggregate_averages_each_metric(self):
        rows = [{"hit@1": 1.0}, {"hit@1": 0.0}, {"hit@1": 1.0}]
        self.assertAlmostEqual(metrics.aggregate(rows)["hit@1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
