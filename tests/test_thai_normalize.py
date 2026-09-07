"""Repairing Thai text that PDF extraction mangles.

These are not cosmetic.  A vowel left detached splits one word into two
tokens, and both the segmenter and BM25 then index something no query will
ever match.

The Private Use Area characters below are written as escapes on purpose.
They are invisible in an editor, and a test carrying them literally can be
normalised away by a tool that means well — after which the "nothing
survives" assertion still passes while testing nothing at all.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.thai_normalize import fix_pua, has_pua, normalize_text  # noqa: E402

MAI_EK = "\uF70A"     # a tone mark the embedded font moved into the PUA
SARA_UU = "\uF700"    # the vowel sara uu, likewise
UNMAPPED = "\uE001"   # no entry in the table: nothing real to restore


class PuaTests(unittest.TestCase):
    def test_plain_text_is_left_alone(self):
        text = "ยื่นคำร้องที่สำนักส่งเสริมวิชาการ"
        self.assertFalse(has_pua(text))
        self.assertEqual(fix_pua(text), text)

    def test_private_use_characters_are_detected(self):
        self.assertTrue(has_pua("ค" + MAI_EK + "า"))

    def test_a_mapped_glyph_becomes_the_character_it_was_drawing(self):
        self.assertEqual(fix_pua("ท" + MAI_EK + "ด"), "ท่ด")
        self.assertEqual(fix_pua("ค" + SARA_UU + "ณ"), "คูณ")

    def test_an_unmapped_glyph_becomes_a_space_not_a_deletion(self):
        # Dropping it outright would fuse the words on either side into one
        # token that no query can match.
        self.assertEqual(fix_pua("คำ" + UNMAPPED + "ร้อง"), "คำ ร้อง")

    def test_nothing_from_the_private_use_area_survives(self):
        mixed = "ทดสอบ" + MAI_EK + UNMAPPED
        self.assertTrue(has_pua(mixed))
        self.assertFalse(has_pua(fix_pua(mixed)))


class SplitVowelTests(unittest.TestCase):
    def test_a_detached_sara_am_is_rejoined(self):
        # Some PDF fonts draw "ำ" separately, so extraction returns "ส านัก".
        # Neither vowel can begin a Thai word, so the space is always this
        # artefact and never a real boundary.
        self.assertEqual(normalize_text("ส านัก"), "สำนัก")
        self.assertEqual(normalize_text("ค าร้อง"), "คำร้อง")

    def test_a_real_word_boundary_is_kept(self):
        self.assertEqual(normalize_text("ยื่น คำร้อง"), "ยื่น คำร้อง")


class StructureTests(unittest.TestCase):
    def test_blank_lines_survive(self):
        # parse_document tells one passage from the next by the blank line;
        # PyThaiNLP's normaliser collapses runs of newlines, so it is run per
        # line rather than over the whole passage.
        self.assertEqual(normalize_text("หัวข้อ\n\nเนื้อหา"), "หัวข้อ\n\nเนื้อหา")

    def test_empty_input_is_not_an_error(self):
        self.assertEqual(normalize_text(""), "")


if __name__ == "__main__":
    unittest.main()
