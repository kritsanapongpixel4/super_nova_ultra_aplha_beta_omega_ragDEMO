"""Parsing the "ข้อมูลที่เอกสารนี้ไม่ได้ระบุ" list into questions.

The list is read off a PDF, so its items wrap.  Splitting on the newline made
a fragment into a question of its own — the set shipped with "1,000 บาท" and
"สัปดาห์)" as things the system was expected to refuse to answer — and worse,
a wrapped continuation escaped the REG- filter that had just excluded the line
it belonged to, turning a cross-reference into a refusal test whose answer sat
one line above it.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.build_golden_set import _CROSS_REFERENCE, _unwrap  # noqa: E402


class UnwrapTests(unittest.TestCase):
    def test_a_wrap_after_a_connective_is_rejoined(self):
        # 08_ลาพักการศึกษา.pdf, exactly as extracted.
        block = (
            "จำนวนเงินค่าธรรมเนียมลาพักการศึกษา ไม่ได้ระบุในเอกสารฉบับนี้ "
            "แต่เอกสาร REG-13 ระบุไว้ว่า\n"
            "1,000 บาท\n"
            "จำนวนภาคการศึกษาสูงสุดที่ลาพักต่อเนื่องได้"
        )
        items = _unwrap(block)
        self.assertEqual(len(items), 2)
        self.assertIn("1,000 บาท", items[0])
        self.assertEqual(items[1], "จำนวนภาคการศึกษาสูงสุดที่ลาพักต่อเนื่องได้")

    def test_the_rejoined_item_is_still_caught_as_a_cross_reference(self):
        # The point of joining: the fee is stated in REG-13, so this is not a
        # question the system should refuse.  Split, the "1,000 บาท" half
        # carried no REG- and slipped through.
        block = (
            "จำนวนเงินค่าธรรมเนียมลาพักการศึกษา ... แต่เอกสาร REG-13 ระบุไว้ว่า\n"
            "1,000 บาท"
        )
        self.assertTrue(_CROSS_REFERENCE.search(_unwrap(block)[0]))

    def test_a_wrap_after_a_number_is_rejoined(self):
        # 13_ขอกลับเข้าศึกษา.pdf — split, this produced a truncated question
        # plus the fragment "สัปดาห์)".
        block = (
            "กำหนดเวลาที่ต้องยื่นขอกลับเข้าศึกษาอย่างช้าที่สุด "
            "(ระบุเพียงว่าควรยื่นก่อนวันลงทะเบียนอย่างน้อย 2\n"
            "สัปดาห์)"
        )
        items = _unwrap(block)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].endswith("สัปดาห์)"))

    def test_a_wrap_inside_brackets_is_rejoined(self):
        block = "หัวข้อหนึ่ง (คำอธิบาย\nที่ยาวจนขึ้นบรรทัดใหม่)"
        self.assertEqual(len(_unwrap(block)), 1)

    def test_separate_items_stay_separate(self):
        # Every genuine item across the 13 registrar documents.  None of them
        # may be joined: that would silently delete a refusal test.
        block = "\n".join([
            "จำนวนเงินค่าขึ้นทะเบียนบัณฑิต (เอกสารนี้ระบุเฉพาะค่าปรับล่าช้า 500 บาท)",
            "จำนวนหน่วยกิตสูงสุดที่เทียบโอนได้",
            "ระยะเวลาพิจารณาอนุมัติก่อนถึงขั้นตอนชำระเงิน",
            "ค่าธรรมเนียมของคำร้องทั้ง 3 ประเภท",
            "ระยะเวลาที่เงินจะโอนเข้าบัญชีหลังยื่นคำร้อง",
            "เงื่อนไขว่ากรณีใดบ้างที่มีสิทธิ์ขอเงินคืน",
            "จำนวนภาคการศึกษาสูงสุดที่ขอรักษาสภาพต่อเนื่องได้",
            "จำนวนหน่วยกิตสูงสุดที่ลงทะเบียนเกินเกณฑ์ได้",
            "ผลกระทบด้านการเงิน เช่น การคืนเงินค่าเทอมหรือหนี้สินคงค้างหลังลาออก",
            "ค่าธรรมเนียมของคำร้องขอลาออก",
            "วิธีการเสนอชื่อและช่วงเวลาประกาศผลผู้ได้รับเกียรตินิยม",
            "รายละเอียดการสอบสมรรถนะด้านสารสนเทศ (ICT) เช่น วิธีสมัคร เกณฑ์ผ่าน และค่าใช้จ่าย",
            "รายละเอียดการทดสอบสมรรถนะด้านวิชาชีพของแต่ละหลักสูตร",
            "วิธีสมัครสอบ ตารางสอบ และค่าสมัครสอบของแต่ละมาตรฐาน",
            "ช่องทางและกำหนดเวลาการยื่นผลคะแนนให้คณะ",
        ])
        self.assertEqual(len(_unwrap(block)), 15)

    def test_blank_lines_are_not_items(self):
        self.assertEqual(_unwrap("หนึ่ง\n\n  \nสอง"), ["หนึ่ง", "สอง"])


if __name__ == "__main__":
    unittest.main()
