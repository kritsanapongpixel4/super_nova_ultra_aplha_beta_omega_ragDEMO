"""Step 1/7 – Extract text from all source files in data/.

Auto-discovers every supported file (.pdf, .txt, .docx, .md) in the
data/ directory, extracts text, and writes outputs/extracted_text.json.

Run: python pipeline/extract_text.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402
from src.document_loader import extract_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    config.ensure_dirs()

    # 1. Show discovered files
    print(f"📂 ค้นหาไฟล์ใน {config.DATA_DIR}")
    if not config.SOURCE_FILES:
        print("❌ ไม่พบไฟล์ที่รองรับในโฟลเดอร์ data/")
        print(f"   (นามสกุลที่รองรับ: {config.SUPPORTED_EXTENSIONS})")
        sys.exit(1)

    print(f"   พบ {len(config.SOURCE_FILES)} ไฟล์:")
    for f in config.SOURCE_FILES:
        print(f"     • {f.name} ({f.stat().st_size / 1024:.0f} KB)")

    # 2. Extract text from all files
    records = extract_all(config.SOURCE_FILES)
    print(f"\n✅ แยกข้อความได้ทั้งหมด {len(records)} records")

    # 2b. A PDF that is a stack of scans, or that embeds a font with no
    #     ToUnicode table, still "succeeds" — it just yields almost nothing.
    #     Silence there is the failure mode, so name the files instead: they
    #     go into the index, contribute nothing, and the gap only shows up
    #     later as a question the system cannot answer.
    chars_by_source: dict[str, int] = {}
    for record in records:
        source = record.get("source", "?")
        chars_by_source[source] = chars_by_source.get(source, 0) + len(
            record.get("text", "")
        )
    for path in config.SOURCE_FILES:
        chars = chars_by_source.get(path.name, 0)
        kb = path.stat().st_size / 1024
        # 2,000 characters per megabyte is far below anything a real text
        # PDF produces; the working files here sit 20-100x above it.
        if kb > 500 and chars < kb * 2:
            logging.warning(
                "⚠️  %s: %.0f KB แต่ดึงข้อความได้แค่ %d ตัวอักษร "
                "— น่าจะเป็นไฟล์สแกนหรือฟอนต์ไม่มีตาราง ToUnicode (ต้องใช้ OCR)",
                path.name,
                kb,
                chars,
            )

    # 3. Golden set status
    if config.GOLDEN_SET_FILE:
        print(f"🏆 Golden set: {config.GOLDEN_SET_FILE.name}")
    else:
        print("ℹ️  ไม่พบ golden_set.json — ข้ามการประเมินด้วย golden set")

    # 4. Write output
    with open(config.EXTRACTED_TEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"💾 บันทึกไว้ที่ {config.EXTRACTED_TEXT_FILE}")


if __name__ == "__main__":
    main()
