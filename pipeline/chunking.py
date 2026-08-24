"""Step 2/7 – Split the extracted text into chunks.

Reads outputs/extracted_text.json, writes outputs/chunks.json.
Uses PyThaiNLP + LangChain RecursiveCharacterTextSplitter for
Thai‑aware chunking (see src/text_splitter.py).

Run: python pipeline/chunking.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402
from src.text_splitter import split_records  # noqa: E402


def main() -> None:
    config.ensure_dirs()

    # 1. Load extracted records
    if not config.EXTRACTED_TEXT_FILE.exists():
        print(f"❌ ไม่พบไฟล์ {config.EXTRACTED_TEXT_FILE}")
        print("   กรุณารัน pipeline/extraction.py ก่อน")
        sys.exit(1)

    with open(config.EXTRACTED_TEXT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"📄 โหลด {len(records)} records จาก {config.EXTRACTED_TEXT_FILE.name}")

    # 2. Split into chunks using Thai‑aware recursive splitter
    chunks = split_records(records, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"✂️  ได้ทั้งหมด {len(chunks)} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    # 3. Write output
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"💾 บันทึกผลลัพธ์ไว้ที่ {config.CHUNKS_FILE}")


if __name__ == "__main__":
    main()
