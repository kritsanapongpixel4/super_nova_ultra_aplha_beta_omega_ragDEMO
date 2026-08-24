"""Entry point — ask the RAG system a question.

Usage:
    python main.py                       # interactive chat loop
    python main.py "คำถามของคุณ"          # single question
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Windows consoles default to cp1252, which cannot encode Thai or the emoji
# used below — without this every print() would raise UnicodeEncodeError.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def _print_sources(sources: list[dict]) -> None:
    """Show where the answer came from, numbered to match the [n] citations."""
    if not sources:
        return
    print("\n📎 แหล่งอ้างอิง:")
    for index, chunk in enumerate(sources, start=1):
        head = chunk["text"].splitlines()[0][:70]
        where = f"หน้า/บรรทัด {chunk.get('line_start')}-{chunk.get('line_end')}"
        print(f"   [{index}] {chunk.get('source', '?')}  {where}")
        print(f"       {head}")


def ask(question: str) -> None:
    """Answer one question and print it with its sources."""
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline.from_config()
    result = pipeline.answer(question)

    print(f"\n❓ {result['question']}\n")
    print(result["answer"])
    _print_sources(result["sources"])


def chat() -> None:
    """Interactive loop that keeps conversation history between turns."""
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline.from_config(use_memory=True)

    print("\n💬 โหมดสนทนา — พิมพ์คำถามได้เลย")
    print("   /clear ล้างประวัติการสนทนา   /exit หรือ Ctrl+C เพื่อออก\n")

    while True:
        try:
            question = input("คุณ > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nลาก่อนครับ")
            return

        if not question:
            continue
        if question in {"/exit", "/quit"}:
            print("ลาก่อนครับ")
            return
        if question == "/clear":
            if pipeline.memory:
                pipeline.memory.clear()
            print("ล้างประวัติแล้ว\n")
            continue

        result = pipeline.answer(question)
        print(f"\nระบบ > {result['answer']}")
        _print_sources(result["sources"])
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG system.")
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    args = parser.parse_args()

    try:
        if args.question:
            ask(args.question)
        else:
            chat()
    except (FileNotFoundError, RuntimeError) as exc:
        # from_config() raises these when the index is missing or out of date;
        # a traceback would bury the one line that says how to fix it.
        print(f"❌ {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
