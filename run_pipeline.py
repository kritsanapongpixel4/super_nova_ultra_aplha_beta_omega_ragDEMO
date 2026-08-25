"""Run the whole pipeline, steps 1-7, and leave a record of what happened.

    python run_pipeline.py                      # every step, default model
    python run_pipeline.py --model qwen3-0.6b   # the same for another model
    python run_pipeline.py --steps 3-7          # only the model-dependent half
    python run_pipeline.py --question "..."     # what steps 5-7 should search for

Steps 1 and 2 (extract, chunk) produce the same files whatever the embedding
model is, so a second model only needs 3-7.  ``--steps`` exists for that: the
benchmark reuses one extraction across every model rather than re-parsing 824
pages of PDF eight times.

Everything is timed, every step's output goes to logs/runs/, anything that
goes wrong also goes to logs/errors/, and the run is summarised in the
experiment journal at logs/EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from src import cli, journal  # noqa: E402
from src.embedding_model import pick_device  # noqa: E402
from src.run_logger import RunLogger  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

DEFAULT_QUESTION = "วิชาปฏิบัติการควบคุมเวอร์ชันมี CLO อะไรบ้าง"

# (number, title, module path, needs the question as argv)
STEPS = [
    (1, "แยกข้อความจาก PDF", "pipeline.extract_text", False),
    (2, "ตัด chunk", "pipeline.chunking", False),
    (3, "สร้าง embeddings", "pipeline.create_embeddings", False),
    (4, "สร้าง FAISS index", "pipeline.create_vector_db", False),
    (5, "แปลงคำถามเป็นเวกเตอร์", "pipeline.query_embedding", True),
    (6, "ค้นหาด้วย similarity", "pipeline.similarity_search", True),
    (7, "retrieval ครบระบบ", "pipeline.complete_retrieval", True),
]


def parse_steps(text: str) -> set[int]:
    """Turn '3-7' or '1,2,5' into a set of step numbers."""
    wanted: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            wanted.update(range(int(lo), int(hi) + 1))
        elif part:
            wanted.add(int(part))
    return wanted


def run_step(module_path: str, question: str | None) -> None:
    """Import a step module and call its main(), with argv set up for it.

    The step scripts read sys.argv themselves (that is how they take a
    question and a --model), so the argv they need is staged here rather
    than threading a parameter through seven modules that do not share a
    signature.
    """
    import importlib

    saved = sys.argv
    argv = [f"{module_path}.py", "--model", config.EMBEDDING_KEY]
    if config.EMBEDDING_DEVICE:
        argv += ["--device", config.EMBEDDING_DEVICE]
    if question:
        argv.append(question)
    sys.argv = argv
    try:
        module = importlib.import_module(module_path)
        module.main()
    finally:
        sys.argv = saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline steps 1-7.")
    parser.add_argument(
        "--steps", default="1-7", help="which steps to run, e.g. 3-7 or 1,2 (default: 1-7)"
    )
    parser.add_argument(
        "--question", default=DEFAULT_QUESTION, help="question used by steps 5-7"
    )
    cli.add_model_arg(parser)
    args = parser.parse_args()
    spec = cli.apply(args)

    wanted = parse_steps(args.steps)
    steps = [s for s in STEPS if s[0] in wanted]
    if not steps:
        print(f"❌ --steps {args.steps} ไม่ตรงกับขั้นตอนใดเลย (มี 1-7)")
        sys.exit(1)

    config.ensure_dirs()

    # Resolve now, and record the answer rather than the request.  Logging
    # "auto" hid a run that fell back to CPU and took 978s where CUDA takes
    # 70 — the journal said device=auto for both, so nothing looked wrong.
    device = pick_device(config.EMBEDDING_DEVICE)

    started = time.perf_counter()
    timings: dict[str, float] = {}
    failed: list[int] = []

    with RunLogger(
        f"pipeline-{spec.key}",
        model=spec.key,
        hf_id=spec.hf_id,
        device=device,
        steps=args.steps,
        sources=len(config.SOURCE_FILES),
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    ) as run:
        print(f"\n🧬 โมเดล {spec.key} ({spec.hf_id})")
        print(f"📂 {len(config.SOURCE_FILES)} ไฟล์ จาก {config.SOURCE_CATEGORIES}")
        print(f"⚙️  อุปกรณ์: {device}")
        if device == "cpu":
            run.problem(
                "running-on-cpu",
                "ไม่พบ CUDA — ขั้นสร้าง embeddings จะช้ากว่า GPU ราว 10-15 เท่า "
                "(ลองรันด้วย .venv-gpu)",
            )
        print(f"📝 log: {run.log_path.name}\n")

        for number, title, module_path, needs_question in steps:
            header = f"{number}/7 {title}"
            print(f"\n{'='*62}\n▶ {header}\n{'='*62}")
            step_started = time.perf_counter()
            try:
                with run.step(header):
                    run_step(module_path, args.question if needs_question else None)
            except SystemExit as exc:
                # The step scripts call sys.exit(1) on a missing input file.
                # That is a real failure, but it should not take the rest of
                # the run — and above all it must end up in the log.
                if exc.code:
                    failed.append(number)
                    run.problem(
                        "step-failed",
                        f"ขั้นที่ {number} ({title}) จบด้วย exit code {exc.code}",
                        step=number,
                    )
                    print(f"❌ ขั้นที่ {number} ล้มเหลว — ดู logs/errors/{run.error_path.name}")
            except Exception as exc:
                failed.append(number)
                run.problem(
                    "step-error",
                    f"ขั้นที่ {number} ({title}): {type(exc).__name__}: {exc}",
                    step=number,
                )
                import traceback

                traceback.print_exc()
                print(f"❌ ขั้นที่ {number} ล้มเหลว — ดู logs/errors/{run.error_path.name}")
            timings[header] = round(time.perf_counter() - step_started, 2)

        total = time.perf_counter() - started
        run.event("total", elapsed_s=round(total, 2), failed_steps=failed)

        print(f"\n{'='*62}")
        for header, seconds in timings.items():
            print(f"  {header:34s} {seconds:8.1f}s")
        print(f"  {'รวม':34s} {total:8.1f}s")
        # "No failed steps" and "nothing went wrong" are different claims.  A
        # run where two PDFs yielded almost no text finishes every step and is
        # still not clean; saying so is the entire point of keeping the logs.
        if failed:
            print(f"\n⚠️  ขั้นที่ล้มเหลว: {failed}  →  logs/errors/{run.error_path.name}")
        elif run.warnings:
            print(f"\n✅ ครบทุกขั้น แต่มีคำเตือน {run.warnings} รายการ "
                  f"→  logs/errors/{run.error_path.name}")
        else:
            print("\n🎉 ครบทุกขั้นโดยไม่มีคำเตือนใด ๆ")

    journal.append(
        f"pipeline steps {args.steps}: {spec.key}",
        ok=not failed,
        what=f"รัน pipeline ขั้น {args.steps} ด้วย embedding model {spec.key}",
        how=(
            f"{len(config.SOURCE_FILES)} ไฟล์จาก {config.SOURCE_CATEGORIES}, "
            f"chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}, "
            f"max_seq={spec.max_seq_length}, device={device}"
        ),
        result=(
            f"รวม {total:.1f}s"
            + (f", ล้มเหลวขั้น {failed}" if failed else ", ครบทุกขั้น")
        ),
        model=spec.key,
        hf_id=spec.hf_id,
        step_seconds=timings,
        total_seconds=round(total, 2),
        failed_steps=failed,
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
