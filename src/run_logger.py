"""Run logging — every run leaves a file behind, especially the ones that fail.

A traceback that only ever existed in a terminal that has since been closed
is worth nothing when the same bug comes back a week later.  This module
sends every run's output to ``logs/`` as well as the console:

    logs/runs/<run-name>-<timestamp>.log     full transcript, DEBUG and up
    logs/errors/<run-name>-<timestamp>.log   written only if something broke
    logs/runs/<run-name>-<timestamp>.json    machine-readable run summary

The error file is deliberately separate.  Scrolling a 4,000-line index build
looking for the one PDF that failed is how problems get missed; if
``logs/errors/`` is empty the run was clean, and if it is not, the file holds
only the parts that were not.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
RUNS_DIR = LOGS_DIR / "runs"
ERRORS_DIR = LOGS_DIR / "errors"

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


class _CountingHandler(logging.Handler):
    """Counts warnings and errors so the summary can say 'clean' honestly."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.warnings = 0
        self.errors = 0
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.errors += 1
        else:
            self.warnings += 1
        # Keep a bounded copy for the JSON summary — an index build that
        # warns on every one of 4,000 chunks should not produce a 40 MB file.
        if len(self.records) < 500:
            self.records.append(
                f"{record.levelname} {record.name}: {record.getMessage()}"
            )


class RunLogger:
    """Context manager that tees a run to logs/ and records how it ended.

        with RunLogger("build-index", model="bge-m3") as run:
            run.event("extract", files=9)
            ...

    Leaving the block writes the JSON summary.  An exception is logged with
    its traceback, recorded in the summary, and then re-raised — this is a
    recorder, never a swallower.
    """

    def __init__(self, name: str, **context: Any) -> None:
        self.name = name
        self.context = context
        self.started_at = datetime.now()
        stamp = self.started_at.strftime("%Y%m%d-%H%M%S")
        self.stem = f"{name}-{stamp}"

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ERRORS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = RUNS_DIR / f"{self.stem}.log"
        self.error_path = ERRORS_DIR / f"{self.stem}.log"
        self.json_path = RUNS_DIR / f"{self.stem}.json"

        self.events: list[dict[str, Any]] = []
        self.problems: list[dict[str, Any]] = []
        self._handlers: list[logging.Handler] = []
        self._counter = _CountingHandler()
        self._t0 = 0.0

    # ── Setup / teardown ────────────────────────────────────────────────

    def __enter__(self) -> "RunLogger":
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        # Root at DEBUG turns on every library's chatter too.  The HTTP
        # clients underneath huggingface-hub and google-genai log a line per
        # request, which buries the run's own output in both the console and
        # the log file — hundreds of redirect traces around the three lines
        # that say what the model actually did.
        for noisy in ("httpx", "httpcore", "urllib3", "filelock", "fsspec"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))

        # delay=True so an untroubled run leaves no empty error file behind —
        # the presence of a file in logs/errors/ is itself the signal.
        error_handler = logging.FileHandler(
            self.error_path, encoding="utf-8", delay=True
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))

        for handler in (file_handler, error_handler, self._counter):
            root.addHandler(handler)
            self._handlers.append(handler)

        # Only add a console handler if nothing has set one up already,
        # or scripts that called basicConfig() would print every line twice.
        if not any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in root.handlers
        ):
            console = logging.StreamHandler(sys.stderr)
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter("%(message)s"))
            root.addHandler(console)
            self._handlers.append(console)

        self._t0 = time.perf_counter()
        logging.getLogger("run").info(
            "▶ เริ่ม %s  (log: %s)", self.name, self.log_path.name
        )
        logging.getLogger("run").debug("environment: %s", json.dumps(_environment()))
        if self.context:
            logging.getLogger("run").debug(
                "context: %s", json.dumps(self.context, default=str)
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self._t0
        log = logging.getLogger("run")

        if exc is not None:
            self.problems.append(
                {
                    "kind": "fatal",
                    "type": exc_type.__name__,
                    "message": str(exc),
                    "traceback": "".join(
                        traceback.format_exception(exc_type, exc, tb)
                    ),
                }
            )
            log.error("✖ %s ล้มเหลวใน %.1fs: %s", self.name, elapsed, exc)
            log.error("%s", "".join(traceback.format_exception(exc_type, exc, tb)))
        else:
            log.info("✔ %s เสร็จใน %.1fs", self.name, elapsed)

        summary = {
            "run": self.name,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "elapsed_s": round(elapsed, 2),
            "ok": exc is None,
            "context": self.context,
            "environment": _environment(),
            "events": self.events,
            "problems": self.problems,
            "warnings": self._counter.warnings,
            "errors": self._counter.errors,
            "log_messages": self._counter.records,
        }
        self.json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        root = logging.getLogger()
        for handler in self._handlers:
            handler.close()
            root.removeHandler(handler)
        self._handlers.clear()
        return False  # never swallow

    # ── Recording ───────────────────────────────────────────────────────

    def event(self, name: str, **data: Any) -> None:
        """Record a structured milestone — a step finishing, a count, a timing."""
        entry = {"t": round(time.perf_counter() - self._t0, 2), "event": name, **data}
        self.events.append(entry)
        logging.getLogger("run").debug("event %s", json.dumps(entry, default=str))

    def problem(self, kind: str, message: str, **data: Any) -> None:
        """Record something that went wrong but did not stop the run.

        A PDF that yields no text, a model that will not download, a chunk
        the tokenizer chokes on — the run continues, but the whole point of
        keeping logs is that these are still visible afterwards.
        """
        entry = {"kind": kind, "message": message, **data}
        self.problems.append(entry)
        logging.getLogger("run").warning("[%s] %s", kind, message)

    def step(self, title: str) -> "_Step":
        """Time one named step: ``with run.step("chunking"): ...``."""
        return _Step(self, title)


class _Step:
    def __init__(self, run: RunLogger, title: str) -> None:
        self.run = run
        self.title = title
        self._t0 = 0.0

    def __enter__(self) -> "_Step":
        self._t0 = time.perf_counter()
        logging.getLogger("run").info("── %s", self.title)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self._t0
        if exc is None:
            self.run.event("step", step=self.title, elapsed_s=round(elapsed, 2))
            logging.getLogger("run").info("   ⏱️  %s: %.1fs", self.title, elapsed)
        else:
            self.run.event(
                "step_failed",
                step=self.title,
                elapsed_s=round(elapsed, 2),
                error=f"{exc_type.__name__}: {exc}",
            )
        return False


def _environment() -> dict[str, Any]:
    """What the run was executed on — the first question asked of any timing."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
    except Exception:  # torch is optional for scripts that never load a model
        info["torch"] = None
    return info


def configure_console(level: int = logging.INFO) -> None:
    """Set up plain console logging for scripts run outside a RunLogger."""
    logging.basicConfig(level=level, format="%(message)s")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
