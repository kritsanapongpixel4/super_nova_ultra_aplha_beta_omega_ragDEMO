"""Shared command-line handling for picking the embedding model.

Every entry point — build_index.py, main.py, the seven pipeline scripts,
the benchmarks — takes the same ``--model`` flag and resolves it the same
way, so there is one answer to "how do I switch model?" rather than nine.

Two shapes are supported because the scripts have two shapes.  Ones built on
argparse call :func:`add_model_arg` then :func:`apply`.  Ones that treat
``sys.argv`` as a free-text question (``python pipeline/similarity_search.py
วิชานี้มี CLO อะไรบ้าง``) call :func:`take_model_flag`, which lifts the flag
out and hands back the rest of the arguments untouched.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from . import model_registry


def add_model_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--model/-m`` to an argparse parser."""
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        metavar="KEY",
        help=(
            "embedding model to use — a registry key ("
            + ", ".join(model_registry.RUNNABLE)
            + ") or any HuggingFace id.  Defaults to RAG_EMBED_MODEL, then "
            f"{model_registry.DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEV",
        help="cuda | cpu (default: cuda when available)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print the model registry with sizes and feasibility, then exit",
    )
    return parser


def apply(args: argparse.Namespace) -> Any:
    """Honour ``--list-models``/``--model``/``--device`` and return the spec.

    Imports config lazily: config reads the environment at import time, so
    the caller gets a consistent answer whether the model came from the flag
    or from RAG_EMBED_MODEL.
    """
    if getattr(args, "list_models", False):
        print(model_registry.table())
        raise SystemExit(0)

    import config

    if getattr(args, "device", None):
        config.EMBEDDING_DEVICE = args.device
    return use(getattr(args, "model", None))


def use(key: str | None) -> Any:
    """Point config at *key*, or leave the environment's choice in place."""
    import config

    if key:
        return config.use_model(key)
    return config.EMBEDDING_SPEC


def take_model_flag(argv: list[str] | None = None) -> tuple[Any, list[str]]:
    """Pull ``--model``/``--device`` out of *argv*; return (spec, remaining).

    For the scripts whose remaining arguments are a question, not options.
    Both ``--model qwen3-0.6b`` and ``--model=qwen3-0.6b`` are accepted.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    def pop(flag: str) -> str | None:
        """Remove *flag* and its value from args, returning the value."""
        for i, item in enumerate(args):
            if item == flag and i + 1 < len(args):
                value = args[i + 1]
                del args[i : i + 2]
                return value
            if item.startswith(flag + "="):
                value = item.split("=", 1)[1]
                del args[i]
                return value
        return None

    model = pop("--model") or pop("-m")
    device = pop("--device")

    import config

    if device:
        config.EMBEDDING_DEVICE = device
    return use(model), args
