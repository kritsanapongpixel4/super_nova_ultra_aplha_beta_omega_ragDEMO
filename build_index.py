"""Build every index the RAG system needs, from the raw source file.

Usage:
    python build_index.py            # build only if the dataset changed
    python build_index.py --force    # always rebuild
"""

import argparse

import config


def build(force: bool = False) -> None:
    """Run the full offline pipeline: extract -> chunk -> embed -> index."""
    config.ensure_dirs()

    # TODO: 1. load + parse the source file        -> src.document_loader
    # TODO: 2. split into chunks                   -> src.text_splitter
    # TODO: 3. embed the chunks                    -> src.embedding_model
    # TODO: 4. write FAISS index + chunk_store     -> src.vector_store
    # TODO: 5. write BM25 index                    -> src.hybrid_retriever
    # TODO: 6. write index_meta.json fingerprint   -> src.index_meta
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG indexes.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the index fingerprint still matches the dataset",
    )
    args = parser.parse_args()
    build(force=args.force)


if __name__ == "__main__":
    main()
