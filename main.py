"""Entry point — ask the RAG system a question.

Usage:
    python main.py                       # interactive chat loop
    python main.py "คำถามของคุณ"          # single question
"""

import argparse


def ask(question: str) -> None:
    """Answer one question and print it with its sources."""
    # TODO: build RAGPipeline from src.rag_pipeline and print result
    raise NotImplementedError


def chat() -> None:
    """Interactive loop that keeps conversation history between turns."""
    # TODO: build RAGPipeline with src.memory.ConversationMemory attached
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG system.")
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    args = parser.parse_args()

    if args.question:
        ask(args.question)
    else:
        chat()


if __name__ == "__main__":
    main()
