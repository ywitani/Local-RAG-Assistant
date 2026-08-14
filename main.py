from __future__ import annotations

import argparse
from rag_engine import DemoRAG, FoundryLocalRAG


def print_sources(retrieved):
    print("\nSources used:")
    for doc, score in retrieved:
        print(f"- {doc.source} | chunk {doc.chunk_id} | similarity={score:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Local RAG Assistant using Microsoft Foundry Local")
    parser.add_argument("--demo", action="store_true", help="Run without Foundry Local model downloads")
    parser.add_argument("--folder", default="knowledge_base", help="Folder containing .txt, .md, and .pdf files")
    args = parser.parse_args()

    print("=" * 66)
    print("Local RAG Assistant using Microsoft Foundry Local")
    print("=" * 66)

    if args.demo:
        print("Running in DEMO mode. This validates the app flow without local model downloads.\n")
        rag = DemoRAG(args.folder)
    else:
        print("Running with Microsoft Foundry Local. First run may download models.\n")
        rag = FoundryLocalRAG(args.folder)

    try:
        print("Ask a question about the files in the knowledge_base folder.")
        print("Type 'quit' to exit.\n")

        while True:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in {"q", "quit", "exit"}:
                break

            answer, sources = rag.answer(question)
            print(f"\nAssistant: {answer}")
            print_sources(sources)
            print()
    finally:
        rag.close()
        print("Goodbye.")


if __name__ == "__main__":
    main()
