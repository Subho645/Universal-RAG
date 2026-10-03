"""
CLI for Universal RAG
Usage:
  python cli.py ingest path/to/file_or_folder
  python cli.py query "What is BASUDEB's phone number?"
  python cli.py chat      # interactive mode
"""

import sys
import json
import argparse
from rag_pipeline import UniversalRAG

def main():
    parser = argparse.ArgumentParser(description="Universal RAG CLI")
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents into the vector DB")
    p_ingest.add_argument("paths", nargs="+", help="Files or directories to ingest")
    p_ingest.add_argument("--chunk-size", type=int, default=500)
    p_ingest.add_argument("--chunk-overlap", type=int, default=50)
    p_ingest.add_argument("--embed-model", default="nomic-embed-text")

    # query
    p_query = sub.add_parser("query", help="Ask a single question")
    p_query.add_argument("question", help="Your question")
    p_query.add_argument("--top-k", type=int, default=3)
    p_query.add_argument("--llm-model", default="llama3")
    p_query.add_argument("--embed-model", default="nomic-embed-text")

    # chat
    p_chat = sub.add_parser("chat", help="Interactive Q&A session")
    p_chat.add_argument("--llm-model", default="llama3")
    p_chat.add_argument("--embed-model", default="nomic-embed-text")

    args = parser.parse_args()

    if args.command == "ingest":
        rag = UniversalRAG(
            embedding_model=args.embed_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        rag.ingest(args.paths)
        print("\nIngestion complete.")

    elif args.command == "query":
        rag = UniversalRAG(
            embedding_model=args.embed_model,
            llm_model=args.llm_model,
            top_k=args.top_k,
        )
        result = rag.query(args.question)
        print(f"\nAnswer: {result['answer']}")
        print("\nSources:")
        for s in result["sources"]:
            print(f"  - {s['source']} (relevance: {s['score']})")

    elif args.command == "chat":
        rag = UniversalRAG(
            embedding_model=args.embed_model,
            llm_model=args.llm_model,
        )
        rag.interactive()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
