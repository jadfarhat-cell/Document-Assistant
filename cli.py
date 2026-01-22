#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="RAG Document Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py ingest document.pdf
  python cli.py ingest ./docs --directory
  python cli.py query "What is the main topic?"
  python cli.py stats
  python cli.py serve --api
  python cli.py serve --ui
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="File or directory path")
    ingest_parser.add_argument(
        "-d", "--directory", action="store_true",
        help="Treat path as directory"
    )

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument(
        "-k", "--top-k", type=int, default=5,
        help="Number of documents to retrieve"
    )

    subparsers.add_parser("stats", help="Show system statistics")

    serve_parser = subparsers.add_parser("serve", help="Start a server")
    serve_group = serve_parser.add_mutually_exclusive_group(required=True)
    serve_group.add_argument("--api", action="store_true", help="Start FastAPI server")
    serve_group.add_argument("--ui", action="store_true", help="Start Gradio UI")

    subparsers.add_parser("clear", help="Clear all indexed documents")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    from src.rag_pipeline import RAGPipeline
    from src.config import settings

    pipeline = RAGPipeline(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
        model=settings.ollama_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )

    if args.command == "ingest":
        path = Path(args.path)
        if args.directory:
            print(f"Ingesting directory: {path}")
            results = pipeline.ingest_directory(str(path))
            print(f"Ingested {len(results)} files:")
            for r in results:
                print(f"  - {r['filename']}: {r['chunks']} chunks")
        else:
            print(f"Ingesting file: {path}")
            result = pipeline.ingest_file(str(path))
            print(f"Ingested: {result['filename']}")
            print(f"  Type: {result['type']}")
            print(f"  Chunks: {result['chunks']}")

    elif args.command == "query":
        print(f"Query: {args.question}\n")

        if not pipeline.llm.check_connection():
            print("Error: Ollama is not running. Start it with 'ollama serve'")
            sys.exit(1)

        result = pipeline.query(args.question, top_k=args.top_k)
        print("Answer:")
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(f"  - {source}")

    elif args.command == "stats":
        stats = pipeline.get_stats()
        print("System Statistics:")
        print(f"  Documents indexed: {stats['vector_store']['count']} chunks")
        print(f"  Ollama connected: {stats['ollama_connected']}")
        print(f"  Current model: {stats['current_model']}")
        if stats['available_models']:
            print(f"  Available models: {', '.join(stats['available_models'])}")

    elif args.command == "serve":
        if args.api:
            print("Starting FastAPI server on http://localhost:8000")
            print("API docs at http://localhost:8000/docs")
            import uvicorn
            uvicorn.run(
                "src.api.main:app",
                host=settings.api_host,
                port=settings.api_port,
                reload=True,
            )
        elif args.ui:
            print("Starting Gradio UI on http://localhost:7861")
            from ui.app import demo
            demo.launch(server_name="0.0.0.0", server_port=7861)

    elif args.command == "clear":
        pipeline.clear()
        print("Cleared all indexed documents.")


if __name__ == "__main__":
    main()
