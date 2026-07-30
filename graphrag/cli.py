"""Command line interface: ``python -m graphrag <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .config import GraphRAGConfig
from .pipeline import GraphRAG

DEMO_DOCUMENTS: List[str] = [
    "Machine learning is a subset of artificial intelligence that enables "
    "computers to learn patterns from data without being explicitly programmed.",
    "Deep learning is a branch of machine learning that uses neural networks "
    "with multiple layers to model complex patterns in large datasets.",
    "Natural language processing allows computers to understand and generate "
    "human language. Modern natural language processing systems are built on "
    "deep learning and transformer architectures.",
    "Computer vision enables machines to interpret visual information from "
    "images and video. Convolutional neural networks power most computer "
    "vision systems and are trained with deep learning.",
    "Reinforcement learning trains agents through rewards and penalties in an "
    "environment. Reinforcement learning is a branch of machine learning used "
    "for robotics and game playing.",
    "Retrieval augmented generation combines a retrieval system with a "
    "language model so answers are grounded in source documents. Knowledge "
    "graphs improve retrieval augmented generation by linking related entities.",
]


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _config_from_args(args: argparse.Namespace) -> GraphRAGConfig:
    return GraphRAGConfig.from_env(
        llm_model=getattr(args, "model", None),
        extractor=getattr(args, "extractor", None),
        storage_dir=getattr(args, "storage", None),
        chunk_size=getattr(args, "chunk_size", None),
        retrieval_top_k=getattr(args, "top_k", None),
    )


def _load_system(args: argparse.Namespace) -> GraphRAG:
    storage = Path(args.storage)
    if not storage.exists():
        raise SystemExit(
            f"No index found at {storage}. Run 'python -m graphrag index <path>' first."
        )
    return GraphRAG.load(storage, config=_config_from_args(args))


def cmd_index(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    storage = Path(config.storage_dir)
    system = (
        GraphRAG.load(storage, config=config)
        if args.append and storage.exists()
        else GraphRAG(config=config)
    )
    added = system.add_path(args.path)
    system.save(storage)
    print(f"Indexed {added} new chunks from {args.path}")
    print(json.dumps(system.stats(), indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    system = _load_system(args)
    result = system.query(args.question, top_k=args.top_k, generate=not args.no_generate)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"\nQuestion: {args.question}")
    print(f"\nAnswer:\n{result['answer']}")
    if result["entities"]:
        print(f"\nSeed entities: {', '.join(result['entities'])}")
    print("\nSources:")
    for index, item in enumerate(result["retrieved"], start=1):
        signals = " ".join(f"{k}={v:.4f}" for k, v in item["signals"].items())
        print(f"  [{index}] {item['source']} (score={item['score']:.4f} {signals})")
        print(f"      {item['text'][:180]}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    system = _load_system(args)
    print(json.dumps(system.stats(), indent=2))
    return 0


def cmd_entities(args: argparse.Namespace) -> int:
    system = _load_system(args)
    if args.entity:
        related = system.related_entities(args.entity, depth=args.depth)
        if not related:
            print(f"'{args.entity}' is not in the graph (or has no neighbours).")
            return 1
        print(f"Entities within {args.depth} hop(s) of '{args.entity}':")
        for key in related:
            print(f"  - {system.graph.entity_name(key)}")
        return 0

    stats = system.graph.stats()
    print(f"{stats['entities']} entities, {stats['relationships']} relationships")
    for item in stats["top_entities"]:
        print(f"  - {item['entity']} (degree {item['degree']})")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    system = _load_system(args)
    output = Path(args.output)
    if output.suffix.lower() == ".graphml":
        system.graph.export_graphml(output)
    else:
        system.graph.save(output)
    print(f"Exported graph to {output}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    system = GraphRAG(config=config)
    system.add_documents(DEMO_DOCUMENTS, source="demo")

    print("=" * 68)
    print("GraphRAG - Graph-based Retrieval Augmented Generation")
    print("=" * 68)
    print(json.dumps(system.stats(), indent=2))

    questions = [
        "What is machine learning?",
        "How is deep learning related to computer vision?",
        "Why do knowledge graphs help retrieval augmented generation?",
    ]
    for question in questions:
        result = system.query(question)
        print(f"\nQuestion: {question}")
        print(f"Answer:   {result['answer']}")
        print(f"Entities: {', '.join(result['entities']) or '-'}")
        print(f"Sources:  {len(result['sources'])} passages")

    if args.save:
        target = system.save(args.storage)
        print(f"\nSaved demo index to {target}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphrag",
        description="Graph-based retrieval augmented generation.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--storage",
        default="storage",
        help="index directory (default: storage)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="index a file or directory")
    index_parser.add_argument("path", help="file or directory of text/markdown")
    index_parser.add_argument(
        "--append", action="store_true", help="add to an existing index"
    )
    index_parser.add_argument("--chunk-size", type=int, dest="chunk_size")
    index_parser.add_argument("--extractor", choices=["rule", "llm"])
    index_parser.add_argument("--model", help="Claude model id for llm extraction")
    index_parser.set_defaults(func=cmd_index)

    query_parser = subparsers.add_parser("query", help="ask a question")
    query_parser.add_argument("question")
    query_parser.add_argument("--top-k", type=int, dest="top_k")
    query_parser.add_argument("--model", help="Claude model id")
    query_parser.add_argument(
        "--no-generate", action="store_true", help="retrieve only"
    )
    query_parser.add_argument("--json", action="store_true", help="raw JSON output")
    query_parser.set_defaults(func=cmd_query)

    stats_parser = subparsers.add_parser("stats", help="show index statistics")
    stats_parser.set_defaults(func=cmd_stats)

    entities_parser = subparsers.add_parser("entities", help="inspect the graph")
    entities_parser.add_argument("entity", nargs="?", help="entity to expand")
    entities_parser.add_argument("--depth", type=int, default=1)
    entities_parser.set_defaults(func=cmd_entities)

    export_parser = subparsers.add_parser("export", help="export the graph")
    export_parser.add_argument("output", help="target .json or .graphml file")
    export_parser.set_defaults(func=cmd_export)

    demo_parser = subparsers.add_parser("demo", help="run an end-to-end demo")
    demo_parser.add_argument("--model", help="Claude model id")
    demo_parser.add_argument("--extractor", choices=["rule", "llm"])
    demo_parser.add_argument(
        "--save", action="store_true", help="persist the demo index to --storage"
    )
    demo_parser.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
