"""
GraphRAG - Graph-based Retrieval Augmented Generation
A RAG system using knowledge graphs for enhanced information retrieval.

This is the demo entrypoint. The implementation lives in the ``graphrag``
package; use ``python -m graphrag --help`` for the full command line.
"""

import sys

from graphrag.cli import main as cli_main

if __name__ == "__main__":
    argv = sys.argv[1:] or ["demo"]
    raise SystemExit(cli_main(argv))
