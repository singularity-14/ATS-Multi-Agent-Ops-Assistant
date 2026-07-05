"""
MCP Documentation Server — provides RAG-based search over accelerator documentation.
Exposes tools: search_accelerator_docs, get_document_by_id, list_doc_categories
"""
import json
import logging
import os
from pathlib import Path
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

DOCS_PATH = Path('data/docs/accelerator_docs.json')


class DocumentationServer:
    """MCP-compatible documentation tool server with simple keyword search."""

    name = 'docs'

    def __init__(self):
        self.docs = self._load_docs()
        logger.info(f'Loaded {len(self.docs)} accelerator documents')

    def _load_docs(self) -> list[dict]:
        """Load accelerator docs from JSON file, returning empty list if not found."""
        if DOCS_PATH.exists():
            with open(DOCS_PATH, encoding='utf-8') as f:
                data = json.load(f)
            # Support both a bare list and a wrapped {"documents": [...]} structure
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get('documents', [])
            logger.warning('Unexpected format in docs file; expected list or dict with "documents" key.')
            return []
        logger.warning(f'Docs file not found at {DOCS_PATH}. Starting with empty document set.')
        return []

    def search_accelerator_docs(self, query: str, top_k: int = 3) -> str:
        """Search accelerator operations documentation using keyword matching.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.

        Returns:
            Formatted string of matching documents ranked by relevance.
        """
        if not self.docs:
            return 'Documentation database is empty. No documents available to search.'

        query_terms = query.lower().split()
        scored: list[tuple[int, dict]] = []

        for doc in self.docs:
            # Build a combined searchable text blob from title, content, and keywords
            text = (
                f"{doc.get('title', '')} "
                f"{doc.get('content', '')} "
                f"{' '.join(doc.get('keywords', []))}"
            ).lower()

            # BM25-style term frequency scoring (simplified)
            score = sum(text.count(term) for term in query_terms)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = scored[:top_k]

        if not results:
            return 'No relevant documentation found for the query.'

        output = []
        for score, doc in results:
            section = (
                f"## {doc.get('title', 'Untitled')}\n"
                f"Category: {doc.get('category', 'N/A')}\n"
                f"Subsystem: {doc.get('subsystem', 'N/A')}\n"
                f"Relevance Score: {score}\n\n"
                f"{doc.get('content', '')}\n"
            )
            output.append(section)

        return '\n---\n'.join(output)

    def get_document_by_id(self, doc_id: str) -> str:
        """Retrieve a specific document by its unique ID.

        Args:
            doc_id: The unique document identifier.

        Returns:
            JSON-formatted document, or an error message if not found.
        """
        for doc in self.docs:
            if doc.get('id') == doc_id:
                return json.dumps(doc, indent=2)
        return f'Document {doc_id} not found.'

    def list_doc_categories(self) -> str:
        """List all available documentation categories with document counts.

        Returns:
            Formatted string listing each category and its document count.
        """
        if not self.docs:
            return 'Documentation database is empty. No categories available.'

        categories: dict[str, int] = {}
        for doc in self.docs:
            cat = doc.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1

        lines = [
            f'- {cat}: {count} document{"s" if count != 1 else ""}'
            for cat, count in sorted(categories.items())
        ]
        return 'Available documentation categories:\n' + '\n'.join(lines)

    def get_tools(self) -> dict[str, Any]:
        """Return a mapping of tool names to callable methods."""
        return {
            'search_accelerator_docs': self.search_accelerator_docs,
            'get_document_by_id': self.get_document_by_id,
            'list_doc_categories': self.list_doc_categories,
        }


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_server: DocumentationServer | None = None


def get_docs_server() -> DocumentationServer:
    """Return the module-level singleton DocumentationServer instance."""
    global _server
    if _server is None:
        _server = DocumentationServer()
    return _server


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    server = get_docs_server()
    print(server.list_doc_categories())
    print(server.search_accelerator_docs('beam injection energy'))
