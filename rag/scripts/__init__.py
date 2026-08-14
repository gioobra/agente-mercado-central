"""
Mercado Central 24h - RAG Scripts Package
"""

from .rag_pdf_processor import (
    DOCUMENT_METADATA_MAP,
    check_pdftotext_installed,
    clean_page_text,
    create_chunks_from_sections,
    detect_sections,
    extract_pdf_pages,
    process_all_pdfs,
)
from .vector_indexer import GoogleGenAIEmbeddingFunction, MockEmbeddingFunction, VectorIndexer
from .hybrid_search import HybridSearcher, PORTUGUESE_STOPWORDS, normalize_text, tokenize_portuguese
from .reranker import ReRanker
from .grounded_qa_agent import GroundedQAAgent

__all__ = [
    "DOCUMENT_METADATA_MAP",
    "check_pdftotext_installed",
    "clean_page_text",
    "detect_sections",
    "create_chunks_from_sections",
    "extract_pdf_pages",
    "process_all_pdfs",
    "GoogleGenAIEmbeddingFunction",
    "MockEmbeddingFunction",
    "VectorIndexer",
    "HybridSearcher",
    "PORTUGUESE_STOPWORDS",
    "normalize_text",
    "tokenize_portuguese",
    "ReRanker",
    "GroundedQAAgent",
]

