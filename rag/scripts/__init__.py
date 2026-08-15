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
from .hybrid_search import (
    HybridSearcher,
    PORTUGUESE_MONTHS,
    PORTUGUESE_STOPWORDS,
    calculate_recency_score,
    normalize_text,
    parse_date_value,
    tokenize_portuguese,
)
from .reranker import ReRanker
from .hallucination_checker import HallucinationChecker
from .contact_catalog import (
    CORPORATE_CONTACT_CATALOG,
    route_fallback_contact,
    format_fallback_message,
    normalize_catalog_text,
)
from .multichannel_formatter import (
    format_multichannel_response,
    extract_tldr_and_details,
    sanitize_channel_name,
    format_citation_line,
)
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
    "PORTUGUESE_MONTHS",
    "PORTUGUESE_STOPWORDS",
    "normalize_text",
    "tokenize_portuguese",
    "parse_date_value",
    "calculate_recency_score",
    "ReRanker",
    "HallucinationChecker",
    "CORPORATE_CONTACT_CATALOG",
    "route_fallback_contact",
    "format_fallback_message",
    "normalize_catalog_text",
    "format_multichannel_response",
    "extract_tldr_and_details",
    "sanitize_channel_name",
    "format_citation_line",
    "GroundedQAAgent",
]
