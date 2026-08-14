"""
Adversarial Test Suite - Tier 5 Hardening
Coverage:
- Section 1: PDF Processor (rag_pdf_processor.py) [test_adv_pdf_*]
- Section 2: Vector Indexer (vector_indexer.py) [test_adv_indexer_*]
- Section 3: Hybrid Searcher (hybrid_search.py) [test_adv_hybrid_*]
- Section 4: ReRanker (reranker.py) [test_adv_reranker_*]
- Section 5: Grounded QA Agent (grounded_qa_agent.py) [test_adv_qa_*]

Tests edge cases, boundary conditions, invalid inputs, unhandled exceptions,
and potential logic defects.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
import pytest
import numpy as np

from rag.scripts.rag_pdf_processor import (
    check_pdftotext_installed,
    clean_page_text,
    detect_sections,
    create_chunks_from_sections,
    extract_pdf_pages,
    process_all_pdfs,
    DOCUMENT_METADATA_MAP,
)
import rag.scripts.rag_pdf_processor as pdf_proc_module

from rag.scripts.vector_indexer import (
    VectorIndexer,
    MockEmbeddingFunction,
    GoogleGenAIEmbeddingFunction,
)
import rag.scripts.vector_indexer as vi_mod

from rag.scripts.hybrid_search import (
    HybridSearcher,
    PORTUGUESE_STOPWORDS,
    normalize_text,
    tokenize_portuguese,
)
from rag.scripts.reranker import ReRanker
from rag.scripts.grounded_qa_agent import GroundedQAAgent


# ============================================================================
# SECTION 1: ADVERSARIAL TESTS FOR RAG PDF PROCESSOR (test_adv_pdf_*)
# ============================================================================

def test_adv_pdf_check_pdftotext_installed_return_type():
    """Adversarial Test: Verify check_pdftotext_installed returns a boolean."""
    res = check_pdftotext_installed()
    assert isinstance(res, bool)


def test_adv_pdf_extract_pages_nonexistent_file():
    """Adversarial Test: Extract pages from non-existent PDF file should raise FileNotFoundError."""
    non_existent = Path("/non/existent/path/doc_9999.pdf")
    with pytest.raises((FileNotFoundError, RuntimeError)):
        extract_pdf_pages(non_existent)


def test_adv_pdf_extract_pages_directory_path(tmp_path: Path):
    """Adversarial Test: Pass a directory path to extract_pdf_pages instead of a file."""
    dir_path = tmp_path / "test_dir_pdf"
    dir_path.mkdir()
    with pytest.raises((RuntimeError, OSError, FileNotFoundError)):
        extract_pdf_pages(dir_path)


def test_adv_pdf_extract_pages_corrupted_pdf(tmp_path: Path, monkeypatch):
    """Adversarial Test: Pass corrupted binary content as PDF file."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"\x00\x01\x02\x03CORRUPTED_PDF_HEADER_INVALID")
    
    monkeypatch.setattr(pdf_proc_module, "check_pdftotext_installed", lambda: True)
    
    class MockProcessFail:
        returncode = 1
        stdout = ""
        stderr = "Syntax Error: May not be a PDF file (or corrupt page tree)"
    
    monkeypatch.setattr("subprocess.run", lambda cmd, capture_output, text: MockProcessFail())
    
    with pytest.raises(RuntimeError) as exc_info:
        extract_pdf_pages(corrupt_file)
    assert "Syntax Error" in str(exc_info.value)


def test_adv_pdf_extract_pages_all_blank_pages(tmp_path: Path, monkeypatch):
    """Adversarial Test: pdftotext output contains only form feeds and whitespace."""
    blank_pdf = tmp_path / "blank.pdf"
    blank_pdf.write_text("%PDF-1.4 blank content")
    
    monkeypatch.setattr(pdf_proc_module, "check_pdftotext_installed", lambda: True)
    
    class MockProcessBlank:
        returncode = 0
        stdout = "   \n\t  \f\f \n \f"
        stderr = ""
        
    monkeypatch.setattr("subprocess.run", lambda cmd, capture_output, text: MockProcessBlank())
    
    pages = extract_pdf_pages(blank_pdf)
    assert pages == [], "All-blank pages should produce an empty pages list"


def test_adv_pdf_clean_page_text_none_input():
    """Adversarial Test: clean_page_text with None input should raise TypeError or AttributeError."""
    with pytest.raises((AttributeError, TypeError)):
        clean_page_text(None)  # type: ignore


def test_adv_pdf_clean_page_text_unicode_whitespace():
    """Adversarial Test: clean_page_text handling non-breaking space \\xa0 and unusual whitespace."""
    text_with_nbsp = "Mercado\xa0Central\xa024h   com\t\tespaços   especiais."
    cleaned = clean_page_text(text_with_nbsp)
    assert "Mercado" in cleaned
    assert "Central" in cleaned
    assert "24h" in cleaned


def test_adv_pdf_clean_page_text_digit_stripping_boundary():
    """Adversarial Test: Standalone digit removal boundary (3 digits stripped vs 4 digits kept)."""
    raw_text = "100\nLinha A\n999\nLinha B\n1000\nLinha C\n1"
    cleaned = clean_page_text(raw_text)
    lines = cleaned.split("\n")
    assert "100" not in lines
    assert "999" not in lines
    assert "1000" in lines
    assert "1" not in lines


def test_adv_pdf_clean_page_text_pure_noise():
    """Adversarial Test: Input with exclusively corporate headers and page numbers."""
    noise = "MERCADO CENTRAL 24H\nPágina 1 de 10\n100\nMERCADO CENTRAL 24H LTDA\nPágina 2"
    cleaned = clean_page_text(noise)
    assert cleaned == ""


def test_adv_pdf_clean_page_text_case_variations():
    """Adversarial Test: Header removal with mixed casing variations."""
    mixed = "Mercado Central 24h\npágina 3 de 15\nConteúdo substantivo mantido."
    cleaned = clean_page_text(mixed)
    assert "Mercado Central 24h" not in cleaned
    assert "página 3 de 15" not in cleaned
    assert "Conteúdo substantivo mantido." in cleaned


def test_adv_pdf_detect_sections_missing_dict_keys():
    """Adversarial Test: detect_sections called with dicts missing page_num or text keys."""
    invalid_pages = [{"text": "Sem page num"}, {"page_num": 2}]
    with pytest.raises(KeyError):
        detect_sections(invalid_pages)  # type: ignore


def test_adv_pdf_detect_sections_discontinuous_pages():
    """Adversarial Test: detect_sections with non-contiguous page numbers (e.g. 1, 5, 12)."""
    pages = [
        {"page_num": 1, "text": "1. INTRODUÇÃO\nTexto da introdução."},
        {"page_num": 5, "text": "Texto continuado na página cinco."},
        {"page_num": 12, "text": "2. DA OPERAÇÃO\nTexto da seção dois."}
    ]
    sections = detect_sections(pages)
    assert len(sections) == 2
    sec1 = sections[0]
    assert sec1["title"] == "1. INTRODUÇÃO"
    assert sec1["page_start"] == 1
    assert sec1["page_end"] == 5
    sec2 = sections[1]
    assert sec2["title"] == "2. DA OPERAÇÃO"
    assert sec2["page_start"] == 12
    assert sec2["page_end"] == 12


def test_adv_pdf_detect_sections_header_length_boundary():
    """Adversarial Test: Section header pattern matching regex length boundary (<100 vs >=100 chars)."""
    h_99 = "1. " + "A" * 95  # 98 chars
    h_100 = "1. " + "B" * 97  # 100 chars
    pages = [{"page_num": 1, "text": f"{h_99}\nTexto A.\n\n{h_100}\nTexto B."}]
    sections = detect_sections(pages)
    titles = [s["title"] for s in sections]
    assert h_99 in titles
    assert h_100 not in titles


def test_adv_pdf_create_chunks_long_single_paragraph():
    """Adversarial Test: Single paragraph without double-newlines exceeding max_chars (1200)."""
    long_paragraph = "Este é um parágrafo extremamente longo sem nenhuma quebra de linha dupla. " * 30  # ~2150 chars
    sections = [{"title": "1. SEÇÃO LONGA", "text": long_paragraph, "page_start": 1, "page_end": 1}]
    chunks = create_chunks_from_sections(
        sections=sections,
        file_name="LongPara.pdf",
        file_path="/path/LongPara.pdf",
        doc_meta={},
        max_chars=1200,
        overlap_chars=200,
    )
    assert len(chunks) >= 2
    oversized_chunks = [c for c in chunks if len(c["text"]) > 1200]
    assert len(oversized_chunks) == 0, "Single long paragraph should be sub-chunked so no chunk exceeds max_chars"


def test_adv_pdf_create_chunks_none_doc_meta():
    """Adversarial Test: create_chunks_from_sections with doc_meta=None (Handled safely without AttributeError)."""
    sections = [{"title": "1. INTRO", "text": "Texto curto", "page_start": 1, "page_end": 1}]
    chunks = create_chunks_from_sections(
        sections=sections,
        file_name="Test.pdf",
        file_path="/path/Test.pdf",
        doc_meta=None,  # type: ignore
    )
    assert len(chunks) == 1
    assert chunks[0]["category"] == "Geral"


def test_adv_pdf_create_chunks_invalid_overlap_or_max_chars():
    """Adversarial Test: create_chunks_from_sections where overlap_chars >= max_chars."""
    sections = [{"title": "1. INTRO", "text": "Texto de teste para validar overlap alto.", "page_start": 1, "page_end": 1}]
    chunks = create_chunks_from_sections(
        sections=sections,
        file_name="TestOverlap.pdf",
        file_path="/path/TestOverlap.pdf",
        doc_meta={},
        max_chars=50,
        overlap_chars=100,
    )
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_adv_pdf_create_chunks_filename_with_accents_and_symbols():
    """Adversarial Test: create_chunks_from_sections with accented characters and special symbols in file_name."""
    sections = [{"title": "1. INTRO", "text": "Texto", "page_start": 1, "page_end": 1}]
    chunks = create_chunks_from_sections(
        sections=sections,
        file_name="Política_de_Reembolso_&_Devoluções (2026).pdf",
        file_path="/path/Política.pdf",
        doc_meta={},
    )
    assert len(chunks) == 1
    assert chunks[0]["file_name"] == "Política_de_Reembolso_&_Devoluções (2026).pdf"
    assert chunks[0]["chunk_id"].startswith("Política_de_Reembolso_&_Devoluções (2026)_CHK_")


def test_adv_pdf_process_all_pdfs_nonexistent_directory():
    """Adversarial Test: process_all_pdfs with non-existent input directory."""
    non_existent_dir = Path("/path/does/not/exist/at/all/12345")
    out_dir = Path("/tmp/out_test_12345")
    process_all_pdfs(non_existent_dir, out_dir)
    assert (out_dir / "processed_rag_chunks.json").exists()


# ============================================================================
# SECTION 2: ADVERSARIAL TESTS FOR VECTOR INDEXER (test_adv_indexer_*)
# ============================================================================

def test_adv_indexer_null_metadata_fields(temp_chroma_db):
    """Adversarial Test: Chunk dict with None in integer metadata fields (e.g. page_start=None).
    Exposes TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'.
    """
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    chunk_with_none_meta = [{
        "chunk_id": "NULL_META_001",
        "text": "Texto com metadados nulos",
        "file_name": "test.pdf",
        "page_start": None,
        "page_end": None,
        "char_count": None,
        "word_count": None,
    }]
    with pytest.raises(TypeError):
        indexer.index_chunks(chunk_with_none_meta)


def test_adv_indexer_invalid_string_metadata_integers(temp_chroma_db):
    """Adversarial Test: Chunk dict with non-numeric string in page_start.
    Exposes ValueError: invalid literal for int() with base 10.
    """
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    chunk_invalid_str = [{
        "chunk_id": "STR_META_001",
        "text": "Texto com página em formato string",
        "page_start": "invalid_page_num",
    }]
    with pytest.raises(ValueError):
        indexer.index_chunks(chunk_invalid_str)


def test_adv_indexer_zero_batch_size(temp_chroma_db, mock_chunks):
    """Adversarial Test: index_chunks called with batch_size=0.
    Exposes ValueError: range() arg 3 must not be zero.
    """
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    with pytest.raises(ValueError):
        indexer.index_chunks(mock_chunks, batch_size=0)


def test_adv_indexer_negative_batch_size(temp_chroma_db, mock_chunks):
    """Adversarial Test: index_chunks called with negative batch_size (-10)."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    res = indexer.index_chunks(mock_chunks, batch_size=-10)
    assert res == 0, "Negative batch_size range produces empty loop and returns count 0"


def test_adv_indexer_search_zero_top_k(temp_chroma_db, mock_chunks):
    """Adversarial Test: search called with top_k=0 on indexed database.
    Exposes ValueError in ChromaDB n_results=0.
    """
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    with pytest.raises(ValueError):
        indexer.search("Frete grátis", top_k=0)


def test_adv_indexer_search_negative_top_k(temp_chroma_db, mock_chunks):
    """Adversarial Test: search called with top_k=-5 on indexed database."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    with pytest.raises(ValueError):
        indexer.search("Frete grátis", top_k=-5)


def test_adv_indexer_mock_embedding_none_input():
    """Adversarial Test: MockEmbeddingFunction called with None in input list is handled safely."""
    embedder = MockEmbeddingFunction(dimension=768)
    vecs = embedder([None])  # type: ignore
    assert len(vecs) == 1
    assert len(vecs[0]) == 768


def test_adv_indexer_mock_embedding_custom_dimension_and_norm():
    """Adversarial Test: MockEmbeddingFunction with custom dimensions (128, 1536) and L2 normalization."""
    emb_128 = MockEmbeddingFunction(dimension=128)
    vecs_128 = emb_128(["Texto 1", "Texto 2"])
    assert len(vecs_128[0]) == 128
    norm_128 = np.linalg.norm(vecs_128[0])
    assert pytest.approx(norm_128, 1e-5) == 1.0

    emb_1536 = MockEmbeddingFunction(dimension=1536)
    vecs_1536 = emb_1536(["Texto 1"])
    assert len(vecs_1536[0]) == 1536
    norm_1536 = np.linalg.norm(vecs_1536[0])
    assert pytest.approx(norm_1536, 1e-5) == 1.0


def test_adv_indexer_index_chunks_malformed_json_file(tmp_path: Path):
    """Adversarial Test: index_chunks with JSON file path containing syntax error."""
    indexer = VectorIndexer(use_mock=True)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{this is not valid json")
    with pytest.raises(json.JSONDecodeError):
        indexer.index_chunks(str(bad_json))


def test_adv_indexer_index_chunks_dict_instead_of_list_json(tmp_path: Path):
    """Adversarial Test: index_chunks with JSON file path containing a Dict instead of a List."""
    indexer = VectorIndexer(use_mock=True)
    dict_json = tmp_path / "dict.json"
    dict_json.write_text('{"key": "value"}')
    res = indexer.index_chunks(dict_json)
    assert res == 0, "Dict input instead of List should be caught by Guard 3 and return 0"


def test_adv_indexer_search_empty_and_whitespace_query(temp_chroma_db, mock_chunks):
    """Adversarial Test: search with empty string query '' or whitespace '   '."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    
    res_empty = indexer.search("", top_k=2)
    assert isinstance(res_empty, list)
    
    res_space = indexer.search("   ", top_k=2)
    assert isinstance(res_space, list)


def test_adv_indexer_google_embedder_api_error_fallback(temp_chroma_db, monkeypatch):
    """Adversarial Test: GoogleGenAIEmbeddingFunction raising RuntimeError during embed_texts falls back to mock."""
    indexer = VectorIndexer(use_mock=False, db_path=temp_chroma_db)
    
    class FakeGoogleEmbedder:
        client = True
        def embed_texts(self, texts):
            raise RuntimeError("API quota exceeded 429")
            
    indexer.google_embedder = FakeGoogleEmbedder()  # type: ignore
    indexer.use_mock = False
    
    vecs = indexer.get_embeddings(["Texto de teste"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 768, "Fallback to mock embedder should return 768-dim vector"


def test_adv_indexer_upsert_duplicate_chunk_ids(temp_chroma_db):
    """Adversarial Test: Indexing duplicate chunk_ids in succession."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.clear_collection()
    chunks1 = [{"chunk_id": "DUP_001", "text": "Versão 1 do texto"}]
    chunks2 = [{"chunk_id": "DUP_001", "text": "Versão 2 atualizada do texto"}]
    
    count1 = indexer.index_chunks(chunks1)
    assert count1 == 1
    count2 = indexer.index_chunks(chunks2)
    assert count2 == 1, "Upsert should replace existing ID, total collection count remains 1"
    
    res = indexer.search("atualizada", top_k=1)
    assert len(res) == 1
    assert res[0]["text"] == "Versão 2 atualizada do texto"


def test_adv_indexer_clear_collection_and_reindex(temp_chroma_db, mock_chunks):
    """Adversarial Test: Index chunks, search, clear collection, verify 0, re-index and search."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.clear_collection()
    indexer.index_chunks(mock_chunks)
    assert indexer.collection.count() == len(mock_chunks)
    
    indexer.clear_collection()
    assert indexer.collection.count() == 0
    assert indexer.search("VIP", top_k=2) == []
    
    indexer.index_chunks(mock_chunks)
    assert indexer.collection.count() == len(mock_chunks)


# ============================================================================
# SECTION 3: ADVERSARIAL TESTS FOR HYBRID SEARCHER (test_adv_hybrid_*)
# ============================================================================

def test_adv_hybrid_empty_corpus_init(temp_chroma_db):
    """Adversarial Test: Initialize HybridSearcher with empty chunks list []."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=[])
    res = searcher.search("Frete grátis", top_k=5)
    assert res == []


def test_adv_hybrid_missing_chunk_id(temp_chroma_db):
    """Adversarial Test: Chunks list missing 'chunk_id' key should raise KeyError."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    bad_chunks = [{"text": "Texto sem chunk_id"}]
    with pytest.raises(KeyError):
        HybridSearcher(vector_indexer=indexer, chunks_data=bad_chunks)


def test_adv_hybrid_none_text_in_chunk(temp_chroma_db):
    """Adversarial Test: Chunk with text=None should raise TypeError or AttributeError during tokenization."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    none_text_chunk = [{"chunk_id": "CHK_NONE_01", "text": None}]
    with pytest.raises((TypeError, AttributeError)):
        HybridSearcher(vector_indexer=indexer, chunks_data=none_text_chunk)  # type: ignore


def test_adv_hybrid_alpha_clamping(temp_chroma_db, mock_chunks):
    """Adversarial Test: Alpha outside [0, 1] range should be clamped to [0.0, 1.0]."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    
    searcher_neg = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks, alpha=-0.5)
    assert searcher_neg.alpha == 0.0
    
    searcher_over = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks, alpha=2.5)
    assert searcher_over.alpha == 1.0
    
    res = searcher_neg.search("Frete grátis", alpha=-2.0)
    assert isinstance(res, list)


def test_adv_hybrid_query_none(temp_chroma_db, mock_chunks):
    """Adversarial Test: search with query=None raises TypeError/AttributeError."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    with pytest.raises((AttributeError, TypeError)):
        searcher.search(query=None)  # type: ignore


def test_adv_hybrid_nonexistent_and_invalid_json_file(tmp_path: Path, temp_chroma_db):
    """Adversarial Test: Non-existent file path raises FileNotFoundError; invalid JSON raises JSONDecodeError."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    
    non_existent = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        HybridSearcher(vector_indexer=indexer, chunks_data=non_existent)
        
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{broken json syntax")
    with pytest.raises(json.JSONDecodeError):
        HybridSearcher(vector_indexer=indexer, chunks_data=invalid_json)


def test_adv_hybrid_metadata_filter_no_match(temp_chroma_db, mock_chunks):
    """Adversarial Test: search with metadata_filter that matches no documents returns empty list."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    
    res = searcher.search("Frete", metadata_filter={"file_name": "Inexistente.pdf"})
    assert res == []


def test_adv_hybrid_bm25_all_zero_scores(temp_chroma_db, mock_chunks):
    """Adversarial Test: Query with zero token matches in corpus returns sparse_score 0 without div by zero."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    
    res = searcher.search("xyz999unmatchedwordqqq")
    assert len(res) > 0
    for r in res:
        assert r["sparse_score"] == 0.0


def test_adv_hybrid_zero_and_negative_top_k(temp_chroma_db, mock_chunks):
    """Adversarial Test: search with top_k=0 or negative top_k."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    
    res_zero = searcher.search("Frete", top_k=0)
    assert res_zero == []
    
    res_neg = searcher.search("Frete", top_k=-5)
    assert res_neg == []


# ============================================================================
# SECTION 4: ADVERSARIAL TESTS FOR RERANKER (test_adv_reranker_*)
# ============================================================================

def test_adv_reranker_model_name_overwrites_rrf_method():
    """Adversarial Test: ReRanker(method='rrf', model_name='invalid_model') overwriting method to 'hybrid_fusion'.
    Exposes white-box defect where user-selected 'rrf' method is destroyed upon failed model load.
    """
    reranker = ReRanker(method="rrf", model_name="invalid_nonexistent_model_123")
    assert reranker.method == "rrf", f"Method should remain 'rrf' but got '{reranker.method}'"


def test_adv_reranker_none_values_in_chunk_fields():
    """Adversarial Test: search_results with None in text, section_title, or file_name.
    Exposes AttributeError: 'NoneType' object has no attribute 'lower'.
    """
    reranker = ReRanker(method="hybrid_fusion")
    none_fields_results = [{
        "chunk_id": "CHK_NONE_FIELDS_01",
        "text": None,
        "section_title": None,
        "file_name": None,
        "hybrid_score": 0.5,
    }]
    with pytest.raises(AttributeError):
        reranker.rerank(query="Frete", search_results=none_fields_results)  # type: ignore


def test_adv_reranker_none_in_scores_sorting():
    """Adversarial Test: search_results with None in dense_score or sparse_score.
    Exposes TypeError during sorting or comparison.
    """
    reranker = ReRanker(method="rrf")
    none_score_results = [
        {"chunk_id": "CHK_NONE_SCORE_01", "text": "Texto 1", "dense_score": None, "sparse_score": None},
        {"chunk_id": "CHK_NONE_SCORE_02", "text": "Texto 2", "dense_score": 0.8, "sparse_score": 0.5},
    ]
    with pytest.raises(TypeError):
        reranker.rerank(query="Frete", search_results=none_score_results)  # type: ignore


def test_adv_reranker_negative_rrf_k_zero_division():
    """Adversarial Test: rrf_k = -1 causes ZeroDivisionError when rank = 1 (rrf_k + rank = 0).
    Exposes ZeroDivisionError: float division by zero.
    """
    reranker = ReRanker(method="rrf", rrf_k=-1)
    results = [{"chunk_id": "CHK_01", "text": "Texto", "dense_score": 0.9, "sparse_score": 0.5}]
    with pytest.raises(ZeroDivisionError):
        reranker.rerank(query="Frete", search_results=results)


def test_adv_reranker_negative_top_k_slicing():
    """Adversarial Test: top_k = -1 returns all items except last item due to Python negative slicing [:top_k]."""
    reranker = ReRanker(method="hybrid_fusion")
    results = [
        {"chunk_id": f"CHK_0{i}", "text": f"Texto {i}", "hybrid_score": 0.5 + i*0.1}
        for i in range(1, 5)
    ]
    reranked = reranker.rerank(query="Texto", search_results=results, top_k=-1)
    assert reranked == [], f"top_k=-1 should return empty list but returned {len(reranked)} items"


def test_adv_reranker_missing_chunk_id_key():
    """Adversarial Test: search_results item missing 'chunk_id' raises KeyError."""
    reranker = ReRanker(method="hybrid_fusion")
    bad_results = [{"text": "Sem chunk_id", "hybrid_score": 0.8}]
    with pytest.raises(KeyError):
        reranker.rerank(query="Frete", search_results=bad_results)


def test_adv_reranker_none_query():
    """Adversarial Test: query=None raises AttributeError in query.lower()."""
    reranker = ReRanker(method="hybrid_fusion")
    results = [{"chunk_id": "CHK_01", "text": "Texto", "hybrid_score": 0.5}]
    with pytest.raises(AttributeError):
        reranker.rerank(query=None, search_results=results)  # type: ignore


def test_adv_reranker_empty_or_none_search_results():
    """Adversarial Test: empty list or None search_results returns empty list."""
    reranker = ReRanker(method="hybrid_fusion")
    assert reranker.rerank("query", []) == []
    assert reranker.rerank("query", None) == []  # type: ignore


def test_adv_reranker_custom_feature_fusion_weights():
    """Adversarial Test: ReRanker with custom feature fusion weights computes expected score."""
    reranker = ReRanker(
        method="hybrid_fusion",
        hybrid_weight=0.5,
        title_weight=0.3,
        file_weight=0.1,
        text_weight=0.1,
    )
    item = {
        "chunk_id": "CHK_WEIGHTS",
        "text": "texto sobre entrega expressa",
        "section_title": "entrega expressa",
        "file_name": "guia.pdf",
        "hybrid_score": 0.8,
    }
    reranked = reranker.rerank(query="entrega expressa", search_results=[item], top_k=1)
    assert len(reranked) == 1
    # expected: (0.8 * 0.5) + (2/2 * 0.3) + (0/2 * 0.1) + (2/2 * 0.1) = 0.4 + 0.3 + 0.0 + 0.1 = 0.8
    assert reranked[0]["rerank_score"] == 0.8


# ============================================================================
# SECTION 5: ADVERSARIAL TESTS FOR GROUNDED QA AGENT (test_adv_qa_*)
# ============================================================================

def test_adv_qa_none_in_chunk_text_or_title(temp_chroma_db, mock_chunks):
    """Adversarial Test: GroundedQAAgent with None in text or section_title raises TypeError in _is_query_grounded."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    bad_chunks = [{"chunk_id": "CHK_NONE", "text": None, "section_title": None, "rerank_score": 0.8}]
    with pytest.raises(TypeError):
        agent._is_query_grounded("entrega", bad_chunks)  # type: ignore


def test_adv_qa_empty_or_whitespace_query_grounding_false_positive(temp_chroma_db, mock_chunks):
    """Adversarial Test: Empty '' or whitespace '   ' query declared grounded (True) if top_score >= 0.30.
    Exposes white-box defect where empty queries get flagged as grounded and generate answers.
    """
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    top_chunks = [{"chunk_id": "CHK_HIGH", "text": "Texto relevante", "rerank_score": 0.5}]
    
    grounded_empty = agent._is_query_grounded("", top_chunks)
    assert not grounded_empty, "Empty query '' should never be grounded"
    
    grounded_space = agent._is_query_grounded("   ", top_chunks)
    assert not grounded_space, "Whitespace query '   ' should never be grounded"


def test_adv_qa_none_rerank_score(temp_chroma_db, mock_chunks):
    """Adversarial Test: rerank_score=None in chunk raises TypeError in comparison top_score < 0.45."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    bad_score_chunks = [{"chunk_id": "CHK_SCORE_NONE", "text": "Texto sem relacao com a busca", "rerank_score": None}]
    with pytest.raises(TypeError):
        agent._is_query_grounded("palavra_absolutamente_inexistente_xyz_123", bad_score_chunks)  # type: ignore


def test_adv_qa_extractive_fallback_missing_fields(temp_chroma_db, mock_chunks):
    """Adversarial Test: Extractive fallback handles missing page_start/end, section_title, file_name safely."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    minimal_chunk = [{
        "chunk_id": "CHK_MINIMAL",
        "text": "Texto explicativo sobre o frete grátis do aplicativo.",
        "rerank_score": 0.8,
    }]
    
    res = agent._generate_extractive_answer("frete grátis", minimal_chunk)
    assert "Documento Oficial" in res["answer"]
    assert "Seção" in res["answer"]
    assert len(res["citations"]) == 1


def test_adv_qa_llm_api_generic_exception_uncaught(temp_chroma_db, mock_chunks):
    """Adversarial Test: Custom 3rd party exception in Gemini API call is caught and falls back to extractive.
    Exposes white-box defect where line 248 only catches (RuntimeError, AttributeError, ValueError).
    """
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    class ConnectionTimeoutError(Exception):
        pass
        
    class FakeGenAIModels:
        def generate_content(self, **kwargs):
            raise ConnectionTimeoutError("504 Gateway Timeout")
            
    class FakeGenAIClient:
        models = FakeGenAIModels()
        
    agent.genai_client = FakeGenAIClient()
    
    # Should catch ConnectionTimeoutError and fallback to extractive answer instead of crashing
    res = agent._generate_llm_answer("frete grátis", "contexto", mock_chunks)
    assert isinstance(res, dict)
    assert "answer" in res


def test_adv_qa_zero_search_or_rerank_k(temp_chroma_db, mock_chunks):
    """Adversarial Test: top_search_k=0 or top_rerank_k=0 returns default no-info response."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    res_zero_search = agent.answer("Frete grátis", top_search_k=0)
    assert "não encontrei informações oficiais" in res_zero_search["answer"]
    
    res_zero_rerank = agent.answer("Frete grátis", top_rerank_k=0)
    assert isinstance(res_zero_rerank, dict)


def test_adv_qa_special_characters_and_script_injection(temp_chroma_db, mock_chunks):
    """Adversarial Test: Query containing script tags, SQL injection characters, and emojis."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    evil_query = "<script>alert('xss')</script> SELECT * FROM users; DROP TABLE chunks; 🚚💨"
    res = agent.answer(evil_query)
    assert isinstance(res, dict)
    assert "answer" in res


def test_adv_qa_extremely_long_query(temp_chroma_db, mock_chunks):
    """Adversarial Test: Very long query (>10,000 characters)."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    long_query = "Qual é o frete grátis para Cliente VIP? " * 300  # ~12,000 chars
    res = agent.answer(long_query)
    assert isinstance(res, dict)
    assert "answer" in res


def test_adv_qa_no_api_key_initialization(monkeypatch, temp_chroma_db, mock_chunks):
    """Adversarial Test: GroundedQAAgent without API keys initializes genai_client=None and runs extractive fallback."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    assert agent.genai_client is None
    
    res = agent.answer("Qual é o frete grátis Cliente VIP Diamante?")
    assert "Com base na documentação oficial" in res["answer"]
    assert len(res["citations"]) > 0
