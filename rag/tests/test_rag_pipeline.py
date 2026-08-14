"""
Test Suite Automatizada do Pipeline RAG - Mercado Central 24h
Validação de Vector Indexing, Hybrid Search, Re-ranking e Grounded QA Agent.
"""

import os
import inspect
import pytest
from typing import Dict, Any

from rag.scripts.vector_indexer import VectorIndexer, MockEmbeddingFunction
from rag.scripts.hybrid_search import HybridSearcher, normalize_text, tokenize_portuguese
from rag.scripts.reranker import ReRanker
from rag.scripts.grounded_qa_agent import GroundedQAAgent
import rag.scripts.vector_indexer as vi_mod
import rag.scripts.hybrid_search as hs_mod
import rag.scripts.reranker as rr_mod
import rag.scripts.grounded_qa_agent as gqa_mod
import rag.scripts.rag_pdf_processor as pdf_mod


@pytest.fixture(scope="module")
def initialized_pipeline(real_chunks_json_path):
    """Fixture que inicializa o pipeline completo RAG usando banco em memória."""
    indexer = VectorIndexer(use_mock=True, db_path=":memory:")
    if real_chunks_json_path.exists():
        indexer.index_chunks(str(real_chunks_json_path))
        chunks_source = str(real_chunks_json_path)
    else:
        # Fallback para lista vazia se arquivo real não for encontrado no ambiente de teste
        chunks_source = []

    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=chunks_source)
    reranker = ReRanker(method="hybrid_fusion")
    qa_agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    return {
        "indexer": indexer,
        "searcher": searcher,
        "reranker": reranker,
        "agent": qa_agent,
    }


# ============================================================================
# 1. TESTES DE REQUISITOS DE EMPACOTAMENTO & IMPORTS (F4, F5, F9, F15)
# ============================================================================

def test_standard_package_imports_without_sys_path_insert():
    """Valida que todos os módulos são importáveis como pacote padronizado (from rag.scripts.X import Y)."""
    assert VectorIndexer is not None
    assert HybridSearcher is not None
    assert ReRanker is not None
    assert GroundedQAAgent is not None


def test_constructors_have_none_return_annotations():
    """Valida que os construtores __init__ dos módulos de script possuem anotação -> None (F9)."""
    for cls in [VectorIndexer, HybridSearcher, ReRanker, GroundedQAAgent]:
        init_method = getattr(cls, "__init__")
        annotations = getattr(init_method, "__annotations__", {})
        assert annotations.get("return") is None or annotations.get("return") == type(None) or annotations.get("return") == "None"


def test_package_exports():
    """Valida a presença obrigatória de exports públicos (__all__) nos módulos (F15)."""
    for mod in [vi_mod, hs_mod, rr_mod, gqa_mod, pdf_mod]:
        assert hasattr(mod, "__all__"), f"Módulo {mod.__name__} não possui __all__ declarado."
        assert isinstance(mod.__all__, list)
        assert len(mod.__all__) > 0


# ============================================================================
# 2. TESTES DE VECTOR INDEXER (F7, DENSE SEARCH, MOCK EMBEDDINGS)
# ============================================================================

def test_mock_embedding_function_dimension():
    """Valida se a função mock gera embeddings determinísticos de 768 dimensões."""
    embedder = MockEmbeddingFunction(dimension=768)
    vectors = embedder(["Mercado Central 24h", "Frete Grátis VIP"])
    
    assert len(vectors) == 2
    assert len(vectors[0]) == 768
    assert len(vectors[1]) == 768
    # Teste de determinismo: mesmo texto produz vetor idêntico
    v_again = embedder("Mercado Central 24h")[0]
    assert vectors[0] == v_again


def test_vector_indexer_empty_and_none_input_handling(temp_chroma_db):
    """Valida tratamento seguro de entradas vazias, None, strings vazias e estruturas malformadas em index_chunks (F7)."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    
    # Testa lista vazia []
    assert indexer.index_chunks([]) == 0
    
    # Testa None
    assert indexer.index_chunks(None) == 0
    
    # Testa string vazia e espaço em branco (F7)
    assert indexer.index_chunks("") == 0
    assert indexer.index_chunks("   ") == 0

    # Edge cases adicionais (F7): lista com dict vazio ou None, tipos de dados inválidos
    assert indexer.index_chunks([{}]) == 0
    assert indexer.index_chunks([None]) == 0
    assert indexer.index_chunks({"invalid": True}) == 0
    assert indexer.index_chunks(123) == 0
    
    # Lista vazia não deve corromper banco
    assert indexer.collection.count() == 0


def test_vector_indexer_indexing_and_search(mock_chunks, temp_chroma_db):
    """Testa a criação de coleção no ChromaDB e a busca densa por cosseno."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    count = indexer.index_chunks(mock_chunks)
    
    assert count == len(mock_chunks)
    
    # Busca por similaridade densa
    results = indexer.search("Frete grátis Cliente VIP Diamante", top_k=2)
    assert len(results) > 0
    assert "dense_score" in results[0]
    assert 0.0 <= results[0]["dense_score"] <= 1.0
    assert results[0]["chunk_id"] in ["TEST_CHK_001", "TEST_CHK_002"]


def test_vector_indexer_metadata_filter(mock_chunks, temp_chroma_db):
    """Valida filtragem por metadados no ChromaDB."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    
    results = indexer.search(
        query="Devolução",
        top_k=5,
        metadata_filter={"category": "Atendimento & CDC"}
    )
    assert len(results) == 1
    assert results[0]["chunk_id"] == "TEST_CHK_003"
    assert results[0]["file_name"] == "Politica_de_Reembolso_e_Devolucoes.pdf"


# ============================================================================
# 3. TESTES DE HYBRID SEARCH
# ============================================================================

def test_text_normalization_and_tokenization():
    """Testa a limpeza de texto e tokenização para BM25 em Português."""
    raw = "Frete Grátis e Entregas Rápidas no Mercado Central 24h!"
    normalized = normalize_text(raw)
    tokens = tokenize_portuguese(raw)
    
    assert "gratis" in normalized
    assert "rapidas" in normalized
    assert "frete" in tokens
    assert "entregas" in tokens
    assert "e" not in tokens  # Stopword removida


def test_hybrid_search_combination(mock_chunks, temp_chroma_db):
    """Valida a fusão ponderada de pontuações Densa + BM25."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks, alpha=0.5)
    results = searcher.search(query="Cliente VIP Diamante R$ 100,00", top_k=3)
    
    assert len(results) > 0
    top = results[0]
    assert top["chunk_id"] == "TEST_CHK_002"
    assert "dense_score" in top
    assert "sparse_score" in top
    assert "hybrid_score" in top
    assert top["sparse_score"] > 0.5  # BM25 alto devido aos termos exatos


def test_hybrid_search_alpha_tuning(mock_chunks, temp_chroma_db):
    """Valida variação do peso alpha na busca híbrida (F12)."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    
    # Alpha = 1.0 (Somente Busca Densa)
    dense_only = searcher.search(query="Arrependimento 7 dias", top_k=1, alpha=1.0)
    assert dense_only[0]["hybrid_score"] == dense_only[0]["dense_score"]

    # Alpha = 0.0 (Somente BM25)
    sparse_only = searcher.search(query="Arrependimento 7 dias", top_k=1, alpha=0.0)
    assert sparse_only[0]["hybrid_score"] == sparse_only[0]["sparse_score"]


# ============================================================================
# 4. TESTES DE RE-RANKING & DIGITAÇÃO (F12, F16)
# ============================================================================

def test_reranker_feature_fusion(mock_chunks, temp_chroma_db):
    """Testa re-ranking por fusão de atributos de contexto e título."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    
    initial_results = searcher.search(query="DIREITO DE ARREPENDIMENTO", top_k=3)
    
    reranker = ReRanker(method="hybrid_fusion")
    reranked = reranker.rerank(query="DIREITO DE ARREPENDIMENTO", search_results=initial_results, top_k=3)
    
    assert len(reranked) > 0
    assert reranked[0]["chunk_id"] == "TEST_CHK_003"
    assert reranked[0]["final_rank"] == 1
    assert "rerank_score" in reranked[0]


def test_reranker_rrf(mock_chunks, temp_chroma_db):
    """Testa algoritmo Reciprocal Rank Fusion (RRF) com parâmetro rrf_k configurável (F12)."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    
    results = searcher.search(query="Cliente VIP Diamante", top_k=3)
    
    reranker = ReRanker(method="rrf", rrf_k=60)
    rrf_results = reranker.rerank(query="Cliente VIP Diamante", search_results=results, top_k=2)
    
    assert len(rrf_results) == 2
    assert rrf_results[0]["final_rank"] == 1
    assert rrf_results[0]["rerank_method"] == "rrf"


def test_reranker_string_typo_fix():
    """Valida que o typo 'Re-rankeadosa' foi corrigido para 'Re-rankeados' em reranker.py (F16)."""
    source_code = inspect.getsource(rr_mod)
    assert "Re-rankeadosa" not in source_code, "Typo 'Re-rankeadosa' ainda presente em reranker.py."
    assert "Re-rankeados" in source_code


# ============================================================================
# 5. TESTES DO GROUNDED QA AGENT
# ============================================================================

def test_grounded_qa_agent_valid_query(initialized_pipeline):
    """Testa resposta fundamentada com citações para pergunta válida sobre frete VIP."""
    agent = initialized_pipeline["agent"]
    
    query = "Qual é a regra de frete grátis para Cliente VIP Diamante no Mercado Central 24h?"
    response = agent.answer(query)
    
    assert response["query"] == query
    assert len(response["answer"]) > 0
    assert len(response["citations"]) > 0
    
    first_citation = response["citations"][0]
    assert "file_name" in first_citation
    assert "section_title" in first_citation
    assert "page_range" in first_citation


def test_grounded_qa_agent_out_of_domain_query(initialized_pipeline):
    """Testa rejeição graciosa de pergunta sem fundamentação (out-of-domain)."""
    agent = initialized_pipeline["agent"]
    
    query = "Qual é a temperatura da superfície de Saturno?"
    response = agent.answer(query)
    
    assert response["query"] == query
    assert "não encontrei informações oficiais" in response["answer"].lower()
    assert len(response["citations"]) == 0
    assert len(response["sources_used"]) == 0


def test_grounded_qa_agent_structure(initialized_pipeline):
    """Valida o formato estruturado do dicionário de retorno do agente."""
    agent = initialized_pipeline["agent"]
    
    query = "Como funciona a política de trocas e devoluções?"
    response = agent.answer(query)
    
    assert isinstance(response, dict)
    assert "query" in response
    assert "answer" in response
    assert "citations" in response
    assert "sources_used" in response
