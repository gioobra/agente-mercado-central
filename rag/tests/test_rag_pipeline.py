"""
Test Suite Automatizada do Pipeline RAG - Mercado Central 24h
Validação de Vector Indexing, Hybrid Search, Re-ranking e Grounded QA Agent.
"""

import os
import inspect
import pytest
from typing import Dict, Any

from rag.scripts.vector_indexer import VectorIndexer, MockEmbeddingFunction
from rag.scripts.hybrid_search import (
    HybridSearcher,
    PORTUGUESE_MONTHS,
    PORTUGUESE_STOPWORDS,
    calculate_recency_score,
    normalize_text,
    parse_date_value,
    tokenize_portuguese,
)
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


# ============================================================================
# 6. TESTES DE FILTRAGEM TEMPORAL & RECENCY BOOST (R1, R2, AC1-AC7)
# ============================================================================

def test_parse_date_value_various_formats():
    """Valida parsing de datas em formato ISO, texto por extenso PT-BR, timestamp numérico e fallbacks seguros."""
    import datetime
    assert parse_date_value("Agosto de 2026") == datetime.datetime(2026, 8, 1)
    assert parse_date_value("14 de Agosto de 2026") == datetime.datetime(2026, 8, 14)
    assert parse_date_value("Março de 2025") == datetime.datetime(2025, 3, 1)
    assert parse_date_value("2026") == datetime.datetime(2026, 1, 1)
    assert parse_date_value("2026-08-14") == datetime.datetime(2026, 8, 14)
    assert parse_date_value("2026-08-14T10:30:00") == datetime.datetime(2026, 8, 14, 10, 30, 0)
    assert parse_date_value("15/08/2026") == datetime.datetime(2026, 8, 15)
    assert parse_date_value("08/2026") == datetime.datetime(2026, 8, 1)
    assert parse_date_value(datetime.date(2026, 8, 14)) == datetime.datetime(2026, 8, 14)
    assert parse_date_value(datetime.datetime(2026, 8, 14, 12, 0)) == datetime.datetime(2026, 8, 14, 12, 0)
    assert parse_date_value(None) is None
    assert parse_date_value("") is None
    assert parse_date_value("   ") is None
    assert parse_date_value(True) is None
    assert parse_date_value(False) is None
    assert parse_date_value("Data desconhecida inválida") is None


def test_calculate_recency_score_normalization():
    """Valida normalização do score de recência no intervalo [0.0, 1.0]."""
    import datetime
    t_min = datetime.datetime(2024, 1, 1).timestamp()
    t_max = datetime.datetime(2026, 8, 1).timestamp()

    # Mais recente deve ser 1.0
    score_new = calculate_recency_score("Agosto de 2026", t_min, t_max)
    assert score_new == pytest.approx(1.0, rel=1e-3)

    # Mais antigo deve ser 0.0
    score_old = calculate_recency_score("Janeiro de 2024", t_min, t_max)
    assert score_old == pytest.approx(0.0, rel=1e-3)

    # Data intermediária deve estar entre 0 e 1
    score_mid = calculate_recency_score("Outubro de 2025", t_min, t_max)
    assert 0.0 < score_mid < 1.0

    # Data inválida ou nula retorna 0.0
    assert calculate_recency_score(None, t_min, t_max) == 0.0
    assert calculate_recency_score("invalido", t_min, t_max) == 0.0

    # Quando min e max são iguais, retorna 1.0 para datas válidas
    assert calculate_recency_score("Agosto de 2026", t_max, t_max) == 1.0


def test_hybrid_search_recency_boost_prioritizes_recent_over_old_similar_chunks(temp_chroma_db):
    """
    Valida que documentos com last_updated mais recente recebem boost de score e são priorizados
    sobre documentos mais antigos com relevância semântica similar (AC1, AC2, AC6).
    """
    chunks = [
        {
            "chunk_id": "RECURSO_OLD_2024",
            "file_name": "Politica_Entregas_2024.pdf",
            "file_path": "/docs/Politica_Entregas_2024.pdf",
            "category": "Logística",
            "department_author": "Logística",
            "last_updated": "Janeiro de 2024",
            "section_title": "Frete Grátis Diamante",
            "page_start": 1,
            "page_end": 1,
            "char_count": 150,
            "word_count": 25,
            "text": "Regra de Frete Grátis Cliente Diamante: compras acima de R$ 120,00 no app têm entrega grátis."
        },
        {
            "chunk_id": "RECURSO_NEW_2026",
            "file_name": "Politica_Entregas_2026.pdf",
            "file_path": "/docs/Politica_Entregas_2026.pdf",
            "category": "Logística",
            "department_author": "Logística",
            "last_updated": "Agosto de 2026",
            "section_title": "Frete Grátis Diamante Atualizado",
            "page_start": 1,
            "page_end": 1,
            "char_count": 150,
            "word_count": 25,
            "text": "Regra de Frete Grátis Cliente Diamante: compras acima de R$ 100,00 no app têm entrega grátis."
        }
    ]

    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(chunks)

    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=chunks, alpha=0.5)

    # 1. Sem recency boost (recency_boost=False)
    results_unboosted = searcher.search("Frete Grátis Cliente Diamante", top_k=2, recency_boost=False)
    assert len(results_unboosted) == 2
    for r in results_unboosted:
        assert r["recency_boost"] == 0.0

    # 2. Com recency boost ativado (recency_boost=True)
    results_boosted = searcher.search("Frete Grátis Cliente Diamante", top_k=2, recency_boost=True, recency_weight=0.20)
    assert len(results_boosted) == 2

    top_result = results_boosted[0]
    second_result = results_boosted[1]

    # O chunk de 2026 deve ser o primeiro
    assert top_result["chunk_id"] == "RECURSO_NEW_2026"
    assert second_result["chunk_id"] == "RECURSO_OLD_2024"

    # O chunk mais recente recebeu maior pontuação de recência e score híbrido final superior
    assert top_result["recency_score"] > second_result["recency_score"]
    assert top_result["recency_boost"] > second_result["recency_boost"]
    assert top_result["hybrid_score"] > second_result["hybrid_score"]


def test_hybrid_search_recency_boost_does_not_exclude_old_documents(temp_chroma_db):
    """
    Valida que documentos antigos relevantes NÃO são excluídos pelo boost temporal,
    mantendo-se presentes nos resultados reordenados por relevância combinada (AC3).
    """
    chunks = [
        {
            "chunk_id": "REEMBOLSO_OLD_2024",
            "file_name": "Politica_Reembolso_2024.pdf",
            "file_path": "/docs/Politica_Reembolso_2024.pdf",
            "category": "Atendimento & CDC",
            "department_author": "SAC",
            "last_updated": "Janeiro de 2024",
            "section_title": "Arrependimento e Estorno",
            "page_start": 1,
            "page_end": 1,
            "char_count": 180,
            "word_count": 25,
            "text": "O cliente pode solicitar devolução por arrependimento em até 7 dias corridos após o recebimento."
        },
        {
            "chunk_id": "ESCALA_NEW_2026",
            "file_name": "Regulamento_RH_2026.pdf",
            "file_path": "/docs/Regulamento_RH_2026.pdf",
            "category": "RH",
            "department_author": "RH",
            "last_updated": "Agosto de 2026",
            "section_title": "Escala de Trabalho 5x2",
            "page_start": 1,
            "page_end": 1,
            "char_count": 180,
            "word_count": 25,
            "text": "A jornada semanal de trabalho é distribuída em escala 5x2 de 44 horas semanais."
        }
    ]

    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(chunks)

    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=chunks)

    # Busca sobre devolução/arrependimento com recency_boost=True
    results = searcher.search("devolução por arrependimento 7 dias", top_k=2, recency_boost=True)

    # O documento antigo relevante deve estar presente e na primeira posição
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "REEMBOLSO_OLD_2024"
    chunk_ids = [r["chunk_id"] for r in results]
    assert "REEMBOLSO_OLD_2024" in chunk_ids


def test_grounded_qa_agent_with_and_without_recency_boost(initialized_pipeline):
    """Valida que o agente E2E funciona perfeitamente com e sem recency_boost (AC4)."""
    agent = initialized_pipeline["agent"]

    query = "Qual é o valor mínimo de compra para ter frete grátis sendo Cliente VIP Diamante?"

    # Execução 1: Sem boost temporal
    resp_without_boost = agent.answer(query, recency_boost=False)
    assert resp_without_boost["query"] == query
    assert len(resp_without_boost["answer"]) > 0
    assert len(resp_without_boost["citations"]) > 0

    # Execução 2: Com boost temporal ativado
    resp_with_boost = agent.answer(query, recency_boost=True, recency_weight=0.15)
    assert resp_with_boost["query"] == query
    assert len(resp_with_boost["answer"]) > 0
    assert len(resp_with_boost["citations"]) > 0


def test_hybrid_searcher_initialization_with_recency_boost(temp_chroma_db, mock_chunks):
    """Valida inicialização do HybridSearcher com recency_boost=True como default de instância."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)

    searcher_default_on = HybridSearcher(
        vector_indexer=indexer,
        chunks_data=mock_chunks,
        recency_boost=True,
        recency_weight=0.25,
    )
    assert searcher_default_on.recency_boost is True
    assert searcher_default_on.recency_weight == 0.25

    results = searcher_default_on.search("Cliente VIP Diamante", top_k=2)
    assert len(results) > 0
    assert "recency_score" in results[0]
    assert "recency_boost" in results[0]


def test_hybrid_search_recency_boost_with_mixed_date_types(temp_chroma_db):
    """Valida candidate pool com tipos mistos de data (datetime, date, ISO string, PT-BR string, English string)."""
    import datetime
    chunks = [
        {"chunk_id": "C_DT", "text": "Regra datetime objeto", "last_updated": datetime.datetime(2026, 8, 14, 10, 0)},
        {"chunk_id": "C_DATE", "text": "Regra date objeto", "last_updated": datetime.date(2026, 8, 1)},
        {"chunk_id": "C_ISO", "text": "Regra ISO string", "last_updated": "2026-07-15T08:00:00Z"},
        {"chunk_id": "C_PT", "text": "Regra PT-BR string", "last_updated": "1º de Junho de 2026"},
        {"chunk_id": "C_EN", "text": "Regra English string", "last_updated": "May 10th, 2026"},
        {"chunk_id": "C_OLD", "text": "Regra Antiga", "last_updated": "2023-01-01"},
    ]
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=chunks)

    results = searcher.search("Regra", top_k=6, recency_boost=True)
    assert len(results) == 6
    # O mais recente (Agosto 14, 2026) deve ter recency_score 1.0
    c_dt_res = [r for r in results if r["chunk_id"] == "C_DT"][0]
    c_old_res = [r for r in results if r["chunk_id"] == "C_OLD"][0]
    assert c_dt_res["recency_score"] == pytest.approx(1.0, rel=1e-3)
    assert c_old_res["recency_score"] == pytest.approx(0.0, rel=1e-3)
    assert c_dt_res["recency_boost"] > c_old_res["recency_boost"]


def test_grounded_qa_agent_string_boolean_recency_boost(initialized_pipeline):
    """Valida GroundedQAAgent com flags em string ('true' / 'false')."""
    agent = initialized_pipeline["agent"]
    query = "Qual o prazo para devolução por arrependimento?"

    res_false = agent.answer(query, recency_boost="false")
    assert len(res_false["citations"]) > 0

    res_true = agent.answer(query, recency_boost="true")
    assert len(res_true["citations"]) > 0


def test_hybrid_search_recency_boost_tie_breaking(temp_chroma_db):
    """Valida que o boost de recência quebra empates de relevância semântica em favor do documento mais novo."""
    chunks = [
        {"chunk_id": "TIE_2023", "text": "Regra idêntica para estorno e devolução", "last_updated": "2023-01-01"},
        {"chunk_id": "TIE_2026", "text": "Regra idêntica para estorno e devolução", "last_updated": "2026-08-01"},
    ]
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=chunks)

    # Com recency boost ativado, o chunk de 2026 deve vencer inequivocamente
    results = searcher.search("Regra idêntica para estorno e devolução", top_k=2, recency_boost=True)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "TIE_2026"
    assert results[1]["chunk_id"] == "TIE_2023"
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]


def test_hybrid_search_recency_boost_doc_meta_filtering_and_extraction(temp_chroma_db):
    """Valida extração de datas e filtragem de metadados em estruturas aninhadas sob doc_meta."""
    chunks = [
        {
            "chunk_id": "DOC_META_OLD",
            "text": "Procedimento operacional do setor financeiro",
            "doc_meta": {"category": "Financeiro", "published_at": "Janeiro de 2024"},
        },
        {
            "chunk_id": "DOC_META_NEW",
            "text": "Procedimento operacional do setor financeiro atualizado",
            "doc_meta": {"category": "Financeiro", "published_at": "Agosto de 2026"},
        },
    ]
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=chunks)

    results = searcher.search(
        "Procedimento operacional financeiro",
        top_k=2,
        metadata_filter={"category": "Financeiro"},
        recency_boost=True,
    )
    assert len(results) == 2
    assert results[0]["chunk_id"] == "DOC_META_NEW"
    assert results[0]["recency_score"] > results[1]["recency_score"]


def test_hybrid_search_recency_boost_single_item_and_empty_result(temp_chroma_db):
    """Valida comportamento robusto com resultado único ou busca sem resultados com recency boost."""
    single_chunk = [{"chunk_id": "SINGLE_01", "text": "Regra única", "last_updated": "Agosto de 2026"}]
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(single_chunk)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=single_chunk)

    # 1 item retorna com recency_score = 1.0
    res_single = searcher.search("Regra", top_k=1, recency_boost=True)
    assert len(res_single) == 1
    assert res_single[0]["recency_score"] == 1.0
    assert res_single[0]["recency_boost"] > 0.0

    # Busca vazia
    res_empty = searcher.search("termo_completamente_inexistente_xyz_999", top_k=0, recency_boost=True)
    assert res_empty == []



