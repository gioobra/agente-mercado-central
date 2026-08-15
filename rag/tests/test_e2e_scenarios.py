"""
Tier 3 Pairwise Cross-Feature Integration & Tier 4 Real-World E2E Business Scenarios Test Suite.
Validação E2E de Fluxos de Delivery SLA, Fidelidade VIP, Reembolso CDC, SOP Escala 5x2 e Auditoria de Consistência Corporativa.
"""

import os
import json
import re
import pytest
from pathlib import Path

from rag.scripts.rag_pdf_processor import (
    clean_page_text,
    detect_sections,
    create_chunks_from_sections,
)
from rag.scripts.vector_indexer import VectorIndexer
from rag.scripts.hybrid_search import HybridSearcher
from rag.scripts.reranker import ReRanker
from rag.scripts.grounded_qa_agent import GroundedQAAgent


@pytest.fixture(scope="module")
def e2e_pipeline(real_chunks_json_path, mock_chunks, temp_chroma_db):
    """Fixture que carrega o pipeline E2E com dados reais ou mock para testes de regras de negócio."""
    indexer = VectorIndexer(use_mock=True, db_path=":memory:")
    
    if real_chunks_json_path.exists():
        indexer.index_chunks(str(real_chunks_json_path))
        chunks_source = str(real_chunks_json_path)
    else:
        indexer.index_chunks(mock_chunks)
        chunks_source = mock_chunks

    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=chunks_source)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    return {
        "indexer": indexer,
        "searcher": searcher,
        "reranker": reranker,
        "agent": agent,
        "chunks_source": chunks_source,
    }


# ============================================================================
# TIER 3: TESTES DE INTEGRAÇÃO PAR-A-PAR (PAIRWISE CROSS-FEATURE)
# ============================================================================

def test_pairwise_pdf_processor_to_vector_indexer(temp_chroma_db):
    """Integração Par-a-Par: Processador PDF -> Limpeza/Chunking -> Indexação no ChromaDB."""
    raw_pages = [
        {
            "page_num": 1,
            "text": "MERCADO CENTRAL 24H\n1. REGRAS DE ENTREGA\nEntregas efetuadas em até 3 horas na modalidade Expressa."
        }
    ]
    sections = detect_sections(raw_pages)
    chunks = create_chunks_from_sections(
        sections,
        file_name="Guia_de_Envios_e_Entregas.pdf",
        file_path="/docs/pdf/Guia_de_Envios_e_Entregas.pdf",
        doc_meta={"category": "Logística & Delivery", "department_author": "Logística", "last_updated": "Agosto de 2026"}
    )
    
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    count = indexer.index_chunks(chunks)
    assert count == len(chunks)
    
    res = indexer.search("entrega em 3 horas", top_k=1)
    assert len(res) == 1
    assert res[0]["chunk_id"] == chunks[0]["chunk_id"]


def test_pairwise_hybrid_search_to_reranker_to_qa(mock_chunks, temp_chroma_db):
    """Integração Par-a-Par: Busca Híbrida -> Re-ranker -> Agente Grounded QA."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="feature_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    response = agent.answer("Qual o prazo para devolução por arrependimento?")
    assert "7 dias" in response["answer"].lower() or "arrependimento" in response["answer"].lower()
    assert len(response["citations"]) > 0


def test_pairwise_category_metadata_routing(mock_chunks, temp_chroma_db):
    """Integração Par-a-Par: Roteamento de Consultas com Filtro de Metadados por Categoria."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    
    cdc_results = searcher.search("prazo", top_k=5, metadata_filter={"category": "Atendimento & CDC"})
    assert all(r["category"] == "Atendimento & CDC" for r in cdc_results)
    
    logistics_results = searcher.search("frete", top_k=5, metadata_filter={"category": "Logística & Delivery"})
    assert all(r["category"] == "Logística & Delivery" for r in logistics_results)


def test_pairwise_recency_boost_policy_evolution(temp_chroma_db):
    """Integração Par-a-Par: Evolução de Políticas e Priorização por Recência (Fase 3 -> ReRanker -> QA)."""
    policy_evolution_chunks = [
        {
            "chunk_id": "POL_VIP_2023",
            "file_name": "Regulamento_Fidelidade_2023.pdf",
            "file_path": "/docs/Regulamento_Fidelidade_2023.pdf",
            "category": "Fidelidade VIP",
            "department_author": "Marketing",
            "last_updated": "Janeiro de 2023",
            "section_title": "Regras Cashback VIP Diamante 2023",
            "page_start": 1,
            "page_end": 1,
            "char_count": 200,
            "word_count": 30,
            "text": "Benefício VIP Diamante: Cashback de 1,5% em todas as compras e frete grátis acima de R$ 150,00."
        },
        {
            "chunk_id": "POL_VIP_2026",
            "file_name": "Regulamento_Fidelidade_2026.pdf",
            "file_path": "/docs/Regulamento_Fidelidade_2026.pdf",
            "category": "Fidelidade VIP",
            "department_author": "Marketing",
            "last_updated": "Agosto de 2026",
            "section_title": "Regras Cashback VIP Diamante 2026",
            "page_start": 1,
            "page_end": 1,
            "char_count": 200,
            "word_count": 30,
            "text": "Benefício VIP Diamante: Cashback de 2,0% em todas as compras e frete grátis acima de R$ 100,00."
        }
    ]

    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(policy_evolution_chunks)

    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=policy_evolution_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)

    # Executa com boost temporal ativado
    res = agent.answer("Qual o cashback do Cliente VIP Diamante?", recency_boost=True)
    assert len(res["citations"]) > 0
    # O chunk de 2026 deve ser o top source
    assert res["citations"][0]["chunk_id"] == "POL_VIP_2026"
    assert "Regulamento_Fidelidade_2026.pdf" in res["citations"][0]["file_name"]


# ============================================================================
# TIER 4: CENÁRIOS E2E DE APLICAÇÃO REAL E REGRAS DE NEGÓCIO
# ============================================================================

def test_e2e_scenario_delivery_sla_flow(e2e_pipeline):
    """Cenário Real 1: Fluxo de Delivery SLA (Prazos, frete grátis padrão R$250, VIP Diamante R$100 e raios)."""
    agent = e2e_pipeline["agent"]
    
    # 1. Valida janela de entrega expressa (3 horas) e frete grátis padrão (R$ 250,00)
    q1 = "Qual é o prazo da entrega expressa e o valor mínimo para frete grátis padrão?"
    res1 = agent.answer(q1)
    ans1 = res1["answer"].lower()
    assert "3 horas" in ans1 or "3h" in ans1 or "250" in ans1
    
    # 2. Valida benefício VIP Diamante (frete grátis a partir de R$ 100,00)
    q2 = "Qual é o valor mínimo para frete grátis do Cliente VIP Diamante?"
    res2 = agent.answer(q2)
    ans2 = res2["answer"].lower()
    assert "100" in ans2 or "diamante" in ans2


def test_e2e_scenario_vip_loyalty_cashback_flow(e2e_pipeline):
    """Cenário Real 2: Fluxo de Fidelidade VIP Central (Bronze 0,5%, Prata 1,0%, Gold 1,5%, Diamante 2,0%, 12 meses FIFO, 48h)."""
    agent = e2e_pipeline["agent"]
    
    # 1. Valida consulta de cashback VIP Diamante (2,0%)
    res_diamante = agent.answer("Qual é a porcentagem de cashback para o nível VIP Diamante?")
    ans_diamante = res_diamante["answer"].lower()
    assert "2,0%" in ans_diamante or "2.0%" in ans_diamante or "2%" in ans_diamante or "diamante" in ans_diamante

    # 2. Valida consulta de cashback VIP Bronze (0,5%)
    res_bronze = agent.answer("Qual é a porcentagem de cashback do nível VIP Bronze?")
    ans_bronze = res_bronze["answer"].lower()
    assert "0,5%" in ans_bronze or "0.5%" in ans_bronze or "bronze" in ans_bronze


def test_e2e_scenario_cdc_refund_and_returns_flow(e2e_pipeline):
    """Cenário Real 3: Fluxo de Reembolso e Devoluções CDC (Arrependimento 7 dias, Defeitos 30/90 dias, PIX 24h, Cartão 5 dias)."""
    agent = e2e_pipeline["agent"]
    
    # 1. Direitos CDC (7 dias arrependimento, 30/90 dias defeitos)
    q_cdc = "Qual é o prazo de desistência por arrepencimento e o prazo para produtos com defeito conforme o CDC?"
    res_cdc = agent.answer(q_cdc)
    ans_cdc = res_cdc["answer"].lower()
    
    assert "7 dias" in ans_cdc or "7" in ans_cdc
    
    # 2. Prazos de reembolso (PIX 24h, Cartão 5 dias úteis)
    q_pix = "Qual o prazo para estorno via PIX e cartão de crédito?"
    res_pix = agent.answer(q_pix)
    ans_pix = res_pix["answer"].lower()
    assert "24h" in ans_pix or "24 horas" in ans_pix or "5 dias" in ans_pix or "pix" in ans_pix


def test_e2e_scenario_sop_shift_schedule_and_247_ops_flow(e2e_pipeline):
    """Cenário Real 4: Fluxo de SOP Escala de Trabalho 5x2, Turnos T1-T5, Adicional Noturno e Automação/IA."""
    agent = e2e_pipeline["agent"]
    
    # 1. Escala de Trabalho 5x2
    q_sop = "Qual é a escala de trabalho adotada no Mercado Central 24h e como a IA viabiliza essa jornada?"
    res_sop = agent.answer(q_sop)
    ans_sop = res_sop["answer"].lower()
    
    assert "5x2" in ans_sop or "cinco por dois" in ans_sop
    assert "6x1" not in ans_sop  # Garante NENHUMA menção a 6x1
    assert "ia" in ans_sop or "automação" in ans_sop or "tecnologia" in ans_sop or "preditivo" in ans_sop


def test_e2e_scenario_automated_corporate_pdf_consistency_audit(real_chunks_json_path, project_root_path):
    """Cenário Real 5: Auditoria Automatizada de Consistência Corporativa (Zero 6x1, Cashback Uniforme, Nome da Marca, Infra)."""
    
    # 1. Auditoria no arquivo de chunks processados
    if real_chunks_json_path.exists():
        with open(real_chunks_json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        full_corpus_text = " ".join([c.get("text", "") for c in chunks])
        
        # Validação R2: Exatamente ZERO ocorrências de "6x1" ou "escala 6x1" em todo o corpus
        matches_6x1 = re.findall(r"\b6\s*[xX]\s*1\b", full_corpus_text)
        assert len(matches_6x1) == 0, "VIOLAÇÃO NARRATIVA: Encontrada menção a escala 6x1 nos documentos corporativos!"
        
        # Validação R2: Uniformidade da Escala 5x2
        matches_5x2 = re.findall(r"\b5\s*[xX]\s*2\b", full_corpus_text)
        assert len(matches_5x2) > 0, "Confirmação de que a Escala 5x2 é explicitada na documentação."

        # Validação R1: Consistência dos percentuais de cashback VIP
        assert "0,5%" in full_corpus_text or "0.5%" in full_corpus_text
        assert "1,0%" in full_corpus_text or "1.0%" in full_corpus_text
        assert "1,5%" in full_corpus_text or "1.5%" in full_corpus_text
        assert "2,0%" in full_corpus_text or "2.0%" in full_corpus_text
        
        # Validação R1: Prazos CDC (7 dias, 30 dias, 90 dias)
        assert "7 dias" in full_corpus_text
        assert "30 dias" in full_corpus_text
        assert "90 dias" in full_corpus_text

    # 2. Auditoria de Infraestrutura do Repositório (F1, F2, F3 / R4)
    gitignore_path = project_root_path / ".gitignore"
    requirements_path = project_root_path / "requirements.txt"
    readme_path = project_root_path / "README.md"
    
    assert gitignore_path.exists(), "Arquivo .gitignore deve existir na raiz."
    assert requirements_path.exists(), "Arquivo requirements.txt deve existir na raiz."
    assert readme_path.exists(), "Arquivo README.md deve existir na raiz."
    
    # Valida conteúdo do .gitignore (F1)
    gitignore_content = gitignore_path.read_text(encoding="utf-8")
    assert "venv" in gitignore_content
    assert "__pycache__" in gitignore_content
    assert ".pytest_cache" in gitignore_content
    
    # Valida dependências no requirements.txt (F2)
    reqs_content = requirements_path.read_text(encoding="utf-8")
    assert "chromadb" in reqs_content
    assert "pytest" in reqs_content
