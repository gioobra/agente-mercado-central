#!/usr/bin/env python3
"""
Test Suite: E2E Enhancements (R1 + R2 + R3 + R4)
Validação de ponta a ponta dos novos recursos do Agente de QA do Mercado Central 24h:
- Limiares de Confiança & Detecção de Alucinação (R1)
- Catálogo de Contatos Corporativos & Roteamento de Fallback por Intenção (R2)
- Formatação de Respostas Multicanal (Chat, Email, Teams/Slack) (R3)
- Conformidade estrita com o Contrato de Retorno e Não-Regressão (R4)
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from rag.scripts.vector_indexer import VectorIndexer
from rag.scripts.hybrid_search import HybridSearcher
from rag.scripts.reranker import ReRanker
from rag.scripts.grounded_qa_agent import GroundedQAAgent
from rag.scripts.contact_catalog import CORPORATE_CONTACT_CATALOG


@pytest.fixture
def enhanced_agent(temp_chroma_db, mock_chunks):
    """Fixture de inicialização determinística do agente com chunks do corpus oficial."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    return GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker, confidence_threshold=0.35)


# ============================================================================
# 1. TESTES DE RESPOSTA FUNDAMENTADA MULTICANAL (GROUNDED GENERATION)
# ============================================================================

def test_e2e_grounded_answer_chat_channel(enhanced_agent):
    """T-E2E.01: Valida resposta fundamentada no formato padrão 'chat'."""
    query = "Qual é o valor mínimo de compra para ter frete grátis sendo Cliente VIP Diamante?"
    res = enhanced_agent.answer(query, channel="chat")

    assert res["query"] == query
    assert res["channel"] == "chat"
    assert res["is_fallback"] is False
    assert res["fallback_department"] is None
    assert res["hallucination_check"]["is_grounded"] is True
    assert len(res["citations"]) > 0

    # Estrutura tripartite no chat
    assert "**Resumo Direto:**" in res["answer"]
    assert "**Detalhamento:**" in res["answer"]
    assert "**Fontes Consultadas:**" in res["answer"]
    assert "[Fonte:" in res["answer"]


def test_e2e_grounded_answer_email_channel(enhanced_agent):
    """T-E2E.02: Valida resposta fundamentada no formato corporativo 'email'."""
    query = "Qual é o valor mínimo de compra para ter frete grátis sendo Cliente VIP Diamante?"
    res = enhanced_agent.answer(query, channel="email")

    assert res["channel"] == "email"
    assert res["is_fallback"] is False
    assert len(res["citations"]) > 0

    # Estrutura tripartite no email
    assert res["answer"].startswith("Prezado(a) colaborador(a),")
    assert "**Resumo Executivo:**" in res["answer"]
    assert "**Detalhamento:**" in res["answer"]
    assert "**Base Normativa e Fontes:**" in res["answer"]
    assert "Atenciosamente,\nEquipe de Atendimento - Mercado Central 24h" in res["answer"]


def test_e2e_grounded_answer_teams_slack_channel(enhanced_agent):
    """T-E2E.03: Valida resposta fundamentada no formato de mensageria 'teams_slack'."""
    query = "Qual é o prazo para devolução por arrependimento conforme o CDC?"
    res = enhanced_agent.answer(query, channel="teams_slack")

    assert res["channel"] == "teams_slack"
    assert res["is_fallback"] is False
    assert len(res["citations"]) > 0

    # Estrutura tripartite no teams/slack
    assert "**[RESUMO]**" in res["answer"]
    assert "**[DETALHAMENTO]**" in res["answer"]
    assert "**[FONTES]**" in res["answer"]


# ============================================================================
# 2. TESTES DE ROTEAMENTO DE FALLBACK POR INTENÇÃO (CONFIDENCE & INTENT GATE)
# ============================================================================

def test_e2e_fallback_routing_rh_department(enhanced_agent):
    """T-E2E.04: Valida fallback roteado para RH quando busca está abaixo do threshold."""
    query = "Qual o procedimento para solicitação de férias e adiantamento salarial?"
    res = enhanced_agent.answer(query, confidence_threshold=0.99, channel="chat")

    assert res["is_fallback"] is True
    assert res["fallback_department"] == "rh"
    assert len(res["citations"]) == 0
    assert "rh@mercadocentral24h.com.br" in res["answer"]
    assert "Não encontrei essa informação nos documentos disponíveis" in res["answer"]
    assert "0800-CENTRAL" in res["answer"]


def test_e2e_fallback_routing_compliance_department(enhanced_agent):
    """T-E2E.05: Valida fallback roteado para Jurídico & Compliance em canal email."""
    query = "Quero fazer uma denúncia anônima de corrupção e desvio de conduta de fornecedor"
    res = enhanced_agent.answer(query, confidence_threshold=0.99, channel="email")

    assert res["is_fallback"] is True
    assert res["fallback_department"] == "juridico_compliance"
    assert res["channel"] == "email"
    assert "etica@mercadocentral24h.com.br" in res["answer"]
    assert "Camila Ferreira" in res["answer"]
    assert res["answer"].startswith("Prezado(a) colaborador(a),")


def test_e2e_fallback_routing_dpo_department(enhanced_agent):
    """T-E2E.06: Valida fallback roteado para DPO / LGPD em canal teams_slack."""
    query = "Como revogar consentimento de dados pessoais e cookies da LGPD?"
    res = enhanced_agent.answer(query, confidence_threshold=0.99, channel="teams_slack")

    assert res["is_fallback"] is True
    assert res["fallback_department"] == "dpo_lgpd"
    assert res["channel"] == "teams_slack"
    assert "dpo@mercadocentral24h.com.br" in res["answer"]
    assert "**[DEPARTAMENTO RECOMENDADO]**" in res["answer"]


def test_e2e_fallback_routing_compras_department(enhanced_agent):
    """T-E2E.07: Valida fallback roteado para Compras e Suprimentos."""
    query = "Qual a janela de agendamento de docas e cotação de compras de perecíveis?"
    res = enhanced_agent.answer(query, confidence_threshold=0.99, channel="chat")

    assert res["is_fallback"] is True
    assert res["fallback_department"] == "compras_fornecedores"
    assert "compras.geral@mercadocentral24h.com.br" in res["answer"]


def test_e2e_fallback_routing_fiscal_department(enhanced_agent):
    """T-E2E.08: Valida fallback roteado para Fiscal / NFe."""
    query = "Para onde encaminhar o arquivo XML da NF-e e validar DANFE na SEFAZ?"
    res = enhanced_agent.answer(query, confidence_threshold=0.99, channel="chat")

    assert res["is_fallback"] is True
    assert res["fallback_department"] == "fiscal_nfe"
    assert "nfe@mercadocentral24h.com.br" in res["answer"]
    assert "José Oliveira" in res["answer"]


def test_e2e_fallback_routing_sac_department(enhanced_agent):
    """T-E2E.09: Valida fallback roteado para SAC e Atendimento Delivery."""
    query = "Meu pedido de compras está atrasado, como falar com o SAC Delivery?"
    res = enhanced_agent.answer(query, confidence_threshold=0.99, channel="chat")

    assert res["is_fallback"] is True
    assert res["fallback_department"] == "sac_delivery"
    assert "sac.sp@mercadocentral24h.com.br" in res["answer"]


def test_e2e_fallback_out_of_domain_defaults_to_ouvidoria(enhanced_agent):
    """T-E2E.10: Valida fallback universal para Ouvidoria Geral em consultas fora de domínio."""
    query = "Qual é a velocidade média da luz no vácuo em metros por segundo?"
    res = enhanced_agent.answer(query, channel="chat")

    assert res["is_fallback"] is True
    assert res["fallback_department"] == "ouvidoria_fallback"
    assert "ouvidoria@mercadocentral24h.com.br" in res["answer"]
    assert "0800-CENTRAL" in res["answer"]
    assert len(res["citations"]) == 0


# ============================================================================
# 3. TESTES DE INTERCEPTAÇÃO DE ALUCINAÇÃO (POST-GENERATION CHECKER)
# ============================================================================

def test_e2e_hallucination_interception_triggers_routed_fallback(enhanced_agent):
    """T-E2E.11: Valida que alucinação detectada pós-geração aciona fallback roteado para a área correta."""
    # Simula resposta do LLM com entidade alucinada (escala 6x1 inventada para RH)
    class HallucinatingRHResponse:
        text = "A jornada dos operadores é de 6x1 com jornada semanal de 60 horas."

    class FakeModels:
        def generate_content(self, **kwargs):
            return HallucinatingRHResponse()

    class FakeClient:
        models = FakeModels()

    enhanced_agent.genai_client = FakeClient()

    res = enhanced_agent.answer("Qual é a jornada de trabalho semanal e a escala?", channel="chat")

    assert res["is_fallback"] is True
    assert res["hallucination_check"]["is_grounded"] is False
    assert res["hallucination_check"]["reason"] == "hallucination_detected"
    assert res["fallback_department"] == "rh"
    assert "rh@mercadocentral24h.com.br" in res["answer"]


# ============================================================================
# 4. TESTES DE CONFORMIDADE COM O CONTRATO DE RETORNO DO AGENTE
# ============================================================================

def test_e2e_response_dict_contract_completeness(enhanced_agent):
    """T-E2E.12: Valida presença de todas as chaves exigidas pelo contrato em PROJECT.md."""
    query = "Como funciona o frete grátis?"
    res = enhanced_agent.answer(query, channel="chat")

    required_keys = [
        "query",
        "answer",
        "citations",
        "sources_used",
        "confidence_score",
        "confidence_threshold",
        "is_fallback",
        "fallback_department",
        "hallucination_check",
        "channel",
    ]
    for k in required_keys:
        assert k in res, f"Chave obrigatória ausente no retorno de answer(): '{k}'"

    assert isinstance(res["query"], str)
    assert isinstance(res["answer"], str)
    assert isinstance(res["citations"], list)
    assert isinstance(res["sources_used"], list)
    assert isinstance(res["confidence_score"], float)
    assert isinstance(res["confidence_threshold"], float)
    assert isinstance(res["is_fallback"], bool)
    assert isinstance(res["hallucination_check"], dict)
    assert isinstance(res["channel"], str)
