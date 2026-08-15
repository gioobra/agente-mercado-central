#!/usr/bin/env python3
"""
Test Suite: Milestone 1 (M1 - R1)
Limiar de Confiança (Confidence Thresholding) e Verificador de Consistência Pós-Geração (Hallucination Checker)
"""

import math
import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from rag.scripts.vector_indexer import VectorIndexer
from rag.scripts.hybrid_search import HybridSearcher
from rag.scripts.reranker import ReRanker
from rag.scripts.grounded_qa_agent import GroundedQAAgent
from rag.scripts.hallucination_checker import HallucinationChecker


# ============================================================================
# 1. TESTES DO LIMIAR DE CONFIANÇA (CONFIDENCE THRESHOLDING)
# ============================================================================

def test_confidence_threshold_default_initialization(temp_chroma_db, mock_chunks):
    """T-R1.01: Valida que GroundedQAAgent inicializa com confidence_threshold padrão de 0.35."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    assert hasattr(agent, "confidence_threshold")
    assert agent.confidence_threshold == 0.35


def test_confidence_threshold_custom_initialization(temp_chroma_db, mock_chunks):
    """Valida customização do confidence_threshold no construtor."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker, confidence_threshold=0.60)
    assert agent.confidence_threshold == 0.60


def test_confidence_threshold_override_in_answer(temp_chroma_db, mock_chunks):
    """T-R1.02: Valida sobrescrita de confidence_threshold em tempo de execução via answer()."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker, confidence_threshold=0.35)
    
    # Com threshold alto (0.95), mesmo uma query válida deve cair no fallback
    res = agent.answer("Qual é o frete grátis para Cliente VIP Diamante?", confidence_threshold=0.95)
    assert res["is_fallback"] is True
    assert res["confidence_threshold"] == 0.95
    assert len(res["citations"]) == 0
    assert "não encontrei informações oficiais" in res["answer"].lower()


def test_confidence_threshold_below_bypasses_llm_call(temp_chroma_db, mock_chunks):
    """T-R1.03: Valida que busca abaixo do limiar de confiança NUNCA invoca o modelo LLM."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker, confidence_threshold=0.90)
    
    # Mock do cliente Gemini
    mock_genai_client = MagicMock()
    mock_models = MagicMock()
    mock_genai_client.models = mock_models
    agent.genai_client = mock_genai_client
    
    res = agent.answer("Qual a política de férias?", confidence_threshold=0.90)
    
    # generate_content NÃO deve ser chamado pois o score fica abaixo de 0.90
    assert mock_models.generate_content.call_count == 0
    assert res["is_fallback"] is True
    assert res["confidence_score"] < 0.90


def test_confidence_threshold_above_proceeds_to_generation(temp_chroma_db, mock_chunks):
    """T-R1.04: Valida que score acima do threshold executa normalmente com citações."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker, confidence_threshold=0.20)
    
    res = agent.answer("Qual o prazo para devolução por arrependimento?")
    assert res["is_fallback"] is False
    assert res["confidence_score"] >= 0.20
    assert len(res["citations"]) > 0
    assert "hallucination_check" in res
    assert res["hallucination_check"]["is_grounded"] is True


def test_confidence_threshold_boundary_values(temp_chroma_db, mock_chunks):
    """T-R1.05: Valida limites estritos de 0.0 (sempre aceita se grounded) e 1.0 (sempre rejeita se score < 1.0)."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    # threshold=0.0 -> passa
    res_zero = agent.answer("Qual o frete grátis Cliente VIP Diamante?", confidence_threshold=0.0)
    assert res_zero["is_fallback"] is False
    
    # threshold=1.0 -> rejeita se score < 1.0
    res_one = agent.answer("Qual o frete grátis Cliente VIP Diamante?", confidence_threshold=1.0)
    assert res_one["is_fallback"] is True


def test_confidence_threshold_sanitization():
    """T-R1.06: Valida sanitização robusta de inputs inválidos no threshold."""
    # Valores fora de faixa são limitados a [0.0, 1.0]
    assert GroundedQAAgent._sanitize_confidence_threshold(-0.5) == 0.0
    assert GroundedQAAgent._sanitize_confidence_threshold(1.8) == 1.0
    assert GroundedQAAgent._sanitize_confidence_threshold("0.45") == 0.45
    assert GroundedQAAgent._sanitize_confidence_threshold(None, default=0.35) == 0.35
    assert GroundedQAAgent._sanitize_confidence_threshold(float("nan"), default=0.35) == 0.35
    assert GroundedQAAgent._sanitize_confidence_threshold(float("inf"), default=0.35) == 0.35
    assert GroundedQAAgent._sanitize_confidence_threshold("invalido", default=0.35) == 0.35


def test_confidence_threshold_empty_search_or_rerank_results(temp_chroma_db, mock_chunks):
    """T-R1.07: Valida que ausência de resultados de busca aciona fallback imediatamente com score 0.0."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=[])
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    res = agent.answer("Pergunta qualquer", top_search_k=0)
    assert res["is_fallback"] is True
    assert res["confidence_score"] == 0.0
    assert len(res["citations"]) == 0
    assert "não encontrei informações oficiais" in res["answer"].lower()


# ============================================================================
# 2. TESTES DO HALLUCINATION CHECKER (SENTENCE TOKENIZER & ENTITY GROUNDING)
# ============================================================================

def test_sentence_tokenizer_portuguese_edge_cases():
    """T-R1.08: Valida divisão sentencial em PT-BR sem quebra indevida em abreviações e decimais."""
    checker = HallucinationChecker()
    
    text = (
        "O Art. 49 do CDC estabelece o prazo de 7 dias para arrependimento. "
        "O frete grátis do Mercado Central Ltda. é de R$ 250,00 para compras em SP. "
        "A jornada é na escala 5x2 (ex. seg a sex), conforme pág. 12 do SOP. "
        "O Dr. Silva aprovou o bônus de 2,0% para clientes VIP."
    )
    
    sentences = checker.split_sentences(text)
    assert len(sentences) == 4
    assert "Art. 49" in sentences[0]
    assert "R$ 250,00" in sentences[1]
    assert "escala 5x2" in sentences[2]
    assert "pág. 12" in sentences[2]
    assert "2,0%" in sentences[3]


def test_sentence_tokenizer_bullet_points_and_newlines():
    """Valida divisão sentencial com quebras de linha e marcadores de lista."""
    checker = HallucinationChecker()
    
    text = (
        "• Primeira regra: entrega em 3 horas para raio de 15km.\n"
        "• Segunda regra: cashback de 1,5% no nível Gold.\n"
        "• Terceira regra: devolução em 30 dias para defeito aparente."
    )
    
    sentences = checker.split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("Primeira regra")
    assert sentences[1].startswith("Segunda regra")
    assert sentences[2].startswith("Terceira regra")


def test_entity_extraction_critical_tokens():
    """Valida extração de moedas, porcentagens, escalas, prazos e artigos legais."""
    checker = HallucinationChecker()
    
    sample = (
        "Segundo o Art. 49 do CDC e a Lei 13.709/2018, o cliente VIP com cashback de 2,0% "
        "recebe frete grátis acima de R$ 100,00 na escala 5x2 no prazo de 24h ou 5 dias úteis."
    )
    
    entities = checker.extract_entities(sample)
    assert "R$ 100,00" in entities["currencies"]
    assert "2,0%" in entities["percentages"]
    assert "5x2" in entities["shifts"]
    assert any("24h" in d or "5 dias" in d for d in entities["durations"])
    assert any("Art. 49" in a or "Lei 13.709" in a for a in entities["articles_and_laws"])


def test_hallucination_checker_fully_grounded_answer():
    """T-R1.09: Valida aprovação de resposta 100% fundamentada nos chunks de contexto."""
    checker = HallucinationChecker()
    
    context = [
        {
            "file_name": "Guia_de_Envios_e_Entregas.pdf",
            "section_title": "Prazos e Modalidades",
            "text": "A entrega expressa é efetuada em até 3 horas para pedidos realizados até as 18h no raio de 15km. O frete padrão é grátis para compras acima de R$ 250,00.",
        },
        {
            "file_name": "Regulamento_Fidelidade_2026.pdf",
            "section_title": "Frete VIP",
            "text": "Clientes VIP Diamante possuem frete grátis a partir de R$ 100,00.",
        }
    ]
    
    answer = (
        "Com base na documentação oficial do Mercado Central 24h, segue o detalhamento:\n"
        "• A entrega expressa ocorre em até 3 horas para compras feitas até as 18h.\n"
        "• O valor mínimo para frete grátis padrão é de R$ 250,00, enquanto clientes VIP Diamante têm frete grátis acima de R$ 100,00.\n"
        "[Fonte: Guia_de_Envios_e_Entregas.pdf, Seção: Prazos e Modalidades, Pág. 1]"
    )
    
    is_grounded, evaluations = checker.check_response(answer, context)
    assert is_grounded is True
    assert len(evaluations) >= 3
    # Nenhuma entidade não suportada
    assert all(len(e.get("ungrounded_entities", [])) == 0 for e in evaluations)


def test_hallucination_checker_detects_fabricated_scale():
    """T-R1.10: Valida detecção de escala de trabalho alucinada (ex: 6x1 quando documento prevê 5x2)."""
    checker = HallucinationChecker()
    
    context = [
        {
            "file_name": "Regulamento_Interno_e_SOP.pdf",
            "section_title": "Escala de Trabalho",
            "text": "Todos os colaboradores do Mercado Central 24h atuam sob a jornada padrão 5x2 com 44 horas semanais.",
        }
    ]
    
    hallucinated_answer = (
        "Conforme as regras internas, os colaboradores cumprem jornada na escala 6x1 com folga semanal."
    )
    
    is_grounded, evaluations = checker.check_response(hallucinated_answer, context)
    assert is_grounded is False
    assert any("escala 6x1" in str(e.get("ungrounded_entities", [])) for e in evaluations)


def test_hallucination_checker_detects_fabricated_currency_and_discount():
    """T-R1.11: Valida detecção de valores monetários e porcentagens inventadas."""
    checker = HallucinationChecker()
    
    context = [
        {
            "file_name": "Guia_de_Envios_e_Entregas.pdf",
            "section_title": "Frete Grátis",
            "text": "O frete é gratuito em compras acima de R$ 250,00. Clientes VIP Diamante recebem frete grátis acima de R$ 100,00.",
        }
    ]
    
    # Resposta inventando R$ 500,00 e 50% de desconto
    fake_answer = (
        "O frete grátis é concedido para valores acima de R$ 500,00 com 50% de desconto adicional na entrega."
    )
    
    is_grounded, evaluations = checker.check_response(fake_answer, context)
    assert is_grounded is False
    assert any("R$ 500,00" in str(e.get("ungrounded_entities", [])) or "50%" in str(e.get("ungrounded_entities", [])) for e in evaluations)


def test_hallucination_checker_detects_fabricated_duration():
    """Valida detecção de prazos de entrega ou SLA inventados (ex: drone em 10 minutos)."""
    checker = HallucinationChecker()
    
    context = [
        {
            "file_name": "Guia_de_Envios_e_Entregas.pdf",
            "section_title": "Entrega Expressa",
            "text": "A entrega expressa é realizada em até 3 horas na Grande São Paulo.",
        }
    ]
    
    fake_sla_answer = "A entrega ultra-rápida é realizada por drone no prazo de 10 minutos."
    is_grounded, evaluations = checker.check_response(fake_sla_answer, context)
    assert is_grounded is False
    assert any("10 minutos" in str(e.get("ungrounded_entities", [])) or e.get("overlap_score", 1.0) < 0.35 for e in evaluations)


def test_hallucination_checker_low_lexical_overlap_rejection():
    """T-R1.13: Valida rejeição de resposta com baixíssimo overlap léxico com os documentos."""
    checker = HallucinationChecker(semantic_overlap_threshold=0.40)
    
    context = [
        {
            "file_name": "Politica_de_Privacidade_LGPD.pdf",
            "section_title": "Direitos dos Titulares",
            "text": "O titular pode solicitar a exclusão de dados pessoais via dpo@mercadocentral24h.com.br.",
        }
    ]
    
    unrelated_answer = "Astronomia e telescópios espaciais observam galáxias distantes e estrelas binárias no espaço."
    is_grounded, evaluations = checker.check_response(unrelated_answer, context)
    assert is_grounded is False


def test_hallucination_checker_empty_and_null_inputs():
    """Valida resiliência contra entradas vazias e nulas."""
    checker = HallucinationChecker()
    
    assert checker.split_sentences("") == []
    assert checker.split_sentences("   ") == []
    assert checker.split_sentences(None) == []
    
    is_grounded, evals = checker.check_response("", [])
    assert is_grounded is False
    
    is_grounded, evals = checker.check_response("Alguma resposta", [])
    assert is_grounded is False


def test_hallucination_checker_verify_alias_contract():
    """Valida conformidade do método verify() conforme contrato em PROJECT.md."""
    checker = HallucinationChecker()
    context = [{"file_name": "Doc.pdf", "section_title": "Sec", "text": "Texto explicativo 5x2."}]
    
    res = checker.verify("Texto explicativo na escala 5x2.", context)
    assert isinstance(res, dict)
    assert "is_grounded" in res
    assert "overlap_score" in res
    assert res["is_grounded"] is True


# ============================================================================
# 3. TESTES DE INTEGRAÇÃO E2E (AGENT + HALLUCINATION CHECKER INTERCEPTION)
# ============================================================================

def test_hallucination_checker_qa_agent_intercepts_llm_hallucination(temp_chroma_db, mock_chunks):
    """T-R1.14: Valida que GroundedQAAgent intercepta e rejeita alucinação gerada por LLM mockado."""
    indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
    indexer.index_chunks(mock_chunks)
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
    reranker = ReRanker(method="hybrid_fusion")
    agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)
    
    # Mock LLM que retorna alucinação com escala 6x1 e frete R$ 999,00
    class HallucinatingResponse:
        text = "A jornada dos operadores é de 6x1 e o frete grátis exige R$ 999,00."
    
    class FakeGenAIModels:
        def generate_content(self, **kwargs):
            return HallucinatingResponse()
            
    class FakeGenAIClient:
        models = FakeGenAIModels()
        
    agent.genai_client = FakeGenAIClient()
    
    # Executa consulta
    res = agent.answer("Qual é a jornada de trabalho e o frete grátis?")
    
    # O agente DEVE rejeitar a resposta alucinada e disparar fallback seguro
    assert res["is_fallback"] is True
    assert res["hallucination_check"]["is_grounded"] is False
    assert res["hallucination_check"]["reason"] == "hallucination_detected"
    assert "não encontrei informações oficiais" in res["answer"].lower()
    assert len(res["citations"]) == 0


def test_hallucination_checker_corporate_greetings_and_framing():
    """T-R1.15: Valida que saudações formais, cabeçalhos e fórmulas de enquadramento são reconhecidos como framing."""
    checker = HallucinationChecker()
    context = [
        {
            "file_name": "Regulamento_Interno_e_SOP.pdf",
            "section_title": "Jornada de Trabalho",
            "text": "A escala dos operadores é 5x2 com 44 horas semanais.",
        }
    ]
    
    greetings = [
        "Prezado(a) colaborador, seguem as orientações conforme os regulamentos vigentes.",
        "Prezado colaborador, segue o detalhamento da sua dúvida.",
        "Olá, colaborador!",
        "Bom dia!",
        "Boa tarde!",
        "Atenciosamente, Equipe Mercado Central 24h.",
        "Resumo Executivo:",
        "Base Normativa:",
        "Detalhamento:",
        "Citações de Fontes:",
        "[Fonte: Regulamento_Interno_e_SOP.pdf, Seção: Jornada de Trabalho, Pág. 4]",
        "Documento Oficial: Regulamento_Interno_e_SOP.pdf | Seção: Jornada | Páginas: 4-5",
    ]
    for g in greetings:
        eval_res = checker.verify(g, context)
        assert eval_res["is_grounded"] is True, f"Framing rejeitado indevidamente: {g}"
        assert eval_res["is_framing"] is True, f"Deveria ter is_framing=True: {g}"


def test_sentence_tokenizer_terminal_abbreviations_boundary_splitting():
    """T-R1.16: Valida que abreviações terminais (Ltda., Cia., S.A.) quebram sentença quando no fim da oração."""
    checker = HallucinationChecker()
    text = (
        "O regulamento foi aprovado pelo Mercado Central Ltda. "
        "A jornada padrão é na escala 5x2, conforme pág. 4 do SOP. "
        "A diretoria da Cia. aprovou a instrução normativa n.º 12/2026."
    )
    sentences = checker.split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "O regulamento foi aprovado pelo Mercado Central Ltda."
    assert sentences[1] == "A jornada padrão é na escala 5x2, conforme pág. 4 do SOP."
    assert sentences[2] == "A diretoria da Cia. aprovou a instrução normativa n.º 12/2026."


def test_hallucination_checker_cross_domain_currency_grounding():
    """T-R1.17: Valida que moedas não declaradas como monetárias ou com baixa correspondência não passam."""
    checker = HallucinationChecker()
    context = [
        {
            "file_name": "Regulamento_Fidelidade_2026.pdf",
            "section_title": "Benefícios VIP",
            "text": "Clientes VIP Diamante possuem frete grátis a partir de R$ 100,00.",
        }
    ]
    # Reivindicação de cashback falso usando o número 100 do frete
    attack = "O cashback creditado é de R$ 100,00 no cadastro inicial."
    is_ok, evals = checker.check_response(attack, context)
    assert is_ok is False

