#!/usr/bin/env python3
"""
Adversarial Stress Test Suite: Milestone 1 (M1 - R1)
Confidence Thresholding & Post-Generation Hallucination Checker Verification

Authors: Empirical Challenger 1
Target Modules:
  - rag/scripts/hallucination_checker.py
  - rag/scripts/grounded_qa_agent.py
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
# FIXTURES & ADVERSARIAL CONTEXT CORPUS
# ============================================================================

@pytest.fixture
def complex_corpus_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": "doc_rh_001",
            "file_name": "Regulamento_Interno_e_SOP.pdf",
            "section_title": "Jornada, Turnos e Escala de Trabalho",
            "page_start": 4,
            "page_end": 5,
            "text": (
                "A jornada de trabalho padrão para operadores e balconistas do Mercado Central 24h "
                "é estruturada sob a escala 5x2, com carga horária de 44 horas semanais e intervalo intrajornada "
                "de 1 hora para refeição e descanso. Horas extras requerem autorização prévia da gerência de RH."
            ),
        },
        {
            "chunk_id": "doc_reembolso_002",
            "file_name": "Politica_de_Reembolso_e_Devolucoes.pdf",
            "section_title": "Direito de Arrependimento e Devoluções",
            "page_start": 2,
            "page_end": 3,
            "text": (
                "Em estrita consonância com o Art. 49 do CDC (Lei 8.078/1990), o consumidor tem o prazo "
                "de 7 dias corridos a contar do recebimento para manifestar arrependimento e solicitar reembolso integral. "
                "Para produtos com vício aparente, o prazo é de 30 dias para bens não duráveis."
            ),
        },
        {
            "chunk_id": "doc_fidelidade_003",
            "file_name": "Regulamento_Fidelidade_2026.pdf",
            "section_title": "Benefícios e Níveis de Cashback",
            "page_start": 1,
            "page_end": 2,
            "text": (
                "O Programa de Fidelidade Mais Central oferece frete grátis em compras acima de R$ 250,00 "
                "para a categoria Padrão, e frete grátis a partir de R$ 100,00 para Clientes VIP Diamante. "
                "Clientes Ouro recebem cashback de 1,5% e Clientes Diamante recebem cashback de 2,0% em todos os pedidos."
            ),
        },
        {
            "chunk_id": "doc_logistica_004",
            "file_name": "Guia_de_Envios_e_Entregas.pdf",
            "section_title": "Modalidades e Prazos de Entrega",
            "page_start": 6,
            "page_end": 7,
            "text": (
                "A Entrega Expressa 24h do Mercado Central 24h garante o recebimento em até 3 horas "
                "para compras faturadas até as 18h no perímetro urbano da Grande São Paulo e Rio de Janeiro."
            ),
        },
        {
            "chunk_id": "doc_lgpd_005",
            "file_name": "Politica_de_Privacidade_LGPD.pdf",
            "section_title": "Canal do Encarregado DPO",
            "page_start": 10,
            "page_end": 11,
            "text": (
                "Nos termos da Lei 13.709/2018 (LGPD), solicitações de titulares de dados devem ser "
                "encaminhadas ao Encarregado pelo e-mail dpo@mercadocentral24h.com.br e serão respondidas em até 15 dias."
            ),
        },
    ]


# ============================================================================
# 1. ADVERSARIAL CHALLENGES: FALSE ENTITY INJECTION & FABRICATION ATTACKS
# ============================================================================

class TestAdversarialEntityInjections:
    """Stress-test HallucinationChecker against hostile entity substitutions and fabrications."""

    def test_rejects_fabricated_currency_r999(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        attack_sentence = (
            "Para obter frete grátis no plano VIP Diamante, a compra mínima necessária é de R$ 999,00."
        )
        res = checker.verify(attack_sentence, complex_corpus_chunks)
        assert res["is_grounded"] is False, "Deve rejeitar moeda falsa R$ 999,00"
        assert any("R$ 999,00" in u for u in res["ungrounded_entities"])

    def test_rejects_fabricated_currency_decimal_substitutions(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        attacks = [
            "O valor mínimo para frete grátis é de R$ 250,50 conforme política.",
            "O cashback fixo creditado é de R$ 88,00 no cadastro inicial.",
            "O frete fixo custa R$ 12,90 para entrega econômica.",
        ]
        for attack in attacks:
            is_ok, evals = checker.check_response(attack, complex_corpus_chunks)
            assert is_ok is False, f"Falha ao rejeitar injeção monetária: {attack}"

    def test_rejects_fabricated_shift_scales_6x1_and_12x36(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        attacks = [
            "Os operadores trabalham na escala 6x1 com descanso semanal remunerado.",
            "A equipe de logística opera em regime de escala 12x36 ininterrupta.",
            "A escala 4x3 foi instituída para todos os balconistas do Mercado Central 24h.",
        ]
        for attack in attacks:
            is_ok, evals = checker.check_response(attack, complex_corpus_chunks)
            assert is_ok is False, f"Falha ao rejeitar escala de trabalho inventada: {attack}"

    def test_rejects_fabricated_sla_durations(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        attacks = [
            "A entrega expressa é efetuada no prazo recorde de 10 minutos na capital.",
            "O prazo para devolução por arrependimento pelo CDC é de 10 dias úteis.",
            "O titular de dados da LGPD terá resposta no prazo de 48 horas.",
            "O direito de troca de produtos com defeito expira em 6 meses.",
        ]
        for attack in attacks:
            is_ok, evals = checker.check_response(attack, complex_corpus_chunks)
            assert is_ok is False, f"Falha ao rejeitar SLA/Duração fabricada: {attack}"

    def test_rejects_fabricated_percentages_discounts(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        attacks = [
            "Clientes VIP Diamante recebem 50% de desconto adicional no frete.",
            "O cashback para o nível Ouro foi elevado para 5,0% em toda a loja.",
            "Há um desconto fixo de 10% para compras pagas via Pix.",
        ]
        for attack in attacks:
            is_ok, evals = checker.check_response(attack, complex_corpus_chunks)
            assert is_ok is False, f"Falha ao rejeitar porcentagem/desconto inventado: {attack}"

    def test_rejects_fabricated_laws_and_articles(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        attacks = [
            "O direito de arrependimento é fundamentado no Art. 18 do CDC.",
            "O tratamento de dados é regido pela Lei 14.133/2021 de licitações.",
            "Conforme o Art. 5º da Constituição Federal, o reembolso é garantido.",
        ]
        for attack in attacks:
            is_ok, evals = checker.check_response(attack, complex_corpus_chunks)
            assert is_ok is False, f"Falha ao rejeitar artigo/lei inventada: {attack}"

    def test_rejects_subtly_altered_numeric_facts(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        # Context has 44 hours, 7 days, 15 days, 30 days
        attacks = [
            "A jornada semanal dos operadores é fixada em 40 horas semanais.",
            "A jornada semanal dos operadores é fixada em 36 horas semanais.",
        ]
        for attack in attacks:
            is_ok, evals = checker.check_response(attack, complex_corpus_chunks)
            assert is_ok is False, f"Falha ao rejeitar número semanal alterado: {attack}"


# ============================================================================
# 2. ADVERSARIAL CHALLENGES: PARAPHRASE ROBUSTNESS & FALSE POSITIVE RESISTANCE
# ============================================================================

class TestParaphraseAndPortugueseSyntaxRobustness:
    """Ensure legitimate Portuguese paraphrasing and syntactic variations are NOT falsely rejected."""

    def test_active_vs_passive_voice_paraphrase(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        # Context: "A jornada de trabalho padrão para operadores e balconistas do Mercado Central 24h é estruturada sob a escala 5x2, com carga horária de 44 horas semanais"
        paraphrase_1 = (
            "Os operadores e balconistas cumprem uma jornada de 44 horas semanais na escala 5x2."
        )
        res1 = checker.verify(paraphrase_1, complex_corpus_chunks)
        assert res1["is_grounded"] is True, f"Paráfrase legítima foi rejeitada indevidamente: {res1}"

    def test_inverted_clause_paraphrase(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        # Context: "Em estrita consonância com o Art. 49 do CDC (Lei 8.078/1990), o consumidor tem o prazo de 7 dias corridos a contar do recebimento para manifestar arrependimento"
        paraphrase_2 = (
            "Para manifestar o arrependimento da compra, o consumidor dispõe de 7 dias corridos segundo o Art. 49 do CDC."
        )
        res2 = checker.verify(paraphrase_2, complex_corpus_chunks)
        assert res2["is_grounded"] is True, f"Inversão de oração legítima foi rejeitada: {res2}"

    def test_frequent_corporate_framing_sentences_accepted(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        framing_sentences = [
            "Com base na documentação oficial do Mercado Central 24h, segue o detalhamento:",
            "Para mais informações sobre as diretrizes corporativas, consulte os canais internos.",
            "Atenciosamente, Assistente Corporativo Mercado Central 24h.",
            "Desculpe, mas não encontrei informações oficiais sobre esse assunto.",
            "Prezado(a) colaborador, seguem as orientações conforme os regulamentos vigentes.",
            "Prezado colaborador, segue o detalhamento:",
            "Olá, colaborador!",
            "Olá colaborador",
            "Bom dia!",
            "Boa tarde!",
            "Atenciosamente, Equipe Mercado Central 24h.",
            "Resumo Executivo:",
            "Base Normativa:",
            "Detalhamento:",
            "Citações de Fontes:",
            "[Fonte: Regulamento_Interno_e_SOP.pdf, Seção: Jornada e Escala, Pág. 1]",
            "Documento Oficial: Regulamento.pdf | Seção: RH | Páginas: 1-2",
        ]
        for s in framing_sentences:
            res = checker.verify(s, complex_corpus_chunks)
            assert res["is_grounded"] is True, f"Sentença de enquadramento foi rejeitada: {s}"
            assert res["is_framing"] is True, f"Sentença deveria ser classificada como is_framing=True: {s}"

    def test_greetings_and_framing_do_not_cause_false_hallucination_rejection(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        response = (
            "Prezado(a) colaborador, seguem as informações solicitadas.\n"
            "O valor mínimo para frete grátis padrão é de R$ 250,00 no Mercado Central 24h.\n"
            "Atenciosamente,\n"
            "Equipe Mercado Central 24h"
        )
        is_ok, evals = checker.check_response(response, complex_corpus_chunks)
        assert is_ok is True, f"Resposta com saudação corporativa e framing foi rejeitada: {evals}"
        assert any(e["is_framing"] for e in evals)

    def test_mixed_grounded_response_with_citations_passes(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        grounded_multi_sentence = (
            "Com base na documentação oficial do Mercado Central 24h, segue o detalhamento:\n"
            "• O prazo para requerer reembolso por arrependimento é de 7 dias corridos, conforme o Art. 49 do CDC.\n"
            "  [Fonte: Politica_de_Reembolso_e_Devolucoes.pdf, Seção: Direito de Arrependimento e Devoluções, Págs. 2-3]\n"
            "• Clientes VIP Diamante possuem benefício de frete grátis em pedidos a partir de R$ 100,00 com cashback de 2,0%.\n"
            "  [Fonte: Regulamento_Fidelidade_2026.pdf, Seção: Benefícios e Níveis de Cashback, Págs. 1-2]"
        )
        is_grounded, evals = checker.check_response(grounded_multi_sentence, complex_corpus_chunks)
        assert is_grounded is True
        assert len(evals) >= 4

    def test_brand_name_24h_does_not_trigger_false_duration_alarm(self, complex_corpus_chunks):
        checker = HallucinationChecker()
        # "Mercado Central 24h" contains "24h", which should not be flagged as a false duration
        sentence = (
            "O Mercado Central 24h oferece aos clientes Diamante cashback de 2,0% nas compras."
        )
        res = checker.verify(sentence, complex_corpus_chunks)
        assert res["is_grounded"] is True
        assert len(res["ungrounded_entities"]) == 0


# ============================================================================
# 3. ADVERSARIAL CHALLENGES: PT-BR SENTENCE TOKENIZER & PUNCTUATION
# ============================================================================

class TestComplexPortugueseSentenceSegmentation:
    """Stress-test sentence boundary detection on Portuguese constructions."""

    def test_abbreviations_inside_clauses_protected(self):
        checker = HallucinationChecker()
        text = (
            "O Dr. Oliveira e a Sra. Santos analisaram o Art. 49 do CDC. "
            "A entrega expressa (ex. SP e RJ) custa R$ 250,00 na escala 5x2, conforme pág. 4 do SOP. "
            "A diretoria aprovou a nova instrução normativa n.º 12/2026."
        )
        sentences = checker.split_sentences(text)
        assert len(sentences) == 3, f"Esperado 3 sentenças, obtido {len(sentences)}: {sentences}"
        assert "Art. 49 do CDC" in sentences[0]
        assert "R$ 250,00" in sentences[1]
        assert "pág. 4" in sentences[1]
        assert "n.º 12/2026" in sentences[2]

    def test_multiple_abbreviations_in_single_paragraph(self):
        checker = HallucinationChecker()
        text = (
            "O Dr. Oliveira e a Sra. Santos analisaram o Art. 49 do CDC no Mercado Central Ltda. "
            "A entrega expressa (ex. SP e RJ) custa R$ 250,00 na escala 5x2, conforme pág. 4 do SOP. "
            "A diretoria da Cia. aprovou a nova instrução normativa n.º 12/2026."
        )
        sentences = checker.split_sentences(text)
        assert len(sentences) == 3, f"Esperado 3 sentenças, obtido {len(sentences)}: {sentences}"
        assert "Mercado Central Ltda." in sentences[0]
        assert "conforme pág. 4 do SOP." in sentences[1]
        assert "diretoria da Cia. aprovou" in sentences[2]

    def test_decimal_values_in_various_portuguese_formats(self):
        checker = HallucinationChecker()
        text = (
            "O valor total foi de R$ 1.500,50 com taxa de 2.5% ao mês. "
            "Para itens de R$ 0,99 a margem é de 10.0% conforme tabela."
        )
        sentences = checker.split_sentences(text)
        assert len(sentences) == 2, f"Esperado 2 sentenças, obtido {len(sentences)}: {sentences}"

    def test_complex_legal_and_acronym_punctuations(self):
        checker = HallucinationChecker()
        text = (
            "Segundo a L.G.P.D. e o C.D.C., o protocolo S.O.P. deve ser seguido. "
            "As fls. 15 e 16 tratam das penalidades aplicáveis pela diretoria."
        )
        sentences = checker.split_sentences(text)
        assert len(sentences) == 2, f"Esperado 2 sentenças, obtido {len(sentences)}: {sentences}"

    def test_dirty_and_adversarial_whitespace_and_newlines(self):
        checker = HallucinationChecker()
        raw = "\n\n  • Item 1: escala 5x2.\n\n\n  * Item 2: frete R$ 250,00.\r\n- Item 3: 7 dias.\n\n"
        sentences = checker.split_sentences(raw)
        assert len(sentences) == 3
        assert sentences[0] == "Item 1: escala 5x2."
        assert sentences[1] == "Item 2: frete R$ 250,00."
        assert sentences[2] == "Item 3: 7 dias."


# ============================================================================
# 4. ADVERSARIAL CHALLENGES: CONFIDENCE THRESHOLD & GATE RESILIENCE
# ============================================================================

class TestConfidenceThresholdGateAdversarial:
    """Stress-test confidence threshold gate with boundary, corrupted, and adversarial queries."""

    @pytest.mark.parametrize("bad_query", [
        "",
        "   ",
        "\t\n\r",
        "???",
        "!!!",
        "...",
        "a",
        "1234567890",
        "asdfghjklqwertyuiopzxcvbnm",
        "' OR '1'='1' --",
        "<script>alert(1)</script>",
        "SELECT * FROM chunks WHERE id = 1;",
        "🚀🌟🔥🎉💯",
    ])
    def test_adversarial_and_garbage_queries_safely_fallback(self, temp_chroma_db, mock_chunks, bad_query):
        indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
        indexer.index_chunks(mock_chunks)
        searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
        reranker = ReRanker(method="hybrid_fusion")
        agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)

        res = agent.answer(bad_query)
        assert isinstance(res, dict)
        assert res["is_fallback"] is True
        assert res["citations"] == []
        assert "não encontrei informações oficiais" in res["answer"].lower()

    @pytest.mark.parametrize("invalid_threshold, expected_sanitized", [
        (-100.0, 0.0),
        (-0.0001, 0.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.0001, 1.0),
        (999.0, 1.0),
        (float("nan"), 0.35),
        (float("inf"), 0.35),
        (float("-inf"), 0.35),
        ("0.42", 0.42),
        ("not_a_float", 0.35),
        ([], 0.35),
        ({}, 0.35),
        (None, 0.35),
    ])
    def test_confidence_threshold_extreme_sanitizations(self, invalid_threshold, expected_sanitized):
        sanitized = GroundedQAAgent._sanitize_confidence_threshold(invalid_threshold, default=0.35)
        assert math.isclose(sanitized, expected_sanitized, abs_tol=1e-5)

    def test_synthetic_rerank_score_gate_precision(self, temp_chroma_db, mock_chunks):
        """Test exact boundary transitions of confidence_threshold."""
        indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
        indexer.index_chunks(mock_chunks)
        searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
        
        # Test custom reranker injecting synthetic score
        class MockScoreReranker:
            def __init__(self, synthetic_score: float):
                self.synthetic_score = synthetic_score
                
            def rerank(self, query: str, search_results: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
                results = []
                for idx, r in enumerate(search_results[:top_k]):
                    item = dict(r)
                    item["rerank_score"] = self.synthetic_score - (idx * 0.05)
                    results.append(item)
                return results

        # 1. Score just below threshold (0.3499 vs 0.3500) -> MUST FALLBACK
        agent_below = GroundedQAAgent(
            indexer=indexer,
            searcher=searcher,
            reranker=MockScoreReranker(synthetic_score=0.3499),
            confidence_threshold=0.35,
        )
        res_below = agent_below.answer("Qual é o frete grátis?")
        assert res_below["is_fallback"] is True
        assert res_below["confidence_score"] == 0.3499
        assert res_below["hallucination_check"]["reason"] == "confidence_below_threshold"

        # 2. Score equal to threshold (0.3500 vs 0.3500) -> PROCEEDS
        agent_equal = GroundedQAAgent(
            indexer=indexer,
            searcher=searcher,
            reranker=MockScoreReranker(synthetic_score=0.3500),
            confidence_threshold=0.35,
        )
        res_equal = agent_equal.answer("Qual o frete grátis Cliente VIP Diamante?")
        # Grounded query with sufficient score proceeds
        assert res_equal["confidence_score"] == 0.3500


# ============================================================================
# 5. ADVERSARIAL CHALLENGES: E2E LLM INTERCEPTION & CONTRACT CONFORMANCE
# ============================================================================

class TestE2ELLMInterceptionAndContract:
    """Stress-test full agent pipeline with corrupted LLM outputs."""

    def test_interception_of_subtly_corrupted_llm_generation(self, temp_chroma_db, mock_chunks):
        """Simulate LLM generating mostly true response but corrupting a single critical entity."""
        indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
        indexer.index_chunks(mock_chunks)
        searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
        reranker = ReRanker(method="hybrid_fusion")
        agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)

        # Mock LLM generating 1 valid sentence and 1 poisoned sentence with fake monetary threshold R$ 999,00
        class PoisonedGenAIResponse:
            text = (
                "O prazo de devolução por arrependimento é de 7 dias corridos conforme o Art. 49 do CDC.\n"
                "Para frete grátis VIP Diamante, a compra deve ser de no mínimo R$ 999,00."
            )

        class PoisonedGenAIModels:
            def generate_content(self, **kwargs):
                return PoisonedGenAIResponse()

        class PoisonedGenAIClient:
            models = PoisonedGenAIModels()

        agent.genai_client = PoisonedGenAIClient()

        res = agent.answer("Qual o prazo de devolução e valor do frete grátis VIP?")
        
        # Must be intercepted by HallucinationChecker
        assert res["is_fallback"] is True
        assert res["hallucination_check"]["is_grounded"] is False
        assert res["hallucination_check"]["reason"] == "hallucination_detected"
        assert len(res["citations"]) == 0
        assert "não encontrei informações oficiais" in res["answer"].lower()

    def test_full_response_dict_contract_keys(self, temp_chroma_db, mock_chunks):
        """Validate exact keys and schema returned by GroundedQAAgent.answer()."""
        indexer = VectorIndexer(use_mock=True, db_path=temp_chroma_db)
        indexer.index_chunks(mock_chunks)
        searcher = HybridSearcher(vector_indexer=indexer, chunks_data=mock_chunks)
        reranker = ReRanker(method="hybrid_fusion")
        agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)

        res = agent.answer("Qual é o frete grátis para Cliente VIP Diamante?")
        
        required_keys = {
            "query",
            "answer",
            "citations",
            "sources_used",
            "confidence_score",
            "confidence_threshold",
            "is_fallback",
            "fallback_department",
            "hallucination_check",
        }
        for k in required_keys:
            assert k in res, f"Chave obrigatória ausente no contrato de retorno: '{k}'"

        assert isinstance(res["citations"], list)
        assert isinstance(res["sources_used"], list)
        assert isinstance(res["confidence_score"], float)
        assert isinstance(res["is_fallback"], bool)
        assert isinstance(res["hallucination_check"], dict)
        assert "is_grounded" in res["hallucination_check"]
        assert "reason" in res["hallucination_check"]
