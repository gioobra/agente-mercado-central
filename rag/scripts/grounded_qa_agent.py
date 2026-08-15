#!/usr/bin/env python3
"""
Grounded QA Agent Módulo - Mercado Central 24h
Agente de QA com Grounding, limiar de confiança, verificação de alucinação sentença por sentença,
roteamento corporativo de contatos por intenção e formatação adaptável multicanal (chat, email, teams_slack).
"""

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from rag.scripts.hybrid_search import HybridSearcher, PORTUGUESE_STOPWORDS, normalize_text, _normalize_recency_params
    from rag.scripts.reranker import ReRanker
    from rag.scripts.vector_indexer import VectorIndexer
    from rag.scripts.hallucination_checker import HallucinationChecker
    from rag.scripts.contact_catalog import (
        CORPORATE_CONTACT_CATALOG,
        route_fallback_contact,
        format_fallback_message,
    )
    from rag.scripts.multichannel_formatter import (
        format_multichannel_response,
        extract_tldr_and_details,
        sanitize_channel_name,
    )
except ImportError:
    from hybrid_search import HybridSearcher, PORTUGUESE_STOPWORDS, normalize_text, _normalize_recency_params
    from reranker import ReRanker
    from vector_indexer import VectorIndexer
    from hallucination_checker import HallucinationChecker
    from contact_catalog import (
        CORPORATE_CONTACT_CATALOG,
        route_fallback_contact,
        format_fallback_message,
    )
    from multichannel_formatter import (
        format_multichannel_response,
        extract_tldr_and_details,
        sanitize_channel_name,
    )

logger = logging.getLogger("GroundedQAAgent")

__all__ = [
    "GroundedQAAgent",
    "HallucinationChecker",
    "CORPORATE_CONTACT_CATALOG",
    "route_fallback_contact",
    "format_fallback_message",
    "format_multichannel_response",
    "extract_tldr_and_details",
    "sanitize_channel_name",
]


class GroundedQAAgent:
    """
    Agente Assistente de QA com Grounding RAG, Limiar de Confiança, Verificação Sentencial de Consistência,
    Roteamento de Fallback por Intenção e Formatação Multicanal.
    """

    def __init__(
        self,
        indexer: VectorIndexer,
        searcher: HybridSearcher,
        reranker: ReRanker,
        model_name: str = "gemini-2.5-flash",
        recency_boost: Optional[Union[bool, float, str]] = None,
        recency_weight: Optional[float] = None,
        confidence_threshold: float = 0.35,
        hallucination_checker: Optional[HallucinationChecker] = None,
    ) -> None:
        self.indexer: VectorIndexer = indexer
        self.searcher: HybridSearcher = searcher
        self.reranker: ReRanker = reranker
        self.model_name: str = model_name

        default_agent_boost = getattr(searcher, "recency_boost", False)
        default_agent_weight = getattr(searcher, "recency_weight", 0.15)
        self.recency_boost, self.recency_weight = _normalize_recency_params(
            recency_boost=recency_boost,
            recency_weight=recency_weight,
            default_boost=default_agent_boost,
            default_weight=default_agent_weight,
        )

        self.confidence_threshold: float = self._sanitize_confidence_threshold(confidence_threshold, default=0.35)
        self.hallucination_checker: HallucinationChecker = hallucination_checker or HallucinationChecker()

        self.api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.genai_client: Any = None

        if self.api_key:
            try:
                from google import genai
                self.genai_client = genai.Client(api_key=self.api_key)
                logger.info(f"GenAI Client inicializado com sucesso ({self.model_name}).")
            except (ImportError, ValueError, AttributeError, RuntimeError) as e:
                logger.warning(f"Erro ao inicializar cliente Google GenAI: {e}. Usando motor extrativo de fallback.")
        else:
            logger.info("Chave GEMINI_API_KEY não configurada. Agente executará no modo Extrativo de Fallback.")

    @staticmethod
    def _sanitize_confidence_threshold(val: Any, default: float = 0.35) -> float:
        """Sanitiza e normaliza o valor do limiar de confiança no intervalo [0.0, 1.0]."""
        if val is None:
            return default
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return default
            return max(0.0, min(1.0, f))
        except (ValueError, TypeError):
            return default

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Formata os chunks de contexto recuperados para injeção no prompt ou síntese."""
        context_blocks: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            p_start = chunk.get("page_start", 1)
            p_end = chunk.get("page_end", 1)
            page_str = f"Pág. {p_start}" if p_start == p_end else f"Págs. {p_start}-{p_end}"

            block = (
                f"--- DOCUMENTO [{idx}] ---\n"
                f"Arquivo: {chunk.get('file_name', 'N/A')}\n"
                f"Seção: {chunk.get('section_title', 'N/A')}\n"
                f"Páginas: {page_str}\n"
                f"ID: {chunk.get('chunk_id', 'N/A')}\n"
                f"Conteúdo:\n{chunk.get('text', '').strip()}\n"
            )
            context_blocks.append(block)
        return "\n".join(context_blocks)

    def _is_query_grounded(self, query: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        Verifica se a consulta possui fundamentação suficiente nos trechos retornados.
        Analisa a cobertura de palavras-chave no contexto e o score de re-ranking.
        """
        if not query or not str(query).strip() or not chunks:
            return False

        top_chunk = chunks[0]
        top_score = top_chunk.get("rerank_score", 0.0)

        domain_acronyms = {"ia", "ai", "rh", "ti", "t1", "t2", "t3", "t4", "t5", "sp", "rj", "nf"}
        synonym_map = {
            "ia": ["ia", "inteligencia", "artificial"],
            "ai": ["ai", "ia", "inteligencia", "artificial"],
            "rh": ["rh", "recursos", "humanos"],
            "dpo": ["dpo", "privacidade", "dados", "lgpd"],
        }

        norm_q = normalize_text(query)
        raw_tokens = [t for t in norm_q.split() if (len(t) > 2 or t in domain_acronyms) and t not in PORTUGUESE_STOPWORDS]
        q_tokens = list(raw_tokens)
        for t in raw_tokens:
            if t in synonym_map:
                q_tokens.extend(synonym_map[t])

        if not q_tokens:
            return top_score >= 0.30

        combined_text = " ".join([normalize_text(c.get("text", "") + " " + c.get("section_title", "")) for c in chunks[:5]])
        matched_tokens = sum(1 for t in q_tokens if t in combined_text)
        token_coverage = matched_tokens / len(q_tokens)

        # Se menos de 30% das palavras expandidas foram encontradas e o score é baixo, não está fundamentado
        if token_coverage < 0.30 and top_score < 0.35:
            return False

        return True

    def _generate_extractive_answer(
        self,
        query: str,
        reranked_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Motor de Geração Extrativa de Fallback quando a API LLM estiver offline.
        Sintetiza respostas baseadas nos trechos re-rankeados com citações estritas e texto limpo.
        """
        if not self._is_query_grounded(query, reranked_chunks):
            return {
                "query": query,
                "answer": (
                    "Desculpe, mas não encontrei informações oficiais sobre esse assunto nas diretrizes e "
                    "documentos corporativos do Mercado Central 24h. Por favor, consulte a equipe de Recursos Humanos "
                    "ou a Ouvidoria Geral para mais detalhes."
                ),
                "citations": [],
                "sources_used": [],
            }

        # Chunks relevantes para síntese
        substantive_chunks = [
            c for c in reranked_chunks 
            if len(c.get("text", "")) >= 40 and (c.get("rerank_score") is None or c.get("rerank_score", 0.0) >= 0.15)
        ]
        if not substantive_chunks:
            substantive_chunks = reranked_chunks[:3]

        def _clean_and_merge_paragraphs(text: str) -> List[str]:
            if not text:
                return []
            text = re.sub(r"<[^>]+>", "", text).strip()
            raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
            merged = []
            buf = ""
            for line in raw_lines:
                if line in ("•", "-", "*", "—"):
                    continue
                line = re.sub(r"^[•\-\*]+\s*", "", line).strip()
                if not line:
                    continue
                if buf:
                    if buf.endswith((".", "!", "?", ":")) or re.match(r"^\d+[\.\)]\s+", line):
                        merged.append(buf)
                        buf = line
                    else:
                        buf = f"{buf} {line}"
                else:
                    buf = line
            if buf:
                merged.append(buf)
            return merged

        def _is_noisy_header(p: str, sec_title: str) -> bool:
            p_strip = p.strip()
            if p_strip.endswith("?") and len(p_strip) < 120:
                return True
            p_clean = re.sub(r"^\d+[\.\)]\s*", "", p_strip).strip().lower()
            if p_clean == sec_title.strip().lower() or len(p_clean) < 20:
                return True
            if not p_strip.endswith((".", "!", "?", ":", ")", '"', "'")) and len(p_strip) < 60:
                return True
            if re.match(r"^\d+(\.\d+)*\.\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ\s,–-]+$", p_strip):
                return True
            if "Modalidade" in p_strip and "Disponibilidade" in p_strip:
                return True
            return False

        all_points: List[str] = []
        seen_keys = set()
        citations: List[Dict[str, Any]] = []
        sources_used: List[Dict[str, Any]] = []

        for chunk in substantive_chunks[:4]:
            file_name = chunk.get("file_name") or "Documento Oficial"
            section = chunk.get("section_title") or "Seção"
            p_start = chunk.get("page_start", 1)
            p_end = chunk.get("page_end", 1)
            page_str = f"Pág. {p_start}" if p_start == p_end else f"Págs. {p_start}-{p_end}"

            citation_tag = f"[Fonte: {file_name}, Seção: {section}, {page_str}]"

            paragraphs = _clean_and_merge_paragraphs(chunk.get("text", ""))
            chunk_points: List[str] = []
            for p in paragraphs:
                if _is_noisy_header(p, str(section)):
                    continue
                p_clean = re.sub(r"^\d+[\.\)]\s*", "", p).strip()
                p_clean = re.sub(r"\s+", " ", p_clean)
                key = p_clean[:45].lower()
                if key not in seen_keys and len(p_clean) >= 25:
                    seen_keys.add(key)
                    chunk_points.append(f"{p_clean}\n  {citation_tag}")

            if chunk_points:
                all_points.extend(chunk_points[:2])

            citations.append({
                "file_name": file_name,
                "section_title": section,
                "page_range": page_str,
                "chunk_id": chunk.get("chunk_id"),
            })
            sources_used.append({
                "chunk_id": chunk.get("chunk_id"),
                "file_name": file_name,
                "section_title": section,
                "page_start": p_start,
                "page_end": p_end,
                "rerank_score": chunk.get("rerank_score"),
            })

        if not all_points:
            # Fallback para primeiro trecho bruto limpo
            for c in substantive_chunks[:2]:
                raw_clean = re.sub(r"<[^>]+>", "", c.get("text", "")).strip()
                if raw_clean:
                    fn = c.get("file_name") or "Documento Oficial"
                    sec = c.get("section_title") or "Seção"
                    all_points.append(f"{raw_clean[:200]}\n  [Fonte: {fn}, Seção: {sec}, Pág. {c.get('page_start', 1)}]")

        intro = "Com base na documentação oficial do Mercado Central 24h, segue o detalhamento para a sua consulta:"
        details = "\n\n".join([f"• {p}" for p in all_points[:6]])

        full_answer = f"{intro}\n\n{details}"

        return {
            "query": query,
            "answer": full_answer.strip(),
            "citations": citations,
            "sources_used": sources_used,
        }

    def _generate_llm_answer(
        self,
        query: str,
        context_str: str,
        reranked_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Gera resposta com o Google Gemini API aplicando grounding e prompt de citação estrita."""
        if not self._is_query_grounded(query, reranked_chunks):
            return {
                "query": query,
                "answer": (
                    "Desculpe, mas não encontrei informações oficiais sobre esse assunto na "
                    "documentação do Mercado Central 24h."
                ),
                "citations": [],
                "sources_used": [],
            }

        system_instruction = (
            "Você é o Assistente Corporativo Inteligente do Mercado Central 24h. "
            "Sua tarefa é responder a pergunta do colaborador estritamente com base nos documentos de contexto fornecidos.\n\n"
            "REGRAS DE GROUNDING E CITAÇÃO OBRIGATÓRIAS:\n"
            "1. Responda apenas com informações contidas no contexto.\n"
            "2. Se a informação não constar explicitamente nos documentos, responda: "
            "'Desculpe, mas não encontrei informações oficiais sobre esse assunto na documentação do Mercado Central 24h.'\n"
            "3. Cada afirmação deve ser seguida de citação explícita no formato: "
            "[Fonte: Nome_do_PDF.pdf, Seção: Titulo_da_Secao, Págs. X-Y].\n"
            "4. Mantenha um tom profissional, direto e em Português do Brasil.\n"
        )

        user_prompt = f"CONTEXTO DOS DOCUMENTOS:\n{context_str}\n\nPERGUNTA DO COLABORADOR:\n{query}"

        try:
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={"system_instruction": system_instruction},
            )
            answer_text = response.text.strip()

            citations: List[Dict[str, Any]] = []
            sources_used: List[Dict[str, Any]] = []
            for c in reranked_chunks:
                p_start = c.get("page_start", 1)
                p_end = c.get("page_end", 1)
                page_str = f"Pág. {p_start}" if p_start == p_end else f"Págs. {p_start}-{p_end}"

                citations.append({
                    "file_name": c.get("file_name"),
                    "section_title": c.get("section_title"),
                    "page_range": page_str,
                    "chunk_id": c.get("chunk_id"),
                })
                sources_used.append({
                    "chunk_id": c.get("chunk_id"),
                    "file_name": c.get("file_name"),
                    "section_title": c.get("section_title"),
                    "page_start": p_start,
                    "page_end": p_end,
                    "rerank_score": c.get("rerank_score"),
                })

            return {
                "query": query,
                "answer": answer_text,
                "citations": citations,
                "sources_used": sources_used,
            }
        except Exception as e:
            logger.error(f"Erro na chamada Gemini LLM: {e}. Executando fallback extrativo.")
            return self._generate_extractive_answer(query, reranked_chunks)

    def answer(
        self,
        query: str,
        top_search_k: int = 20,
        top_rerank_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        recency_boost: Optional[Union[bool, float, str]] = None,
        recency_weight: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        channel: str = "chat",
    ) -> Dict[str, Any]:
        """
        Pipeline completo do QA Agent:
        1. Hybrid Search (com suporte a priorização por recência)
        2. Re-ranking contextual
        3. Validação de Limiar de Confiança (Confidence Gate)
        4. Grounded Generation (Gemini ou Fallback Extrativo)
        5. Verificação de Alucinação Pós-Geração Sentença por Sentença
        6. Roteamento Inteligente de Fallback por Intenção (R2)
        7. Formatação Multicanal Adaptável (R3)
        """
        logger.info(f"Executando QA Agent para a consulta: '{query}' (canal: {channel})...")

        clean_channel = sanitize_channel_name(channel)

        effective_recency_boost, effective_recency_weight = _normalize_recency_params(
            recency_boost=recency_boost,
            recency_weight=recency_weight,
            default_boost=self.recency_boost,
            default_weight=self.recency_weight,
        )

        effective_confidence_threshold = (
            self._sanitize_confidence_threshold(confidence_threshold, default=self.confidence_threshold)
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        # 1. Busca Híbrida
        hybrid_results = self.searcher.search(
            query=query,
            top_k=top_search_k,
            metadata_filter=metadata_filter,
            recency_boost=effective_recency_boost,
            recency_weight=effective_recency_weight,
        )

        if not hybrid_results:
            routed_dept = route_fallback_contact(query)
            fallback_text = format_fallback_message(query, routed_dept, channel=clean_channel)
            return {
                "query": query,
                "answer": fallback_text,
                "citations": [],
                "sources_used": [],
                "confidence_score": 0.0,
                "confidence_threshold": effective_confidence_threshold,
                "is_fallback": True,
                "fallback_department": routed_dept.get("department_key"),
                "channel": clean_channel,
                "hallucination_check": {
                    "is_grounded": False,
                    "reason": "no_search_results",
                    "sentences": [],
                },
            }

        # 2. Re-ranking dos melhores trechos com seleção adaptativa por complexidade
        candidate_k = max(top_rerank_k, 8)
        all_reranked_chunks = self.reranker.rerank(
            query=query,
            search_results=hybrid_results,
            top_k=candidate_k,
        )

        if not all_reranked_chunks:
            routed_dept = route_fallback_contact(query)
            fallback_text = format_fallback_message(query, routed_dept, channel=clean_channel)
            return {
                "query": query,
                "answer": fallback_text,
                "citations": [],
                "sources_used": [],
                "confidence_score": 0.0,
                "confidence_threshold": effective_confidence_threshold,
                "is_fallback": True,
                "fallback_department": routed_dept.get("department_key"),
                "channel": clean_channel,
                "hallucination_check": {
                    "is_grounded": False,
                    "reason": "no_rerank_results",
                    "sentences": [],
                },
            }

        # Filtragem Dinâmica Adaptativa:
        # Pergunta simples/focada -> seleciona 2 a 3 fontes altamente aderentes.
        # Pergunta complexa/multidocumental -> expande automaticamente para 4 a 8 fontes relevantes.
        top_score = all_reranked_chunks[0].get("rerank_score", 0.0)
        reranked_chunks: List[Dict[str, Any]] = []
        seen_docs: Set[str] = set()

        for idx, c in enumerate(all_reranked_chunks):
            score = c.get("rerank_score", 0.0)
            doc = c.get("file_name", "")

            # Inclui os primeiros 2 se tiverem score razoável
            if idx < 2 and score >= 0.15:
                reranked_chunks.append(c)
                seen_docs.add(doc)
                continue

            # Se tiver alta relevância relativa (>= 70% do top_score)
            if score >= top_score * 0.70 and score >= 0.22:
                reranked_chunks.append(c)
                seen_docs.add(doc)
            # Se for de um documento diferente com relevância substantiva (pergunta multidisciplinar)
            elif doc not in seen_docs and score >= 0.28 and score >= top_score * 0.55:
                reranked_chunks.append(c)
                seen_docs.add(doc)

            if len(reranked_chunks) >= 8:
                break

        if not reranked_chunks:
            reranked_chunks = all_reranked_chunks[:2]

        top_score = float(reranked_chunks[0].get("rerank_score", 0.0))

        # 3. Portão de Limiar de Confiança Pré-Geração
        if top_score < effective_confidence_threshold or not self._is_query_grounded(query, reranked_chunks):
            routed_dept = route_fallback_contact(query)
            fallback_text = format_fallback_message(query, routed_dept, channel=clean_channel)
            return {
                "query": query,
                "answer": fallback_text,
                "citations": [],
                "sources_used": [],
                "confidence_score": top_score,
                "confidence_threshold": effective_confidence_threshold,
                "is_fallback": True,
                "fallback_department": routed_dept.get("department_key"),
                "channel": clean_channel,
                "hallucination_check": {
                    "is_grounded": False,
                    "reason": "confidence_below_threshold",
                    "sentences": [],
                },
            }

        # 4. Formata contexto
        context_str = self._format_context(reranked_chunks)

        # 5. Geração Grounded
        if self.genai_client:
            raw_response = self._generate_llm_answer(query, context_str, reranked_chunks)
        else:
            raw_response = self._generate_extractive_answer(query, reranked_chunks)

        # Se a resposta gerada já for uma recusa interna
        if not raw_response.get("citations") and "não encontrei informações oficiais" in raw_response.get("answer", "").lower():
            routed_dept = route_fallback_contact(query)
            fallback_text = format_fallback_message(query, routed_dept, channel=clean_channel)
            return {
                "query": query,
                "answer": fallback_text,
                "citations": [],
                "sources_used": [],
                "confidence_score": top_score,
                "confidence_threshold": effective_confidence_threshold,
                "is_fallback": True,
                "fallback_department": routed_dept.get("department_key"),
                "channel": clean_channel,
                "hallucination_check": {
                    "is_grounded": False,
                    "reason": "query_not_grounded",
                    "sentences": [],
                },
            }

        # 6. Verificação de Alucinação Pós-Geração Sentença por Sentença
        is_grounded, sentence_evals = self.hallucination_checker.check_response(
            answer=raw_response.get("answer", ""),
            context_chunks=reranked_chunks,
        )

        if not is_grounded:
            logger.warning(f"Resposta rejeitada pelo HallucinationChecker para consulta '{query}'.")
            routed_dept = route_fallback_contact(query)
            fallback_text = format_fallback_message(query, routed_dept, channel=clean_channel)
            return {
                "query": query,
                "answer": fallback_text,
                "citations": [],
                "sources_used": [],
                "confidence_score": top_score,
                "confidence_threshold": effective_confidence_threshold,
                "is_fallback": True,
                "fallback_department": routed_dept.get("department_key"),
                "channel": clean_channel,
                "hallucination_check": {
                    "is_grounded": False,
                    "reason": "hallucination_detected",
                    "sentences": sentence_evals,
                },
            }

        # 7. Formatação Multicanal Tripartite da Resposta Grounded Válida
        tldr, details = extract_tldr_and_details(raw_response.get("answer", ""))
        formatted_answer = format_multichannel_response(
            tldr=tldr,
            details=details,
            citations=raw_response.get("citations", []),
            channel=clean_channel,
        )

        return {
            "query": query,
            "answer": formatted_answer,
            "citations": raw_response.get("citations", []),
            "sources_used": raw_response.get("sources_used", []),
            "confidence_score": top_score,
            "confidence_threshold": effective_confidence_threshold,
            "is_fallback": False,
            "fallback_department": None,
            "channel": clean_channel,
            "hallucination_check": {
                "is_grounded": True,
                "reason": "passed",
                "sentences": sentence_evals,
            },
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    script_dir = Path(__file__).parent.resolve()
    json_data_path = script_dir.parent / "data" / "processed_rag_chunks.json"

    print("--- Inicializando Pipeline Completo RAG com R2 e R3 ---")
    indexer = VectorIndexer(use_mock=True)
    indexer.index_chunks(str(json_data_path))

    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=str(json_data_path))
    reranker = ReRanker(method="hybrid_fusion")
    qa_agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)

    # Teste 1: Pergunta sobre Frete Grátis Cliente VIP (chat)
    q1 = "Qual é o valor mínimo de compra para ter frete grátis sendo Cliente VIP Diamante?"
    res1 = qa_agent.answer(q1, channel="chat")
    print("\n" + "=" * 60)
    print(f"PERGUNTA: {res1['query']} (Canal: {res1['channel']})")
    print(f"RESPOSTA:\n{res1['answer']}")

    # Teste 2: Pergunta em canal Email
    res2 = qa_agent.answer(q1, channel="email")
    print("\n" + "=" * 60)
    print(f"RESPOSTA EMAIL:\n{res2['answer']}")

    # Teste 3: Fallback roteado para RH em canal Teams
    q3 = "Qual a data de pagamento do salário e do vale adiantamento?"
    res3 = qa_agent.answer(q3, confidence_threshold=0.99, channel="teams_slack")
    print("\n" + "=" * 60)
    print(f"FALLBACK TEAMS (RH):\n{res3['answer']}")
