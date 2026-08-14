#!/usr/bin/env python3
"""
Grounded QA Agent Módulo - Mercado Central 24h
Agente de QA com Grounding e citação estrita das fontes (Nome do PDF, Seção, Faixa de Páginas).
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rag.scripts.hybrid_search import HybridSearcher, PORTUGUESE_STOPWORDS, normalize_text
    from rag.scripts.reranker import ReRanker
    from rag.scripts.vector_indexer import VectorIndexer
except ImportError:
    from hybrid_search import HybridSearcher, PORTUGUESE_STOPWORDS, normalize_text
    from reranker import ReRanker
    from vector_indexer import VectorIndexer

logger = logging.getLogger("GroundedQAAgent")

__all__ = [
    "GroundedQAAgent",
]


class GroundedQAAgent:
    """
    Agente Assistente de QA com Grounding RAG e Atribuição de Fontes.
    Garante respostas estritamente baseadas na documentação corporativa do Mercado Central 24h.
    """

    def __init__(
        self,
        indexer: VectorIndexer,
        searcher: HybridSearcher,
        reranker: ReRanker,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self.indexer: VectorIndexer = indexer
        self.searcher: HybridSearcher = searcher
        self.reranker: ReRanker = reranker
        self.model_name: str = model_name

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

        norm_q = normalize_text(query)
        q_tokens = [t for t in norm_q.split() if len(t) > 2 and t not in PORTUGUESE_STOPWORDS]

        if not q_tokens:
            return top_score >= 0.30

        combined_text = " ".join([normalize_text(c.get("text", "") + " " + c.get("section_title", "")) for c in chunks[:3]])
        matched_tokens = sum(1 for t in q_tokens if t in combined_text)
        token_coverage = matched_tokens / len(q_tokens)

        # Se menos da metade das palavras da consulta foram encontradas e o score é mediano/baixo, não está fundamentado
        if token_coverage < 0.50 and top_score < 0.45:
            return False

        return True

    def _generate_extractive_answer(
        self,
        query: str,
        reranked_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Motor de Geração Extrativa de Fallback quando a API LLM estiver offline.
        Sintetiza respostas baseadas nos trechos re-rankeados com citações estritas.
        """
        if not self._is_query_grounded(query, reranked_chunks):
            return {
                "query": query,
                "answer": (
                    "Desculpe, mas não encontrei informações oficiais sobre esse assunto nas diretrizes e "
                    "documentos corporativos do Mercado Central 24h. Por favor, consulte a equipe de Recursos Humanos "
                    "ou a Diretoria Operacional para mais detalhes."
                ),
                "citations": [],
                "sources_used": [],
            }

        relevant_chunks = [c for c in reranked_chunks if c.get("rerank_score", 0.0) >= 0.20][:3]
        if not relevant_chunks:
            relevant_chunks = reranked_chunks[:1]

        answer_paragraphs: List[str] = [
            "Com base na documentação oficial do Mercado Central 24h, segue o detalhamento para a sua consulta:\n"
        ]

        citations: List[Dict[str, Any]] = []
        sources_used: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(relevant_chunks, start=1):
            file_name = chunk.get("file_name", "Documento Oficial")
            section = chunk.get("section_title", "Seção")
            p_start = chunk.get("page_start", 1)
            p_end = chunk.get("page_end", 1)
            page_str = f"Pág. {p_start}" if p_start == p_end else f"Págs. {p_start}-{p_end}"

            text_snippet = chunk.get("text", "").strip()
            lines = text_snippet.split("\n")
            short_text = "\n".join(lines[:6]) if len(lines) > 6 else text_snippet

            citation_tag = f"[Fonte: {file_name}, Seção: {section}, {page_str}]"

            answer_paragraphs.append(
                f"• {short_text}\n  {citation_tag}\n"
            )

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

        full_answer = "\n".join(answer_paragraphs)

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
        top_search_k: int = 15,
        top_rerank_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Pipeline completo do QA Agent:
        1. Hybrid Search
        2. Re-ranking
        3. Grounded Generation (Gemini ou Fallback Extrativo)
        """
        logger.info(f"Executando QA Agent para a consulta: '{query}'...")

        # 1. Busca Híbrida
        hybrid_results = self.searcher.search(
            query=query,
            top_k=top_search_k,
            metadata_filter=metadata_filter,
        )

        if not hybrid_results:
            return {
                "query": query,
                "answer": "Desculpe, mas não encontrei informações oficiais sobre esse assunto na documentação do Mercado Central 24h.",
                "citations": [],
                "sources_used": [],
            }

        # 2. Re-ranking dos melhores trechos
        reranked_chunks = self.reranker.rerank(
            query=query,
            search_results=hybrid_results,
            top_k=top_rerank_k,
        )

        # 3. Formata contexto
        context_str = self._format_context(reranked_chunks)

        # 4. Geração Grounded
        if self.genai_client:
            return self._generate_llm_answer(query, context_str, reranked_chunks)
        else:
            return self._generate_extractive_answer(query, reranked_chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    script_dir = Path(__file__).parent.resolve()
    json_data_path = script_dir.parent / "data" / "processed_rag_chunks.json"

    print("--- Inicializando Pipeline Completo RAG ---")
    indexer = VectorIndexer(use_mock=True)
    indexer.index_chunks(str(json_data_path))

    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=str(json_data_path))
    reranker = ReRanker(method="hybrid_fusion")
    qa_agent = GroundedQAAgent(indexer=indexer, searcher=searcher, reranker=reranker)

    # Teste 1: Pergunta sobre Frete Grátis Cliente VIP
    q1 = "Qual é o valor mínimo de compra para ter frete grátis sendo Cliente VIP Diamante?"
    res1 = qa_agent.answer(q1)

    print("\n" + "=" * 60)
    print(f"PERGUNTA: {res1['query']}")
    print("-" * 60)
    print(f"RESPOSTA:\n{res1['answer']}")
    print("-" * 60)
    print("CITAÇÕES REGISTRADAS:")
    for c in res1["citations"]:
        print(f"  • [{c['file_name']}] {c['section_title']} ({c['page_range']})")
    print("=" * 60)

    # Teste 2: Pergunta fora do escopo corporativo
    q2 = "Qual é a distância média entre a Terra e Marte em quilômetros?"
    res2 = qa_agent.answer(q2)

    print("\n" + "=" * 60)
    print(f"PERGUNTA (Fora do Domínio): {res2['query']}")
    print("-" * 60)
    print(f"RESPOSTA:\n{res2['answer']}")
    print("=" * 60)
