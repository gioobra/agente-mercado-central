#!/usr/bin/env python3
"""
Re-ranker Módulo - Mercado Central 24h
Re-ranking de resultados da Busca Híbrida via Reciprocal Rank Fusion (RRF) e
Cross-Encoder / Feature Score Fusion para elevar os trechos mais relevantes.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger("ReRanker")

__all__ = [
    "ReRanker",
]


try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


class ReRanker:
    """
    Módulo para reordenamento (re-ranking) refinado de trechos de documentos.
    Suporta Reciprocal Rank Fusion (RRF) e Cross-Encoder / Feature-Based Fusion.
    """

    def __init__(
        self,
        method: str = "hybrid_fusion",  # Opções: "rrf", "cross_encoder", "hybrid_fusion"
        model_name: Optional[str] = None,
        rrf_k: int = 60,
        hybrid_weight: float = 0.45,
        title_weight: float = 0.25,
        file_weight: float = 0.10,
        text_weight: float = 0.20,
    ) -> None:
        """
        :param method: Método de re-ranking ("rrf", "cross_encoder", ou "hybrid_fusion").
        :param model_name: Nome do modelo SentenceTransformer/CrossEncoder se disponível.
        :param rrf_k: Constante de suavização para RRF (padrão 60).
        :param hybrid_weight: Peso para o score híbrido na fusão de atributos.
        :param title_weight: Peso para correspondência de título de seção.
        :param file_weight: Peso para correspondência de nome de arquivo.
        :param text_weight: Peso para cobertura no corpo do texto.
        """
        self.method: str = method.lower()
        self.rrf_k: int = rrf_k
        self.hybrid_weight: float = hybrid_weight
        self.title_weight: float = title_weight
        self.file_weight: float = file_weight
        self.text_weight: float = text_weight
        self.cross_encoder_model: Any = None

        if self.method == "cross_encoder":
            if CrossEncoder and model_name:
                try:
                    self.cross_encoder_model = CrossEncoder(model_name)
                    logger.info(f"CrossEncoder carregado: {model_name}")
                except (RuntimeError, ValueError, OSError, TypeError) as e:
                    logger.warning(f"Erro ao carregar CrossEncoder '{model_name}': {e}. Fallback para hybrid_fusion.")
                    self.method = "hybrid_fusion"
            else:
                self.method = "hybrid_fusion"

    def _reciprocal_rank_fusion(
        self,
        dense_ranked: List[Dict[str, Any]],
        sparse_ranked: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Aplica Reciprocal Rank Fusion (RRF) sobre os dois ránkings."""
        rrf_scores: Dict[str, float] = {}
        items_map: Dict[str, Dict[str, Any]] = {}

        # Processa ranking denso
        for rank, item in enumerate(dense_ranked, start=1):
            cid = item["chunk_id"]
            items_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Processa ranking esparso (ordenado por sparse_score)
        sparse_sorted = sorted(sparse_ranked, key=lambda x: x.get("sparse_score", 0.0), reverse=True)
        for rank, item in enumerate(sparse_sorted, start=1):
            cid = item["chunk_id"]
            items_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Ordena pelo score RRF
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        reranked_results: List[Dict[str, Any]] = []
        for final_rank, cid in enumerate(sorted_cids[:top_k], start=1):
            item = dict(items_map[cid])
            item["final_rank"] = final_rank
            item["rerank_score"] = round(float(rrf_scores[cid]), 6)
            item["rerank_method"] = "rrf"
            reranked_results.append(item)

        return reranked_results

    def _feature_score_fusion(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Re-ranking por fusão de features contextuais:
        - Relevância de título/seção (se a query bate no título da seção)
        - Densidade de palavras-chave exatas da query no texto
        - Score híbrido original (dense + sparse)
        """
        domain_acronyms = {"ia", "ai", "rh", "ti", "t1", "t2", "t3", "t4", "t5", "sp", "rj", "nf", "cd", "pj", "pf"}
        synonym_map = {
            "ia": ["ia", "inteligencia", "artificial"],
            "ai": ["ai", "ia", "inteligencia", "artificial"],
            "rh": ["rh", "recursos", "humanos"],
            "dpo": ["dpo", "privacidade", "dados", "lgpd"],
            "cdc": ["cdc", "consumidor", "codigo"],
            "sac": ["sac", "atendimento", "suporte"],
        }

        raw_words = set(re.findall(r"\w+", query.lower()))
        query_words_clean = {w for w in raw_words if len(w) > 2 or w in domain_acronyms}
        
        # Expande sinônimos
        expanded_words = set(query_words_clean)
        for w in query_words_clean:
            if w in synonym_map:
                expanded_words.update(synonym_map[w])

        scored_candidates: List[Dict[str, Any]] = []
        for item in results:
            cid = item["chunk_id"]
            text = item.get("text", "").lower()
            section_title = item.get("section_title", "").lower()
            file_name = item.get("file_name", "").lower()

            hybrid_score = item.get("hybrid_score", 0.0)

            # Boost se os termos da busca estão no título da seção
            title_matches = sum(1 for w in expanded_words if w in section_title)
            title_boost = (title_matches / len(expanded_words)) * self.title_weight if expanded_words else 0.0

            # Boost se os termos da busca estão no nome do arquivo
            file_matches = sum(1 for w in expanded_words if w in file_name)
            file_boost = (file_matches / len(expanded_words)) * self.file_weight if expanded_words else 0.0

            # Cobertura de palavras no corpo do texto
            text_matches = sum(1 for w in expanded_words if w in text)
            text_coverage = (text_matches / len(expanded_words)) * self.text_weight if expanded_words else 0.0

            # Pontuação final do Re-ranking
            rerank_score = (hybrid_score * self.hybrid_weight) + title_boost + file_boost + text_coverage

            new_item = dict(item)
            new_item["rerank_score"] = round(float(rerank_score), 4)
            scored_candidates.append(new_item)

        # Ordena pela pontuação re-rankeada decrescente
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        for final_rank, item in enumerate(scored_candidates[:top_k], start=1):
            item["final_rank"] = final_rank
            item["rerank_method"] = "feature_fusion"

        return scored_candidates[:top_k]

    def rerank(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Executa o re-ranking dos trechos recuperados.
        """
        if not search_results or top_k <= 0:
            return []

        if self.method == "cross_encoder" and self.cross_encoder_model:
            pairs = [[query, r.get("text", "")] for r in search_results]
            scores = self.cross_encoder_model.predict(pairs)

            # Normalização Sigmoid dos scores do CrossEncoder
            norm_scores = 1.0 / (1.0 + np.exp(-scores))

            scored: List[Dict[str, Any]] = []
            for r, sc in zip(search_results, norm_scores):
                item = dict(r)
                item["rerank_score"] = round(float(sc), 4)
                item["rerank_method"] = "cross_encoder"
                scored.append(item)

            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            for rank, item in enumerate(scored[:top_k], start=1):
                item["final_rank"] = rank
            return scored[:top_k]

        elif self.method == "rrf":
            # Extrai ranking denso (ordenado por dense_score)
            dense_sorted = sorted(search_results, key=lambda x: x.get("dense_score", 0.0), reverse=True)
            return self._reciprocal_rank_fusion(dense_sorted, search_results, top_k=top_k)

        else:
            # Padrão: feature_fusion (combina hybrid_score + matching de seção/título/texto)
            return self._feature_score_fusion(query, search_results, top_k=top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    try:
        from rag.scripts.hybrid_search import HybridSearcher
        from rag.scripts.vector_indexer import VectorIndexer
    except ImportError:
        from hybrid_search import HybridSearcher
        from vector_indexer import VectorIndexer

    script_dir = Path(__file__).parent.resolve()
    json_data_path = script_dir.parent / "data" / "processed_rag_chunks.json"

    print("--- Testando ReRanker ---")
    indexer = VectorIndexer(use_mock=True)
    indexer.index_chunks(str(json_data_path))
    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=str(json_data_path))

    query = "Como funciona a política de devolução e reembolso?"
    initial_results = searcher.search(query=query, top_k=10)

    reranker = ReRanker(method="hybrid_fusion")
    top_reranked = reranker.rerank(query=query, search_results=initial_results, top_k=5)

    logger.info(f"\nResultados Re-rankeados para: '{query}'")
    for r in top_reranked:
        print(
            f"Rank #{r['final_rank']} | Score Re-rank: {r['rerank_score']:.4f} (Hybrid: {r['hybrid_score']:.4f}) | "
            f"{r['file_name']} - Seção: {r['section_title']} (Pág {r['page_start']})"
        )
