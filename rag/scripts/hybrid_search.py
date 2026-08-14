#!/usr/bin/env python3
"""
Hybrid Search Módulo - Mercado Central 24h
Busca Híbrida combinando busca vetorial densa (cosseno) com BM25 esparso (palavras-chave).
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from rag.scripts.vector_indexer import VectorIndexer
except ImportError:
    from vector_indexer import VectorIndexer

logger = logging.getLogger("HybridSearcher")

__all__ = [
    "PORTUGUESE_STOPWORDS",
    "normalize_text",
    "tokenize_portuguese",
    "HybridSearcher",
]


# Stopwords básicas em Português
PORTUGUESE_STOPWORDS: Set[str] = {
    "a", "ao", "aos", "aquela", "aqueles", "aquilo", "as", "até", "com", "como",
    "da", "das", "de", "dela", "dele", "deles", "depois", "do", "dos", "e", "ela",
    "elas", "ele", "eles", "em", "entre", "era", "essa", "essas", "esse", "esses",
    "esta", "estas", "este", "estes", "eu", "foi", "há", "já", "lhe", "mais",
    "mas", "me", "mesmo", "meu", "minha", "muito", "na", "nas", "nem", "no",
    "nos", "nós", "nossa", "nosso", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelas", "pelo", "pelos", "por", "qual", "quando", "que", "quem",
    "se", "seja", "sem", "seu", "seus", "só", "sua", "suas", "também", "te",
    "tem", "temos", "ter", "um", "uma", "você", "vocês"
}


def normalize_text(text: str) -> str:
    """Remove acentos, caracteres especiais e converte para minúsculas."""
    if text is None:
        raise TypeError("text não pode ser None.")
    if not text:
        return ""
    # Decomposição NFD para separar caracteres de acentos
    nfd_form = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(c for c in nfd_form if unicodedata.category(c) != "Mn")
    # Substitui não alfanuméricos por espaço
    cleaned = re.sub(r"[^\w\s]", " ", without_accents)
    return cleaned.strip()


def tokenize_portuguese(text: str) -> List[str]:
    """Tokeniza texto em português para a busca BM25."""
    clean = normalize_text(text)
    tokens = clean.split()
    filtered = [t for t in tokens if len(t) > 1 and t not in PORTUGUESE_STOPWORDS]
    return filtered if filtered else tokens


class HybridSearcher:
    """
    Módulo de Busca Híbrida que combina buscas vetoriais densas (ChromaDB)
    e buscas esparsas por palavras-chave (BM25).
    """

    def __init__(
        self,
        vector_indexer: VectorIndexer,
        chunks_data: Union[str, Path, List[Dict[str, Any]]],
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        """
        :param vector_indexer: Instância configurada do VectorIndexer.
        :param chunks_data: Lista de dicionários de chunks ou caminho para o JSON.
        :param alpha: Peso entre Busca Densa (alpha) e BM25 (1 - alpha). Padrão = 0.5.
        :param rrf_k: Constante para fusion rank de compatibilidade (padrão 60).
        """
        self.vector_indexer: VectorIndexer = vector_indexer
        self.alpha: float = max(0.0, min(1.0, alpha))
        self.rrf_k: int = rrf_k

        if isinstance(chunks_data, (str, Path)):
            chunk_path = Path(chunks_data)
            if not chunk_path.exists():
                raise FileNotFoundError(f"Arquivo de chunks não encontrado: {chunk_path}")
            try:
                with open(chunk_path, "r", encoding="utf-8") as f:
                    self.chunks: List[Dict[str, Any]] = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Erro ao carregar arquivo de chunks JSON '{chunk_path}': {e}")
                raise
        else:
            self.chunks = chunks_data

        # Mapeamento rápido de chunk_id -> chunk dict
        self.chunks_map: Dict[str, Dict[str, Any]] = {c["chunk_id"]: c for c in self.chunks}

        # Inicializa BM25
        self.corpus_tokens: List[List[str]] = [tokenize_portuguese(c["text"]) for c in self.chunks]
        has_tokens: bool = any(len(tokens) > 0 for tokens in self.corpus_tokens)
        if BM25Okapi and self.chunks and has_tokens:
            try:
                self.bm25: Any = BM25Okapi(self.corpus_tokens)
                logger.info(f"Indexador BM25 (rank_bm25) inicializado com {len(self.chunks)} documentos.")
            except ZeroDivisionError:
                self.bm25 = None
                logger.warning("ZeroDivisionError ao inicializar BM25. Usando fallback.")
        else:
            self.bm25 = None
            logger.warning("Biblioteca rank_bm25 não encontrada ou corpus sem tokens. Usando fallback.")

    def _get_bm25_scores(self, query: str) -> np.ndarray:
        """Calcula as pontuações BM25 para cada chunk do corpus."""
        query_tokens = tokenize_portuguese(query)
        if not query_tokens:
            return np.zeros(len(self.chunks))

        if self.bm25:
            return np.array(self.bm25.get_scores(query_tokens), dtype=np.float32)
        else:
            # Fallback simples de sobreposição de termos caso rank_bm25 não esteja instalado
            scores: List[float] = []
            q_set = set(query_tokens)
            for tokens in self.corpus_tokens:
                match_count = sum(1 for t in tokens if t in q_set)
                score = match_count / (len(tokens) + 1.0)
                scores.append(score)
            return np.array(scores, dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executa a busca híbrida combinando Dense Cosine Similarity + BM25 Sparse Score.
        """
        effective_alpha = self.alpha if alpha is None else max(0.0, min(1.0, alpha))

        if not self.chunks or top_k <= 0:
            return []

        # 1. Busca Densa (Vector Search)
        # Recupera os top K * 3 pré-candidatos para fusão
        candidate_k = max(top_k * 3, 50)
        dense_results = self.vector_indexer.search(
            query=query,
            top_k=candidate_k,
            metadata_filter=metadata_filter,
        )

        dense_scores_map: Dict[str, float] = {r["chunk_id"]: r["dense_score"] for r in dense_results}

        # 2. Busca BM25 (Sparse Keyword Search)
        bm25_raw_scores = self._get_bm25_scores(query)

        # Normalização Min-Max das pontuações BM25 no intervalo [0, 1]
        min_bm25 = float(np.min(bm25_raw_scores))
        max_bm25 = float(np.max(bm25_raw_scores))
        range_bm25 = max_bm25 - min_bm25

        sparse_scores_map: Dict[str, float] = {}
        for idx, chunk in enumerate(self.chunks):
            cid = chunk["chunk_id"]
            if range_bm25 > 1e-6:
                norm_score = (bm25_raw_scores[idx] - min_bm25) / range_bm25
            else:
                norm_score = 0.0
            sparse_scores_map[cid] = float(norm_score)

        # 3. Filtragem e Fusão dos Resultados
        all_candidate_ids: Set[str] = set(dense_scores_map.keys())

        # Adiciona top candidatos do BM25 se não estiverem na busca densa
        top_bm25_indices = np.argsort(bm25_raw_scores)[::-1][:candidate_k]
        for idx in top_bm25_indices:
            all_candidate_ids.add(self.chunks[idx]["chunk_id"])

        hybrid_results: List[Dict[str, Any]] = []
        for cid in all_candidate_ids:
            chunk = self.chunks_map.get(cid)
            if not chunk:
                continue

            # Aplica filtro de metadados se fornecido
            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    if chunk.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            dense_score = dense_scores_map.get(cid, 0.0)
            sparse_score = sparse_scores_map.get(cid, 0.0)

            # Cálculo do score híbrido ponderado
            hybrid_score = (effective_alpha * dense_score) + ((1.0 - effective_alpha) * sparse_score)

            merged_item = {
                **chunk,
                "dense_score": round(float(dense_score), 4),
                "sparse_score": round(float(sparse_score), 4),
                "hybrid_score": round(float(hybrid_score), 4),
            }
            hybrid_results.append(merged_item)

        # Ordena resultados pelo score híbrido decrescente
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    script_dir = Path(__file__).parent.resolve()
    json_data_path = script_dir.parent / "data" / "processed_rag_chunks.json"

    print("--- Testando HybridSearcher ---")
    indexer = VectorIndexer(use_mock=True)
    indexer.index_chunks(str(json_data_path))

    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=str(json_data_path), alpha=0.5)

    query = "Frete grátis Cliente VIP Diamante"
    results = searcher.search(query=query, top_k=5)

    print(f"\nResultados para a busca híbrida: '{query}'")
    for idx, r in enumerate(results, 1):
        print(
            f"{idx}. [{r['chunk_id']}] Hybrid Score: {r['hybrid_score']:.4f} "
            f"(Dense: {r['dense_score']:.4f}, Sparse BM25: {r['sparse_score']:.4f}) | "
            f"{r['file_name']} (Págs {r['page_start']}-{r['page_end']}): {r['section_title']}"
        )
