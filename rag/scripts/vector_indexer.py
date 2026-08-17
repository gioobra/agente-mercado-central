#!/usr/bin/env python3
"""
Vector Indexer Módulo - Mercado Central 24h
Indexação no ChromaDB utilizando o modelo Google text-embedding-004
com suporte a fallback mock local de 768 dimensões para testes offline.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
import numpy as np

# Configuração do Logger
logger = logging.getLogger("VectorIndexer")

__all__ = [
    "MockEmbeddingFunction",
    "GoogleGenAIEmbeddingFunction",
    "VectorIndexer",
]


class MockEmbeddingFunction:
    """
    Função de embedding local determinística de 768 dimensões para execução offline / testes unitários.
    Gera vetores normalizados baseados em hashes das palavras/palavra-chave do texto.
    """

    def __init__(self, dimension: int = 768) -> None:
        self.dimension: int = dimension

    def __call__(self, input_texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        embeddings: List[List[float]] = []
        for text in input_texts:
            # Deterministic pseudo-random vector based on text MD5/SHA256 hash
            clean_text = str(text).lower().strip() if text is not None else ""
            # Token-based seed pooling for semantic similarity in mock mode
            tokens = clean_text.split()
            vec = np.zeros(self.dimension, dtype=np.float32)

            if tokens:
                for token in tokens:
                    token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                    rng = np.random.RandomState(token_hash % (2**32))
                    vec += rng.randn(self.dimension)
                vec = vec / len(tokens)
            else:
                text_hash = int(hashlib.md5(clean_text.encode("utf-8")).hexdigest(), 16)
                rng = np.random.RandomState(text_hash % (2**32))
                vec = rng.randn(self.dimension)

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())

        return embeddings


class GoogleGenAIEmbeddingFunction:
    """
    Função de embedding utilizando a API do Google Gemini (text-embedding-004).
    Retorna vetores de 768 dimensões.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-embedding-001") -> None:
        self.model_name: str = model_name
        self.api_key: Optional[str] = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client: Any = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Google GenAI Client inicializado com modelo {self.model_name}.")
            except Exception as e:
                logger.warning(f"Erro ao inicializar Google GenAI Client: {e}. Usando fallback local.")
                self.client = None
        else:
            logger.info("Chave de API do Google não encontrada. Fallback ativo.")

    def embed_texts(self, input_texts: List[str]) -> List[List[float]]:
        if not self.client or not input_texts:
            return self._fallback_embed_texts(input_texts)

        embeddings: List[List[float]] = []
        batch_size = 32
        candidate_models = [
            self.model_name,
            "gemini-embedding-001",
            "models/gemini-embedding-001",
            "gemini-embedding-2-preview",
            "text-embedding-004",
            "embedding-001",
        ]
        candidate_models = list(dict.fromkeys(candidate_models))

        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i : i + batch_size]
            batch_emb_values = None

            for model in candidate_models:
                try:
                    response = self.client.models.embed_content(
                        model=model,
                        contents=batch,
                        config={"output_dimensionality": 768},
                    )
                    if response and hasattr(response, "embeddings") and response.embeddings:
                        extracted = [e.values for e in response.embeddings]
                        if len(extracted) == len(batch):
                            batch_emb_values = extracted
                            self.model_name = model
                            break
                except Exception as e:
                    logger.debug(f"Modelo {model} falhou em batch: {e}")
                    continue

            # Se a API não retornou o número exato de embeddings para o batch, usa fallback local
            if batch_emb_values is None or len(batch_emb_values) != len(batch):
                logger.debug(f"Usando fallback determinístico para batch de {len(batch)} itens.")
                batch_emb_values = self._fallback_embed_texts(batch)

            embeddings.extend(batch_emb_values)

        # Garantia final absoluta de consistência 1-para-1
        if len(embeddings) != len(input_texts):
            logger.warning(
                f"Inconsistência de tamanho detectada ({len(embeddings)} vs {len(input_texts)}). "
                "Utilizando fallback local integral (768-dim)."
            )
            return self._fallback_embed_texts(input_texts)

        return embeddings

    def _fallback_embed_texts(self, input_texts: List[str]) -> List[List[float]]:
        """Gera embeddings locais determinísticos de 768 dimensões como fallback seguro."""
        fallback_fn = MockEmbeddingFunction(dimension=768)
        return fallback_fn(input_texts)


class VectorIndexer:
    """
    Classe para gerenciamento da coleção ChromaDB, indexação e busca vetorial densa.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: str = "mercado_central_chunks",
        use_mock: bool = False,
        embedding_model_name: str = "gemini-embedding-001",
    ) -> None:
        self.collection_name: str = collection_name
        self.db_path: Optional[str] = db_path
        self.use_mock: bool = use_mock
        self.mock_embedder: MockEmbeddingFunction = MockEmbeddingFunction(dimension=768)

        # Inicializa cliente ChromaDB
        if db_path and db_path != ":memory:":
            path_obj = Path(db_path)
            path_obj.mkdir(parents=True, exist_ok=True)
            self.client: Any = chromadb.PersistentClient(path=str(path_obj))
            logger.info(f"ChromaDB PersistentClient inicializado em: {path_obj}")
        else:
            self.client = chromadb.EphemeralClient()
            logger.info("ChromaDB EphemeralClient (em memória) inicializado.")

        # Tenta inicializar o provider de Embeddings do Google
        self.google_embedder: Optional[GoogleGenAIEmbeddingFunction] = None
        if not use_mock:
            try:
                self.google_embedder = GoogleGenAIEmbeddingFunction(model_name=embedding_model_name)
                if not self.google_embedder.client:
                    logger.info("Google API indisponível/sem chave. Alternando para MockEmbeddingFunction (768-dim).")
                    self.use_mock = True
            except Exception as e:
                logger.warning(f"Falha ao carregar GoogleGenAIEmbeddingFunction: {e}. Usando mock local.")
                self.use_mock = True
        else:
            logger.info("Modo MOCK explicitamente selecionado para embeddings (768-dim).")

        # Cria ou obtém coleção ChromaDB com distância de cosseno
        self.collection: Any = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings para a lista de textos usando Google API ou Mock Local com garantia estrita de tamanho."""
        if not texts:
            return []
        if not self.use_mock and self.google_embedder and self.google_embedder.client:
            try:
                result = self.google_embedder.embed_texts(texts)
                if result and len(result) == len(texts):
                    return result
                logger.warning(f"Tamanho de embeddings ({len(result)}) incompatível com ({len(texts)}). Usando Mock.")
                return self.mock_embedder(texts)
            except Exception as e:
                logger.warning(f"Erro na API de Embeddings do Google: {e}. Alternando para Mock Local (768-dim).")
                return self.mock_embedder(texts)
        else:
            return self.mock_embedder(texts)

    def index_chunks(
        self,
        chunks_input: Optional[Union[str, Path, List[Dict[str, Any]]]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Indexa uma lista de chunks (ou arquivo JSON com os chunks) no ChromaDB.
        Retorna o número total de chunks indexados. Retorna 0 para entradas vazias ou inválidas.
        """
        if batch_size == 0:
            raise ValueError("batch_size não pode ser zero.")
        if batch_size < 0:
            return 0

        # Guard 1: Entrada nula, vazia ou avaliada como Falsa (None, [], "", etc.)
        if not chunks_input:
            logger.info("Nenhum chunk fornecido para indexação.")
            return 0

        # Guard 2: Trata entrada do tipo string ou Path (caminho JSON) ou lista
        chunks: List[Dict[str, Any]] = []
        if isinstance(chunks_input, (str, Path)):
            clean_str = str(chunks_input).strip()
            if not clean_str:
                logger.info("Nenhum chunk fornecido para indexação.")
                return 0
            json_path = Path(clean_str)
            if not json_path.exists():
                raise FileNotFoundError(f"Arquivo de chunks não encontrado: {json_path}")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Erro ao ler arquivo de chunks JSON {json_path}: {e}")
                raise
        elif isinstance(chunks_input, list):
            chunks = chunks_input
        else:
            logger.info("Nenhum chunk fornecido para indexação.")
            return 0

        # Guard 3: Lista vazia pós-carregamento JSON ou conteúdo não-lista
        if not chunks or not isinstance(chunks, list):
            logger.info("Nenhum chunk fornecido para indexação.")
            return 0

        # Guard 4: Validação de estrutura dos elementos (remove dicts vazios, None ou malformados)
        valid_chunks = [
            c for c in chunks
            if isinstance(c, dict) and c.get("chunk_id") is not None and c.get("text") is not None
        ]

        if not valid_chunks:
            logger.info("Nenhum chunk fornecido para indexação.")
            return 0

        chunks = valid_chunks

        logger.info(f"Iniciando indexação de {len(chunks)} chunks na coleção '{self.collection_name}'...")

        # Prepara listas para inserção em batch
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]

            ids = [c["chunk_id"] for c in batch_chunks]
            documents = [c["text"] for c in batch_chunks]

            # Sanitiza metadados para garantir compatibilidade com ChromaDB
            metadatas = []
            for c in batch_chunks:
                meta = {
                    "file_name": str(c.get("file_name", "")),
                    "file_path": str(c.get("file_path", "")),
                    "category": str(c.get("category", "")),
                    "department_author": str(c.get("department_author", "")),
                    "last_updated": str(c.get("last_updated", "")),
                    "section_title": str(c.get("section_title", "")),
                    "page_start": int(c.get("page_start", 1)),
                    "page_end": int(c.get("page_end", 1)),
                    "char_count": int(c.get("char_count", len(c.get("text", "")))),
                    "word_count": int(c.get("word_count", len(c.get("text", "").split()))),
                }
                metadatas.append(meta)

            embeddings = self.get_embeddings(documents)

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"  ├─ Indexados {min(i + batch_size, len(chunks))}/{len(chunks)} chunks.")

        count: int = self.collection.count()
        logger.info(f"✅ Indexação concluída! Total de documentos na coleção: {count}")
        return count

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executa a busca vetorial densa por similaridade de cosseno.
        Retorna lista de resultados ordenados por similaridade decrescente.
        """
        if top_k <= 0:
            raise ValueError("top_k deve ser um número inteiro maior que zero.")

        if self.collection.count() == 0:
            logger.warning("Coleção está vazia. Retornando busca vazia.")
            return []

        query_embedding = self.get_embeddings([query])[0]

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
        }
        if metadata_filter:
            query_kwargs["where"] = metadata_filter

        results = self.collection.query(**query_kwargs)

        search_results: List[Dict[str, Any]] = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
            documents = results["documents"][0] if "documents" in results and results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(ids)

            for cid, dist, doc, meta in zip(ids, distances, documents, metadatas):
                # Cosine similarity in ChromaDB: cosine_sim = 1.0 - distance
                similarity = max(0.0, min(1.0, float(1.0 - dist)))
                item = {
                    "chunk_id": cid,
                    "text": doc,
                    "dense_score": similarity,
                    "similarity": similarity,
                    **meta,
                }
                search_results.append(item)

        # Sort by similarity descending
        search_results.sort(key=lambda x: x["dense_score"], reverse=True)
        return search_results

    def clear_collection(self) -> None:
        """Limpa a coleção no ChromaDB."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Coleção '{self.collection_name}' limpa com sucesso.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Teste rápido do VectorIndexer
    script_dir = Path(__file__).parent.resolve()
    json_data_path = script_dir.parent / "data" / "processed_rag_chunks.json"

    print("--- Testando VectorIndexer ---")
    indexer = VectorIndexer(use_mock=True)
    count = indexer.index_chunks(str(json_data_path))
    print(f"Total indexado: {count}")

    results = indexer.search("Qual o prazo de entrega expressa?", top_k=3)
    print(f"\nResultados da Busca Dense para 'Qual o prazo de entrega expressa?':")
    for r in results:
        print(f" - [{r['chunk_id']}] Score: {r['dense_score']:.4f} | {r['file_name']} (Pág {r['page_start']}): {r['section_title']}")
