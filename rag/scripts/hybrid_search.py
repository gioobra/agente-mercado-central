#!/usr/bin/env python3
"""
Hybrid Search Módulo - Mercado Central 24h
Busca Híbrida combinando busca vetorial densa (cosseno) com BM25 esparso (palavras-chave).
"""

import datetime
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
    "PORTUGUESE_MONTHS",
    "normalize_text",
    "tokenize_portuguese",
    "parse_date_value",
    "calculate_recency_score",
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

# Mapeamento multilingue de meses (Português, Espanhol, Inglês, Francês, Alemão, Italiano) para índices numéricos (1 a 12)
PORTUGUESE_MONTHS: Dict[str, int] = {
    # Português
    "janeiro": 1, "jan": 1,
    "fevereiro": 2, "fev": 2,
    "marco": 3, "março": 3, "mar": 3,
    "abril": 4, "abr": 4,
    "maio": 5, "mai": 5,
    "junho": 6, "jun": 6,
    "julho": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "setembro": 9, "set": 9, "sete": 9,
    "outubro": 10, "out": 10,
    "novembro": 11, "nov": 11,
    "dezembro": 12, "dez": 12,
    # Espanhol
    "enero": 1, "ene": 1,
    "febrero": 2, "feb": 2,
    "marzo": 3,
    "mayo": 5, "may": 5,
    "junio": 6,
    "julio": 7,
    "septiembre": 9, "setiembre": 9, "sep": 9, "sept": 9,
    "octubre": 10, "oct": 10,
    "noviembre": 11,
    "diciembre": 12, "dic": 12,
    # Inglês
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4, "apr": 4,
    "june": 6,
    "july": 7,
    "august": 8, "aug": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12, "dec": 12,
    # Outros formatos europeus comuns (Francês, Alemão, Italiano)
    "janvier": 1, "januar": 1, "gennaio": 1,
    "fevrier": 2, "febbraio": 2,
    "mars": 3, "maerz": 3, "marz": 3,
    "avril": 4, "aprile": 4,
    "maggio": 5, "giugno": 6, "juin": 6,
    "juillet": 7, "luglio": 7,
    "aout": 8,
    "settembre": 9, "ottobre": 10, "oktober": 10,
    "novembre": 11, "decembre": 12, "dezember": 12, "dicembre": 12,
}


def _normalize_recency_params(
    recency_boost: Any,
    recency_weight: Any = None,
    default_boost: bool = False,
    default_weight: float = 0.15,
) -> Tuple[bool, float]:
    """
    Normaliza parâmetros de recency_boost e recency_weight de forma segura e consistente,
    aceitando bool, float, int, str e tratando adequadamente NaN/Inf.
    """
    import math

    # 1. Determina effective boost & initial weight
    if recency_boost is None:
        eff_boost = default_boost
        raw_weight = default_weight if recency_weight is None else recency_weight
    elif isinstance(recency_boost, bool):
        eff_boost = recency_boost
        raw_weight = default_weight if recency_weight is None else recency_weight
    elif isinstance(recency_boost, (int, float, np.integer, np.floating)):
        val = float(recency_boost)
        if math.isnan(val) or math.isinf(val):
            eff_boost = False
            raw_weight = 0.0
        else:
            eff_boost = val > 0
            raw_weight = val if (recency_weight is None or recency_weight == default_weight) else recency_weight
    elif isinstance(recency_boost, str):
        clean_str = recency_boost.strip().lower()
        if clean_str in ("false", "0", "0.0", "no", "none", "off", "disable", "disabled", "nao", "não", "desativado"):
            eff_boost = False
            raw_weight = default_weight if recency_weight is None else recency_weight
        elif clean_str in ("true", "1", "yes", "sim", "on", "enable", "enabled", "ativo", "ativado"):
            eff_boost = True
            raw_weight = default_weight if recency_weight is None else recency_weight
        else:
            try:
                num_val = float(clean_str)
                if math.isnan(num_val) or math.isinf(num_val):
                    eff_boost = False
                    raw_weight = 0.0
                else:
                    eff_boost = num_val > 0
                    raw_weight = num_val if (recency_weight is None or recency_weight == default_weight) else recency_weight
            except ValueError:
                eff_boost = bool(clean_str)
                raw_weight = default_weight if recency_weight is None else recency_weight
    else:
        eff_boost = bool(recency_boost)
        raw_weight = default_weight if recency_weight is None else recency_weight

    # 2. Normaliza e faz clamp de raw_weight
    try:
        w = float(raw_weight)
        if math.isnan(w) or math.isinf(w):
            eff_weight = 0.0
        else:
            eff_weight = max(0.0, w)
    except (ValueError, TypeError):
        eff_weight = max(0.0, float(default_weight))

    return eff_boost, eff_weight


def parse_date_value(value: Any) -> Optional[datetime.datetime]:
    """
    Converte datas em múltiplos formatos (ISO, PT-BR, ES, EN, numéricos, timestamps, np.datetime64) em datetime naive UTC.
    Retorna None se o valor for inválido, booleano, NaN/Inf, nulo ou número fora do intervalo plausível.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day)
    try:
        if isinstance(value, np.datetime64):
            dt = value.astype("M8[ms]").astype(datetime.datetime)
            if isinstance(dt, datetime.datetime):
                if dt.tzinfo is not None:
                    return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                return dt
    except (ImportError, Exception):
        pass

    if isinstance(value, (int, float, np.integer, np.floating)):
        import math
        if math.isnan(value) or math.isinf(value):
            return None
        num = float(value)
        # Ano de 4 dígitos (ex: 2026 ou 2026.0)
        if 1800 <= num <= 2200 and (isinstance(value, (int, np.integer)) or num.is_integer()):
            return datetime.datetime(int(num), 1, 1)
        # Timestamp Unix em segundos ou milissegundos
        if abs(num) >= 1e8:
            if abs(num) > 1e11:
                num = num / 1000.0
            try:
                return datetime.datetime.fromtimestamp(num, tz=datetime.timezone.utc).replace(tzinfo=None)
            except (ValueError, OSError, OverflowError):
                return None
        return None

    text = str(value).strip()
    if not text:
        return None

    # Tenta padrão ISO
    try:
        clean_iso = text.replace("Z", "+00:00")
        dt_iso = datetime.datetime.fromisoformat(clean_iso)
        if dt_iso.tzinfo is not None:
            return dt_iso.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return dt_iso
    except (ValueError, TypeError):
        pass

    # Tenta timestamp ou ano numérico em string (ex: "1786665600", "2026", "2026.0")
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        try:
            num_val = float(text)
            if 1800 <= num_val <= 2200 and ("." not in text or num_val.is_integer()):
                return datetime.datetime(int(num_val), 1, 1)
            if abs(num_val) >= 1e8:
                if abs(num_val) > 1e11:
                    num_val = num_val / 1000.0
                return datetime.datetime.fromtimestamp(num_val, tz=datetime.timezone.utc).replace(tzinfo=None)
        except (ValueError, OSError, OverflowError):
            pass

    # Normalização de acentos para matching de meses
    norm = unicodedata.normalize("NFD", text.lower())
    norm_no_accents = "".join(c for c in norm if unicodedata.category(c) != "Mn")

    # Match DD/MM/YYYY ou DD-MM-YYYY ou DD.MM.YYYY (com fallback para MM/DD/YYYY)
    m = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", text)
    if m:
        p1, p2, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime.datetime(y, p2, p1)
        except ValueError:
            try:
                return datetime.datetime(y, p1, p2)
            except ValueError:
                pass

    # Match YYYY-MM-DD ou YYYY/MM/DD ou YYYY.MM.DD
    m = re.search(r"\b(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})\b", text)
    if m:
        try:
            return datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Match YYYY/MM ou YYYY-MM ou YYYY.MM
    m = re.search(r"\b(\d{4})[/.-](\d{1,2})\b", text)
    if m:
        try:
            y, m_val = int(m.group(1)), int(m.group(2))
            if 1 <= m_val <= 12 and 1800 <= y <= 2200:
                return datetime.datetime(y, m_val, 1)
        except ValueError:
            pass

    # Match MM/YYYY ou MM-YYYY ou MM.YYYY
    m = re.search(r"\b(\d{1,2})[/.-](\d{4})\b", text)
    if m:
        try:
            m_val, y = int(m.group(1)), int(m.group(2))
            if 1 <= m_val <= 12 and 1800 <= y <= 2200:
                return datetime.datetime(y, m_val, 1)
        except ValueError:
            pass

    # Conectores comuns em PT, ES, EN, FR, DE: "de", "del", "of", "do", "da", "du", "vom", "d'"
    connector_pat = r"(?:[\s/,-]+(?:de\s+|del\s+|of\s+|do\s+|da\s+|du\s+|vom\s+|d\x27\s*)?|[\s/,-]+)"
    ordinal_pat = r"(?:[º°ªo]|st|nd|rd|th|er)?"

    # Match DD de [Mês] de YYYY ou DD-[Mês]-YYYY ou DD/[Mês]/YYYY ou DD [Mês], YYYY ou DD [Mês] del YYYY
    m = re.search(
        r"\b(\d{1,2})" + ordinal_pat + connector_pat + r"([a-z]+)" + connector_pat + r"(\d{4})\b",
        norm_no_accents,
    )
    if m:
        d_val, m_str, y_val = int(m.group(1)), m.group(2), int(m.group(3))
        if m_str in PORTUGUESE_MONTHS:
            try:
                return datetime.datetime(y_val, PORTUGUESE_MONTHS[m_str], d_val)
            except ValueError:
                pass

    # Match [Mês] DD, YYYY ou [Mês] DDth YYYY (ex: "August 14, 2026", "Agosto 14, 2026", "Diciembre 24, 2026")
    m = re.search(
        r"\b([a-z]+)" + connector_pat + r"(\d{1,2})" + ordinal_pat + connector_pat + r"(\d{4})\b",
        norm_no_accents,
    )
    if m:
        m_str, d_val, y_val = m.group(1), int(m.group(2)), int(m.group(3))
        if m_str in PORTUGUESE_MONTHS:
            try:
                return datetime.datetime(y_val, PORTUGUESE_MONTHS[m_str], d_val)
            except ValueError:
                pass

    # Match [Mês] de YYYY ou [Mês]/YYYY ou [Mês]-YYYY ou [Mês] YYYY ou [Mês] del YYYY
    m = re.search(r"\b([a-z]+)" + connector_pat + r"(\d{4})\b", norm_no_accents)
    if m:
        m_str, y_val = m.group(1), int(m.group(2))
        if m_str in PORTUGUESE_MONTHS:
            try:
                return datetime.datetime(y_val, PORTUGUESE_MONTHS[m_str], 1)
            except ValueError:
                pass

    # Match Trimestres / Semestres (ex: Q1 2026, 1T 2026, 1º Trimestre de 2026, 2º Semestre 2026, 2S 2026)
    m = re.search(
        r"\b(?:q([1-4])|([1-4])\s*(?:[º°ªo])?\s*(?:t|tri|trimestre)|([1-2])\s*(?:[º°ªo])?\s*(?:s|sem|semestre))\s*(?:de\s+|del\s+|of\s+|/|-|\s+)?(\d{4})\b",
        norm_no_accents,
    )
    if m:
        q_val = m.group(1) or m.group(2)
        s_val = m.group(3)
        y_val = int(m.group(4))
        if q_val:
            month = (int(q_val) - 1) * 3 + 1
            return datetime.datetime(y_val, month, 1)
        elif s_val:
            month = (int(s_val) - 1) * 6 + 1
            return datetime.datetime(y_val, month, 1)

    # Match apenas o ano YYYY (1800 a 2200)
    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2}|2200)\b", text)
    if m:
        return datetime.datetime(int(m.group(1)), 1, 1)

    return None


def calculate_recency_score(
    chunk_date: Any,
    min_timestamp: Optional[float] = None,
    max_timestamp: Optional[float] = None,
) -> float:
    """
    Calcula a pontuação de recência normalizada no intervalo [0.0, 1.0].
    Documentos mais recentes recebem pontuações mais próximas de 1.0.
    Garante proteção estrita contra NaN, Inf e valores inválidos.
    """
    parsed_dt = parse_date_value(chunk_date)
    if parsed_dt is None:
        return 0.0

    try:
        ts = parsed_dt.timestamp()
    except (ValueError, OSError, OverflowError):
        return 0.0

    if min_timestamp is not None and max_timestamp is not None:
        import math
        if math.isnan(min_timestamp) or math.isnan(max_timestamp) or math.isinf(min_timestamp) or math.isinf(max_timestamp):
            return 1.0
        if max_timestamp > min_timestamp:
            norm = (ts - min_timestamp) / (max_timestamp - min_timestamp)
            if math.isnan(norm) or math.isinf(norm):
                return 1.0
            return max(0.0, min(1.0, float(norm)))
        else:
            return 1.0

    return 1.0


def _is_usable_date_value(val: Any) -> bool:
    """Verifica se o valor é não-nulo, não-booleano e não é string vazia/em branco."""
    if val is None or isinstance(val, bool):
        return False
    if isinstance(val, str) and not val.strip():
        return False
    return True


def _extract_chunk_date(chunk: Dict[str, Any]) -> Any:
    """Extrai valor de data de um chunk considerando campos raiz ou dicionários aninhados de metadados."""
    if not isinstance(chunk, dict):
        return None
    keys = (
        "last_updated",
        "updated_at",
        "date",
        "data",
        "created_at",
        "published_at",
        "publish_date",
        "publication_date",
        "timestamp",
    )
    for k in keys:
        val = chunk.get(k)
        if _is_usable_date_value(val):
            return val
    meta = chunk.get("metadata")
    if isinstance(meta, dict):
        for k in keys:
            val = meta.get(k)
            if _is_usable_date_value(val):
                return val
    doc_meta = chunk.get("doc_meta")
    if isinstance(doc_meta, dict):
        for k in keys:
            val = doc_meta.get(k)
            if _is_usable_date_value(val):
                return val
    return None


# Siglas e Acrônimos de Domínio Corporativo com 2 letras preservadas
DOMAIN_ACRONYMS: Set[str] = {
    "ia", "ai", "rh", "ti", "t1", "t2", "t3", "t4", "t5", "sp", "rj", "nf", "cd", "pj", "pf"
}

# Expansão de Sinônimos / Siglas para Busca Abrangente
SYNONYM_EXPANSION_MAP: Dict[str, List[str]] = {
    "ia": ["ia", "inteligencia", "artificial"],
    "ai": ["ai", "ia", "inteligencia", "artificial"],
    "rh": ["rh", "recursos", "humanos", "gestao", "pessoas"],
    "dpo": ["dpo", "privacidade", "dados", "lgpd", "encarregado"],
    "cdc": ["cdc", "consumidor", "codigo"],
    "sac": ["sac", "atendimento", "suporte", "cliente"],
    "nfe": ["nfe", "nota", "fiscal"],
    "sop": ["sop", "procedimento", "operacional", "normas"],
    "faq": ["faq", "perguntas", "frequentes"],
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
    """Tokeniza texto em português para a busca BM25 preservando siglas corporativas e expandindo sinônimos."""
    clean = normalize_text(text)
    tokens = clean.split()
    filtered: List[str] = []
    for t in tokens:
        if t in PORTUGUESE_STOPWORDS:
            continue
        if len(t) > 1 or t in DOMAIN_ACRONYMS:
            if t in SYNONYM_EXPANSION_MAP:
                filtered.extend(SYNONYM_EXPANSION_MAP[t])
            else:
                filtered.append(t)
    return filtered if filtered else tokens


class HybridSearcher:
    """
    Módulo de Busca Híbrida que combina buscas vetoriais densas (ChromaDB)
    e buscas esparsas por palavras-chave (BM25), com suporte a priorização por recência (boost temporal).
    """

    def __init__(
        self,
        vector_indexer: VectorIndexer,
        chunks_data: Union[str, Path, List[Dict[str, Any]]],
        alpha: float = 0.5,
        rrf_k: int = 60,
        recency_boost: Union[bool, float, str] = False,
        recency_weight: float = 0.15,
    ) -> None:
        """
        :param vector_indexer: Instância configurada do VectorIndexer.
        :param chunks_data: Lista de dicionários de chunks ou caminho para o JSON.
        :param alpha: Peso entre Busca Densa (alpha) e BM25 (1 - alpha). Padrão = 0.5.
        :param rrf_k: Constante para fusion rank de compatibilidade (padrão 60).
        :param recency_boost: Se True (ou float > 0), ativa a priorização por recência por padrão.
        :param recency_weight: Peso do boost de recência no score final (padrão 0.15).
        """
        self.vector_indexer: VectorIndexer = vector_indexer
        self.alpha: float = max(0.0, min(1.0, alpha))
        self.rrf_k: int = rrf_k
        self.recency_boost, self.recency_weight = _normalize_recency_params(
            recency_boost=recency_boost,
            recency_weight=recency_weight,
            default_boost=False,
            default_weight=0.15,
        )

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
        recency_boost: Optional[Union[bool, float, str]] = None,
        recency_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executa a busca híbrida combinando Dense Cosine Similarity + BM25 Sparse Score,
        com suporte a priorização por recência de documentos via score boost.
        """
        effective_alpha = self.alpha if alpha is None else max(0.0, min(1.0, alpha))

        # Determina parâmetros efetivos de boost de recência
        effective_recency_boost, effective_recency_weight = _normalize_recency_params(
            recency_boost=recency_boost,
            recency_weight=recency_weight,
            default_boost=self.recency_boost,
            default_weight=self.recency_weight,
        )

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

        # Pré-filtra candidatos por metadados e coleta timestamps para normalização temporal
        filtered_chunks: List[Dict[str, Any]] = []
        valid_timestamps: List[float] = []

        for cid in all_candidate_ids:
            chunk = self.chunks_map.get(cid)
            if not chunk:
                continue

            # Aplica filtro de metadados se fornecido
            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    c_val = chunk.get(k)
                    if c_val is None and isinstance(chunk.get("metadata"), dict):
                        c_val = chunk.get("metadata", {}).get(k)
                    if c_val is None and isinstance(chunk.get("doc_meta"), dict):
                        c_val = chunk.get("doc_meta", {}).get(k)
                    if c_val != v:
                        match = False
                        break
                if not match:
                    continue

            filtered_chunks.append(chunk)
            chunk_date_val = _extract_chunk_date(chunk)
            dt = parse_date_value(chunk_date_val)
            if dt is not None:
                valid_timestamps.append(dt.timestamp())

        min_ts = min(valid_timestamps) if valid_timestamps else None
        max_ts = max(valid_timestamps) if valid_timestamps else None

        hybrid_results: List[Dict[str, Any]] = []
        for chunk in filtered_chunks:
            cid = chunk["chunk_id"]
            dense_score = dense_scores_map.get(cid, 0.0)
            sparse_score = sparse_scores_map.get(cid, 0.0)

            # Cálculo do score híbrido ponderado base
            base_hybrid_score = (effective_alpha * dense_score) + ((1.0 - effective_alpha) * sparse_score)

            # Cálculo da recência e score com boost temporal
            chunk_date_val = _extract_chunk_date(chunk)
            recency_score = calculate_recency_score(chunk_date_val, min_ts, max_ts)
            recency_boost_val = (effective_recency_weight * recency_score) if effective_recency_boost else 0.0
            final_hybrid_score = base_hybrid_score + recency_boost_val

            merged_item = {
                **chunk,
                "dense_score": round(float(dense_score), 4),
                "sparse_score": round(float(sparse_score), 4),
                "raw_hybrid_score": round(float(base_hybrid_score), 4),
                "recency_score": round(float(recency_score), 4),
                "recency_boost": round(float(recency_boost_val), 4),
                "hybrid_score": round(float(final_hybrid_score), 4),
            }
            hybrid_results.append(merged_item)

        # Ordena resultados pelo score híbrido (boosted quando ativo) decrescente
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    script_dir = Path(__file__).parent.resolve()
    json_data_path = script_dir.parent / "data" / "processed_rag_chunks.json"

    print("--- Testando HybridSearcher ---")
    indexer = VectorIndexer(use_mock=True)
    indexer.index_chunks(str(json_data_path))

    searcher = HybridSearcher(vector_indexer=indexer, chunks_data=str(json_data_path), alpha=0.5, recency_boost=True)

    query = "Frete grátis Cliente VIP Diamante"
    results = searcher.search(query=query, top_k=5)

    print(f"\nResultados para a busca híbrida com recency boost: '{query}'")
    for idx, r in enumerate(results, 1):
        print(
            f"{idx}. [{r['chunk_id']}] Hybrid Score: {r['hybrid_score']:.4f} "
            f"(Raw: {r['raw_hybrid_score']:.4f}, Recency Score: {r['recency_score']:.4f}, Boost: {r['recency_boost']:.4f}) | "
            f"{r['file_name']} ({r.get('last_updated', 'N/A')}): {r['section_title']}"
        )

