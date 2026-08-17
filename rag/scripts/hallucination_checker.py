#!/usr/bin/env python3
"""
Hallucination Checker Módulo - Mercado Central 24h
Verificador de consistência pós-geração sentença por sentença contra os trechos recuperados.
Valida overlap léxico-semântico e grounding de entidades críticas (números, moedas, prazos, escalas, leis).
"""

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from rag.scripts.hybrid_search import PORTUGUESE_STOPWORDS, normalize_text, tokenize_portuguese
except ImportError:
    from hybrid_search import PORTUGUESE_STOPWORDS, normalize_text, tokenize_portuguese

logger = logging.getLogger("HallucinationChecker")

__all__ = [
    "HallucinationChecker",
]

# Abreviações comuns em PT-BR e terminologias corporativas/legais
HONORIFIC_ABBREVIATIONS = [
    r"Dr\.",
    r"Dra\.",
    r"Sr\.",
    r"Sra\.",
    r"Prof\.",
    r"Profa\.",
]

# Padrões de enquadramento / saudações corporativas / cabeçalhos estruturais
FRAMING_PATTERNS = [
    # Saudações corporativas formais e informais
    r"\bprezado(?:\s*\(?a\)?|\s*a|\s*as|\s*os)?\s*(?:colaborador(?:a)?|equipe|time)?\b",
    r"\bprezad[oa]s?\b",
    r"\bol[aá](?:,\s*|\s+)?(?:colaborador(?:a)?|equipe|time|a todos)?\b",
    r"\bbom\s+dia\b",
    r"\bboa\s+tarde\b",
    r"\bboa\s+noite\b",
    # Assinaturas e sign-offs institucionais
    r"\batenciosamente\b",
    r"\bcordialmente\b",
    r"\bsauda[cç][oõ]es\b",
    r"\bequipe\s+mercado\s+central(?:\s*24\s*h)?\b",
    r"\bassistente\s+corporativo(?:\s+mercado\s+central(?:\s*24\s*h)?)?\b",
    r"\bdiretoria\s+operacional\b",
    r"\brecursos\s+humanos\b",
    r"\bdepartamento\s+de\s+(?:recursos\s+humanos|rh)\b",
    r"\bsetor\s+de\s+rh\b",
    r"\bouvidoria\s+geral(?:\s+0800-central)?\b",
    # Cabeçalhos e seções estruturais multicanal
    r"^(?:#{1,6}\s*)?resumo\s+executivo(?:\s*:)?$",
    r"\bresumo\s+executivo\b",
    r"^(?:#{1,6}\s*)?resumo\s+direto(?:\s*:)?$",
    r"\bresumo\s+direto\b",
    r"^(?:#{1,6}\s*)?tl;?dr(?:\s*:)?$",
    r"\btl;?dr\b",
    r"^(?:#{1,6}\s*)?detalhamento(?:\s+e\s+contextualiza[cç][aã]o)?(?:\s*:)?$",
    r"\bdetalhamento(?:\s+e\s+contextualiza[cç][aã]o)?\b",
    r"^(?:#{1,6}\s*)?cita[cç][oõ]es(?:\s+de\s+fontes)?(?:\s*:)?$",
    r"\bcita[cç][oõ]es(?:\s+de\s+fontes)?\b",
    r"^(?:#{1,6}\s*)?base\s+normativa(?:\s+e\s+fontes)?(?:\s*:)?$",
    r"\bbase\s+normativa\b",
    r"^(?:#{1,6}\s*)?fontes(?:\s+consultadas|\s+oficiais)?(?:\s*:)?$",
    r"\bfontes\s+consultadas\b",
    r"^(?:#{1,6}\s*)?diretrizes\s+oficiais(?:\s*:)?$",
    r"\bdiretrizes\s+oficiais\b",
    r"^(?:#{1,6}\s*)?orienta[cç][oõ]es(?:\s+gerais)?(?:\s*:)?$",
    # Fórmulas de introdução, transição e fallback
    r"com\s+base\s+n[ao]s?\s+(?:documenta[cç][aã]o|documentos)\s+oficia(?:l|is)",
    r"segue\s+(?:o\s+detalhamento|as?\s+informa[cç][oõ]es|as?\s+orienta[cç][oõ]es|a\s+resposta)",
    r"seguem\s+(?:o\s+detalhamento|as?\s+informa[cç][oõ]es|as?\s+orienta[cç][oõ]es|a\s+resposta)",
    r"para\s+(?:mais|maiores)\s+informa[cç][oõ]es",
    r"n[aã]o\s+encontrei\s+informa[cç][oõ]es\s+oficiais",
    r"n[aã]o\s+localizei\s+informa[cç][oõ]es",
    r"desculpe(?:-me)?,?\s*mas",
    r"lamento(?:-me)?,?\s*mas",
]


class HallucinationChecker:
    """
    Verificador Sentença por Sentença de Alucinações e Grounding Factual.
    Avalia a consistência semântica e ancoragem de entidades críticas em respostas do agente QA.
    """

    def __init__(
        self,
        semantic_overlap_threshold: float = 0.35,
        min_sentence_grounding_ratio: float = 0.70,
    ) -> None:
        self.semantic_overlap_threshold: float = max(0.0, min(1.0, float(semantic_overlap_threshold)))
        self.min_sentence_grounding_ratio: float = max(0.0, min(1.0, float(min_sentence_grounding_ratio)))

    def split_sentences(self, text: str) -> List[str]:
        """
        Divide o texto em sentenças para o idioma Português (PT-BR),
        protegendo abreviações, decimais, artigos de leis, valores monetários e listas,
        enquanto permite quebras em abreviações no final de sentenças (ex: 'Ltda. A entrega...').
        """
        if not text or not isinstance(text, str) or not text.strip():
            return []

        cleaned = text.strip()

        # 1. Proteger decimais numéricos (ex: 2.0, 2.5%, 1.500,50, 250.00)
        cleaned = re.sub(r"(\d+)\.(\d+)", r"\1<<DOT>>\2", cleaned)

        # 2. Proteger siglas e abreviações com pontos internos (ex: S.O.P., C.D.C., L.G.P.D.)
        # Mas preservar o último ponto se estiver no final de sentença seguido de espaço e maiúscula
        def _mask_acronym(m: re.Match) -> str:
            matched = m.group(0)
            parts = matched.split(".")
            if len(parts) > 2:
                return "<<DOT>>".join(parts[:-1]) + "."
            return matched.replace(".", "<<DOT>>")

        cleaned = re.sub(r"\b(?:[A-Za-z]\.){2,}", _mask_acronym, cleaned)

        # 3. Proteger abreviações de títulos e honoríficos (ex: Dr. Silva, Sra. Santos)
        honorifics_pattern = r"\b(?:Dr|dr|Dra|dra|Sr|sr|Sra|sra|Prof|prof|Profa|profa)\.\s*(?=[A-Za-zÀ-ÿ])"
        cleaned = re.sub(honorifics_pattern, lambda m: m.group(0).replace(".", "<<DOT>>"), cleaned)

        # 4. Proteger abreviações de referências, artigos, páginas, endereços e números
        # Ex: Art. 49, pág. 4, fls. 15, n.º 12, Av. Paulista, etc.
        ref_pattern = r"\b(?:Art|art|pág|págs|Pág|Págs|fl|fls|cap|Cap|par|Par|inc|Inc|sec|Sec|ref|Ref|no|No|nº|Nº|min|hs|seg|v\.g|i\.e|e\.g|vs|v|av|Av|ed|Ed)\.\s*(?=[0-9A-Za-zÀ-ÿº°/])"
        cleaned = re.sub(ref_pattern, lambda m: m.group(0).replace(".", "<<DOT>>"), cleaned)
        cleaned = re.sub(r"\b(n|N)\.º\s*", r"\1<<DOT>>º ", cleaned)

        # 5. Proteger 'ex.' dentro de parênteses ou seguido de dois pontos / termos
        cleaned = re.sub(r"\b(?:ex|Ex)\.\s*(?=[A-Za-z0-9À-ÿ:,\)])", lambda m: m.group(0).replace(".", "<<DOT>>"), cleaned)

        # 6. Abreviações terminais / corporativas (ex: Ltda., Cia., Inc., S.A., etc.)
        # Se seguidas de minúscula ou pontuação, protege o ponto.
        # NÃO usar re.IGNORECASE para que [a-zà-ÿ] não capture letras maiúsculas no lookahead!
        terminal_intra_clause = r"\b(?:Ltda|ltda|Cia|cia|Inc|inc|S\.A|s\.a|SA|sa|S/A|s/a|etc)\.\s*(?=[a-zà-ÿ,\;\-\)])"
        cleaned = re.sub(terminal_intra_clause, lambda m: m.group(0).replace(".", "<<DOT>>"), cleaned)

        # 7. Proteger moeda R$ e US$ se seguida de pontuação
        cleaned = re.sub(r"\bR\$\s*\.", "R$<<DOT>>", cleaned)

        # 8. Dividir por quebras de linha ou terminadores de sentença (. ! ?)
        # seguido de espaço e letra maiúscula, dígito, marcador de lista ou aspas
        raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“•\-\*])|\n+", cleaned)

        sentences: List[str] = []
        for part in raw_parts:
            restored = part.replace("<<DOT>>", ".").strip()
            # Limpa marcadores de lista no início se houver
            restored = re.sub(r"^[•\-\*]\s*", "", restored).strip()
            if restored:
                sentences.append(restored)

        return sentences

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrai entidades críticas de uma sentença:
        - Moedas (R$ X)
        - Porcentagens (X%)
        - Escalas de trabalho (5x2, 6x1)
        - Prazos / Durações / SLA (3 horas, 7 dias, 24h, 10 min)
        - Leis e Artigos (Art. 49, Lei 13.709/2018)
        - Números gerais
        """
        if not text or not isinstance(text, str):
            return {
                "currencies": [],
                "percentages": [],
                "shifts": [],
                "durations": [],
                "articles_and_laws": [],
                "numbers": [],
            }

        # Ignorar '24h' quando for parte do nome institucional 'Mercado Central 24h'
        text_without_brand = re.sub(r"mercado\s+central\s+24\s*h\b", "Mercado Central", text, flags=re.IGNORECASE)

        # Moedas com formato brasileiro (ex: R$ 250,00, R$ 1.500,50, R$ 250, R$ 0,99)
        currencies = re.findall(r"R\$\s*\d{1,3}(?:\.\d{3})*(?:,\d+)?|R\$\s*\d+(?:[.,]\d+)?", text, re.IGNORECASE)
        percentages = re.findall(r"\b\d+(?:[.,]\d+)?\s*%", text)
        shifts = re.findall(r"\b\d+\s*[xX]\s*\d+\b", text)
        durations = re.findall(
            r"\b\d+\s*(?:horas|hora|dias\s+úteis|dias\s+uteis|dias|dia|meses|mês|semanas|semana|minutos|minuto|anos|ano|h|min|hrs)\b",
            text_without_brand,
            re.IGNORECASE,
        )
        articles_and_laws = re.findall(
            r"\b(?:Art(?:igo|\.)?\s*\d+(?:º|°)?|Lei\s*(?:n[ºo\.]*)?\s*\d+[\d\.]*(?:/\d+)?)\b",
            text,
            re.IGNORECASE,
        )
        numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", text_without_brand)

        return {
            "currencies": [c.strip() for c in currencies],
            "percentages": [p.strip() for p in percentages],
            "shifts": [re.sub(r"\s+", "", s.lower()) for s in shifts],
            "durations": [d.strip() for d in durations],
            "articles_and_laws": [a.strip() for a in articles_and_laws],
            "numbers": [n.strip() for n in numbers],
        }

    def _is_framing_or_citation(self, sentence: str) -> bool:
        """
        Verifica se a sentença é apenas fórmula de enquadramento, saudação,
        cabeçalho estrutural ou tag de citação.
        Verifica tanto sobre o texto bruto/lowercased quanto sobre o normalizado.
        """
        if not sentence or not isinstance(sentence, str):
            return True

        raw = sentence.strip()
        if not raw:
            return True

        raw_lower = raw.lower()
        norm = normalize_text(raw)
        norm_compact = " ".join(norm.split())

        # 1. Tags de citação (ex: [Fonte: Arquivo.pdf, Seção: ..., Págs. X])
        if re.search(r"\[\s*fonte\s*:", raw_lower) or re.search(r"\[\s*documento\b", raw_lower):
            return True
        if re.search(r"\[fonte:\s*[^\]]+\]", raw_lower) or re.search(r"\[fonte:\s*[^\]]+\]", norm):
            return True
        if (
            norm_compact.startswith("fonte:")
            or norm_compact.startswith("fonte ")
            or norm_compact.startswith("[fonte")
            or norm_compact.startswith("fontes")
        ):
            return True
        if (
            norm_compact.startswith("documento oficial")
            or norm_compact.startswith("documento:")
            or norm_compact.startswith("documento ")
        ):
            return True
        if norm_compact.startswith("secao:") or norm_compact.startswith("secao "):
            return True
        if (
            norm_compact.startswith("paginas:")
            or norm_compact.startswith("pagina:")
            or norm_compact.startswith("pag:")
            or norm_compact.startswith("pags:")
            or norm_compact.startswith("paginas")
            or norm_compact.startswith("pagina")
        ):
            return True

        # 2. Padrões de enquadramento, saudações e seções estruturais
        for pattern in FRAMING_PATTERNS:
            if re.search(pattern, raw_lower, re.IGNORECASE) or re.search(pattern, norm_compact, re.IGNORECASE):
                return True

        return False

    def _normalize_num_str(self, num_str: str) -> str:
        """Normaliza representações de números (ex: '2,0' -> '2', '250,00' -> '250', '1.500,50' -> '1500.5')."""
        if not num_str or not isinstance(num_str, str):
            return ""
        cleaned = num_str.replace(" ", "").replace("R$", "").replace("r$", "").replace("%", "").strip()
        if "." in cleaned and "," in cleaned:
            cleaned = cleaned.replace(".", "").replace(",", ".")
        elif "," in cleaned:
            cleaned = cleaned.replace(",", ".")
        try:
            val = float(cleaned)
            if val.is_integer():
                return str(int(val))
            return str(val)
        except ValueError:
            return cleaned.lower()

    def _is_negated_entity_mention(self, sentence: str, entity_str: str) -> bool:
        """Verifica se a entidade é mencionada dentro de um contexto explícito de negação/rejeição."""
        norm_s = normalize_text(sentence)
        norm_e = normalize_text(entity_str)
        neg_prefix = rf"\b(?:nao|não|nem|nunca|jamais|rejeita|descarta|sem|invalido|invalida|vedado|proibido|inexistente|dispensa|descarta)\b(?:\s+\w+){{0,5}}\s+{re.escape(norm_e)}\b"
        if re.search(neg_prefix, norm_s):
            return True
        neg_suffix = rf"\b{re.escape(norm_e)}\b(?:\s+\w+){{0,4}}\s+(?:nao|não|inexistente|invalido|proibido|vedado)\b"
        if re.search(neg_suffix, norm_s):
            return True
        return False

    def verify_sentence(
        self,
        sentence: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Avalia se uma única sentença possui suporte factual e semântico no contexto.
        Retorna dicionário detalhado com status de grounding, overlap e entidades verificadas.
        """
        if not sentence or not isinstance(sentence, str) or not sentence.strip():
            return {
                "sentence": sentence,
                "is_grounded": True,
                "is_framing": True,
                "overlap_score": 1.0,
                "entities_found": {},
                "ungrounded_entities": [],
                "reason": "empty_sentence",
            }

        # 1. Sentenças de enquadramento ou citação
        if self._is_framing_or_citation(sentence):
            return {
                "sentence": sentence,
                "is_grounded": True,
                "is_framing": True,
                "overlap_score": 1.0,
                "entities_found": {},
                "ungrounded_entities": [],
                "reason": "framing_or_citation",
            }

        # 2. Montar texto completo e tokens do contexto
        context_texts: List[str] = []
        for c in (context_chunks or []):
            if isinstance(c, dict):
                context_texts.append(str(c.get("file_name", "")))
                context_texts.append(str(c.get("section_title", "")))
                context_texts.append(str(c.get("text", "")))

        full_context_str = " ".join(context_texts)
        norm_context_str = normalize_text(full_context_str)
        context_tokens = set(
            t for t in norm_context_str.split()
            if t not in PORTUGUESE_STOPWORDS and len(t) > 1 and not t.isdigit()
        )

        # 3. Extração e validação de entidades críticas
        sent_entities = self.extract_entities(sentence)
        context_entities = self.extract_entities(full_context_str)

        ungrounded_entities: List[str] = []

        # Validação de escalas de trabalho (ex: 5x2, 6x1)
        for shift in sent_entities["shifts"]:
            if shift not in context_entities["shifts"]:
                if not self._is_negated_entity_mention(sentence, shift):
                    ungrounded_entities.append(f"escala {shift}")

        # Moedas contextuais normalizadas
        context_currency_nums = set()
        for curr in context_entities["currencies"]:
            num_part = re.search(r"\d+(?:[.,]\d+)*", curr)
            if num_part:
                context_currency_nums.add(self._normalize_num_str(num_part.group(0)))

        # Validação de moedas (ex: R$ 250,00, R$ 500,00)
        for curr in sent_entities["currencies"]:
            num_part = re.search(r"\d+(?:[.,]\d+)*", curr)
            if num_part:
                norm_curr_num = self._normalize_num_str(num_part.group(0))
                if norm_curr_num not in context_currency_nums and curr.lower() not in norm_context_str:
                    if not self._is_negated_entity_mention(sentence, curr):
                        ungrounded_entities.append(curr)

        # Porcentagens contextuais normalizadas
        context_pct_nums = set()
        for pct in context_entities["percentages"]:
            num_part = re.search(r"\d+(?:[.,]\d+)*", pct)
            if num_part:
                context_pct_nums.add(self._normalize_num_str(num_part.group(0)))

        # Validação de porcentagens (ex: 2,0%, 50%)
        for pct in sent_entities["percentages"]:
            num_part = re.search(r"\d+(?:[.,]\d+)*", pct)
            if num_part:
                norm_pct_num = self._normalize_num_str(num_part.group(0))
                if norm_pct_num not in context_pct_nums and pct.lower() not in norm_context_str:
                    if not self._is_negated_entity_mention(sentence, pct):
                        ungrounded_entities.append(pct)

        # Durações contextuais normalizadas
        context_dur_nums = set()
        for dur in context_entities["durations"]:
            for n in re.findall(r"\d+", dur):
                context_dur_nums.add(self._normalize_num_str(n))

        # Validação de durações / SLA (ex: 10 minutos, 3 horas)
        for dur in sent_entities["durations"]:
            norm_dur = normalize_text(dur)
            dur_nums = re.findall(r"\d+", dur)
            num_present = all(self._normalize_num_str(n) in context_dur_nums for n in dur_nums)
            if not num_present and norm_dur not in norm_context_str:
                if not self._is_negated_entity_mention(sentence, dur):
                    ungrounded_entities.append(dur)

        # Artigos e Leis contextuais normalizados
        context_law_nums = set()
        for law in context_entities["articles_and_laws"]:
            for n in re.findall(r"\d+", law):
                context_law_nums.add(self._normalize_num_str(n))

        # Validação de artigos e leis (ex: Art. 49, Lei 13.709)
        for art in sent_entities["articles_and_laws"]:
            norm_art = normalize_text(art)
            art_nums = re.findall(r"\d+", art)
            num_present = all(self._normalize_num_str(n) in context_law_nums for n in art_nums)
            if not num_present and norm_art not in norm_context_str:
                if not self._is_negated_entity_mention(sentence, art):
                    ungrounded_entities.append(art)

        # 4. Overlap léxico de termos substantivos (exclui números puros e stopwords)
        sent_norm = normalize_text(sentence)
        sent_tokens = [
            t for t in sent_norm.split()
            if t not in PORTUGUESE_STOPWORDS and len(t) > 2 and not t.isdigit()
        ]

        if not sent_tokens:
            overlap_score = 1.0 if not ungrounded_entities else 0.0
        else:
            matched = sum(1 for t in sent_tokens if t in context_tokens)
            overlap_score = matched / len(sent_tokens)

        # 5. Decisão de Grounding para a Sentença
        is_grounded = True
        reason = "grounded"

        if ungrounded_entities:
            is_grounded = False
            reason = f"entidades_nao_suportadas: {', '.join(ungrounded_entities)}"
        elif len(sent_tokens) >= 3 and overlap_score < self.semantic_overlap_threshold:
            is_grounded = False
            reason = f"baixo_overlap_lexico ({overlap_score:.2f} < {self.semantic_overlap_threshold:.2f})"

        return {
            "sentence": sentence,
            "is_grounded": is_grounded,
            "is_framing": False,
            "overlap_score": round(overlap_score, 4),
            "entities_found": sent_entities,
            "ungrounded_entities": ungrounded_entities,
            "reason": reason,
        }

    def verify(
        self,
        sentence: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Alias para verify_sentence para compatibilidade de contrato."""
        return self.verify_sentence(sentence, context_chunks)

    def check_response(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Avalia uma resposta completa sentença por sentença contra os trechos recuperados.
        Retorna:
            (is_grounded: bool, sentence_evaluations: List[Dict[str, Any]])
        """
        if not answer or not isinstance(answer, str) or not answer.strip():
            return False, []

        if not context_chunks:
            sentences = self.split_sentences(answer)
            evals = [
                {
                    "sentence": s,
                    "is_grounded": False,
                    "is_framing": self._is_framing_or_citation(s),
                    "overlap_score": 0.0,
                    "entities_found": self.extract_entities(s),
                    "ungrounded_entities": [],
                    "reason": "context_chunks_empty",
                }
                for s in sentences
            ]
            return False, evals

        sentences = self.split_sentences(answer)
        if not sentences:
            return False, []

        sentence_evals: List[Dict[str, Any]] = []
        for s in sentences:
            eval_res = self.verify_sentence(s, context_chunks)
            sentence_evals.append(eval_res)

        substantive_evals = [e for e in sentence_evals if not e.get("is_framing", False)]

        if not substantive_evals:
            # Apenas frases de enquadramento/citação
            return True, sentence_evals

        # Se houver qualquer entidade crítica não suportada em qualquer sentença, rejeita imediatamente
        has_ungrounded_entity = any(len(e.get("ungrounded_entities", [])) > 0 for e in substantive_evals)
        if has_ungrounded_entity:
            return False, sentence_evals

        # Proporção de sentenças substantivas válidas
        grounded_count = sum(1 for e in substantive_evals if e.get("is_grounded", False))
        grounding_ratio = grounded_count / len(substantive_evals)

        is_response_grounded = grounding_ratio >= self.min_sentence_grounding_ratio

        return is_response_grounded, sentence_evals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    checker = HallucinationChecker()

    sample_context = [
        {
            "file_name": "Regulamento_Interno_e_SOP.pdf",
            "section_title": "Jornada e Escala de Trabalho",
            "text": "A jornada de trabalho dos operadores é cumprida na escala 5x2, totalizando 44 horas semanais.",
        },
        {
            "file_name": "Politica_de_Reembolso_e_Devolucoes.pdf",
            "section_title": "Direito de Arrependimento CDC",
            "text": "Conforme o Art. 49 do CDC, o prazo para devolução por arrependimento é de 7 dias corridos.",
        },
    ]

    # Teste 1: Resposta Grounded
    valid_ans = (
        "Com base na documentação oficial do Mercado Central 24h, segue o detalhamento para a sua consulta:\n"
        "• A jornada de trabalho é realizada na escala 5x2 com 44 horas semanais.\n"
        "  [Fonte: Regulamento_Interno_e_SOP.pdf, Seção: Jornada e Escala, Pág. 1]\n"
        "• O prazo de arrependimento é de 7 dias segundo o Art. 49 do CDC."
    )
    ok, evals = checker.check_response(valid_ans, sample_context)
    print(f"Teste 1 (Válido) -> Grounded: {ok}")
    for e in evals:
        print(f"  [{e['is_grounded']}] {e['sentence']} -> {e['reason']}")

    # Teste 2: Resposta com Alucinação de Escala (6x1 em vez de 5x2) e Prazo falso (10 dias)
    hallucinated_ans = (
        "A jornada dos operadores é executada na escala 6x1 e o prazo de devolução é de 10 dias úteis."
    )
    ok2, evals2 = checker.check_response(hallucinated_ans, sample_context)
    print(f"\nTeste 2 (Alucinado) -> Grounded: {ok2}")
    for e in evals2:
        print(f"  [{e['is_grounded']}] {e['sentence']} -> {e['reason']}")

