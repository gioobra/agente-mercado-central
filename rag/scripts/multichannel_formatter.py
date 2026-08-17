#!/usr/bin/env python3
"""
Formatador de Respostas Multicanal (R3)
Mercado Central 24h - Módulo RAG Corporativo
Formata respostas do assistente em estrutura tripartite (TL;DR, Detalhamento, Citações)
adaptadas para múltiplos canais ('chat', 'email', 'teams_slack').
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def sanitize_channel_name(channel: Optional[str]) -> str:
    """Sanitiza o nome do canal para um dos formatos suportados ('chat', 'email', 'teams_slack')."""
    if not channel or not isinstance(channel, str):
        return "chat"
    cleaned = channel.strip().lower()
    if cleaned in ("email", "mail", "e-mail"):
        return "email"
    if cleaned in ("teams_slack", "teams", "slack", "teams/slack", "teamsslack"):
        return "teams_slack"
    return "chat"


def format_citation_line(citation: Dict[str, Any], channel: str = "chat") -> str:
    """Formata uma linha individual de citação de fonte conforme o canal."""
    file_name = citation.get("file_name") or "Documento Oficial"
    section = citation.get("section_title") or "Seção Geral"
    page_range = citation.get("page_range")
    if not page_range:
        p_start = citation.get("page_start", 1)
        p_end = citation.get("page_end", p_start)
        page_range = f"Pág. {p_start}" if p_start == p_end else f"Págs. {p_start}-{p_end}"

    if channel == "email":
        return f"• Documento: {file_name} — Seção: {section} ({page_range})"
    elif channel == "teams_slack":
        return f"• *{file_name}* — Seção: _{section}_ (`{page_range}`)"
    else:  # chat
        return f"• [Fonte: {file_name}, Seção: {section}, {page_range}]"


def extract_tldr_and_details(raw_text: str) -> Tuple[str, str]:
    """
    Separa de forma heurística e robusta o resumo (TL;DR) e o detalhamento de uma resposta gerada.
    """
    if not raw_text or not str(raw_text).strip():
        return "", ""

    text = str(raw_text).strip()
    # Converte tags strong/b para markdown ou remove tags HTML residuais
    text = re.sub(r"<\s*strong\s*>(.*?)<\s*/\s*strong\s*>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*b\s*>(.*?)<\s*/\s*b\s*>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)

    # Remove citações embutidas no final do texto para evitar duplicação
    cleaned_lines: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Se for linha de citação solta tipo "[Fonte: ...]", ignora aqui pois irá para a seção de fontes
        if re.match(r"^\[Fonte:\s*.*?\]$", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    clean_content = "\n".join(cleaned_lines).strip()

    # Se contiver marcadores prévios de Resumo / Detalhamento
    resumo_match = re.search(r"\*\*Resumo\s*(?:Direto|Executivo)?:\*\*\s*(.*?)(?=\*\*Detalhamento|\n\n|$)", clean_content, re.DOTALL | re.IGNORECASE)
    detalhes_match = re.search(r"\*\*Detalhamento(?:\s*Normativo)?:\*\*\s*(.*?)(?=\*\*Fontes|\*\*Base\s*Normativa|$)", clean_content, re.DOTALL | re.IGNORECASE)

    if resumo_match and detalhes_match:
        tldr = resumo_match.group(1).strip()
        details = detalhes_match.group(1).strip()
        return tldr, details

    # Caso padrão: divide por parágrafos ou bullets
    paragraphs = [p.strip() for p in clean_content.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [clean_content]

    if len(paragraphs) == 1:
        first_block = paragraphs[0]
        # Se começar com introdução institucional
        intro_match = re.match(r"^(Com base na documentação oficial[^:\n]*:?)(.*)", first_block, re.DOTALL)
        if intro_match:
            tldr = intro_match.group(1).strip()
            details = intro_match.group(2).strip()
            if not details:
                details = tldr
            return tldr, details
        # Caso contrário, divide por sentenças
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", first_block) if s.strip()]
        if len(sentences) > 1:
            first_sent = sentences[0]
            # Se a primeira sentença for muito curta (ex: "Não.", "Sim.", "Correto."), junta com a próxima
            if len(first_sent.split()) <= 4 and len(sentences) > 1:
                tldr = f"{first_sent} {sentences[1]}".strip()
                details = "\n".join(sentences[2:]).strip() if len(sentences) > 2 else tldr
                return tldr, details
            return first_sent, "\n".join(sentences[1:]).strip()
        return first_block, first_block

    tldr = paragraphs[0]
    # Se o primeiro parágrafo for muito curto (ex: "Não."), junta com o próximo
    if len(tldr.split()) <= 4 and len(paragraphs) > 1:
        tldr = f"{tldr} {paragraphs[1]}".strip()
        details = "\n\n".join(paragraphs[2:]).strip() if len(paragraphs) > 2 else tldr
    else:
        details = "\n\n".join(paragraphs[1:]).strip() if len(paragraphs) > 1 else tldr

    return tldr, details


def format_multichannel_response(
    tldr: str,
    details: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    channel: str = "chat",
) -> str:
    """
    Constrói a resposta estruturada tripartite (Resumo, Detalhamento, Citações de Fontes)
    no formato específico do canal corporativo solicitado ('chat', 'email', 'teams_slack').
    """
    clean_channel = sanitize_channel_name(channel)
    clean_tldr = (tldr or "").strip()
    clean_details = (details or "").strip()
    citations_list = citations or []

    # Se tldr e details forem vazios
    if not clean_tldr and not clean_details:
        return "Nenhuma informação disponível para exibição."

    if not clean_tldr:
        clean_tldr = clean_details
    if not clean_details:
        clean_details = clean_tldr

    # Formata bloco de citações
    citation_lines: List[str] = []
    for c in citations_list:
        formatted_c = format_citation_line(c, channel=clean_channel)
        if formatted_c not in citation_lines:
            citation_lines.append(formatted_c)

    if not citation_lines:
        if clean_channel == "email":
            citations_block = "• Nenhuma fonte documental direta citada."
        elif clean_channel == "teams_slack":
            citations_block = "• *Documentação Corporativa* — _Diretrizes Internas_"
        else:
            citations_block = "• [Fonte: Diretrizes Corporativas do Mercado Central 24h]"
    else:
        citations_block = "\n".join(citation_lines)

    # 1. CANAL EMAIL (Corporativo Formal com Saudação e Assinatura)
    if clean_channel == "email":
        return (
            f"Prezado(a) colaborador(a),\n\n"
            f"**Resumo Executivo:**\n"
            f"{clean_tldr}\n\n"
            f"**Detalhamento:**\n"
            f"{clean_details}\n\n"
            f"**Base Normativa e Fontes:**\n"
            f"{citations_block}\n\n"
            f"Atenciosamente,\n"
            f"Equipe de Atendimento - Mercado Central 24h"
        )

    # 2. CANAL TEAMS / SLACK (Mensageria Corporativa Concisa em Blocos)
    elif clean_channel == "teams_slack":
        return (
            f"**[RESUMO]**\n"
            f"{clean_tldr}\n\n"
            f"**[DETALHAMENTO]**\n"
            f"{clean_details}\n\n"
            f"**[FONTES]**\n"
            f"{citations_block}"
        )

    # 3. CANAL CHAT (Padrão Interativo com Tags e Bullets)
    else:
        return (
            f"**Resumo Direto:**\n"
            f"{clean_tldr}\n\n"
            f"**Detalhamento:**\n"
            f"{clean_details}\n\n"
            f"**Fontes Consultadas:**\n"
            f"{citations_block}"
        )


if __name__ == "__main__":
    sample_tldr = "O frete é grátis para compras acima de R$ 250,00 e para Clientes VIP Diamante em compras a partir de R$ 100,00."
    sample_details = (
        "• Modalidade Expressa: entrega em até 3 horas na Grande SP.\n"
        "• Modalidade Agendada: janelas de 2 horas entre 08h e 22h.\n"
        "• Clique & Retire: retirada na loja em até 1 hora sem taxa."
    )
    sample_citations = [
        {"file_name": "Guia_de_Envios_e_Entregas.pdf", "section_title": "Prazos e Modalidades", "page_range": "Págs. 1-2"},
        {"file_name": "Regulamento_Fidelidade_2026.pdf", "section_title": "Benefícios Diamante", "page_range": "Pág. 3"},
    ]

    print("=== FORMATO CHAT ===")
    print(format_multichannel_response(sample_tldr, sample_details, sample_citations, channel="chat"))
    print("\n=== FORMATO EMAIL ===")
    print(format_multichannel_response(sample_tldr, sample_details, sample_citations, channel="email"))
    print("\n=== FORMATO TEAMS/SLACK ===")
    print(format_multichannel_response(sample_tldr, sample_details, sample_citations, channel="teams_slack"))
