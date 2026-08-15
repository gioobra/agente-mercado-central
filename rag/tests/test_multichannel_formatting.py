#!/usr/bin/env python3
"""
Test Suite: Milestone 3 (M3 - R3)
Formatador de Respostas Multicanal (Chat, Email, Teams/Slack)
Valida a estrutura tripartite (TL;DR, Detalhamento, Citações de Fontes) e adaptação por canal.
"""

import pytest
from typing import Any, Dict, List

from rag.scripts.multichannel_formatter import (
    extract_tldr_and_details,
    format_citation_line,
    format_multichannel_response,
    sanitize_channel_name,
)


# ============================================================================
# 1. TESTES DE FORMATAÇÃO POR CANAL (TRIPARTITE STRUCTURE)
# ============================================================================

@pytest.fixture
def sample_data():
    tldr = "O frete é grátis para compras acima de R$ 250,00 e para Clientes VIP Diamante acima de R$ 100,00."
    details = (
        "• Entrega Expressa: realizada em até 3 horas na Grande SP.\n"
        "• Entrega Agendada: janelas de 2 horas entre 08h e 22h.\n"
        "• Clique & Retire: retirada em até 1 hora sem custo adicional."
    )
    citations = [
        {
            "file_name": "Guia_de_Envios_e_Entregas.pdf",
            "section_title": "Prazos e Modalidades de Entrega",
            "page_range": "Págs. 1-2",
            "chunk_id": "GUIA_ENV_001",
        },
        {
            "file_name": "Politica_de_Reembolso_e_Devolucoes.pdf",
            "section_title": "Benefícios VIP",
            "page_range": "Pág. 3",
            "chunk_id": "REEMB_003",
        },
    ]
    return {"tldr": tldr, "details": details, "citations": citations}


def test_format_chat_channel(sample_data):
    """T-R3.01: Valida formatação tripartite no canal 'chat'."""
    res = format_multichannel_response(
        tldr=sample_data["tldr"],
        details=sample_data["details"],
        citations=sample_data["citations"],
        channel="chat",
    )
    
    assert "**Resumo Direto:**" in res
    assert sample_data["tldr"] in res
    assert "**Detalhamento:**" in res
    assert "• Entrega Expressa: realizada em até 3 horas" in res
    assert "**Fontes Consultadas:**" in res
    assert "• [Fonte: Guia_de_Envios_e_Entregas.pdf, Seção: Prazos e Modalidades de Entrega, Págs. 1-2]" in res
    assert "• [Fonte: Politica_de_Reembolso_e_Devolucoes.pdf, Seção: Benefícios VIP, Pág. 3]" in res


def test_format_email_channel(sample_data):
    """T-R3.02: Valida formatação corporativa formal no canal 'email'."""
    res = format_multichannel_response(
        tldr=sample_data["tldr"],
        details=sample_data["details"],
        citations=sample_data["citations"],
        channel="email",
    )
    
    assert res.startswith("Prezado(a) colaborador(a),")
    assert "**Resumo Executivo:**" in res
    assert sample_data["tldr"] in res
    assert "**Detalhamento:**" in res
    assert sample_data["details"] in res
    assert "**Base Normativa e Fontes:**" in res
    assert "• Documento: Guia_de_Envios_e_Entregas.pdf — Seção: Prazos e Modalidades de Entrega (Págs. 1-2)" in res
    assert "Atenciosamente,\nEquipe de Atendimento - Mercado Central 24h" in res


def test_format_teams_slack_channel(sample_data):
    """T-R3.03: Valida formatação concisa em blocos no canal 'teams_slack'."""
    res = format_multichannel_response(
        tldr=sample_data["tldr"],
        details=sample_data["details"],
        citations=sample_data["citations"],
        channel="teams_slack",
    )
    
    assert "**[RESUMO]**" in res
    assert sample_data["tldr"] in res
    assert "**[DETALHAMENTO]**" in res
    assert sample_data["details"] in res
    assert "**[FONTES]**" in res
    assert "• *Guia_de_Envios_e_Entregas.pdf* — Seção: _Prazos e Modalidades de Entrega_ (`Págs. 1-2`)" in res


def test_format_multichannel_empty_citations(sample_data):
    """T-R3.04: Valida comportamento gracioso quando a lista de citações é vazia."""
    # Chat
    res_chat = format_multichannel_response(
        tldr=sample_data["tldr"],
        details=sample_data["details"],
        citations=[],
        channel="chat",
    )
    assert "**Fontes Consultadas:**" in res_chat
    assert "Diretrizes Corporativas" in res_chat

    # Email
    res_email = format_multichannel_response(
        tldr=sample_data["tldr"],
        details=sample_data["details"],
        citations=[],
        channel="email",
    )
    assert "**Base Normativa e Fontes:**" in res_email
    assert "Nenhuma fonte documental direta citada" in res_email

    # Teams / Slack
    res_teams = format_multichannel_response(
        tldr=sample_data["tldr"],
        details=sample_data["details"],
        citations=[],
        channel="teams_slack",
    )
    assert "**[FONTES]**" in res_teams
    assert "*Documentação Corporativa*" in res_teams


def test_format_multichannel_channel_sanitization(sample_data):
    """T-R3.05: Valida sanitização de canais não reconhecidos ou variações de grafia."""
    # Variações de email
    assert "**Resumo Executivo:**" in format_multichannel_response(sample_data["tldr"], sample_data["details"], sample_data["citations"], channel="EMAIL")
    assert "**Resumo Executivo:**" in format_multichannel_response(sample_data["tldr"], sample_data["details"], sample_data["citations"], channel="e-mail")
    assert "**Resumo Executivo:**" in format_multichannel_response(sample_data["tldr"], sample_data["details"], sample_data["citations"], channel="mail")

    # Variações de teams / slack
    assert "**[RESUMO]**" in format_multichannel_response(sample_data["tldr"], sample_data["details"], sample_data["citations"], channel="teams")
    assert "**[RESUMO]**" in format_multichannel_response(sample_data["tldr"], sample_data["details"], sample_data["citations"], channel="SLACK")
    assert "**[RESUMO]**" in format_multichannel_response(sample_data["tldr"], sample_data["details"], sample_data["citations"], channel="teams/slack")

    # Canal inválido ou nulo aplica default 'chat'
    res_invalid = format_multichannel_response(sample_data["tldr"], sample_data["details"], sample_data["citations"], channel="canal_desconhecido")
    assert "**Resumo Direto:**" in res_invalid

    res_none = format_multichannel_response(sample_data["tldr"], sample_data["details"], sample_data["citations"], channel=None)
    assert "**Resumo Direto:**" in res_none


# ============================================================================
# 2. TESTES DAS FUNÇÕES UTILITÁRIAS DO FORMATADOR
# ============================================================================

def test_sanitize_channel_name_logic():
    """T-R3.06: Valida mapeamento de strings para 'chat', 'email', 'teams_slack'."""
    assert sanitize_channel_name("chat") == "chat"
    assert sanitize_channel_name("CHAT") == "chat"
    assert sanitize_channel_name("email") == "email"
    assert sanitize_channel_name("Mail") == "email"
    assert sanitize_channel_name("e-mail") == "email"
    assert sanitize_channel_name("teams_slack") == "teams_slack"
    assert sanitize_channel_name("teams") == "teams_slack"
    assert sanitize_channel_name("slack") == "teams_slack"
    assert sanitize_channel_name("qualquer_coisa") == "chat"
    assert sanitize_channel_name("") == "chat"
    assert sanitize_channel_name(None) == "chat"


def test_extract_tldr_and_details_with_existing_headers():
    """T-R3.07: Valida extração quando o texto já possui marcadores estruturados."""
    raw = (
        "**Resumo Direto:**\n"
        "A jornada semanal é de 44h na escala 5x2.\n\n"
        "**Detalhamento:**\n"
        "• Turnos de 8h40 com 1h de intervalo para refeição.\n"
        "• Banco de horas compensável em até 6 meses.\n\n"
        "**Fontes Consultadas:**\n"
        "• [Fonte: Regulamento_Interno_e_SOP.pdf, Seção: Jornada, Pág. 4]"
    )
    tldr, details = extract_tldr_and_details(raw)
    assert tldr == "A jornada semanal é de 44h na escala 5x2."
    assert "• Turnos de 8h40" in details
    assert "• [Fonte:" not in details


def test_extract_tldr_and_details_extractive_text():
    """T-R3.08: Valida extração a partir da saída padrão do motor extrativo."""
    raw = (
        "Com base na documentação oficial do Mercado Central 24h, segue o detalhamento para a sua consulta:\n"
        "• O adiantamento salarial é pago no dia 20 de cada mês no valor de 40% do salário nominal.\n"
        "  [Fonte: Manual_de_Perguntas_Frequentes_FAQ.pdf, Seção: Folha de Pagamento, Pág. 5]\n"
    )
    tldr, details = extract_tldr_and_details(raw)
    assert "Com base na documentação oficial" in tldr
    assert "adiantamento salarial" in details


def test_extract_tldr_and_details_empty_and_single_line():
    """T-R3.09: Valida extração com textos curtos, vazios ou de linha única."""
    assert extract_tldr_and_details("") == ("", "")
    assert extract_tldr_and_details("   ") == ("", "")
    assert extract_tldr_and_details(None) == ("", "")

    single = "A tolerância máxima para atrasos no ponto biométrico é de 10 minutos diários."
    tldr, details = extract_tldr_and_details(single)
    assert tldr == single
    assert details == single


def test_format_citation_line_page_range_deduction():
    """T-R3.10: Valida dedução de faixa de páginas quando page_range não é fornecido."""
    c_single = {"file_name": "Doc.pdf", "section_title": "Sec", "page_start": 2, "page_end": 2}
    assert "Pág. 2" in format_citation_line(c_single, channel="chat")

    c_multi = {"file_name": "Doc.pdf", "section_title": "Sec", "page_start": 2, "page_end": 5}
    assert "Págs. 2-5" in format_citation_line(c_multi, channel="chat")
