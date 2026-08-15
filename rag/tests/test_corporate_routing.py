#!/usr/bin/env python3
"""
Test Suite: Milestone 2 (M2 - R2)
Catálogo Corporativo de Contatos & Roteamento de Fallback por Intenção
Valida a integridade dos 7 departamentos, casamento de intenção, desambiguação e formatação de fallback.
"""

import pytest
from typing import Any, Dict

from rag.scripts.contact_catalog import (
    CORPORATE_CONTACT_CATALOG,
    format_fallback_message,
    normalize_catalog_text,
    route_fallback_contact,
)


# ============================================================================
# 1. TESTES DO CATÁLOGO DE CONTATOS CORPORATIVOS (8 DOCUMENTOS OFICIAIS)
# ============================================================================

def test_corporate_contact_catalog_contains_all_seven_departments():
    """T-R2.01: Valida que o catálogo contém exatamente os 7 departamentos oficiais."""
    expected_depts = {
        "rh",
        "juridico_compliance",
        "dpo_lgpd",
        "compras_fornecedores",
        "fiscal_nfe",
        "sac_delivery",
        "ouvidoria_fallback",
    }
    assert set(CORPORATE_CONTACT_CATALOG.keys()) == expected_depts
    assert len(CORPORATE_CONTACT_CATALOG) == 7


def test_corporate_catalog_rh_data_integrity():
    """T-R2.02: Valida os dados oficiais de RH (e-mail, 0800, tópicos de salário/benefícios/escala 5x2)."""
    rh = CORPORATE_CONTACT_CATALOG["rh"]
    assert rh["primary_email"] == "rh@mercadocentral24h.com.br"
    assert "0800-CENTRAL" in rh["primary_phone"]
    assert "0800-236-8725" in rh["primary_phone"]
    assert any("5º dia útil" in t or "5" in t for t in rh["topics"])
    assert any("20" in t for t in rh["topics"])  # adiantamento dia 20
    assert any("5x2" in t for t in rh["topics"])
    assert any("Unimed" in t for t in rh["topics"])
    assert any("Indique um Talento" in t or "150" in t for t in rh["topics"])


def test_corporate_catalog_compliance_data_integrity():
    """T-R2.03: Valida dados de Compliance & Ética (Camila Ferreira, Lei 12.846/2013, brindes R$ 100)."""
    comp = CORPORATE_CONTACT_CATALOG["juridico_compliance"]
    assert comp["responsible"] == "Camila Ferreira"
    assert comp["primary_email"] == "etica@mercadocentral24h.com.br"
    assert "0800-CENTRAL" in comp["primary_phone"]
    assert any("12.846" in t for t in comp["topics"])
    assert any("100" in t for t in comp["topics"])
    assert any("assédio" in t.lower() or "denúncias" in t.lower() for t in comp["topics"])


def test_corporate_catalog_dpo_lgpd_data_integrity():
    """T-R2.04: Valida dados do DPO (e-mail, App Meus Dados, endereço SP, prazos LGPD 15d/5d/5 anos)."""
    dpo = CORPORATE_CONTACT_CATALOG["dpo_lgpd"]
    assert dpo["primary_email"] == "dpo@mercadocentral24h.com.br"
    assert "Meus Dados" in dpo["primary_channel"]
    assert "Gerenciar Privacidade" in dpo["primary_channel"]
    assert "Av. Principal" in dpo["postal_address"]
    assert any("13.709" in t for t in dpo["topics"])
    assert "15 dias" in dpo["hours_sla"]
    assert "5 dias" in dpo["hours_sla"]


def test_corporate_catalog_procurement_suppliers_data_integrity():
    """T-R2.05: Valida dados de Compras (João Silva, Maria Santos, Ricardo Lima, docas)."""
    compras = CORPORATE_CONTACT_CATALOG["compras_fornecedores"]
    assert compras["primary_email"] == "compras.geral@mercadocentral24h.com.br"
    assert "responsible_contacts" in compras
    contacts = compras["responsible_contacts"]
    
    assert contacts["pereciveis"]["name"] == "João Silva"
    assert contacts["pereciveis"]["email"] == "compras.pereciveis@mercadocentral24h.com.br"
    assert "(11) 98888-0001" in contacts["pereciveis"]["phone"]

    assert contacts["secos"]["name"] == "Maria Santos"
    assert contacts["secos"]["email"] == "compras.secos@mercadocentral24h.com.br"
    assert "(11) 98888-0002" in contacts["secos"]["phone"]

    assert contacts["geral_bazar"]["name"] == "Ricardo Lima"
    assert contacts["geral_bazar"]["email"] == "compras.geral@mercadocentral24h.com.br"
    assert "(11) 98888-0003" in contacts["geral_bazar"]["phone"]

    assert "06h00" in compras["hours_sla"]
    assert "16h00" in compras["hours_sla"]


def test_corporate_catalog_fiscal_nfe_data_integrity():
    """T-R2.06: Valida dados do Fiscal & NFe (José Oliveira, nfe@..., XML, 72h SEFAZ, 5 anos)."""
    fiscal = CORPORATE_CONTACT_CATALOG["fiscal_nfe"]
    assert fiscal["responsible"] == "José Oliveira"
    assert fiscal["primary_email"] == "nfe@mercadocentral24h.com.br"
    assert "(11) 98888-0005" in fiscal["primary_phone"]
    assert any("XML" in t for t in fiscal["topics"])
    assert any("72 horas" in t for t in fiscal["topics"])
    assert "72 horas" in fiscal["hours_sla"]


def test_corporate_catalog_sac_delivery_data_integrity():
    """T-R2.07: Valida dados do SAC (SP/RJ, WhatsApp, prazos CDC, reembolso PIX 24h, cartão)."""
    sac = CORPORATE_CONTACT_CATALOG["sac_delivery"]
    assert sac["regional_emails"]["sp"] == "sac.sp@mercadocentral24h.com.br"
    assert sac["regional_emails"]["rj"] == "sac.rj@mercadocentral24h.com.br"
    assert "(11) 9XXXX-XXXX" in sac["primary_phones"]["whatsapp_sp"]
    assert "(21) 9XXXX-XXXX" in sac["primary_phones"]["whatsapp_rj"]
    assert "24/7" in sac["hours_sla"]
    assert any("PIX" in t for t in sac["topics"])
    assert any("CDC" in t for t in sac["topics"])


def test_corporate_catalog_ouvidoria_fallback_data_integrity():
    """T-R2.08: Valida dados da Ouvidoria Geral (0800-CENTRAL, ouvidoria@..., SLA 5 dias úteis)."""
    ouv = CORPORATE_CONTACT_CATALOG["ouvidoria_fallback"]
    assert ouv["primary_email"] == "ouvidoria@mercadocentral24h.com.br"
    assert "0800-CENTRAL" in ouv["primary_phone"]
    assert "0800-236-8725" in ouv["primary_phone"]
    assert "5 dias úteis" in ouv["hours_sla"]


# ============================================================================
# 2. TESTES DO MOTOR DE ROTEAMENTO POR INTENÇÃO (route_fallback_contact)
# ============================================================================

@pytest.mark.parametrize("query,expected_dept_key", [
    ("Quando cai o salário do mês e o adiantamento quinzenal?", "rh"),
    ("Como funciona o plano de saúde Unimed e auxílio-creche?", "rh"),
    ("Quais são os turnos da escala 5x2 e banco de horas?", "rh"),
    ("Como funciona o bônus do programa Indique um Talento de 150 reais?", "rh"),
    ("Como marcar ponto eletrônico biométrico?", "rh"),
    ("Quero fazer uma denúncia anônima sobre conduta antiética de um gestor", "juridico_compliance"),
    ("Colaborador pode receber brindes acima de 100 reais?", "juridico_compliance"),
    ("Regras da Lei Anticorrupção 12.846 e canal confidencial", "juridico_compliance"),
    ("Como falar com a Camila Ferreira do compliance?", "juridico_compliance"),
    ("Como solicitar a exclusão dos meus dados pessoais conforme a LGPD?", "dpo_lgpd"),
    ("Quero revogar o consentimento de cookies e gerenciar privacidade no app", "dpo_lgpd"),
    ("Qual o e-mail do DPO para portabilidade em JSON?", "dpo_lgpd"),
    ("Qual o horário de funcionamento das docas para descarga de fornecedores?", "compras_fornecedores"),
    ("Como homologar cadastro de novo fornecedor e agendar janela?", "compras_fornecedores"),
    ("Como falar com o João Silva sobre cotação de perecíveis e FLV?", "compras_fornecedores"),
    ("Como solicitar antecipação de recebíveis no portal financeiro?", "compras_fornecedores"),
    ("Para onde enviar o arquivo XML da NF-e com a chave de 44 dígitos?", "fiscal_nfe"),
    ("Qual o prazo da manifestação eletrônica na SEFAZ e emissão de CC-e?", "fiscal_nfe"),
    ("Como falar com José Oliveira do recebimento fiscal?", "fiscal_nfe"),
    ("Meu pedido do delivery atrasou, como falar no chat do SAC?", "sac_delivery"),
    ("Qual o prazo de estorno no PIX e cartão para produto com defeito no CDC?", "sac_delivery"),
    ("Como funciona o programa De Olho na Validade para produto vencido?", "sac_delivery"),
    ("Quais os benefícios de cashback do Cliente VIP Diamante?", "sac_delivery"),
    ("O gerente da loja não resolveu minha reclamação, quero falar com o ouvidor de nível 3", "ouvidoria_fallback"),
    ("Quero registrar um elogio formal para a diretoria institucional", "ouvidoria_fallback"),
])
def test_route_fallback_contact_exact_intent_routing(query: str, expected_dept_key: str):
    """T-R2.09: Valida roteamento preciso de intenções para os 7 departamentos."""
    routed = route_fallback_contact(query)
    assert isinstance(routed, dict)
    assert routed["department_key"] == expected_dept_key, (
        f"Query '{query}' roteada incorretamente para '{routed.get('department_key')}' (esperado: '{expected_dept_key}')"
    )
    assert routed["match_score"] > 0.0


def test_route_fallback_contact_out_of_domain_defaults_to_ouvidoria():
    """T-R2.10: Valida que perguntas fora do domínio ou aleatórias caem no fallback universal da Ouvidoria."""
    out_of_domain_queries = [
        "Qual é a distância média entre a Terra e Saturno?",
        "Receita de bolo de chocolate com morango",
        "Como programar uma rede neural convolucional em Rust?",
        "xyz123 random non-sense words",
        "",
        "   ",
    ]
    for q in out_of_domain_queries:
        routed = route_fallback_contact(q)
        assert routed["department_key"] == "ouvidoria_fallback"
        assert routed["primary_email"] == "ouvidoria@mercadocentral24h.com.br"
        assert "0800-CENTRAL" in routed["primary_phone"]


def test_route_fallback_disambiguation_supplier_vs_consumer_exchange():
    """T-R2.11: Valida desambiguação de devoluções (Fornecedor/Doca vs Consumidor/SAC)."""
    # Fornecedor
    q_supp = "Como fazer devolução de mercadoria avariada na conferência cega das docas?"
    routed_supp = route_fallback_contact(q_supp)
    assert routed_supp["department_key"] == "compras_fornecedores"

    # Consumidor
    q_cons = "Comprei um produto estragado no app, como pedir devolução e reembolso?"
    routed_cons = route_fallback_contact(q_cons)
    assert routed_cons["department_key"] == "sac_delivery"


def test_route_fallback_disambiguation_fiscal_xml_vs_order():
    """T-R2.12: Valida desambiguação entre nota fiscal técnica (XML/SEFAZ) e pedido de compra (OC)."""
    q_fiscal = "Qual a regra de validação do arquivo XML da NFe e chave de acesso de 44 dígitos?"
    assert route_fallback_contact(q_fiscal)["department_key"] == "fiscal_nfe"

    q_order = "Qual o procedimento para emissão e aprovação da Ordem de Compra OC de fornecedor?"
    assert route_fallback_contact(q_order)["department_key"] == "compras_fornecedores"


def test_route_fallback_disambiguation_ethics_vs_routine_hr():
    """T-R2.13: Valida desambiguação entre denúncia de conduta/assédio e rotina de RH."""
    q_ethics = "Quero fazer uma denúncia confidencial de assédio moral e pagamento de propina."
    assert route_fallback_contact(q_ethics)["department_key"] == "juridico_compliance"

    q_hr = "Qual a regra disciplinar de advertência verbal e atraso no ponto biométrico?"
    assert route_fallback_contact(q_hr)["department_key"] == "rh"


# ============================================================================
# 3. TESTES DE FORMATAÇÃO DE MENSAGEM DE FALLBACK (format_fallback_message)
# ============================================================================

def test_format_fallback_message_prefix_compliance():
    """T-R2.14: Valida que a mensagem de fallback inicia com o prefixo oficial padronizado."""
    dept = CORPORATE_CONTACT_CATALOG["rh"]
    for ch in ["chat", "email", "teams_slack", "unknown_channel"]:
        msg = format_fallback_message("Dúvida sobre salário", dept, channel=ch)
        assert "não encontrei essa informação nos documentos disponíveis" in msg.lower()
        assert "mercado central 24h" in msg.lower()
        assert "não encontrei informações oficiais" in msg.lower()


def test_format_fallback_message_chat_channel():
    """T-R2.15: Valida formato de fallback para canal 'chat'."""
    dept = CORPORATE_CONTACT_CATALOG["dpo_lgpd"]
    msg = format_fallback_message("Como apagar meus dados?", dept, channel="chat")
    
    assert "• **Departamento**:" in msg
    assert "• **E-mail Oficial**: `dpo@mercadocentral24h.com.br`" in msg
    assert "• **Telefone / WhatsApp**:" in msg
    assert "• **Canal Recomendado**:" in msg
    assert "Ouvidoria Geral" in msg
    assert "0800-CENTRAL" in msg


def test_format_fallback_message_email_channel():
    """T-R2.16: Valida formato de fallback para canal 'email'."""
    dept = CORPORATE_CONTACT_CATALOG["fiscal_nfe"]
    msg = format_fallback_message("Dúvida sobre XML", dept, channel="email")
    
    assert msg.startswith("Prezado(a) colaborador(a),")
    assert "**Resumo da Solicitação:**" in msg
    assert "**Encaminhamento Recomendado:**" in msg
    assert "• **Departamento**: Fiscal & Faturamento" in msg
    assert "• **Responsável**: José Oliveira" in msg
    assert "• **E-mail Institucional**: nfe@mercadocentral24h.com.br" in msg
    assert "**Canal Geral de Ouvidoria:**" in msg
    assert "Atenciosamente,\nEquipe de Atendimento - Mercado Central 24h" in msg


def test_format_fallback_message_teams_slack_channel():
    """T-R2.17: Valida formato de fallback para canal 'teams_slack'."""
    dept = CORPORATE_CONTACT_CATALOG["sac_delivery"]
    msg = format_fallback_message("Rastrear entrega expressa", dept, channel="teams_slack")
    
    assert "**[RESUMO]**" in msg
    assert "**[DEPARTAMENTO RECOMENDADO]**" in msg
    assert "• **Área**: SAC & Atendimento Delivery" in msg
    assert "• **E-mail**: `sac.sp@mercadocentral24h.com.br`" in msg
    assert "**[OUVIDORIA GERAL]**" in msg
    assert "`0800-CENTRAL (0800-236-8725)`" in msg


def test_normalize_catalog_text_accents_and_punctuation():
    """T-R2.18: Valida normalização robusta de texto (remoção de acentos e pontuações)."""
    assert normalize_catalog_text("Salário, Férias & Benefícios!") == "salario ferias beneficios"
    assert normalize_catalog_text("  NF-e / SEFAZ (72h)  ") == "nf e sefaz 72h"
    assert normalize_catalog_text(None) == ""
    assert normalize_catalog_text("") == ""
