#!/usr/bin/env python3
"""
Catálogo Corporativo de Contatos & Motor de Roteamento de Fallback por Intenção (R2)
Mercado Central 24h - Módulo RAG Corporativo
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional


def normalize_catalog_text(text: Optional[str]) -> str:
    """Normaliza texto para minúsculas, sem acentos e sem pontuação extra."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[^\w\s]", " ", ascii_text.lower())
    return " ".join(cleaned.split())


# ============================================================================
# CATÁLOGO CORPORATIVO OFICIAL DE CONTATOS (EXTRAÍDO DOS 8 DOCUMENTOS OFICIAIS)
# ============================================================================

CORPORATE_CONTACT_CATALOG: Dict[str, Dict[str, Any]] = {
    "rh": {
        "department_key": "rh",
        "department_name": "Recursos Humanos (RH) & Departamento Pessoal",
        "responsible": "Equipe de Recursos Humanos & DP",
        "primary_email": "rh@mercadocentral24h.com.br",
        "primary_phone": "0800-CENTRAL (0800-236-8725) / Ramal RH",
        "channels": [
            "E-mail institucional (rh@mercadocentral24h.com.br)",
            "Portal do Colaborador no Aplicativo Interno",
            "Balcão Presencial de RH nas Unidades (SP / RJ)",
            "Totens de Autoatendimento Interno",
        ],
        "primary_channel": "E-mail institucional (rh@mercadocentral24h.com.br) ou Portal do Colaborador no App",
        "hours_sla": "Segunda a Sexta das 08h00 às 18h00 | SLA de resposta até 48 horas úteis (CAT em 24h)",
        "topics": [
            "Remuneração e folha de pagamento (salário até 5º dia útil)",
            "Adiantamento salarial / vale 40% (dia 20)",
            "Benefícios corporativos (Plano Unimed coparticipativo, seguro de vida gratuito, refeitório na loja)",
            "Vale-Transporte (desconto legal de 6%) e Auxílio-Creche (filhos até 5 anos)",
            "Jornada de trabalho e escala padrão 5x2 (44h semanais, 8h40 diárias, turnos T1 a T5)",
            "Ponto eletrônico biométrico e facial com IA de escalas",
            "Banco de horas (+50% dias úteis, +100% domingos/feriados, compensação em 6 meses)",
            "Adicional noturno (20% das 22h às 05h com hora reduzida de 52m30s)",
            "Carreira, promoção e trilha de sucessão (mínimo 6 meses, avaliação >= 4.0)",
            "Programa Indique um Talento (bônus de R$ 150,00 após 90 dias)",
            "Normas disciplinares (advertência verbal/escrita, suspensão 1-3 dias, justa causa)",
            "Segurança do trabalho, EPIs por setor e emissão de CAT em até 24 horas",
            "Capacitação obrigatória em Libras e inclusão",
        ],
        "instructions": (
            "Abra uma solicitação no Portal do Colaborador no App ou envie um e-mail para "
            "rh@mercadocentral24h.com.br com sua matrícula e unidade de lotação."
        ),
        "keywords": [
            "salario", "salario mensal", "holerite", "pagamento", "folha", "folha de pagamento",
            "adiantamento", "adiantamento salarial", "vale", "vale 40", "dia 20", "5 dia util",
            "ferias", "solicitacao de ferias", "ponto", "ponto eletronico", "biometria",
            "reconhecimento facial", "escala", "escala 5x2", "5x2", "turno", "t1", "t2", "t3",
            "t4", "t5", "jornada 44h", "beneficio", "beneficios", "unimed", "plano de saude",
            "refeitorio", "refeitorio proprio", "vale transporte", "vale transporte vt", "vt",
            "desconto 6", "creche", "auxilio creche", "carreira", "trilha de sucessao",
            "promocao", "promocao interna", "admissao", "demissao", "justa causa",
            "advertencia", "advertencia verbal", "advertencia escrita", "suspensao",
            "disciplinar", "bonus", "indique um talento", "150 reais", "banco de horas",
            "adicional noturno", "hora noturna", "52m30s", "epi", "seguranca do trabalho",
            "treinamento", "libras", "cat", "acidente de trabalho", "atestado", "afastamento",
            "convenio farmacia", "convenio otica", "desconto faculdade",
        ],
        "regex_patterns": [
            r"(?i)\b(sal[aá]rio|holerite|adiantamento|f[eé]rias|ponto\s+(eletr[oô]nico|facial|biom[eé]trico))\b",
            r"(?i)\b(escala\s*5x2|plano\s+de\s+sa[uú]de|unimed|aux[ií]lio[- ]creche|vale[- ]transporte)\b",
            r"(?i)\b(admiss[aã]o|demiss[aã]o|justa\s+causa|trilha\s+de\s+sucess[aã]o|indique\s+um\s+talento)\b",
            r"(?i)\b(banco\s+de\s+horas|adicional\s+noturno|acidente\s+de\s+trabalho|abertura\s+de\s+cat)\b",
        ],
    },
    "juridico_compliance": {
        "department_key": "juridico_compliance",
        "department_name": "Jurídico & Compliance (Canal de Ética e Denúncias)",
        "responsible": "Camila Ferreira",
        "primary_email": "etica@mercadocentral24h.com.br",
        "primary_phone": "0800-CENTRAL (0800-236-8725)",
        "channels": [
            "Canal Confidencial de Ética (etica@mercadocentral24h.com.br)",
            "Telefone Gratuito 0800-CENTRAL (0800-236-8725)",
            "Canal de Denúncias com Garantia de Anonimato",
        ],
        "primary_channel": "Canal Confidencial de Ética (etica@mercadocentral24h.com.br | 0800-CENTRAL)",
        "hours_sla": "Recebimento contínuo 24/7 | Confirmação de recebimento / protocolo em até 48 horas úteis",
        "topics": [
            "Canal de denúncias anônimas e estritamente confidenciais",
            "Lei Anticorrupção (Lei nº 12.846/2013) e vedação a propinas/vantagens indevidas",
            "Código de Conduta Ética e Integridade Corporativa",
            "Política de brindes e presentes institucionais (limite máximo de R$ 100,00)",
            "Prevenção e combate ao assédio moral e assédio sexual",
            "Tolerância zero a trabalho análogo ao escravo e trabalho infantil (notificação ao MPT)",
            "Termos e Condições de Uso, propriedade intelectual e representação judicial",
            "Governança corporativa e foro da Comarca de São Paulo/SP",
        ],
        "instructions": (
            "Envie seu relato com garantia total de anonimato para etica@mercadocentral24h.com.br "
            "ou ligue para 0800-CENTRAL (0800-236-8725). Responsável: Camila Ferreira."
        ),
        "keywords": [
            "denuncia", "denuncias", "denuncia anonima", "canal de denuncias", "etica",
            "codigo de etica", "compliance", "corrupcao", "lei anticorrupcao", "lei 12 846",
            "lei 12846", "suborno", "propina", "vantagem indevida", "comissao oculta",
            "conduta", "codigo de conduta", "desvio de conduta", "assedio", "assedio moral",
            "assedio sexual", "fraude", "irregularidade", "brinde", "brindes", "limite de brindes",
            "100 reais brindes", "trabalho escravo", "trabalho analogo", "trabalho infantil",
            "ministerio publico", "mpt", "processo judicial", "representacao judicial",
            "foro", "propriedade intelectual", "camila ferreira",
        ],
        "regex_patterns": [
            r"(?i)\b(den[uú]ncia[s]?|[eé]tica|compliance|corrup[cç][aã]o|suborno|propina|ass[eé]dio)\b",
            r"(?i)\b(lei\s*(n[ºo\.]?\s*)?12\.?846|c[oó]digo\s+de\s+conduta|trabalho\s+(escravo|infantil))\b",
            r"(?i)\b(limite\s+de\s+brinde[s]?|camila\s+ferreira|vantagem\s+indevida)\b",
        ],
    },
    "dpo_lgpd": {
        "department_key": "dpo_lgpd",
        "department_name": "Encarregado de Proteção de Dados (DPO / LGPD)",
        "responsible": "Encarregado de Proteção de Dados (DPO)",
        "primary_email": "dpo@mercadocentral24h.com.br",
        "primary_phone": "0800-CENTRAL (0800-236-8725)",
        "postal_address": "Av. Principal, n.º 1000 – Centro – São Paulo/SP – A/C: DPO",
        "channels": [
            "Aplicativo Mercado Central 24h: Aba 'Meus Dados' -> 'Gerenciar Privacidade'",
            "E-mail institucional do DPO: dpo@mercadocentral24h.com.br",
            "Correspondência Postal Físico (A/C Encarregado DPO)",
        ],
        "primary_channel": "App 'Meus Dados' -> 'Gerenciar Privacidade' ou e-mail dpo@mercadocentral24h.com.br",
        "hours_sla": (
            "Dias úteis das 09h00 às 18h00 | Acesso e confirmação: até 15 dias corridos | "
            "Correção de dados: até 5 dias úteis | Guarda legal fiscal: 5 anos"
        ),
        "topics": [
            "Exercício dos direitos dos titulares (Art. 18 da Lei Geral de Proteção de Dados - Lei nº 13.709/2018)",
            "Confirmação da existência de tratamento e acesso aos dados pessoais",
            "Correção de dados incompletos, inexatos ou desatualizados (5 dias úteis)",
            "Anonimização, bloqueio ou eliminação de dados desnecessários",
            "Portabilidade de dados cadastrais em formato estruturado e legível (JSON / CSV)",
            "Revogação de consentimento para marketing e personalização",
            "Segurança de dados tokenizados de pagamento (cartões e chaves PIX)",
            "Gestão e sigilo de imagens de câmeras de segurança CFTV",
            "Prazos de retenção legal (guarda mínima de 5 anos para documentos fiscais)",
        ],
        "instructions": (
            "Acesse o App na seção 'Meus Dados' -> 'Gerenciar Privacidade' ou envie sua solicitação "
            "formal com cópia de documento de identificação para dpo@mercadocentral24h.com.br."
        ),
        "keywords": [
            "lgpd", "lei geral de protecao de dados", "lei 13 709", "lei 13709", "dpo",
            "encarregado de dados", "privacidade", "dados pessoais", "protecao de dados",
            "titular de dados", "direitos do titular", "artigo 18", "art 18 lgpd", "cookies",
            "consentimento", "revogar consentimento", "excluir dados", "exclusao de dados",
            "apagar meus dados", "anonimizacao", "bloqueio de dados", "portabilidade",
            "portabilidade json", "portabilidade csv", "cftv sigilo", "cameras de seguranca",
            "dados tokenizados", "vazamento de dados",
        ],
        "regex_patterns": [
            r"(?i)\b(lgpd|dpo|dados\s+pessoais|prote[cç][aã]o\s+de\s+dados|privacidade)\b",
            r"(?i)\b(lei\s*(n[ºo\.]?\s*)?13\.?709|titular\s+de\s+dados|revoga[cç][aã]o\s+de\s+consentimento)\b",
            r"(?i)\b(exclus[aã]o\s+de\s+dados|portabilidade\s+de\s+dados|gerenciar\s+privacidade)\b",
        ],
    },
    "compras_fornecedores": {
        "department_key": "compras_fornecedores",
        "department_name": "Compras & Suprimentos (Gestão de Fornecedores)",
        "responsible": "Equipe de Compras & Gestão de Fornecedores",
        "responsible_contacts": {
            "pereciveis": {
                "name": "João Silva",
                "category": "Perecíveis e FLV (Hortifrúti, Carnes, Frios)",
                "email": "compras.pereciveis@mercadocentral24h.com.br",
                "phone": "(11) 98888-0001",
            },
            "secos": {
                "name": "Maria Santos",
                "category": "Mercearia e Bebidas (Secos)",
                "email": "compras.secos@mercadocentral24h.com.br",
                "phone": "(11) 98888-0002",
            },
            "geral_bazar": {
                "name": "Ricardo Lima",
                "category": "Higiene, Bazar e Suprimentos Gerais",
                "email": "compras.geral@mercadocentral24h.com.br",
                "phone": "(11) 98888-0003",
            },
        },
        "primary_email": "compras.geral@mercadocentral24h.com.br",
        "primary_phone": "(11) 98888-0001 / (11) 98888-0002 / (11) 98888-0003",
        "channels": [
            "Portal de Fornecedores e Portal Financeiro",
            "Mesa Setorial de Compras (João Silva, Maria Santos, Ricardo Lima)",
            "E-mail Geral de Compras: compras.geral@mercadocentral24h.com.br",
        ],
        "primary_channel": "Portal de Fornecedores / Portal Financeiro ou compras.geral@mercadocentral24h.com.br",
        "hours_sla": (
            "Docas de Descarga: Seg-Sex 06h00 às 16h00, Sáb 07h00 às 12h00 | "
            "Atendimento Comercial: Seg-Sex 08h00 às 18h00"
        ),
        "topics": [
            "Homologação e cadastro de fornecedores (CNPJ < 90d, Contrato Social, CNDs, Alvará Sanitário)",
            "Emissão de Ordens de Compra (OC), cotações e agendamento de janelas de descarga",
            "Recebimento em docas e conferência cega na portaria",
            "Prazos de pagamento por categoria (Secos 28-42d, Perecíveis 7-14d, Bazar 30-60d)",
            "Programa de antecipação de recebíveis (taxa de 2,5% a.m. pro-rata, portal financeiro)",
            "Metas e KPIs de fornecedores (OTIF > 95%, avarias < 0,5%, fill rate > 98%)",
            "Incentivo a produtores regionais e locais (raio de 100 km, meta 30% em hortifrúti)",
            "Calendário anual de negociações sazonais (Páscoa, Black Friday, Natal)",
        ],
        "instructions": (
            "Acesse o Portal de Fornecedores ou contate o comprador da sua categoria pelo e-mail "
            "compras.geral@mercadocentral24h.com.br informando o número da OC ou CNPJ."
        ),
        "keywords": [
            "fornecedor", "fornecedores", "cadastro de fornecedor", "homologacao de fornecedor",
            "compras", "cotacao", "cotacoes", "ordem de compra", "oc", "pedido de compra",
            "doca", "docas", "descarga", "horario de descarga", "janela de descarga",
            "conferencia cega", "recebimento de mercadorias", "antecipacao de recebiveis",
            "portal financeiro", "portal de fornecedores", "classe a", "classe b", "classe c",
            "prazo de pagamento fornecedor", "otif", "fill rate", "avaria de fornecedor",
            "produtores locais", "produtor regional", "joao silva", "maria santos", "ricardo lima",
            "compras pereciveis", "compras secos",
        ],
        "regex_patterns": [
            r"(?i)\b(fornecedor[es]*|ordem\s+de\s+compra|homologa[cç][aã]o\s+de\s+fornecedor|confer[eê]ncia\s+cega)\b",
            r"(?i)\b(hor[aá]rio\s+de\s+descarga|antecipa[cç][aã]o\s+de\s+receb[ií]veis|doca[s]?|otif|fill\s+rate)\b",
            r"(?i)\b(jo[aã]o\s+silva|maria\s+santos|ricardo\s+lima|compras\.(pereciveis|secos|geral))\b",
        ],
    },
    "fiscal_nfe": {
        "department_key": "fiscal_nfe",
        "department_name": "Fiscal & Faturamento (Recebimento Fiscal)",
        "responsible": "José Oliveira",
        "primary_email": "nfe@mercadocentral24h.com.br",
        "primary_phone": "(11) 98888-0005",
        "channels": [
            "E-mail de Recebimento Fiscal: nfe@mercadocentral24h.com.br",
            "Telefone direto Fiscal: (11) 98888-0005 (José Oliveira)",
            "Portal Integrado de Fornecedores",
        ],
        "primary_channel": "E-mail (nfe@mercadocentral24h.com.br) ou Telefone (11) 98888-0005",
        "hours_sla": (
            "Segunda a Sexta das 06h00 às 17h00, Sábado das 07h00 às 12h00 | "
            "Manifestação SEFAZ em até 72 horas | Guarda legal fiscal de 5 anos"
        ),
        "topics": [
            "Envio obrigatório de arquivo XML no momento da saída do veículo do CD do fornecedor",
            "Validação de DANFE com chave de acesso de 44 dígitos legível",
            "Manifestação Eletrônica do Destinatário (SEFAZ) em até 72 horas após a entrada física",
            "Regras e restrições de Carta de Correção Eletrônica (CC-e)",
            "Procedimento de recusa fiscal e emissão de Nota Fiscal de Devolução (total ou parcial)",
            "Guarda legal obrigatória de documentos fiscais pelo prazo mínimo de 5 anos",
            "Emissão de NFC-e e registro de CPF na nota para consumidores",
        ],
        "instructions": (
            "Envie o arquivo XML e o DANFE com a chave de 44 dígitos para nfe@mercadocentral24h.com.br. "
            "Responsável: José Oliveira, tel. (11) 98888-0005."
        ),
        "keywords": [
            "nfe", "nf e", "nota fiscal", "danfe", "xml", "arquivo xml", "chave de acesso",
            "44 digitos", "sefaz", "manifestacao do destinatario", "manifestacao eletronica",
            "carta de correcao", "carta de correcao eletronica", "cc e", "cce", "faturamento",
            "imposto", "impostos", "tributario", "cfop", "nota fiscal de devolucao",
            "recusa fiscal", "divergencia fiscal", "guarda fiscal", "5 anos fiscal",
            "jose oliveira", "frente de caixa fiscal",
        ],
        "regex_patterns": [
            r"(?i)\b(nf-?e|danfe|arquivo\s+xml|chave\s+de\s+acesso|sefaz|carta\s+de\s+corre[cç][aã]o|cc-e)\b",
            r"(?i)\b(manifesta[cç][aã]o\s+eletr[oô]nica|recusa\s+fiscal|nota\s+fiscal\s+de\s+devolu[cç][aã]o)\b",
            r"(?i)\b(jos[eé]\s+oliveira|nfe@mercadocentral24h\.com\.br)\b",
        ],
    },
    "sac_delivery": {
        "department_key": "sac_delivery",
        "department_name": "SAC & Atendimento Delivery SP / RJ",
        "responsible": "Central de Atendimento ao Consumidor (SAC & Delivery)",
        "primary_email": "sac.sp@mercadocentral24h.com.br",
        "regional_emails": {
            "sp": "sac.sp@mercadocentral24h.com.br",
            "rj": "sac.rj@mercadocentral24h.com.br",
        },
        "primary_phones": {
            "geral": "0800-CENTRAL (0800-236-8725)",
            "whatsapp_sp": "(11) 9XXXX-XXXX",
            "whatsapp_rj": "(21) 9XXXX-XXXX",
        },
        "primary_phone": "0800-CENTRAL (0800-236-8725) | WhatsApp SP: (11) 9XXXX-XXXX / RJ: (21) 9XXXX-XXXX",
        "channels": [
            "Chat Automatizado 24/7 no App 'Mercado Central 24h'",
            "WhatsApp Delivery SP ((11) 9XXXX-XXXX) e RJ ((21) 9XXXX-XXXX)",
            "Central 0800-CENTRAL (0800-236-8725)",
            "Balcão Físico de SAC nas Lojas (São Paulo - Vila Mariana e Cordeiro - RJ)",
        ],
        "primary_channel": "Chat no App 'Mercado Central 24h' (24/7) ou WhatsApp",
        "hours_sla": (
            "Chat App e SAC: 24/7 | Entrega Expressa: 24/7 em até 3h | "
            "Reembolso PIX: até 24h úteis | Estorno Cartão: até 5 dias úteis (até 2 faturas)"
        ),
        "topics": [
            "Atendimento a clientes sobre pedidos, rastreamento e status de entrega",
            "Modalidades de entrega (Expressa 24/7 até 3h, Agendada 08h-22h, Clique & Retire 24/7 até 1h, iFood 07h-23h)",
            "Raio de entrega (SP: até 7 km da Matriz Vila Mariana; RJ: até 15 km da Unidade Cordeiro)",
            "Políticas de frete grátis (compras acima de R$ 250,00 ou VIP Diamante acima de R$ 100,00)",
            "Prazos CDC para trocas e devoluções (alimentos perecíveis 24h/30d, duráveis 90d, arrependimento 7d)",
            "Métodos e prazos de reembolso (PIX até 24h úteis, cartão até 5 dias úteis, vale-compras imediato)",
            "Programa 'De Olho na Validade' (1 item gratuito idêntico se vencido na gôndola)",
            "Programa Cliente VIP Central (Bronze 0,5%, Prata 1,0%, Gold 1,5%, Diamante 2,0% cashback)",
        ],
        "instructions": (
            "Acesse o App 'Mercado Central 24h' na aba 'Meus Pedidos' para atendimento 24/7 via Chat, "
            "ou contate o SAC por e-mail (sac.sp@ / sac.rj@mercadocentral24h.com.br) ou 0800-CENTRAL."
        ),
        "keywords": [
            "sac", "atendimento", "sac delivery", "status de entrega", "meu pedido", "rastreamento",
            "rastrear pedido", "atraso de entrega", "entrega expressa", "entrega agendada",
            "clique e retire", "pickup", "frete", "frete gratis", "frete gratis 250",
            "ifood", "taxa de entrega", "troca", "trocas", "devolucao", "devolucao de produto",
            "reembolso", "estorno", "estorno pix", "estorno cartao", "defeito", "vicio",
            "produto vencido", "produto estragado", "de olho na validade", "direito de arrependimento",
            "7 dias arrependimento", "30 dias cdc", "90 dias duraveis", "cliente vip",
            "vip diamante", "cashback", "pontos vip", "bronze", "prata", "gold",
            "vila mariana", "cordeiro", "whatsapp sp", "whatsapp rj",
        ],
        "regex_patterns": [
            r"(?i)\b(status\s+de\s+entrega|rastreamento|atraso\s+no\s+pedido|frete\s+gr[aá]tis|clique\s*&?\s*retire)\b",
            r"(?i)\b(troca[s]?|devolu[cç][aã]o|reembolso|estorno|de\s+olho\s+na\s+validade|direito\s+de\s+arrependimento)\b",
            r"(?i)\b(cliente\s+vip|cashback|pontos\s+vip|sac\s+delivery|sac\.(sp|rj))\b",
        ],
    },
    "ouvidoria_fallback": {
        "department_key": "ouvidoria_fallback",
        "department_name": "Ouvidoria Geral 0800-CENTRAL (Instância Máxima)",
        "responsible": "Ouvidor Institucional",
        "primary_email": "ouvidoria@mercadocentral24h.com.br",
        "primary_phone": "0800-CENTRAL (0800-236-8725)",
        "channels": [
            "Telefone Gratuito Nacional 0800-CENTRAL (0800-236-8725)",
            "E-mail Oficial da Ouvidoria: ouvidoria@mercadocentral24h.com.br",
        ],
        "primary_channel": "Telefone gratuito 0800-CENTRAL (0800-236-8725) ou e-mail ouvidoria@mercadocentral24h.com.br",
        "hours_sla": "Dias úteis das 09h00 às 18h00 | Resposta conclusiva formal com número de protocolo em até 5 dias úteis",
        "topics": [
            "Atendimento de 3º nível para demandas não solucionadas satisfatoriamente no SAC ou Gerência",
            "Mediação de conflitos corporativos, trabalhistas e de relacionamento",
            "Registro de elogios formais, sugestões institucionais e críticas construtivas à Diretoria",
            "Canal padrão universal de fallback para dúvidas fora do escopo documental",
        ],
        "instructions": (
            "Entre em contato pelo telefone gratuito 0800-CENTRAL (0800-236-8725) ou envie sua mensagem para "
            "ouvidoria@mercadocentral24h.com.br informando o protocolo anterior, caso houver."
        ),
        "keywords": [
            "ouvidoria", "ouvidor", "reclamacao", "reclamacao formal", "reclamacoes",
            "elogio", "elogios", "sugestao", "sugestoes", "insatisfacao", "nivel 3",
            "gerencia nao resolveu", "instancia maxima", "ouvidoria geral", "0800 central",
            "0800 236 8725",
        ],
        "regex_patterns": [
            r"(?i)\b(ouvidoria|reclama[cç][aã]o\s+(grave|geral|n[ií]vel\s*3)|elogio[s]?|sugest[aã]o|sugest[oõ]es)\b",
            r"(?i)\b(0800[- ]central|0800[- ]236[- ]8725|ouvidoria@mercadocentral24h\.com\.br)\b",
        ],
    },
}


# ============================================================================
# MOTOR DE ROTEAMENTO POR INTENÇÃO (R2)
# ============================================================================

def route_fallback_contact(query: str) -> Dict[str, Any]:
    """
    Classifica a pergunta do usuário e determina o departamento corporativo mais adequado.
    Aplica análise léxica ponderada, casamento por expressões regulares e regras de desambiguação.
    Caso a consulta não apresente correspondência com departamentos específicos, retorna Ouvidoria Geral.
    """
    if not query or not str(query).strip():
        fallback_dept = dict(CORPORATE_CONTACT_CATALOG["ouvidoria_fallback"])
        fallback_dept["query"] = query or ""
        fallback_dept["match_score"] = 0.0
        fallback_dept["matched_keywords"] = []
        return fallback_dept

    raw_query = str(query).strip()
    norm_query = normalize_catalog_text(raw_query)
    tokens = set(norm_query.split())

    scores: Dict[str, float] = {}
    matched_kws_map: Dict[str, List[str]] = {}

    for dept_key, dept_data in CORPORATE_CONTACT_CATALOG.items():
        if dept_key == "ouvidoria_fallback":
            # Avaliado apenas se houver palavras explícitas de ouvidoria
            pass

        score = 0.0
        matched_kws: List[str] = []

        # 1. Checagem de palavras-chave / frases
        keywords = dept_data.get("keywords", [])
        for kw in keywords:
            norm_kw = normalize_catalog_text(kw)
            if " " in norm_kw:
                if norm_kw in norm_query:
                    score += 2.5
                    matched_kws.append(kw)
            else:
                if norm_kw in tokens:
                    score += 1.0
                    matched_kws.append(kw)

        # 2. Checagem de regex patterns
        patterns = dept_data.get("regex_patterns", [])
        for pattern in patterns:
            try:
                if re.search(pattern, raw_query) or re.search(pattern, norm_query):
                    score += 3.0
            except re.error:
                continue

        scores[dept_key] = score
        matched_kws_map[dept_key] = matched_kws

    # Desambiguação contextual
    # 1. Trocas e Devoluções: Fornecedor/Docas vs Consumidor/SAC
    is_supplier_exchange = any(t in tokens for t in ["fornecedor", "fornecedores", "doca", "docas", "palete", "conferencia", "ordem", "oc"])
    is_consumer_exchange = any(t in tokens for t in ["cliente", "comprei", "consumidor", "app", "sac", "loja", "casa", "cdc", "cupom"])
    if is_supplier_exchange and not is_consumer_exchange:
        scores["compras_fornecedores"] = scores.get("compras_fornecedores", 0.0) + 3.0
    elif is_consumer_exchange and not is_supplier_exchange:
        scores["sac_delivery"] = scores.get("sac_delivery", 0.0) + 3.0

    # 2. Fiscal vs Compras: XML, DANFE, SEFAZ vs Ordem de Compra
    is_strict_fiscal = any(t in tokens for t in ["xml", "danfe", "sefaz", "cc", "cce", "chave", "44"])
    if is_strict_fiscal:
        scores["fiscal_nfe"] = scores.get("fiscal_nfe", 0.0) + 4.0

    # 3. Compliance vs RH: Assédio, denúncia, propina vs Ponto, férias, salário
    is_strict_compliance = any(t in tokens for t in ["denuncia", "suborno", "propina", "assedio", "corrupcao", "brinde", "brindes", "etica"])
    if is_strict_compliance:
        scores["juridico_compliance"] = scores.get("juridico_compliance", 0.0) + 4.0

    # 4. DPO vs RH: Dados pessoais, privacidade, cookies vs Cadastro interno de funcionário
    is_strict_privacy = any(t in tokens for t in ["lgpd", "dpo", "privacidade", "cookies", "titular", "anonimizacao"])
    if is_strict_privacy:
        scores["dpo_lgpd"] = scores.get("dpo_lgpd", 0.0) + 4.0

    # Seleciona o departamento com maior score
    best_dept_key = None
    best_score = 0.0

    for dept_key, score in scores.items():
        if score > best_score:
            best_score = score
            best_dept_key = dept_key

    # Se nenhum departamento tiver score suficiente (> 0), roteia para Ouvidoria Geral
    if not best_dept_key or best_score <= 0.0:
        result_dept = dict(CORPORATE_CONTACT_CATALOG["ouvidoria_fallback"])
        result_dept["query"] = raw_query
        result_dept["match_score"] = 0.0
        result_dept["matched_keywords"] = []
        return result_dept

    result_dept = dict(CORPORATE_CONTACT_CATALOG[best_dept_key])
    result_dept["query"] = raw_query
    result_dept["match_score"] = float(best_score)
    result_dept["matched_keywords"] = matched_kws_map.get(best_dept_key, [])
    return result_dept


# ============================================================================
# FORMATADOR DE MENSAGEM DE FALLBACK POR CANAL (R2 / R3)
# ============================================================================

def format_fallback_message(query: str, department_info: Dict[str, Any], channel: str = "chat") -> str:
    """
    Formata a mensagem oficial de recusa e roteamento para os contatos corporativos.
    Inicia obrigatoriamente com a mensagem padronizada de ausência de informações nos documentos
    e adapta o layout para os canais 'chat', 'email' e 'teams_slack'.
    """
    clean_channel = (channel or "chat").lower().strip()
    if clean_channel not in ("chat", "email", "teams_slack"):
        clean_channel = "chat"

    dept_name = department_info.get("department_name", "Ouvidoria Geral 0800-CENTRAL")
    responsible = department_info.get("responsible", "Equipe Responsável")
    primary_email = department_info.get("primary_email", "ouvidoria@mercadocentral24h.com.br")
    primary_phone = department_info.get("primary_phone", "0800-CENTRAL (0800-236-8725)")
    primary_channel_desc = department_info.get("primary_channel", "Telefone 0800-CENTRAL ou E-mail")
    hours_sla = department_info.get("hours_sla", "Segunda a Sexta em horário comercial")
    instructions = department_info.get("instructions", "Entre em contato através dos canais oficiais informados.")

    # Mensagem padronizada de abertura
    prefix = (
        "Não encontrei essa informação nos documentos disponíveis do Mercado Central 24h "
        "(não encontrei informações oficiais suficientes sobre esse tema)."
    )

    if clean_channel == "email":
        return (
            f"Prezado(a) colaborador(a),\n\n"
            f"{prefix}\n\n"
            f"**Resumo da Solicitação:**\n"
            f"Não foi possível localizar respaldo documental conclusivo para a consulta informada nos regulamentos e manuais indexados.\n\n"
            f"**Encaminhamento Recomendado:**\n"
            f"• **Departamento**: {dept_name}\n"
            f"• **Responsável**: {responsible}\n"
            f"• **E-mail Institucional**: {primary_email}\n"
            f"• **Telefone / Contato**: {primary_phone}\n"
            f"• **Canal Preferencial**: {primary_channel_desc}\n"
            f"• **Horário / SLA**: {hours_sla}\n"
            f"• **Procedimento**: {instructions}\n\n"
            f"**Canal Geral de Ouvidoria:**\n"
            f"Nossa Ouvidoria Geral está disponível pelo telefone gratuito 0800-CENTRAL (0800-236-8725) "
            f"ou pelo e-mail ouvidoria@mercadocentral24h.com.br (SLA de resposta com protocolo em até 5 dias úteis).\n\n"
            f"Atenciosamente,\n"
            f"Equipe de Atendimento - Mercado Central 24h"
        )

    elif clean_channel == "teams_slack":
        return (
            f"{prefix}\n\n"
            f"**[RESUMO]**\n"
            f"Não localizamos informações documentadas para a sua consulta.\n\n"
            f"**[DEPARTAMENTO RECOMENDADO]**\n"
            f"• **Área**: {dept_name}\n"
            f"• **E-mail**: `{primary_email}`\n"
            f"• **Contato**: {primary_phone}\n"
            f"• **Horário / SLA**: {hours_sla}\n"
            f"• **Orientação**: {instructions}\n\n"
            f"**[OUVIDORIA GERAL]**\n"
            f"• Telefone gratuito: `0800-CENTRAL (0800-236-8725)` | E-mail: `ouvidoria@mercadocentral24h.com.br`"
        )

    else:  # chat
        return (
            f"{prefix}\n\n"
            f"Para esclarecer sua dúvida, recomendamos entrar em contato diretamente com a área responsável:\n\n"
            f"• **Departamento**: {dept_name}\n"
            f"• **E-mail Oficial**: `{primary_email}`\n"
            f"• **Telefone / WhatsApp**: {primary_phone}\n"
            f"• **Canal Recomendado**: {primary_channel_desc}\n"
            f"• **Horário / SLA**: {hours_sla}\n"
            f"• **Orientação**: {instructions}\n\n"
            f"Se preferir, nossa **Ouvidoria Geral** está à disposição pelo telefone gratuito `0800-CENTRAL (0800-236-8725)` "
            f"ou e-mail `ouvidoria@mercadocentral24h.com.br`."
        )


if __name__ == "__main__":
    test_queries = [
        "Qual o dia do pagamento do salário e vale?",
        "Quero fazer uma denúncia anônima de fraude",
        "Como solicitar exclusão dos meus dados pessoais?",
        "Qual o horário de entrega das docas para fornecedor?",
        "Como enviar o arquivo XML da nota fiscal?",
        "Como funciona o reembolso do PIX no SAC?",
        "Qual a distância média da Terra à Lua?",
    ]

    print("=== TESTE DO MOTOR DE ROTEAMENTO DE CONTATOS ===")
    for q in test_queries:
        routed = route_fallback_contact(q)
        print(f"\nQuery: {q}")
        print(f"-> Dept: {routed['department_name']} ({routed['department_key']}) | Score: {routed['match_score']}")
        print(f"-> E-mail: {routed['primary_email']} | Telefone: {routed['primary_phone']}")
