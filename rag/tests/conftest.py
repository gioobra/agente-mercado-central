"""
Shared pytest fixtures for Mercado Central 24h RAG Test Suite.
"""

import os
import sys
import shutil
import tempfile
import pytest
from pathlib import Path

# Project root is resolved via standard package imports and pytest pythonpath
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "rag" / "scripts"



@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """Retorna o caminho Path absoluto da raiz do projeto MercadoCentral."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def real_chunks_json_path(project_root_path: Path) -> Path:
    """Retorna o caminho para o arquivo processed_rag_chunks.json real."""
    return project_root_path / "rag" / "data" / "processed_rag_chunks.json"


@pytest.fixture(scope="module")
def mock_chunks():
    """Fixture com conjunto sintético e determinístico de chunks para testes unitários/integração."""
    return [
        {
            "chunk_id": "TEST_CHK_001",
            "file_name": "Guia_de_Envios_e_Entregas.pdf",
            "file_path": "/docs/pdf/Guia_de_Envios_e_Entregas.pdf",
            "category": "Logística & Delivery",
            "department_author": "Logística & Operações Digitais",
            "last_updated": "Agosto de 2026",
            "section_title": "2. MODALIDADES DE ENTREGA E PRAZOS",
            "page_start": 1,
            "page_end": 2,
            "char_count": 250,
            "word_count": 40,
            "text": "Entrega Expressa App (24/7): Receba em até 3 horas. Frete Grátis para compras a partir de R$ 250,00 no App."
        },
        {
            "chunk_id": "TEST_CHK_002",
            "file_name": "Guia_de_Envios_e_Entregas.pdf",
            "file_path": "/docs/pdf/Guia_de_Envios_e_Entregas.pdf",
            "category": "Logística & Delivery",
            "department_author": "Logística & Operações Digitais",
            "last_updated": "Agosto de 2026",
            "section_title": "2. Benefício Cliente VIP Diamante",
            "page_start": 2,
            "page_end": 2,
            "char_count": 200,
            "word_count": 30,
            "text": "Clientes cadastrados no nível Diamante possuem Frete Grátis Ilimitado em compras a partir de R$ 100,00 no App próprio. O cashback Diamante é de 2,0%."
        },
        {
            "chunk_id": "TEST_CHK_003",
            "file_name": "Politica_de_Reembolso_e_Devolucoes.pdf",
            "file_path": "/docs/pdf/Politica_de_Reembolso_e_Devolucoes.pdf",
            "category": "Atendimento & CDC",
            "department_author": "Atendimento ao Cliente (SAC) / Financeiro",
            "last_updated": "Agosto de 2026",
            "section_title": "3. DIREITO DE ARREPENDIMENTO",
            "page_start": 1,
            "page_end": 1,
            "char_count": 300,
            "word_count": 45,
            "text": "O cliente tem o prazo de 7 dias corridos a contar da data de recebimento do produto para solicitar a devolução por arrependimento conforme Art. 49 do CDC. PIX estorno em 24h."
        },
        {
            "chunk_id": "TEST_CHK_004",
            "file_name": "Regulamento_Interno_e_SOP.pdf",
            "file_path": "/docs/pdf/Regulamento_Interno_e_SOP.pdf",
            "category": "RH, Operações & SOP",
            "department_author": "Recursos Humanos / Diretoria de Operações",
            "last_updated": "Agosto de 2026",
            "section_title": "1. JORNADA DE TRABALHO E ESCALA 5X2",
            "page_start": 2,
            "page_end": 3,
            "char_count": 350,
            "word_count": 55,
            "text": "A empresa adota exclusivamente a Escala 5x2 para todos os colaboradores (8h40 por dia, 1h de almoço, totalizando 44h semanais). Esta jornada é viabilizada por escalonamento preditivo por IA e self-checkouts autônomos."
        },
        {
            "chunk_id": "TEST_CHK_005",
            "file_name": "Manual_de_Perguntas_Frequentes_FAQ.pdf",
            "file_path": "/docs/pdf/Manual_de_Perguntas_Frequentes_FAQ.pdf",
            "category": "RH, Operações & Atendimento",
            "department_author": "Comunicação Corporativa / Recursos Humanos",
            "last_updated": "Agosto de 2026",
            "section_title": "PROGRAMA CLIENTE VIP CENTRAL",
            "page_start": 3,
            "page_end": 4,
            "char_count": 280,
            "word_count": 42,
            "text": "O programa possui 4 níveis: Bronze (0,5% cashback), Prata (1,0% cashback), Gold (1,5% cashback) e Diamante (2,0% cashback). Os créditos têm validade de 12 meses FIFO e liberação em 48h."
        }
    ]


@pytest.fixture(scope="module")
def temp_chroma_db():
    """Cria um diretório temporário para banco ChromaDB persistente isolado por teste."""
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_pdf_path(project_root_path: Path) -> Path:
    """Retorna o caminho para um PDF real existente na pasta docs/pdf/."""
    pdf_path = project_root_path / "docs" / "pdf" / "Guia_de_Envios_e_Entregas.pdf"
    return pdf_path


@pytest.fixture(scope="function")
def sample_text() -> str:
    """Fixture com texto bruto em Português contendo ruídos, cabeçalhos e numeração de páginas."""
    return """
MERCADO CENTRAL 24H LTDA
Página 1 de 5

1. MODALIDADES DE ENTREGA E PRAZOS
A entrega expressa funciona 24 horas por dia, 7 dias por semana no Mercado Central 24h.
O prazo de entrega é de até 3 horas para compras realizadas pelo aplicativo móvel.

Página 2 de 5
12
MERCADO CENTRAL 24H
2. FRETE GRÁTIS E BENEFÍCIOS VIP
Compras acima de R$ 250,00 contam com frete grátis para entregas padrão na região metropolitana.
Clientes VIP Diamante desfrutam de frete grátis em pedidos acima de R$ 100,00 com 2,0% de cashback.
    """
