"""
Suíte de Testes Unitários Dedicada ao Processador de PDFs (rag_pdf_processor.py).
Cobre limpeza de ruídos, detecção de seções lógicas, divisão de chunks,
compatibilidade com pathlib.Path, verificação de binários de sistema e tratamento de erros.
"""

import json
import os
import shutil
import pytest
from pathlib import Path

from rag.scripts.rag_pdf_processor import (
    clean_page_text,
    detect_sections,
    create_chunks_from_sections,
    extract_pdf_pages,
    process_all_pdfs,
    DOCUMENT_METADATA_MAP,
)
import rag.scripts.rag_pdf_processor as pdf_proc_module


# ============================================================================
# 1. TESTES DE VERIFICAÇÃO DE BINÁRIO DE SISTEMA (pdftotext / poppler-utils)
# ============================================================================

def test_check_pdftotext_installed():
    """Valida que a função check_pdftotext_installed existe em rag_pdf_processor.py e retorna bool correto (F6)."""
    assert hasattr(pdf_proc_module, "check_pdftotext_installed"), "rag_pdf_processor.py deve definir check_pdftotext_installed()"
    is_installed = pdf_proc_module.check_pdftotext_installed()
    assert isinstance(is_installed, bool)
    expected_binary = shutil.which("pdftotext") is not None
    assert is_installed == expected_binary


def test_check_pdftotext_installed_mocked_true_and_false(monkeypatch):
    """Valida retorno do check_pdftotext_installed ao simular presença (Path) e ausência (None) do binário."""
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pdftotext" if cmd == "pdftotext" else None)
    assert pdf_proc_module.check_pdftotext_installed() is True

    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    assert pdf_proc_module.check_pdftotext_installed() is False


def test_check_pdftotext_not_installed_raises_exact_runtime_error(monkeypatch):
    """Valida que se pdftotext não estiver instalado, RuntimeError é lançado com a mensagem exata do contrato (F6)."""
    monkeypatch.setattr(pdf_proc_module, "check_pdftotext_installed", lambda: False)
    expected_msg = (
        "O utilitário binário 'pdftotext' (pacote poppler-utils) não está instalado no sistema. "
        "Instale-o com 'sudo apt-get install poppler-utils'."
    )
    with pytest.raises(RuntimeError) as exc_info:
        extract_pdf_pages("qualquer_arquivo.pdf")
    assert str(exc_info.value) == expected_msg

    with pytest.raises(RuntimeError) as exc_info_proc:
        process_all_pdfs(".", ".")
    assert str(exc_info_proc.value) == expected_msg


def test_extract_pdf_pages_missing_file_raises_error(tmp_path: Path):
    """Garante erro explicativo (RuntimeError/FileNotFoundError) ao tentar extrair PDF inexistente."""
    non_existent_pdf = tmp_path / "arquivo_inexistente_12345.pdf"
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        extract_pdf_pages(str(non_existent_pdf))


def test_extract_pdf_pages_direct_path_object(tmp_path: Path, monkeypatch):
    """Valida aceitação direta de objeto pathlib.Path (PEP 484 Union[str, Path]) na extração."""
    fake_pdf = tmp_path / "test_doc.pdf"
    fake_pdf.write_text("%PDF-1.4 mock content")
    
    monkeypatch.setattr(pdf_proc_module, "check_pdftotext_installed", lambda: True)
    
    class MockCompletedProcess:
        returncode = 0
        stdout = "Conteúdo da Página 1\fConteúdo da Página 2\f"
        stderr = ""
        
    monkeypatch.setattr("subprocess.run", lambda cmd, capture_output, text: MockCompletedProcess())
    
    pages = extract_pdf_pages(fake_pdf)
    assert len(pages) == 2
    assert pages[0]["page_num"] == 1
    assert pages[0]["text"] == "Conteúdo da Página 1"


def test_extract_pdf_pages_filters_empty_pages(tmp_path: Path, monkeypatch):
    """Garante que páginas vazias ou contendo apenas espaços em branco no pdftotext sejam descartadas."""
    fake_pdf = tmp_path / "test_empty_pages.pdf"
    fake_pdf.write_text("%PDF-1.4 mock content")
    
    monkeypatch.setattr(pdf_proc_module, "check_pdftotext_installed", lambda: True)
    
    class MockCompletedProcess:
        returncode = 0
        stdout = "Página Válida 1\f   \n\t  \fPágina Válida 3\f"
        stderr = ""
        
    monkeypatch.setattr("subprocess.run", lambda cmd, capture_output, text: MockCompletedProcess())
    
    pages = extract_pdf_pages(fake_pdf)
    assert len(pages) == 2
    assert pages[0]["page_num"] == 1
    assert pages[1]["page_num"] == 3


def test_extract_pdf_pages_subprocess_nonzero_exit_raises_runtime_error(tmp_path: Path, monkeypatch):
    """Valida disparo de RuntimeError quando pdftotext falha com código de retorno não-zero."""
    fake_pdf = tmp_path / "corrupted.pdf"
    fake_pdf.write_text("corrupted content")
    
    monkeypatch.setattr(pdf_proc_module, "check_pdftotext_installed", lambda: True)
    
    class MockCompletedProcess:
        returncode = 1
        stdout = ""
        stderr = "Syntax Error: Command line argument invalid"
        
    monkeypatch.setattr("subprocess.run", lambda cmd, capture_output, text: MockCompletedProcess())
    
    with pytest.raises(RuntimeError) as exc_info:
        extract_pdf_pages(fake_pdf)
    assert "Syntax Error" in str(exc_info.value)


def test_extract_pdf_pages_subprocess_oserror_wrapped(tmp_path: Path, monkeypatch):
    """Valida captura e envelopamento de OSError em RuntimeError ao executar pdftotext."""
    fake_pdf = tmp_path / "test_oserror.pdf"
    fake_pdf.write_text("%PDF-1.4")
    
    monkeypatch.setattr(pdf_proc_module, "check_pdftotext_installed", lambda: True)
    
    def mock_run_raise(*args, **kwargs):
        raise OSError("Permission denied / Execution error")
        
    monkeypatch.setattr("subprocess.run", mock_run_raise)
    
    with pytest.raises(RuntimeError) as exc_info:
        extract_pdf_pages(fake_pdf)
    assert "Permission denied" in str(exc_info.value)


# ============================================================================
# 2. TESTES DE LIMPEZA DE TEXTO (clean_page_text)
# ============================================================================

def test_clean_page_text_removes_headers_footers_and_page_numbers(sample_text: str):
    """Valida remoção de cabeçalhos repetitivos, rodapés e números de página isolados."""
    cleaned = clean_page_text(sample_text)
    
    # Verifica que nomes corporativos em cabeçalhos e números de página foram removidos
    assert "MERCADO CENTRAL 24H LTDA" not in cleaned
    assert "Página 1 de 5" not in cleaned
    assert "Página 2 de 5" not in cleaned
    
    # Verifica preservação do conteúdo substantivo
    assert "1. MODALIDADES DE ENTREGA E PRAZOS" in cleaned
    assert "2. FRETE GRÁTIS E BENEFÍCIOS VIP" in cleaned
    assert "entrega expressa funciona 24 horas" in cleaned
    assert "R$ 250,00" in cleaned
    assert "VIP Diamante" in cleaned


def test_clean_page_text_whitespace_normalization():
    """Valida normalização de múltiplos espaços em branco e quebras de linha excessivas."""
    raw = "   Texto   com    múltiplos    espaços\n\n\n\n\nOutro   parágrafo.   "
    cleaned = clean_page_text(raw)
    
    assert "Texto com múltiplos espaços" in cleaned
    assert "\n\n" in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_page_text_empty_input():
    """Valida comportamento seguro ao receber string vazia ou apenas quebras de linha."""
    assert clean_page_text("") == ""
    assert clean_page_text("\n\n   \n") == ""


def test_clean_page_text_pure_noise_returns_empty():
    """Garante retorno de string vazia quando o texto contém EXCLUSIVAMENTE ruídos de cabeçalho/rodapé."""
    noise_text = "MERCADO CENTRAL 24H LTDA\nPágina 1 de 10\n42\n"
    assert clean_page_text(noise_text) == ""


def test_clean_page_text_case_insensitive_headers():
    """Valida remoção de cabeçalhos e números de página independentemente de maiúsculas/minúsculas."""
    raw = "mercado central 24h\nPÁGINA 5 DE 10\nConteúdo legítimo preservado."
    cleaned = clean_page_text(raw)
    assert "mercado central 24h" not in cleaned
    assert "PÁGINA 5 DE 10" not in cleaned
    assert cleaned == "Conteúdo legítimo preservado."


def test_clean_page_text_page_number_digit_boundary():
    """Testa limite de remoção de números isolados (<=3 dígitos removidos, >3 dígitos mantidos)."""
    raw = "999\nLinha normal\n1000\n1."
    cleaned = clean_page_text(raw)
    lines = cleaned.split("\n")
    assert "999" not in lines
    assert "1000" in lines
    assert "1." in lines


def test_clean_page_text_preserves_in_body_mentions():
    """Verifica que menções à marca no meio de frases não são removidas inadvertidamente."""
    raw = "A jornada no Mercado Central 24h é baseada na escala 5x2."
    cleaned = clean_page_text(raw)
    assert cleaned == "A jornada no Mercado Central 24h é baseada na escala 5x2."


# ============================================================================
# 3. TESTES DE DETECÇÃO REGEX DE SEÇÕES (detect_sections)
# ============================================================================

def test_detect_sections_parses_titles_and_page_ranges():
    """Valida identificação de títulos estruturais por Regex e rastreamento de páginas."""
    pages = [
        {
            "page_num": 1,
            "text": "SUMÁRIO\nEste é o sumário executivo do manual de fornecedores.\n\n1. DISPOSIÇÕES GERAIS\nRegras de compras e suprimentos na página um."
        },
        {
            "page_num": 2,
            "text": "Continuação do texto das disposições gerais na página dois.\n\n2. ALÇADAS DE APROVAÇÃO\nCompras acima de R$ 50k exigem gerente."
        }
    ]
    
    sections = detect_sections(pages)
    assert len(sections) >= 3
    
    titles = [s["title"] for s in sections]
    assert any("SUMÁRIO" in t for t in titles)
    assert any("1. DISPOSIÇÕES GERAIS" in t for t in titles)
    assert any("2. ALÇADAS DE APROVAÇÃO" in t for t in titles)
    
    # Valida rastreamento de páginas inicial e final
    disp_sec = [s for s in sections if "1. DISPOSIÇÕES GERAIS" in s["title"]][0]
    assert disp_sec["page_start"] == 1
    assert disp_sec["page_end"] == 2


def test_detect_sections_handles_various_regex_headers():
    """Testa detecção de diversos formatos de cabeçalho: N. TÍTULO, SUMÁRIO, ANEXO, etc."""
    pages = [
        {
            "page_num": 1,
            "text": "SUMÁRIO DA DOCUMENTAÇÃO\nVisão geral.\n\n1. ATENDIMENTO AO CLIENTE\nRegras do SAC.\n\nANEXO I: TABELA DE PRAZOS\nPrazo de troca CDC."
        }
    ]
    sections = detect_sections(pages)
    sec_titles = [s["title"] for s in sections]
    assert any("1. ATENDIMENTO AO CLIENTE" in t for t in sec_titles)
    assert any("ANEXO I:" in t for t in sec_titles)


def test_detect_sections_empty_pages_list():
    """Valida retorno de lista vazia ao passar lista de páginas vazia."""
    assert detect_sections([]) == []


def test_detect_sections_no_headers_fallback_title():
    """Valida atribuição do título padrão 'Introdução / Cabeçalho' quando a página não possui títulos regex."""
    pages = [{"page_num": 1, "text": "Texto sem nenhum título formatado."}]
    sections = detect_sections(pages)
    assert len(sections) == 1
    assert sections[0]["title"] == "Introdução / Cabeçalho"
    assert sections[0]["text"] == "Texto sem nenhum título formatado."


def test_detect_sections_ignores_long_header_lines():
    """Valida que linhas iniciando com padrão regex mas com >= 100 caracteres NÃO viram títulos de seção."""
    long_line = "1. " + "Esta é uma frase muito longa " * 4  # > 100 chars
    pages = [{"page_num": 1, "text": f"{long_line}\n\n2. TÍTULO VÁLIDO\nConteúdo."}]
    sections = detect_sections(pages)
    titles = [s["title"] for s in sections]
    assert long_line not in titles
    assert any("2. TÍTULO VÁLIDO" in t for t in titles)


def test_detect_sections_markdown_and_block_headers():
    """Testa detecção de títulos em formatos Markdown (# Header) e BLOCO [A-Z]:."""
    pages = [{
        "page_num": 1,
        "text": "# Título Principal\nConteúdo intro.\n\nBLOCO A: POLÍTICA DE COMPRAS\nRegras de compras."
    }]
    sections = detect_sections(pages)
    titles = [s["title"] for s in sections]
    assert "# Título Principal" in titles
    assert "BLOCO A: POLÍTICA DE COMPRAS" in titles


# ============================================================================
# 4. TESTES DE ESTRUTURA E DIVISÃO DE CHUNKS (create_chunks_from_sections)
# ============================================================================

def test_create_chunks_from_sections_under_max_chars():
    """Valida criação de chunk único para seções curtas (< max_chars)."""
    sections = [{
        "title": "1. VISÃO GERAL",
        "text": "Texto curto para teste de chunk único.",
        "page_start": 1,
        "page_end": 1,
    }]
    doc_meta = {
        "category": "Testes",
        "department_author": "QA",
        "last_updated": "Agosto de 2026"
    }
    
    chunks = create_chunks_from_sections(
        sections,
        file_name="Documento_Teste.pdf",
        file_path="/path/Documento_Teste.pdf",
        doc_meta=doc_meta,
        max_chars=1200,
        overlap_chars=200
    )
    
    assert len(chunks) == 1
    c = chunks[0]
    assert c["chunk_id"] == "Documento_Teste_CHK_001"
    assert c["file_name"] == "Documento_Teste.pdf"
    assert c["category"] == "Testes"
    assert c["department_author"] == "QA"
    assert c["section_title"] == "1. VISÃO GERAL"
    assert c["char_count"] == len("Texto curto para teste de chunk único.")
    assert c["word_count"] == 7
    assert c["page_start"] == 1
    assert c["page_end"] == 1


def test_create_chunks_from_sections_splitting_and_overlap():
    """Valida divisão de seções grandes em múltiplos chunks com sobreposição (overlap)."""
    # Cria texto longo com múltiplos parágrafos excedendo 500 caracteres
    para1 = "Parágrafo 1. " + "Conteúdo da seção de teste. " * 15
    para2 = "Parágrafo 2. " + "Continuação da explicação detalhada. " * 15
    para3 = "Parágrafo 3. " + "Conclusão da seção com mais detalhes. " * 15
    long_text = f"{para1}\n\n{para2}\n\n{para3}"
    
    sections = [{
        "title": "2. SEÇÃO LONGA DE TESTE",
        "text": long_text,
        "page_start": 2,
        "page_end": 3,
    }]
    
    chunks = create_chunks_from_sections(
        sections,
        file_name="Guia_Longo.pdf",
        file_path="/path/Guia_Longo.pdf",
        doc_meta=DOCUMENT_METADATA_MAP["Guia_de_Envios_e_Entregas.pdf"],
        max_chars=500,
        overlap_chars=100
    )
    
    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"]) > 0
        assert c["section_title"] == "2. SEÇÃO LONGA DE TESTE"
        assert c["category"] == "Logística & Delivery"
        assert c["page_start"] == 2
        assert c["page_end"] == 3


def test_create_chunks_from_sections_empty_sections():
    """Valida retorno de lista vazia ao passar seções vazias."""
    chunks = create_chunks_from_sections(
        sections=[],
        file_name="test.pdf",
        file_path="/path/test.pdf",
        doc_meta={}
    )
    assert chunks == []


def test_create_chunks_from_sections_path_object_compat(tmp_path: Path):
    """Valida passagem de objeto Path em file_path e conversão automática para str no chunk."""
    pdf_path = tmp_path / "Guia.pdf"
    sections = [{"title": "1. INTRO", "text": "Texto de teste", "page_start": 1, "page_end": 1}]
    chunks = create_chunks_from_sections(
        sections=sections,
        file_name="Guia.pdf",
        file_path=pdf_path,
        doc_meta={"category": "Geral"}
    )
    assert chunks[0]["file_path"] == str(pdf_path)


def test_create_chunks_from_sections_missing_metadata_defaults():
    """Valida atribuição correta de valores default quando doc_meta é um dicionário vazio."""
    sections = [{"title": "1. INTRO", "text": "Texto de teste", "page_start": 1, "page_end": 1}]
    chunks = create_chunks_from_sections(
        sections=sections,
        file_name="Desconhecido.pdf",
        file_path="/path/Desconhecido.pdf",
        doc_meta={}
    )
    c = chunks[0]
    assert c["category"] == "Geral"
    assert c["department_author"] == "Mercado Central 24h"
    assert c["last_updated"] == "2026"


def test_create_chunks_from_sections_exact_overlap_content():
    """Valida se o prefixo de sobreposição (overlap) entre chunks consecutivos é preservado."""
    sec_text = "Parágrafo A com dados.\n\nParágrafo B com mais detalhes extensos para forçar a divisão dos chunks em dois ou mais blocos."
    sections = [{"title": "1. TEST", "text": sec_text, "page_start": 1, "page_end": 1}]
    chunks = create_chunks_from_sections(
        sections=sections,
        file_name="Overlap.pdf",
        file_path="/path/Overlap.pdf",
        doc_meta={},
        max_chars=60,
        overlap_chars=20
    )
    if len(chunks) > 1:
        assert chunks[1]["text"][:20] in chunks[0]["text"] or chunks[0]["text"][-20:] in chunks[1]["text"]


def test_create_chunks_from_sections_sequential_chunk_ids():
    """Valida formatação sequencial dos IDs de chunks (CHK_001, CHK_002, CHK_003) entre seções."""
    sections = [
        {"title": "1. SEC ONE", "text": "Texto um", "page_start": 1, "page_end": 1},
        {"title": "2. SEC TWO", "text": "Texto dois", "page_start": 2, "page_end": 2},
    ]
    chunks = create_chunks_from_sections(
        sections=sections,
        file_name="Doc.pdf",
        file_path="/path/Doc.pdf",
        doc_meta={}
    )
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "Doc_CHK_001"
    assert chunks[1]["chunk_id"] == "Doc_CHK_002"


# ============================================================================
# 5. TESTES DE COMPATIBILIDADE PATHLIB & PIPELINE DE END-TO-END PDF
# ============================================================================

def test_pathlib_compatibility(sample_pdf_path: Path):
    """Valida aceitação de objetos pathlib.Path nas funções de extração de PDF."""
    if not sample_pdf_path.exists():
        pytest.skip(f"PDF de amostra não encontrado em {sample_pdf_path}")
    
    # Passa Path object convertido para str se a assinatura exigir ou aceitar Path
    pages = extract_pdf_pages(str(sample_pdf_path))
    assert isinstance(pages, list)
    assert len(pages) > 0
    assert "page_num" in pages[0]
    assert "text" in pages[0]


def test_process_all_pdfs_integration(project_root_path: Path, tmp_path: Path):
    """Testa execução completa do pipeline process_all_pdfs em diretório de PDFs."""
    pdf_dir = project_root_path / "docs" / "pdf"
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        pytest.skip("Diretório docs/pdf/ não possui PDFs para teste.")
    
    output_dir = tmp_path / "rag_data_output"
    process_all_pdfs(str(pdf_dir), str(output_dir))
    
    json_path = output_dir / "processed_rag_chunks.json"
    jsonl_path = output_dir / "processed_rag_chunks.jsonl"
    
    assert json_path.exists()
    assert jsonl_path.exists()
    assert json_path.stat().st_size > 0
    assert jsonl_path.stat().st_size > 0


def test_process_all_pdfs_empty_input_directory(tmp_path: Path, monkeypatch):
    """Valida que o processamento de um diretório sem PDFs gera arquivos JSON/JSONL vazios sem erros."""
    monkeypatch.setattr(pdf_proc_module, "check_pdftotext_installed", lambda: True)
    empty_in = tmp_path / "empty_in"
    empty_in.mkdir()
    out_dir = tmp_path / "empty_out"
    
    process_all_pdfs(empty_in, out_dir)
    
    json_p = out_dir / "processed_rag_chunks.json"
    jsonl_p = out_dir / "processed_rag_chunks.jsonl"
    
    assert json_p.exists()
    assert jsonl_p.exists()
    assert json_p.read_text(encoding="utf-8").strip() == "[]"
    assert jsonl_p.read_text(encoding="utf-8").strip() == ""


def test_process_all_pdfs_accepts_path_objects(tmp_path: Path, monkeypatch):
    """Valida aceite de objetos Path nos parâmetros input_dir e output_dir em process_all_pdfs."""
    monkeypatch.setattr(pdf_proc_module, "check_pdftotext_installed", lambda: True)
    in_dir = tmp_path / "path_obj_in"
    in_dir.mkdir()
    pdf_file = in_dir / "Test.pdf"
    pdf_file.write_text("%PDF-1.4 mock")
    
    class MockCompletedProcess:
        returncode = 0
        stdout = "Conteúdo de teste.\f"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda cmd, capture_output, text: MockCompletedProcess())
    
    out_dir = tmp_path / "path_obj_out"
    process_all_pdfs(in_dir, out_dir)
    assert (out_dir / "processed_rag_chunks.json").exists()


def test_process_all_pdfs_unmapped_file_fallback_metadata(tmp_path: Path, monkeypatch):
    """Valida fallback de metadados quando um arquivo PDF não está no DOCUMENT_METADATA_MAP."""
    monkeypatch.setattr(pdf_proc_module, "check_pdftotext_installed", lambda: True)
    
    in_dir = tmp_path / "custom_pdf_in"
    in_dir.mkdir()
    custom_pdf = in_dir / "Manual_Novo_Desconhecido.pdf"
    custom_pdf.write_text("%PDF-1.4 mock")
    
    class MockCompletedProcess:
        returncode = 0
        stdout = "Conteúdo de teste do PDF customizado.\f"
        stderr = ""
        
    monkeypatch.setattr("subprocess.run", lambda cmd, capture_output, text: MockCompletedProcess())
    
    out_dir = tmp_path / "custom_pdf_out"
    process_all_pdfs(in_dir, out_dir)
    
    chunks = json.loads((out_dir / "processed_rag_chunks.json").read_text(encoding="utf-8"))
    assert len(chunks) == 1
    assert chunks[0]["category"] == "Geral"
    assert chunks[0]["department_author"] == "Mercado Central 24h"
    assert chunks[0]["file_name"] == "Manual_Novo_Desconhecido.pdf"


def test_document_metadata_map_completeness():
    """Valida que DOCUMENT_METADATA_MAP possui todos os 8 PDFs corporativos com metadados obrigatórios."""
    expected_pdfs = [
        "Regulamento_Interno_e_SOP.pdf",
        "Manual_de_Fornecedores_e_Politica_de_Compras.pdf",
        "Manual_de_Perguntas_Frequentes_FAQ.pdf",
        "Politica_Integrada_de_Atendimento_Trocas_Devolucoes_e_Privacidade.pdf",
        "Guia_de_Envios_e_Entregas.pdf",
        "Termos_e_Condicoes_de_Uso.pdf",
        "Politica_de_Privacidade_LGPD.pdf",
        "Politica_de_Reembolso_e_Devolucoes.pdf",
    ]
    assert len(DOCUMENT_METADATA_MAP) == 8
    for pdf_name in expected_pdfs:
        assert pdf_name in DOCUMENT_METADATA_MAP
        meta = DOCUMENT_METADATA_MAP[pdf_name]
        for required_key in ["category", "department_author", "default_title", "last_updated"]:
            assert required_key in meta
            assert len(meta[required_key].strip()) > 0
