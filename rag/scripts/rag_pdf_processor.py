#!/usr/bin/env python3
"""
RAG PDF Processor - Mercado Central 24h
Ingestão, Limpeza de Ruídos, Chunking Lógico e Atribuição de Metadados para RAG.
"""

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("RAGPDFProcessor")

__all__ = [
    "DOCUMENT_METADATA_MAP",
    "check_pdftotext_installed",
    "clean_page_text",
    "detect_sections",
    "create_chunks_from_sections",
    "extract_pdf_pages",
    "process_all_pdfs",
]

# Mapeamento de Metadados de Domínio por Nome de Arquivo
DOCUMENT_METADATA_MAP: Dict[str, Dict[str, str]] = {
    "Regulamento_Interno_e_SOP.pdf": {
        "category": "RH, Operações & SOP",
        "department_author": "Recursos Humanos / Diretoria de Operações",
        "default_title": "Regulamento Interno e Manual de Procedimentos Operacionais (SOP)",
        "last_updated": "Agosto de 2026",
    },
    "Manual_de_Fornecedores_e_Politica_de_Compras.pdf": {
        "category": "Compras & Suprimentos",
        "department_author": "Compras e Suprimentos",
        "default_title": "Manual de Fornecedores e Política de Compras",
        "last_updated": "Agosto de 2026",
    },
    "Manual_de_Perguntas_Frequentes_FAQ.pdf": {
        "category": "RH, Operações & Atendimento",
        "department_author": "Comunicação Corporativa / Recursos Humanos",
        "default_title": "Manual de Perguntas Frequentes (FAQ)",
        "last_updated": "Agosto de 2026",
    },
    "Politica_Integrada_de_Atendimento_Trocas_Devolucoes_e_Privacidade.pdf": {
        "category": "Atendimento, CDC & Privacidade",
        "department_author": "Diretoria de Operações e Relacionamento",
        "default_title": "Política Integrada de Atendimento, Trocas, Devoluções e Privacidade",
        "last_updated": "Agosto de 2026",
    },
    "Guia_de_Envios_e_Entregas.pdf": {
        "category": "Logística & Delivery",
        "department_author": "Logística & Operações Digitais",
        "default_title": "Guia de Envios e Entregas (Delivery & E-commerce)",
        "last_updated": "Agosto de 2026",
    },
    "Termos_e_Condicoes_de_Uso.pdf": {
        "category": "Jurídico & Fidelidade",
        "department_author": "Diretoria Comercial / Jurídico",
        "default_title": "Termos e Condições de Uso da Plataforma e Regulamento Cliente VIP",
        "last_updated": "Agosto de 2026",
    },
    "Politica_de_Privacidade_LGPD.pdf": {
        "category": "LGPD & Proteção de Dados",
        "department_author": "Encarregado de Proteção de Dados (DPO)",
        "default_title": "Política de Privacidade e Proteção de Dados (LGPD)",
        "last_updated": "Agosto de 2026",
    },
    "Politica_de_Reembolso_e_Devolucoes.pdf": {
        "category": "Atendimento & CDC",
        "department_author": "Atendimento ao Cliente (SAC) / Financeiro",
        "default_title": "Política de Trocas, Reembolso e Devoluções",
        "last_updated": "Agosto de 2026",
    },
}


def check_pdftotext_installed() -> bool:
    """Verifica se o utilitário binário pdftotext (poppler-utils) está disponível no PATH do SO."""
    return shutil.which("pdftotext") is not None


def extract_pdf_pages(pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Extrai o texto do PDF página a página usando pdftotext."""
    if not check_pdftotext_installed():
        raise RuntimeError(
            "O utilitário binário 'pdftotext' (pacote poppler-utils) não está instalado no sistema. Instale-o com 'sudo apt-get install poppler-utils'."
        )
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_file}")

    try:
        cmd: List[str] = ["pdftotext", str(pdf_file), "-"]
        res = subprocess.run(cmd, capture_output=True, text=True)
    except (subprocess.SubprocessError, OSError) as e:
        logger.error(f"Falha ao executar pdftotext em {pdf_file}: {e}")
        raise RuntimeError(f"Erro ao ler PDF {pdf_file}: {e}") from e

    if res.returncode != 0:
        logger.error(f"pdftotext retornou código de erro {res.returncode} para {pdf_file}: {res.stderr}")
        raise RuntimeError(f"Erro ao ler PDF {pdf_file}: {res.stderr}")

    # Form-feed (\f) separa as páginas no pdftotext
    raw_pages: List[str] = res.stdout.split("\f")
    pages: List[Dict[str, Any]] = []
    for idx, page_str in enumerate(raw_pages, start=1):
        if page_str.strip():
            pages.append({"page_num": idx, "text": page_str})
    return pages


def clean_page_text(text: str) -> str:
    """Remove ruídos de cabeçalho, rodapé, numeração de páginas e espaços duplicados."""
    lines: List[str] = text.split("\n")
    cleaned_lines: List[str] = []

    for line in lines:
        stripped: str = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        if re.match(r"^(MERCADO CENTRAL 24H|MERCADO CENTRAL 24H LTDA)$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^Página \d+( de \d+)?$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\d+$", stripped) and len(stripped) <= 3:
            continue

        normalized_line: str = re.sub(r"[ \t]+", " ", stripped)
        cleaned_lines.append(normalized_line)

    cleaned_text: str = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def detect_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Agrupa o texto por seções/subseções estruturais mantendo o rastreamento da página."""
    sections: List[Dict[str, Any]] = []
    current_section_title: str = "Introdução / Cabeçalho"
    current_lines: List[str] = []
    current_start_page: int = 1
    current_end_page: int = 1

    section_pattern = re.compile(
        r"^(?:[0-9]+\.|\b[0-9]+\.[0-9]+\.|\bSUMÁRIO\b|\bÍNDICE\b|\bANEXO\b|\bBLOCO [A-Z]:|#{1,4}\s+).*$",
        re.IGNORECASE,
    )

    for p in pages:
        page_num: int = p["page_num"]
        cleaned_text: str = clean_page_text(p["text"])
        lines: List[str] = cleaned_text.split("\n")

        for line in lines:
            line_str: str = line.strip()
            if not line_str:
                if current_lines:
                    current_lines.append("")
                continue

            if section_pattern.match(line_str) and len(line_str) < 100:
                if current_lines:
                    text_block: str = "\n".join(current_lines).strip()
                    if text_block:
                        sections.append(
                            {
                                "title": current_section_title,
                                "text": text_block,
                                "page_start": current_start_page,
                                "page_end": current_end_page,
                            }
                        )
                current_section_title = line_str
                current_lines = [line_str]
                current_start_page = page_num
                current_end_page = page_num
            else:
                current_lines.append(line_str)
                current_end_page = page_num

    if current_lines:
        text_block = "\n".join(current_lines).strip()
        if text_block:
            sections.append(
                {
                    "title": current_section_title,
                    "text": text_block,
                    "page_start": current_start_page,
                    "page_end": current_end_page,
                }
            )

    return sections


def _build_chunk_dict(
    file_name: str,
    file_path: Union[str, Path],
    doc_meta: Optional[Dict[str, Any]],
    sec_title: str,
    p_start: int,
    p_end: int,
    text: str,
    chunk_counter: int,
) -> Dict[str, Any]:
    """Função auxiliar DRY para construção padronizada de dicionário de chunk."""
    meta: Dict[str, Any] = doc_meta or {}
    base_name: str = Path(file_name).stem
    clean_text: str = text.strip()
    return {
        "chunk_id": f"{base_name}_CHK_{chunk_counter:03d}",
        "file_name": file_name,
        "file_path": str(file_path),
        "category": meta.get("category", "Geral"),
        "department_author": meta.get("department_author", "Mercado Central 24h"),
        "last_updated": meta.get("last_updated", "2026"),
        "section_title": sec_title,
        "page_start": p_start,
        "page_end": p_end,
        "char_count": len(clean_text),
        "word_count": len(clean_text.split()),
        "text": clean_text,
    }


def create_chunks_from_sections(
    sections: List[Dict[str, Any]],
    file_name: str,
    file_path: Union[str, Path],
    doc_meta: Optional[Dict[str, Any]] = None,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Dict[str, Any]]:
    """Subdivide seções grandes em chunks menores mantendo integridade de sentenças e metadados."""
    chunks: List[Dict[str, Any]] = []
    chunk_counter: int = 1
    safe_doc_meta: Dict[str, Any] = doc_meta or {}

    for sec in sections:
        sec_title: str = sec["title"]
        sec_text: str = sec["text"]
        p_start: int = sec["page_start"]
        p_end: int = sec["page_end"]

        if len(sec_text) <= max_chars:
            chunks.append(
                _build_chunk_dict(
                    file_name=file_name,
                    file_path=file_path,
                    doc_meta=safe_doc_meta,
                    sec_title=sec_title,
                    p_start=p_start,
                    p_end=p_end,
                    text=sec_text,
                    chunk_counter=chunk_counter,
                )
            )
            chunk_counter += 1
        else:
            paragraphs: List[str] = sec_text.split("\n\n")
            expanded_paragraphs: List[str] = []
            target_sub_len = (
                max(1, max_chars - overlap_chars - 2)
                if max_chars > overlap_chars + 2
                else max(1, max_chars - overlap_chars)
            )
            for p in paragraphs:
                if len(p) <= max_chars:
                    expanded_paragraphs.append(p)
                else:
                    start = 0
                    while start < len(p):
                        sub_p = p[start : start + target_sub_len]
                        if sub_p.strip():
                            expanded_paragraphs.append(sub_p.strip())
                        start += target_sub_len

            current_chunk_text: str = ""

            for p in expanded_paragraphs:
                if len(current_chunk_text) + len(p) + 2 <= max_chars:
                    if current_chunk_text:
                        current_chunk_text += "\n\n" + p
                    else:
                        current_chunk_text = p
                else:
                    if current_chunk_text:
                        chunks.append(
                            _build_chunk_dict(
                                file_name=file_name,
                                file_path=file_path,
                                doc_meta=safe_doc_meta,
                                sec_title=sec_title,
                                p_start=p_start,
                                p_end=p_end,
                                text=current_chunk_text,
                                chunk_counter=chunk_counter,
                            )
                        )
                        chunk_counter += 1
                        overlap_prefix: str = (
                            current_chunk_text[-overlap_chars:]
                            if len(current_chunk_text) > overlap_chars
                            else ""
                        )
                        if overlap_prefix and len(overlap_prefix) + len(p) + 2 <= max_chars:
                            current_chunk_text = (overlap_prefix + "\n\n" + p).strip()
                        else:
                            allowed_overlap_len = max(0, max_chars - len(p) - 2)
                            trimmed_overlap = overlap_prefix[-allowed_overlap_len:] if allowed_overlap_len > 0 else ""
                            if trimmed_overlap:
                                current_chunk_text = (trimmed_overlap + "\n\n" + p).strip()
                            else:
                                current_chunk_text = p
                    else:
                        current_chunk_text = p

            if current_chunk_text:
                chunks.append(
                    _build_chunk_dict(
                        file_name=file_name,
                        file_path=file_path,
                        doc_meta=safe_doc_meta,
                        sec_title=sec_title,
                        p_start=p_start,
                        p_end=p_end,
                        text=current_chunk_text,
                        chunk_counter=chunk_counter,
                    )
                )
                chunk_counter += 1

    return chunks


def process_all_pdfs(input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    if not check_pdftotext_installed():
        raise RuntimeError(
            "O utilitário binário 'pdftotext' (pacote poppler-utils) não está instalado no sistema. Instale-o com 'sudo apt-get install poppler-utils'."
        )

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    pdf_files: List[Path] = sorted(list(input_path.glob("*.pdf")))

    logger.info(f"🔍 Diretório de entrada PDF: {input_path}")
    logger.info(f"📁 Encontrados {len(pdf_files)} arquivos PDF.\n")

    output_path.mkdir(parents=True, exist_ok=True)
    all_chunks: List[Dict[str, Any]] = []

    for pdf_path in pdf_files:
        file_name: str = pdf_path.name
        logger.info(f"📄 Processando: {file_name}")

        doc_meta: Dict[str, Any] = DOCUMENT_METADATA_MAP.get(
            file_name,
            {
                "category": "Geral",
                "department_author": "Mercado Central 24h",
                "default_title": file_name.replace(".pdf", ""),
                "last_updated": "2026",
            },
        )

        try:
            pages = extract_pdf_pages(pdf_path)
            sections = detect_sections(pages)
            chunks = create_chunks_from_sections(sections, file_name, pdf_path.resolve(), doc_meta)

            logger.info(f"   ├─ Páginas: {len(pages)} | Seções: {len(sections)} | Chunks: {len(chunks)}")
            all_chunks.extend(chunks)
        except (subprocess.SubprocessError, OSError, FileNotFoundError, RuntimeError) as e:
            logger.error(f"Erro ao processar PDF {pdf_path}: {e}")
            raise

    json_path = output_path / "processed_rag_chunks.json"
    jsonl_path = output_path / "processed_rag_chunks.jsonl"

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error(f"Erro de E/S ao salvar chunks em {output_path}: {e}")
        raise

    logger.info("\n" + "=" * 60)
    logger.info("✅ PROCESSAMENTO RAG CONCLUÍDO COM SUCESSO!")
    logger.info(f"📊 Total de PDFs Processados: {len(pdf_files)}")
    logger.info(f"🧩 Total de Chunks Criados: {len(all_chunks)}")
    logger.info(f"💾 Arquivos Gerados em: {output_path}")
    logger.info(f"   - {json_path}")
    logger.info(f"   - {jsonl_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent

    input_pdf_dir = project_root / "docs" / "pdf"
    output_rag_dir = project_root / "rag" / "data"

    process_all_pdfs(input_pdf_dir, output_rag_dir)
