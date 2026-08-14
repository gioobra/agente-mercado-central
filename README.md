# Mercado Central 24h - Repositório de Documentos Fiscais, Operacionais e Pipeline RAG

Repositório corporativo contendo a documentação oficial em PDF e o pipeline de ingestão e busca RAG (*Retrieval-Augmented Generation*) para o agente assistente dos colaboradores do **Mercado Central 24h LTDA**.

---

## 📁 Estrutura do Repositório

```text
MercadoCentral/
├── .gitignore                 # Configuração de arquivos e diretórios ignorados pelo Git
├── PROJECT.md                 # Especificação arquitetural, roadmap e contratos do projeto
├── README.md                  # Documentação principal do repositório
├── RELATORIO_AUDITORIA.md     # Relatório consolidado de auditoria e dívida técnica
├── pytest.ini                 # Configurações de execução do Pytest
├── requirements.txt           # Especificação de dependências Python do projeto
├── docs/                      # Documentação Oficial da Empresa
│   └── pdf/                   # Documentos oficiais e normativas corporativas em PDF (8 arquivos)
│       ├── Guia_de_Envios_e_Entregas.pdf
│       ├── Manual_de_Fornecedores_e_Politica_de_Compras.pdf
│       ├── Manual_de_Perguntas_Frequentes_FAQ.pdf
│       ├── Politica_Integrada_de_Atendimento_Trocas_Devolucoes_e_Privacidade.pdf
│       ├── Politica_de_Privacidade_LGPD.pdf
│       ├── Politica_de_Reembolso_e_Devolucoes.pdf
│       ├── Regulamento_Interno_e_SOP.pdf
│       └── Termos_e_Condicoes_de_Uso.pdf
└── rag/                       # Módulo RAG para Agente de Colaboradores
    ├── data/                  # Chunks fatiados e metadados para busca vetorial
    │   ├── processed_rag_chunks.json
    │   └── processed_rag_chunks.jsonl
    ├── scripts/               # Scripts Python do Pipeline RAG (Pacote Python)
    │   ├── __init__.py        # Inicialização do pacote rag.scripts e declaração de __all__
    │   ├── grounded_qa_agent.py  # Agente QA fundamentado com citações de fontes
    │   ├── hybrid_search.py   # Busca híbrida combinada (Densa ChromaDB + Esparsa BM25)
    │   ├── rag_pdf_processor.py # Ingestão, limpeza, estruturação e chunking de PDFs
    │   ├── reranker.py        # Re-ranker RRF e fusão de pontuações de relevância
    │   └── vector_indexer.py  # Gerenciamento do banco vetorial ChromaDB e indexação
    └── tests/                 # Suíte de Testes Automatizados
        ├── __init__.py        # Inicialização do pacote rag.tests
        ├── conftest.py        # Fixtures compartilhadas do Pytest
        ├── test_e2e_scenarios.py # Testes integrados de cenários de negócios ponta a ponta
        ├── test_pdf_processor.py # Suíte dedicada de testes unitários do processador de PDFs
        └── test_rag_pipeline.py  # Testes unitários do pipeline RAG, indexador e reranker
```

---

## 📋 Pré-requisitos de Sistema

Antes de configurar o ambiente Python, certifique-se de que o sistema atende aos seguintes pré-requisitos:

1. **Python 3.10+** (recomendado Python 3.10 ou superior).
2. **poppler-utils (`pdftotext`)**: O script de extração de PDFs (`rag_pdf_processor.py`) utiliza o utilitário binário do sistema `pdftotext` para conversão com fidelidade de layout.

### Instalação do `poppler-utils`:
- **Linux (Ubuntu / Debian / WSL)**:
  ```bash
  sudo apt-get update && sudo apt-get install -y poppler-utils
  ```
- **macOS (Homebrew)**:
  ```bash
  brew install poppler
  ```
- **Verificação**:
  ```bash
  pdftotext -v
  ```

---

## ⚙️ Configuração do Ambiente Virtual e Instalação

1. **Criar o Ambiente Virtual (`venv`)**:
   ```bash
   python3 -m venv venv
   ```

2. **Ativar o Ambiente Virtual**:
   - Linux / macOS:
     ```bash
     source venv/bin/activate
     ```
   - Windows (PowerShell):
     ```cmd
     .\venv\Scripts\Activate.ps1
     ```

3. **Instalar Dependências**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🧪 Execução dos Testes Automatizados

A suíte de testes do repositório garante a cobertura de integridade de ingestão de PDFs, indexação no ChromaDB, busca híbrida, reranking e cenários operacionais de negócio.

Para executar todos os testes automatizados com saída detalhada:

```bash
venv/bin/pytest rag/tests/ -v
```

Ou com o ambiente virtual ativado:
```bash
pytest rag/tests/ -v
```

### Componentes de Teste:
- `rag/tests/test_pdf_processor.py`: Suíte unitária dedicada ao processador de PDFs (validação do `pdftotext`, limpeza de ruídos, marcação de seções, chunking e suporte a `pathlib.Path`).
- `rag/tests/test_rag_pipeline.py`: Testes unitários de importações de pacote, índice vetorial ChromaDB, validação de chunks vazios, busca híbrida (ChromaDB + BM25) e reranker RRF.
- `rag/tests/test_e2e_scenarios.py`: Testes integrados de cenários corporativos (escala 5x2, regras do Cliente VIP Central, prazos de devolução CDC e SLAs de entrega).

---

## ⚡ Execução do Pipeline RAG e Componentes

Os scripts em `rag/scripts/` formam um pacote modular. Você pode executar cada etapa individualmente a partir da raiz do repositório:

1. **Ingestão e Processamento de PDFs**:
   Reprocessa os PDFs corporativos em `docs/pdf/` e gera os chunks fatiados com metadados em `rag/data/`:
   ```bash
   python3 -m rag.scripts.rag_pdf_processor
   ```

2. **Indexação Vetorial (ChromaDB)**:
   Carrega os chunks processados e inicializa o índice vetorial no ChromaDB:
   ```bash
   python3 -m rag.scripts.vector_indexer
   ```

3. **Busca Híbrida (Dense + Sparse BM25)**:
   Executa a recuperação combinada por vetores e palavras-chave BM25:
   ```bash
   python3 -m rag.scripts.hybrid_search
   ```

4. **Re-ranking e Fusão Recíproca (RRF)**:
   Aplica algoritmos de reordenação e pontuação ponderada nos resultados recuperados:
   ```bash
   python3 -m rag.scripts.reranker
   ```

5. **Agente de Respostas Fundamentadas (Grounded QA Agent)**:
   Agente assistente para responder dúvidas de colaboradores com citação de fontes corporativas:
   ```bash
   python3 -m rag.scripts.grounded_qa_agent
   ```
