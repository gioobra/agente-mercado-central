# Mercado Central 24h - Repositório de Documentos Fiscais, Operacionais e Pipeline RAG

Repositório corporativo contendo a documentação oficial em PDF e o pipeline de ingestão e busca RAG (*Retrieval-Augmented Generation*) para o agente assistente dos colaboradores do **Mercado Central 24h LTDA**.

---

## 📁 Estrutura do Repositório

```text
MercadoCentral/
├── .dockerignore              # Exclusões para build da imagem Docker
├── .env.example               # Modelo documentado de variáveis de ambiente
├── .gitignore                 # Configuração de arquivos ignorados pelo Git
├── AUDIT_REPORT.md            # Relatório consolidado de auditoria arquitetural e testes (Nota A+)
├── Dockerfile                 # Imagem Docker multi-stage (Python 3.12 + Streamlit)
├── docker-compose.yml         # Orquestração Docker com persistência de dados
├── PROJECT.md                 # Especificação arquitetural, roadmap e contratos do projeto
├── README.md                  # Documentação principal do repositório
├── RELATORIO_AUDITORIA.md     # Relatório consolidado de auditoria e dívida técnica
├── TEST_INFRA.md              # Documentação de infraestrutura de testes
├── TEST_READY.md              # Checklist de prontidão de testes
├── app.py                     # Interface Web Streamlit (Chat do Colaborador)
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
    │   ├── contact_catalog.py # Catálogo de contatos corporativos e roteamento inteligente
    │   ├── grounded_qa_agent.py  # Agente QA fundamentado com citações de fontes
    │   ├── hallucination_checker.py # Verificador de consistência sentencial e anti-alucinação
    │   ├── hybrid_search.py   # Busca híbrida combinada (Densa ChromaDB + Esparsa BM25)
    │   ├── multichannel_formatter.py # Formatador de respostas (Chat, E-mail, Teams/Slack)
    │   ├── rag_pdf_processor.py # Ingestão, limpeza, estruturação e chunking de PDFs
    │   ├── reranker.py        # Re-ranker RRF e fusão de pontuações de relevância
    │   └── vector_indexer.py  # Gerenciamento do banco vetorial ChromaDB e indexação
    └── tests/                 # Suíte de Testes Automatizados (326 testes)
        ├── __init__.py        # Inicialização do pacote rag.tests
        ├── conftest.py        # Fixtures compartilhadas do Pytest
        ├── test_adversarial_challenge_final.py # Testes adversariais finais de robustez
        ├── test_adversarial_tier5.py # Testes de robustez adversarial e limites
        ├── test_corporate_routing.py # Testes de roteamento de contatos corporativos
        ├── test_e2e_enhancements.py  # Testes de melhorias ponta a ponta
        ├── test_e2e_scenarios.py     # Testes integrados de cenários de negócios
        ├── test_hallucination_and_confidence.py # Testes de limiares e anti-alucinação
        ├── test_m1_adversarial.py    # Testes de consistência sentencial e grounding
        ├── test_multichannel_formatting.py # Testes de formatação multicanal
        ├── test_pdf_processor.py     # Suíte dedicada de testes do processador de PDFs
        └── test_rag_pipeline.py      # Testes unitários do pipeline RAG e reranker
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

## 🔑 Configuração de Variáveis de Ambiente (`.env`)

O projeto pode ser executado em dois modos: **Online** (com LLM Gemini) ou **Offline** (100% gratuito e local, sem custos de API).

### 1. Criar o arquivo `.env`:
```bash
cp .env.example .env
```

### 2. Configurar a Chave (Opcional):
Abra o arquivo `.env` e adicione sua chave de API caso queira o modo generativo online:
```env
GEMINI_API_KEY=sua_chave_gemini_aqui
```

### 💡 Modos de Operação do Agente:
- **Modo Online (com `GEMINI_API_KEY`)**:
  - O agente utiliza o modelo oficial `text-embedding-004` (Google GenAI) e sintetiza respostas em linguagem natural com `gemini-2.5-flash`, mantendo as citações formais `[Fonte: ..., Pág. ...]`.
  - Obtenha sua chave gratuita em: [Google AI Studio](https://aistudio.google.com/app/apikey).
- **Modo Offline / Fallback (sem chave ou offline)**:
  - Se a variável `GEMINI_API_KEY` estiver vazia ou ausente, o sistema ativa automaticamente o **MockEmbeddingFunction determinístico de 768 dimensões** e o **motor de geração extrativa com citação fática de fontes**.
  - **Zero custo, zero dependência de rede e 100% funcional.**

---

## 🐳 Execução com Docker

A forma mais simples de executar o projeto completo sem precisar configurar Python, venv ou dependências manualmente:

### Com Docker Compose (recomendado):
```bash
# Build e execução
docker compose up --build

# Para rodar em segundo plano:
docker compose up --build -d
```

### Com Docker puro:
```bash
# Build da imagem
docker build -t mercado-central-ia .

# Execução do container
docker run -p 8501:8501 --name mercado-central-ia mercado-central-ia
```

### Com chave de API do Gemini (opcional):
```bash
# Via Docker Compose
GEMINI_API_KEY=sua_chave_aqui docker compose up --build

# Via Docker puro
docker run -p 8501:8501 -e GEMINI_API_KEY=sua_chave_aqui mercado-central-ia
```

Acesse a aplicação em: **http://localhost:8501**

> **Nota:** Sem a `GEMINI_API_KEY`, o agente opera em modo extrativo de fallback (respostas geradas diretamente a partir dos trechos recuperados, sem LLM generativa).

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
- `test_pdf_processor.py`: Suíte unitária dedicada ao processador de PDFs (validação do `pdftotext`, limpeza de ruídos, marcação de seções, chunking e suporte a `pathlib.Path`).
- `test_rag_pipeline.py`: Testes unitários de importações de pacote, índice vetorial ChromaDB, validação de chunks vazios, busca híbrida (ChromaDB + BM25) e reranker RRF.
- `test_e2e_scenarios.py` / `test_e2e_enhancements.py`: Testes integrados de cenários corporativos (escala 5x2, regras do Cliente VIP Central, prazos de devolução CDC e SLAs de entrega).
- `test_adversarial_tier5.py` / `test_adversarial_challenge_final.py` / `test_m1_adversarial.py`: Testes adversariais de robustez, limites e consistência de grounding.
- `test_hallucination_and_confidence.py`: Testes de limiares de confiança e controle anti-alucinação.
- `test_corporate_routing.py`: Testes de roteamento de fallback para departamentos corporativos.
- `test_multichannel_formatting.py`: Testes de formatação de respostas multicanal (Chat, E-mail, Teams/Slack).

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

---

## 🖥️ Interface Web do Colaborador (Streamlit)

O projeto inclui uma interface web conversacional moderna, rápida e intuitiva dedicada aos colaboradores do Mercado Central 24h:

### Recursos da Interface:
- 💬 **Chat Web com Histórico de Conversas**: Múltiplas sessões de conversa com contexto contínuo e alternância instantânea entre threads.
- 🤖 **Aviso de Transparência de IA**: Identificação clara de que se trata de um assistente virtual baseado em IA generativa e RAG.
- 📚 **Visualização de Fontes**: Expander em cada resposta detalhando o documento PDF, seção e páginas exatas consultadas.
- 🧠 **Recuperação Adaptativa**: Seleção inteligente de 2 a 8 fontes conforme a complexidade da pergunta.
- 👍/👎 **Botão de Feedback**: Avaliação nativa em cada resposta da IA com feedback instantâneo.
- ⚙️ **Configurações e Filtros**:
  - Seleção de canal de formatação: Chat, E-mail Corporativo Formal, Teams / Slack.
  - Filtro por departamento/categoria de documento.
  - Toggle de boost temporal para priorizar normas recentes.

### Como Executar a Aplicação Web:

```bash
# Com ambiente virtual (desenvolvimento local)
source venv/bin/activate
streamlit run app.py

# Com Docker (recomendado para produção)
docker compose up --build
```

A aplicação será aberta automaticamente no seu navegador padrão em `http://localhost:8501`.
