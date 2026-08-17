# ==============================================================================
# Mercado Central 24h — Dockerfile
# Assistente Virtual IA (RAG) com interface Streamlit
# ==============================================================================

# --------------- Stage 1: Builder (instala dependências) ---------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Instala dependências de sistema necessárias para compilação
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --------------- Stage 2: Runtime (imagem final enxuta) ---------------
FROM python:3.12-slim AS runtime

LABEL maintainer="gioobra"
LABEL description="Mercado Central 24h — Assistente Virtual IA (RAG + Streamlit)"

WORKDIR /app

# Instala dependências de sistema em runtime (poppler-utils para pdftotext)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copia pacotes Python do builder
COPY --from=builder /install /usr/local

# Copia código fonte do projeto
COPY . .

# Cria diretório para o banco vetorial (ChromaDB) em runtime
RUN mkdir -p /app/rag/data/vector_store

# Variáveis de ambiente do Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expõe a porta do Streamlit
EXPOSE 8501

# Healthcheck para monitoramento do container
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de execução padrão
CMD ["streamlit", "run", "app.py"]
