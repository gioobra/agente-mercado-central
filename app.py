#!/usr/bin/env python3
"""
Mercado Central 24h — Interface Web Streamlit
Assistente Virtual de Inteligência Artificial para Colaboradores (RAG Grounded QA).
Inclui gerenciamento de histórico de conversas em sessões múltiplas e respostas limpas e legíveis.
"""

import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from rag.scripts.grounded_qa_agent import GroundedQAAgent
from rag.scripts.hybrid_search import HybridSearcher
from rag.scripts.reranker import ReRanker
from rag.scripts.vector_indexer import VectorIndexer

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("StreamlitApp")

# Configuração da página Streamlit
st.set_page_config(
    page_title="Mercado Central 24h — Assistente IA",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados para design corporativo compacto, elegante e legível
st.markdown(
    """
    <style>
    /* Estilos do Cabeçalho Principal */
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 0.9rem 1.4rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .main-header h1 {
        color: #ffffff;
        margin: 0;
        font-size: 1.35rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .main-header p {
        color: #d8f3dc;
        margin: 0.15rem 0 0 0;
        font-size: 0.82rem;
    }
    .ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background-color: #e8f5e9;
        color: #1b4332;
        padding: 0.15rem 0.55rem;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid #c8e6c9;
        margin-top: 0.25rem;
    }

    /* Otimização Compacta da Barra Lateral (Sidebar) */
    section[data-testid="stSidebar"] {
        width: 320px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.4rem !important;
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 1.0rem !important;
        margin: 0 0 0.1rem 0 !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] h4 {
        font-size: 0.82rem !important;
        margin: 0.25rem 0 0.1rem 0 !important;
        padding: 0 !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] hr {
        margin: 0.25rem 0 !important;
        border-color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stAlert {
        padding: 0.35rem 0.5rem !important;
        font-size: 0.72rem !important;
        line-height: 1.2 !important;
        margin-bottom: 0.2rem !important;
    }
    section[data-testid="stSidebar"] label {
        font-size: 0.76rem !important;
        margin-bottom: 0.05rem !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.15rem !important;
    }
    section[data-testid="stSidebar"] .stCheckbox {
        margin-top: 0.05rem !important;
        margin-bottom: 0.15rem !important;
        font-size: 0.76rem !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        padding: 0.18rem 0.4rem !important;
        font-size: 0.72rem !important;
        min-height: 1.7rem !important;
        line-height: 1.15 !important;
        border-radius: 5px !important;
        margin-bottom: 0.1rem !important;
    }

    /* Histórico de Sessões Ativas */
    .session-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.75rem;
        cursor: pointer;
        margin-bottom: 0.1rem;
    }

    /* Respostas e Fontes */
    .response-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="🔄 Inicializando Base de Conhecimento RAG do Mercado Central 24h...")
def get_rag_agent() -> GroundedQAAgent:
    """Carrega e inicializa o pipeline RAG completo (VectorStore + HybridSearch + Reranker + Agent)."""
    project_root = Path(__file__).parent.resolve()
    chunks_json_path = project_root / "rag" / "data" / "processed_rag_chunks.json"
    vector_db_path = project_root / "rag" / "data" / "vector_store"

    logger.info("Inicializando componentes do RAG para o Streamlit...")

    indexer = VectorIndexer(
        db_path=str(vector_db_path),
        collection_name="mercado_central_rag",
        use_mock=False,
    )
    indexer.index_chunks(str(chunks_json_path))

    searcher = HybridSearcher(
        vector_indexer=indexer,
        chunks_data=str(chunks_json_path),
        alpha=0.5,
    )

    reranker = ReRanker(method="hybrid_fusion")

    agent = GroundedQAAgent(
        indexer=indexer,
        searcher=searcher,
        reranker=reranker,
        confidence_threshold=0.35,
    )
    logger.info("Pipeline RAG carregado com sucesso!")
    return agent


def sanitize_display_markdown(text: str) -> str:
    """Remove quaisquer tags HTML residuais para garantir renderização limpa no chat."""
    if not text:
        return ""
    # Converte strong/b para markdown
    text = re.sub(r"<\s*strong\s*>(.*?)<\s*/\s*strong\s*>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*b\s*>(.*?)<\s*/\s*b\s*>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove qualquer outra tag HTML residual
    text = re.sub(r"<[^>]+>", "", text)
    return text


def create_new_session() -> str:
    """Cria uma nova sessão de conversa vazia e a define como ativa."""
    session_id = str(uuid.uuid4())[:8]
    initial_message = {
        "role": "assistant",
        "content": (
            "Olá, colaborador! 👋 Sou o **Assistente Virtual de Inteligência Artificial** do Mercado Central 24h.\n\n"
            "Estou aqui para tirar dúvidas sobre normas internas, jornada e escala 5x2, turnos T1-T5, benefícios, "
            "procedimentos de atendimento, compras e privacidade com base estrita nos nossos documentos corporativos.\n\n"
            "Como posso ajudar você hoje?"
        ),
        "sources": [],
        "timestamp": datetime.now().strftime("%H:%M"),
        "feedback": None,
        "is_fallback": False,
    }
    st.session_state.conversations[session_id] = {
        "id": session_id,
        "title": "Nova Conversa",
        "created_at": datetime.now().strftime("%d/%m %H:%M"),
        "messages": [initial_message],
    }
    st.session_state.active_session_id = session_id
    return session_id


# Inicialização de Estado de Conversas / Sessões Múltiplas
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_session_id" not in st.session_state or st.session_state.active_session_id not in st.session_state.conversations:
    create_new_session()

if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = {}

active_session = st.session_state.conversations[st.session_state.active_session_id]

# Sidebar — Histórico de Conversas, Configurações e Dúvidas Frequentes
with st.sidebar:
    st.markdown("### 🛒 Mercado Central 24h")
    st.caption("🤖 **Assistente Virtual de IA • Base Oficial RAG**")

    # Botão de Nova Conversa
    if st.button("➕ Nova Conversa", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()

    # Histórico de Sessões Anteriores
    st.markdown("---")
    st.markdown("#### 🗂️ Histórico de Conversas")

    session_ids = list(st.session_state.conversations.keys())
    for s_id in reversed(session_ids):
        conv = st.session_state.conversations[s_id]
        is_current = s_id == st.session_state.active_session_id
        title = conv.get("title", "Conversa")
        # Ícone de destaque para a conversa ativa
        prefix = "💬 " if not is_current else "👉 "
        btn_label = f"{prefix}{title[:22]}"

        col_nav, col_del = st.columns([5, 1])
        with col_nav:
            if st.button(btn_label, key=f"nav_{s_id}", use_container_width=True, disabled=is_current):
                st.session_state.active_session_id = s_id
                st.rerun()
        with col_del:
            if len(session_ids) > 1:
                if st.button("✕", key=f"del_{s_id}", help="Excluir conversa"):
                    del st.session_state.conversations[s_id]
                    if st.session_state.active_session_id == s_id:
                        st.session_state.active_session_id = list(st.session_state.conversations.keys())[0]
                    st.rerun()

    st.markdown("---")
    st.markdown("#### ⚙️ Configurações")

    channel_option = st.selectbox(
        "Formato da Resposta:",
        options=["chat", "email", "teams_slack"],
        format_func=lambda x: {
            "chat": "💬 Chat Conversacional",
            "email": "✉️ E-mail Formal",
            "teams_slack": "👥 Teams / Slack",
        }.get(x, x),
        index=0,
        help="Adapta a estrutura da resposta ao canal desejado.",
    )

    category_filter = st.selectbox(
        "Filtrar Categoria:",
        options=[
            "Todas",
            "RH, Operações & Atendimento",
            "Financeiro & Compras",
            "Jurídico & LGPD",
            "Logística & Atendimento",
            "Institucional & Governança",
        ],
        index=0,
        help="Restringe a busca a documentos de uma área específica.",
    )

    boost_recency = st.checkbox(
        "Priorizar Documentos Recentes",
        value=True,
        help="Aplica boost temporal para documentos com atualizações mais recentes.",
    )

    st.markdown("---")
    st.markdown("#### 💡 Dúvidas Frequentes")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("🗓️ Escala 5x2", use_container_width=True):
            st.session_state.pending_query = "Como funciona a jornada de trabalho na escala 5x2 e os turnos T1 a T5?"
        if st.button("🚚 Entregas", use_container_width=True):
            st.session_state.pending_query = "Qual o prazo da entrega expressa e o valor para frete grátis?"

    with col_q2:
        if st.button("💎 VIP Diamante", use_container_width=True):
            st.session_state.pending_query = "Quais são as regras e porcentagens de cashback do Cliente VIP Diamante?"
        if st.button("🛡️ DPO / LGPD", use_container_width=True):
            st.session_state.pending_query = "Qual o contato do DPO para assuntos de privacidade e LGPD?"

# Cabeçalho Principal da Interface
st.markdown(
    """
    <div class="main-header">
        <h1>🛒 Mercado Central 24h — Assistente do Colaborador</h1>
        <p>Base de Conhecimento RAG Corporativa • Escala 5x2 • Políticas Internas • Atendimento 24/7</p>
        <div class="ai-badge">🤖 Agente de IA • Fundamentação Estrita em Documentos Oficiais</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Renderização das Mensagens da Sessão Ativa
current_messages = active_session.get("messages", [])

for idx, message in enumerate(current_messages):
    role = message["role"]
    avatar = "👤" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):
        clean_content = sanitize_display_markdown(message["content"])
        st.markdown(clean_content)

        # Exibição de Fontes / Documentos Consultados
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"📚 Fontes e Documentos Consultados ({len(sources)})", expanded=False):
                for s_idx, src in enumerate(sources, start=1):
                    file_name = src.get("file_name", "Documento")
                    section = src.get("section_title", "Seção")
                    page_range = src.get("page_range", "")
                    score = src.get("score")
                    score_info = f" • Relevância: `{score:.2f}`" if score is not None else ""

                    st.markdown(
                        f"**{s_idx}. 📄 `{file_name}`**  \n"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;📌 **Seção:** {section}  \n"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;📖 **Página(s):** {page_range}{score_info}"
                    )

        # Botão de Feedback para respostas do assistente
        if role == "assistant" and idx > 0:
            feedback_key = f"fb_{active_session['id']}_{idx}"
            col_fb1, col_fb2 = st.columns([1, 4])
            with col_fb1:
                current_feedback = st.session_state.feedbacks.get(feedback_key)
                selected_feedback = st.feedback(
                    options="thumbs",
                    key=feedback_key,
                )
                if selected_feedback is not None and selected_feedback != current_feedback:
                    st.session_state.feedbacks[feedback_key] = selected_feedback
                    st.toast("Obrigado pelo seu feedback!", icon="✨")

            with col_fb2:
                saved_fb = st.session_state.feedbacks.get(feedback_key)
                if saved_fb == 1:
                    st.caption("✅ Feedback: Resposta útil e correta")
                elif saved_fb == 0:
                    st.caption("⚠️ Feedback: Precisa de melhoria")


# Processamento de Nova Pergunta (via chat_input ou botão de dúvida frequente)
query_to_process: Optional[str] = None

if "pending_query" in st.session_state and st.session_state.pending_query:
    query_to_process = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    user_input = st.chat_input("Digite sua dúvida sobre normas, procedimentos, escalas ou benefícios...")
    if user_input:
        query_to_process = user_input.strip()

if query_to_process:
    # Atualiza o título da conversa se for a primeira pergunta do usuário
    user_messages_count = sum(1 for m in current_messages if m["role"] == "user")
    if user_messages_count == 0:
        clean_title = query_to_process[:26].strip()
        if len(query_to_process) > 26:
            clean_title += "..."
        active_session["title"] = clean_title

    # Registra a mensagem do usuário no histórico da sessão
    active_session["messages"].append(
        {
            "role": "user",
            "content": query_to_process,
            "sources": [],
            "timestamp": datetime.now().strftime("%H:%M"),
        }
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(query_to_process)

    # Executa a geração de resposta fundamentada com o RAG
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Consultando documentos oficiais e validando resposta..."):
            agent = get_rag_agent()

            # Prepara filtros de metadados
            meta_filter = None
            if category_filter != "Todas":
                meta_filter = {"category": category_filter}

            try:
                response = agent.answer(
                    query=query_to_process,
                    channel=channel_option,
                    metadata_filter=meta_filter,
                    recency_boost=boost_recency,
                )

                answer_text = response.get("answer", "Desculpe, ocorreu um erro ao gerar a resposta.")
                clean_answer = sanitize_display_markdown(answer_text)
                citations = response.get("citations", [])
                is_fallback = response.get("is_fallback", False)

                # Renderiza a resposta formatada
                st.markdown(clean_answer)

                # Renderiza fontes
                if citations:
                    with st.expander(f"📚 Fontes e Documentos Consultados ({len(citations)})", expanded=False):
                        for s_idx, src in enumerate(citations, start=1):
                            file_name = src.get("file_name", "Documento")
                            section = src.get("section_title", "Seção")
                            page_range = src.get("page_range", "")
                            st.markdown(
                                f"**{s_idx}. 📄 `{file_name}`**  \n"
                                f"&nbsp;&nbsp;&nbsp;&nbsp;📌 **Seção:** {section}  \n"
                                f"&nbsp;&nbsp;&nbsp;&nbsp;📖 **Página(s):** {page_range}"
                            )

                # Salva no histórico da sessão ativa
                active_session["messages"].append(
                    {
                        "role": "assistant",
                        "content": clean_answer,
                        "sources": citations,
                        "timestamp": datetime.now().strftime("%H:%M"),
                        "feedback": None,
                        "is_fallback": is_fallback,
                    }
                )

            except Exception as e:
                logger.error(f"Erro ao processar consulta no Streamlit: {e}", exc_info=True)
                error_msg = (
                    "Ocorreu um erro interno ao processar sua dúvida. "
                    "Por favor, tente novamente ou consulte o suporte corporativo."
                )
                st.error(error_msg)
                active_session["messages"].append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "timestamp": datetime.now().strftime("%H:%M"),
                        "feedback": None,
                        "is_fallback": True,
                    }
                )

    st.rerun()
