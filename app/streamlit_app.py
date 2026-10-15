
import sys
from pathlib import Path

import streamlit as st


# =====================================================
# PROJECT PATH
# =====================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.ireland_agent import create_services, build_graph


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Ireland Navigator AI",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #102a23;
    }

    [data-testid="stSidebar"] * {
        color: #f5f7f6;
    }

    .hero {
        padding: 24px;
        border-radius: 16px;
        background: linear-gradient(
            120deg, #123c2e, #21674c
        );
        color: white;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }

    .hero p {
        color: #e1f1e9;
        margin-top: 8px;
        margin-bottom: 0;
    }

    .source-card {
        padding: 12px 15px;
        border-radius: 10px;
        border: 1px solid #d6e3db;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# LOAD LANGGRAPH
# =====================================================

@st.cache_resource(show_spinner="Loading AI services...")
def load_agent():
    vector_store, llm_client = create_services()
    graph = build_graph(vector_store, llm_client)
    return graph


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("🇮🇪 Ireland Navigator")
    st.caption("Your AI guide to Irish public services")

    st.divider()

    st.subheader("Explore services")

    suggestions = {
        "🪪 IRP Renewal":
            "How do I renew my Irish Residence Permit?",
        "🚍 Leap Card":
            "What is a Leap Card and how can I use it?",
        "🚗 Driving Licence":
            "What documents do I need for an Irish driving licence?",
    }

    for label, question in suggestions.items():
        if st.button(label, use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    st.caption("AI Model: NVIDIA Nemotron 3 Ultra")
    st.caption("Vector Database: Qdrant Cloud")
    st.caption("Agent Framework: LangGraph")
    st.caption("Temperature: 0.3")


# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
    <div class="hero">
        <h1>🇮🇪 Ireland Navigator AI</h1>
        <p>
            Find information about Irish public services
            with AI-powered answers and official sources.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Answers are based on indexed official webpages. "
    "Always verify important requirements and fees "
    "on the relevant government website.",
    icon="ℹ️",
)


# =====================================================
# HELPER: DISPLAY SOURCES
# =====================================================

def display_sources(sources):
    if not sources:
        return

    with st.expander(
        f"📚 View retrieved sources ({len(sources)})"
    ):
        for source in sources:
            number = source.get("number", "?")
            title = source.get("title", "Official source")
            url = source.get("url", "")

            st.markdown(f"**[Source {number}] {title}**")

            if url.startswith("https://"):
                st.markdown(f"[Open official webpage]({url})")


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))


# =====================================================
# QUESTION INPUT
# =====================================================

typed_question = st.chat_input(
    "Ask about Irish immigration, transport, and more..."
)

pending_question = st.session_state.pending_question

if pending_question:
    question = pending_question
    st.session_state.pending_question = None
else:
    question = typed_question


# =====================================================
# PROCESS QUESTION
# =====================================================

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching official sources and generating an answer..."
        ):
            try:
                graph = load_agent()

                result = graph.invoke({
                    "question": question
                })

                answer = result.get(
                    "answer",
                    "No answer was generated."
                )

                sources = result.get("sources", [])

            except Exception as error:
                answer = (
                    "Sorry, I couldn't process your question. "
                    "Please try again."
                )
                sources = []
                st.error(
                    f"Application error: {error}"
                )

        st.markdown(answer)
        display_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Ireland Navigator AI | "
    "NVIDIA Nemotron • LangGraph • Qdrant Cloud • Streamlit"
)
