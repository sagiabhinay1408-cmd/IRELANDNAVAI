
"""
Phase 7 — Streamlit Chat Interface

Features:
- Interactive chat messages
- Memory-enabled LangGraph agent
- Separate conversation thread per browser session
- Official source links
- New Chat button
- Quick-question shortcuts
"""

import sys
from pathlib import Path

import streamlit as st


# -----------------------------------------------------
# IMPORT PROJECT MODULES
# -----------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from phase4_rag.rag_pipeline import (
    create_rag_services,
)

from phase6_memory.memory_agent import (
    build_memory_graph,
)

from phase6_memory.conversation_memory import (
    create_thread_id,
    create_thread_config,
)


# -----------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------

# Streamlit requires page configuration before
# other page-rendering commands.

st.set_page_config(
    page_title="Ireland Navigator AI",
    page_icon="🇮🇪",
    layout="wide",
)


# -----------------------------------------------------
# CUSTOM PAGE STYLING
# -----------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
    }

    .hero {
        background: linear-gradient(
            120deg,
            #116149,
            #16805c
        );
        color: white;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .hero p {
        color: #f0fff7;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------
# LOAD SERVICES AND AGENT
# -----------------------------------------------------

@st.cache_resource(
    show_spinner="Initializing Ireland Navigator AI..."
)
def load_agent():
    """
    Initialize the models, database and graph once.

    Streamlit reruns the script after interactions.
    Caching avoids repeatedly loading the embedding
    model and rebuilding the LangGraph agent.

    The cached graph is shared across sessions,
    but each session receives its own thread ID.
    """

    services = create_rag_services()

    return build_memory_graph(*services)


# -----------------------------------------------------
# INITIALIZE BROWSER SESSION STATE
# -----------------------------------------------------

# Session state stores interface information across
# Streamlit reruns within the browser session.

if "thread_id" not in st.session_state:

    st.session_state.thread_id = (
        create_thread_id()
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


if "pending_question" not in st.session_state:

    st.session_state.pending_question = None


def start_new_chat():
    """
    Reset the visible chat and use a new thread ID.

    A new thread has no previous conversation context.
    """

    st.session_state.thread_id = (
        create_thread_id()
    )

    st.session_state.messages = []

    st.session_state.pending_question = None


# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

with st.sidebar:

    st.title("🇮🇪 Ireland Navigator AI")

    st.caption(
        "Official-source RAG + LangGraph Memory"
    )

    st.divider()

    if st.button(
        "➕ New Chat",
        use_container_width=True,
    ):

        start_new_chat()

        # Immediately refresh the interface.
        st.rerun()

    st.success("Conversational memory enabled")

    st.divider()

    st.subheader("Quick Questions")

    quick_questions = {
        "🪪 IRP Renewal":
            "How do I renew my Irish Residence Permit?",

        "🚆 Leap Card":
            "What is a Leap Card?",

        "🚗 Driving Licence":
            "How do I apply for an Irish driving licence?",
    }

    for label, question in quick_questions.items():

        if st.button(
            label,
            use_container_width=True,
        ):

            # Store the selected question so the main
            # chat handler can process it.
            st.session_state.pending_question = (
                question
            )

            st.rerun()

    st.divider()

    st.caption(
        "Important: Information may change. "
        "Always verify current requirements "
        "on official websites."
    )


# -----------------------------------------------------
# PAGE HEADER
# -----------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🇮🇪 Ireland Navigator AI</h1>
        <p>
            Your AI assistant for navigating
            Irish public-service information.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------
# INITIALIZE GRAPH
# -----------------------------------------------------

try:
    graph = load_agent()

except Exception as error:

    st.error("Failed to initialize the AI agent.")

    st.exception(error)

    # Stop processing if initialization failed.
    st.stop()


# -----------------------------------------------------
# HELPER: DISPLAY SOURCE LINKS
# -----------------------------------------------------

def display_sources(sources):
    """
    Show source links beneath an assistant answer.

    Duplicate URLs are displayed only once.
    """

    if not sources:
        return

    with st.expander("📚 Official Sources"):

        seen_urls = set()

        for source in sources:

            url = source.get("url", "")

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            title = source.get(
                "title",
                "Official source",
            )

            # Markdown links allow users to open
            # the underlying official webpage.
            st.markdown(f"[{title}]({url})")


# -----------------------------------------------------
# DISPLAY PREVIOUS CHAT MESSAGES
# -----------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            display_sources(
                message.get("sources", [])
            )


# -----------------------------------------------------
# READ USER INPUT
# -----------------------------------------------------

typed_question = st.chat_input(
    "Ask about Irish public services..."
)

pending_question = (
    st.session_state.pending_question
)

if pending_question:

    question = pending_question

    st.session_state.pending_question = None

else:
    question = typed_question


# -----------------------------------------------------
# PROCESS THE QUESTION
# -----------------------------------------------------

if question and question.strip():

    question = question.strip()

    # Save the user's message for display.
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching official sources..."
            ):

                # Use the current session's thread ID.
                config = create_thread_config(
                    st.session_state.thread_id
                )

                # LangGraph retrieves previous state
                # automatically for this thread.
                result = graph.invoke(
                    {"question": question},
                    config=config,
                )

                answer = result.get(
                    "answer",
                    "No answer was generated.",
                )

                sources = result.get(
                    "sources",
                    [],
                )

            st.markdown(answer)

            display_sources(sources)

            # Store the answer for future UI reruns.
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:

            st.error(
                "Something went wrong while "
                "processing the question."
            )

            st.exception(error)


# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.divider()

st.caption(
    "Ireland Navigator AI provides informational "
    "guidance based on collected official webpages. "
    "It is not a government service or a substitute "
    "for current official advice."
)
