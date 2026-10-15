
"""
Phase 6 — LangGraph Agent with Conversational Memory

Workflow:

START
  |
  v
Rewrite Follow-up Question
  |
  v
Classify Topic
  |
  v
Retrieve Official Documents
  |
  v
Check Evidence
  |
  +---- Generate / Fallback
                |
                v
        Save Conversation
                |
                v
               END

The original user question is preserved separately
from the rewritten retrieval question.
"""

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END,
)


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Reuse the existing Phase 5 logic.
from phase5_agent.ireland_agent import (
    classify_topic,
    make_retriever_node,
    check_evidence,
    route_after_evidence,
    make_generation_node,
    fallback_answer,
)

from phase4_rag.rag_pipeline import (
    create_rag_services,
)

from phase6_memory.conversation_memory import (
    create_memory,
    create_thread_id,
    create_thread_config,
    rewrite_followup_question,
    append_exchange,
)


# -----------------------------------------------------
# MEMORY-ENABLED GRAPH STATE
# -----------------------------------------------------

class MemoryAgentState(TypedDict, total=False):
    """
    State shared across the memory-enabled graph.

    question:
        The user's original wording.

    standalone_question:
        The contextualized version used for retrieval.

    history:
        Previous user and assistant messages.
    """

    question: str
    standalone_question: str
    topic: str
    documents: list
    sources: list
    sufficient_evidence: bool
    answer: str
    history: list


# -----------------------------------------------------
# NODE 1: REWRITE FOLLOW-UP QUESTION
# -----------------------------------------------------

def make_contextualize_node(llm_client):
    """
    Create a node that uses conversation history
    to clarify the current user question.
    """

    def contextualize(state: MemoryAgentState):

        original_question = state["question"]

        history = state.get("history", [])

        standalone_question = (
            rewrite_followup_question(
                question=original_question,
                history=history,
                llm_client=llm_client,
            )
        )

        print(
            f"[Memory] Original: {original_question}"
        )

        print(
            f"[Memory] Search: {standalone_question}"
        )

        return {
            "standalone_question": standalone_question
        }

    return contextualize


# -----------------------------------------------------
# NODE 2: CLASSIFY REWRITTEN QUESTION
# -----------------------------------------------------

def classify_with_memory(state: MemoryAgentState):
    """
    Classify the standalone question rather than
    the short original follow-up.
    """

    search_question = state.get(
        "standalone_question",
        state["question"],
    )

    # Reuse Phase 5's classifier.
    return classify_topic({
        "question": search_question
    })


# -----------------------------------------------------
# NODE 3: RETRIEVE USING REWRITTEN QUESTION
# -----------------------------------------------------

def make_memory_retriever(
    embedding_model,
    qdrant_client,
):
    """
    Adapt the Phase 5 retriever to use the
    contextualized question.
    """

    original_retriever = make_retriever_node(
        embedding_model,
        qdrant_client,
    )

    def retrieve(state: MemoryAgentState):

        search_question = state.get(
            "standalone_question",
            state["question"],
        )

        return original_retriever({
            "question": search_question,
            "topic": state.get(
                "topic",
                "unknown",
            ),
        })

    return retrieve


# -----------------------------------------------------
# NODE 4: GENERATE GROUNDED ANSWER
# -----------------------------------------------------

def make_memory_generator(llm_client):
    """
    Reuse Phase 5's answer generator.

    Only retrieved documents provide factual
    evidence for the generated answer.
    """

    original_generator = make_generation_node(
        llm_client
    )

    def generate(state: MemoryAgentState):

        search_question = state.get(
            "standalone_question",
            state["question"],
        )

        return original_generator({
            "question": search_question,
            "documents": state.get(
                "documents",
                [],
            ),
        })

    return generate


# -----------------------------------------------------
# NODE 5: SAVE CONVERSATION
# -----------------------------------------------------

def save_conversation(state: MemoryAgentState):
    """
    Save the user's original question and the
    final answer into conversation history.

    The updated history is then checkpointed
    by LangGraph automatically.
    """

    updated_history = append_exchange(
        history=state.get("history", []),
        question=state["question"],
        answer=state["answer"],
    )

    print(
        f"[Memory] Stored "
        f"{len(updated_history)} messages"
    )

    return {
        "history": updated_history
    }


# -----------------------------------------------------
# BUILD MEMORY GRAPH
# -----------------------------------------------------

def build_memory_graph(
    embedding_model,
    qdrant_client,
    llm_client,
):
    """
    Construct and compile the memory-enabled graph.

    Unlike Phase 5, this graph uses a checkpointer.
    """

    builder = StateGraph(MemoryAgentState)

    builder.add_node(
        "contextualize",
        make_contextualize_node(llm_client),
    )

    builder.add_node(
        "classify",
        classify_with_memory,
    )

    builder.add_node(
        "retrieve",
        make_memory_retriever(
            embedding_model,
            qdrant_client,
        ),
    )

    builder.add_node(
        "check_evidence",
        check_evidence,
    )

    builder.add_node(
        "generate",
        make_memory_generator(llm_client),
    )

    builder.add_node(
        "fallback",
        fallback_answer,
    )

    builder.add_node(
        "save_memory",
        save_conversation,
    )

    # Start by resolving conversational references.
    builder.add_edge(START, "contextualize")

    # Then classify and retrieve evidence.
    builder.add_edge("contextualize", "classify")
    builder.add_edge("classify", "retrieve")
    builder.add_edge("retrieve", "check_evidence")

    # Choose an answer-generation path.
    builder.add_conditional_edges(
        "check_evidence",
        route_after_evidence,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    # Save the exchange regardless of the route.
    builder.add_edge("generate", "save_memory")
    builder.add_edge("fallback", "save_memory")

    builder.add_edge("save_memory", END)

    # MemorySaver stores checkpoints by thread ID.
    checkpointer = create_memory()

    return builder.compile(
        checkpointer=checkpointer
    )


# -----------------------------------------------------
# TERMINAL TEST
# -----------------------------------------------------

def main():

    print("IRELAND NAVIGATOR AI — PHASE 6")

    services = create_rag_services()

    graph = build_memory_graph(*services)

    # Start with a new conversation thread.
    thread_id = create_thread_id()

    config = create_thread_config(thread_id)

    print("Type 'new' for a new conversation.")
    print("Type 'exit' to quit.")

    while True:

        question = input(
            "\nAsk a question: "
        ).strip()

        if question.lower() in ("exit", "quit"):
            break

        if question.lower() == "new":

            # A new thread ID means no prior context.
            thread_id = create_thread_id()
            config = create_thread_config(thread_id)

            print("New conversation started.")
            continue

        if not question:
            continue

        try:
            # Reusing config keeps the same history.
            result = graph.invoke(
                {"question": question},
                config=config,
            )

            print("\nANSWER:\n")
            print(result["answer"])

            print("\nSOURCES:")

            seen_urls = set()

            for source in result.get(
                "sources",
                [],
            ):

                url = source.get("url", "")

                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)

                print(source["title"])
                print(url)

        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
