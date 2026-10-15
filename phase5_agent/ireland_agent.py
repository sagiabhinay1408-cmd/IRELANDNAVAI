
"""
Phase 5 — LangGraph Public-Service Agent

Workflow:

START
  |
  v
Classify Question
  |
  v
Retrieve Documents
  |
  v
Check Evidence
  |
  +---- Sufficient ----> Generate Answer
  |
  +---- Insufficient --> Safe Fallback
                              |
                              v
                             END

This agent reuses the Phase 4 RAG services.
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


from config.settings import MIN_SIMILARITY

from phase3_vectorstore.vector_store import (
    search_documents,
)

from phase4_rag.rag_pipeline import (
    create_rag_services,
    generate_answer,
)


# -----------------------------------------------------
# AGENT STATE
# -----------------------------------------------------

class AgentState(TypedDict, total=False):
    """
    Shared information passed between graph nodes.

    Each node reads the fields it needs and returns
    a dictionary containing its updated fields.
    """

    question: str
    topic: str
    documents: list
    sources: list
    sufficient_evidence: bool
    answer: str


# -----------------------------------------------------
# NODE 1: CLASSIFY TOPIC
# -----------------------------------------------------

def classify_topic(state: AgentState):
    """
    Classify the question using simple keyword rules.

    This is a deterministic baseline classifier,
    not an LLM-powered intent classifier.
    """

    question = state["question"].lower()

    immigration_terms = [
        "irp",
        "immigration",
        "residence permit",
        "visa",
        "renewal",
        "registration",
        "permission to stay",
    ]

    transport_terms = [
        "leap card",
        "bus",
        "luas",
        "public transport",
        "train fare",
        "transport",
    ]

    driving_terms = [
        "driving licence",
        "driver licence",
        "driving test",
        "learner permit",
    ]

    # Check driving first to avoid incorrectly
    # classifying licence questions as immigration.

    if any(
        term in question
        for term in driving_terms
    ):
        topic = "driving"

    elif any(
        term in question
        for term in immigration_terms
    ):
        topic = "immigration"

    elif any(
        term in question
        for term in transport_terms
    ):
        topic = "transport"

    else:
        topic = "unknown"

    print(f"[Agent] Classified topic: {topic}")

    return {"topic": topic}


# -----------------------------------------------------
# NODE 2: RETRIEVE DOCUMENTS
# -----------------------------------------------------

def make_retriever_node(
    embedding_model,
    qdrant_client,
):
    """
    Create a retrieval node with access to services.

    A factory function is used because LangGraph
    nodes normally receive only the graph state.
    """

    def retrieve(state: AgentState):

        topic = state.get("topic", "unknown")

        # Unknown questions search all official topics.
        # Known topics use a matching metadata filter.

        topic_filter = (
            None if topic == "unknown" else topic
        )

        documents = search_documents(
            question=state["question"],
            embedding_model=embedding_model,
            client=qdrant_client,
            topic=topic_filter,
        )

        sources = []

        for index, document in enumerate(
            documents, 1
        ):

            metadata = document["metadata"]

            sources.append({
                "number": index,
                "title": metadata.get(
                    "title",
                    "Official source",
                ),
                "url": metadata.get(
                    "source",
                    ""
                ),
                "score": document["score"],
            })

        return {
            "documents": documents,
            "sources": sources,
        }

    return retrieve


# -----------------------------------------------------
# NODE 3: CHECK EVIDENCE
# -----------------------------------------------------

def check_evidence(state: AgentState):
    """
    Decide whether retrieval results are relevant.

    This is a similarity-based routing heuristic.
    It does not verify every factual claim.
    """

    documents = state.get("documents", [])

    if not documents:
        return {"sufficient_evidence": False}

    # Find the highest retrieved similarity score.
    best_score = max(
        document["score"]
        for document in documents
    )

    sufficient = best_score >= MIN_SIMILARITY

    print(
        f"[Agent] Best score: {best_score:.3f}"
    )

    print(
        f"[Agent] Evidence accepted: {sufficient}"
    )

    return {
        "sufficient_evidence": sufficient
    }


# -----------------------------------------------------
# CONDITIONAL ROUTER
# -----------------------------------------------------

def route_after_evidence(state: AgentState):
    """
    Choose the next node.

    The return value must match a route defined
    in builder.add_conditional_edges().
    """

    if state.get("sufficient_evidence", False):
        return "generate"

    return "fallback"


# -----------------------------------------------------
# NODE 4: GENERATE ANSWER
# -----------------------------------------------------

def make_generation_node(llm_client):
    """
    Create a node that uses NVIDIA Nemotron.
    """

    def generate(state: AgentState):

        # Exclude chunks below the similarity threshold.
        documents = [
            document
            for document in state.get("documents", [])
            if document["score"] >= MIN_SIMILARITY
        ]

        answer = generate_answer(
            question=state["question"],
            documents=documents,
            llm_client=llm_client,
        )

        # Keep displayed sources aligned with
        # the documents actually given to the model.
        sources = []

        for index, document in enumerate(
            documents, 1
        ):

            metadata = document["metadata"]

            sources.append({
                "number": index,
                "title": metadata.get(
                    "title",
                    "Official source",
                ),
                "url": metadata.get(
                    "source",
                    ""
                ),
                "score": document["score"],
            })

        return {
            "answer": answer,
            "sources": sources,
        }

    return generate


# -----------------------------------------------------
# NODE 5: SAFE FALLBACK
# -----------------------------------------------------

def fallback_answer(state: AgentState):
    """
    Return a cautious answer when relevant
    supporting evidence is unavailable.
    """

    topic = state.get("topic", "unknown")

    official_websites = {
        "immigration": "https://www.irishimmigration.ie/",
        "transport": "https://www.transportforireland.ie/",
        "driving": "https://www.ndls.ie/",
    }

    website = official_websites.get(
        topic,
        "https://www.gov.ie/",
    )

    answer = (
        "I couldn't find enough relevant official "
        "information in the current document collection "
        "to answer this reliably.\n\n"
        f"Please check: {website}"
    )

    return {
        "answer": answer,
        "sources": [],
    }


# -----------------------------------------------------
# BUILD LANGGRAPH WORKFLOW
# -----------------------------------------------------

def build_agent_graph(
    embedding_model,
    qdrant_client,
    llm_client,
):
    """
    Register nodes, connect them and compile
    the workflow into an executable LangGraph.
    """

    builder = StateGraph(AgentState)

    # Register every node by its graph name.
    builder.add_node(
        "classify",
        classify_topic,
    )

    builder.add_node(
        "retrieve",
        make_retriever_node(
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
        make_generation_node(llm_client),
    )

    builder.add_node(
        "fallback",
        fallback_answer,
    )

    # Define the fixed beginning of the workflow.
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "retrieve")
    builder.add_edge("retrieve", "check_evidence")

    # Route to generation or fallback.
    builder.add_conditional_edges(
        "check_evidence",
        route_after_evidence,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    # Both outcomes finish the workflow.
    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)

    # Compile turns the graph definition into
    # an executable application.
    return builder.compile()


# -----------------------------------------------------
# TERMINAL TEST
# -----------------------------------------------------

def main():

    print("IRELAND NAVIGATOR AI — PHASE 5")

    services = create_rag_services()

    graph = build_agent_graph(*services)

    while True:

        question = input(
            "\nAsk a question (or 'exit'): "
        ).strip()

        if question.lower() in ("exit", "quit"):
            break

        if not question:
            continue

        try:
            result = graph.invoke({
                "question": question
            })

            print("\nANSWER:\n")
            print(result["answer"])

            for source in result.get(
                "sources", []
            ):
                print(
                    f"\n{source['title']}: "
                    f"{source['url']}"
                )

        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
