
"""
Phase 8 — Basic RAG Evaluation

Evaluation goals:
1. Check whether retrieval returns expected topics.
2. Check whether the agent provides answers.
3. Check whether unsupported questions fall back.
4. Test conversation memory with follow-ups.

These are basic integration checks.

They are not a complete measure of factual accuracy,
hallucination rate or answer faithfulness.
"""

import sys
from pathlib import Path

from phase4_rag.rag_pipeline import (
    create_rag_services,
)

from phase5_agent.ireland_agent import (
    build_agent_graph,
)

from phase6_memory.memory_agent import (
    build_memory_graph,
)

from phase6_memory.conversation_memory import (
    create_thread_id,
    create_thread_config,
)


# -----------------------------------------------------
# EVALUATION QUESTIONS
# -----------------------------------------------------

# Each test contains:
# - A question
# - The expected classification topic
# - Whether the initial collection should contain
#   relevant information for that topic

TEST_CASES = [
    {
        "question": (
            "How do I renew my Irish Residence Permit?"
        ),
        "expected_topic": "immigration",
        "expect_evidence": True,
    },
    {
        "question": "What is a Leap Card?",
        "expected_topic": "transport",
        "expect_evidence": True,
    },
    {
        "question": (
            "How do I apply for an Irish driving licence?"
        ),
        "expected_topic": "driving",
        "expect_evidence": False,
    },
]


# -----------------------------------------------------
# EVALUATE BASIC AGENT
# -----------------------------------------------------

def evaluate_agent(graph):
    """
    Run predefined questions through Phase 5.

    Returns:
        A list of evaluation results.
    """

    results = []

    for test in TEST_CASES:

        question = test["question"]

        print(f"\nTesting: {question}")

        try:
            # Run the full LangGraph agent.
            response = graph.invoke({
                "question": question
            })

            actual_topic = response.get(
                "topic",
                "unknown",
            )

            answer = response.get(
                "answer",
                "",
            )

            evidence_found = response.get(
                "sufficient_evidence",
                False,
            )

            # Classification is correct only when
            # actual and expected topics match.
            topic_correct = (
                actual_topic == test["expected_topic"]
            )

            # Compare the evidence decision with
            # our test expectation.
            evidence_correct = (
                evidence_found
                == test["expect_evidence"]
            )

            # A nonempty answer is a basic operational
            # check, not a factual correctness check.
            has_answer = bool(answer.strip())

            passed = (
                topic_correct
                and evidence_correct
                and has_answer
            )

            results.append({
                "question": question,
                "passed": passed,
                "topic_correct": topic_correct,
                "evidence_correct": evidence_correct,
                "has_answer": has_answer,
            })

            print(
                "Result:",
                "PASS" if passed else "FAIL",
            )

        except Exception as error:

            print(f"ERROR: {error}")

            results.append({
                "question": question,
                "passed": False,
                "error": str(error),
            })

    return results


# -----------------------------------------------------
# EVALUATE CONVERSATIONAL MEMORY
# -----------------------------------------------------

def evaluate_memory(graph):
    """
    Check that the graph stores conversation history
    and processes a follow-up question.

    This checks the memory mechanism, not whether
    the generated answer is factually correct.
    """

    # Use a fresh thread to avoid previous context.
    thread_id = create_thread_id()

    config = create_thread_config(thread_id)

    questions = [
        "How do I renew my IRP?",
        "How much does it cost?",
    ]

    result = None

    for question in questions:

        result = graph.invoke(
            {"question": question},
            config=config,
        )

    history = result.get("history", [])

    standalone_question = result.get(
        "standalone_question",
        "",
    )

    # Two exchanges should create four messages.
    history_saved = len(history) == 4

    # Check that contextualization returned text.
    rewrite_returned = bool(
        standalone_question.strip()
    )

    return {
        "history_saved": history_saved,
        "rewrite_returned": rewrite_returned,
        "standalone_question": standalone_question,
    }


# -----------------------------------------------------
# DISPLAY SUMMARY
# -----------------------------------------------------

def print_summary(results, memory_result):
    """
    Display evaluation totals in the terminal.
    """

    total = len(results)

    passed = sum(
        1
        for result in results
        if result["passed"]
    )

    percentage = (
        (passed / total) * 100
        if total
        else 0
    )

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)

    print(f"Agent tests passed: {passed}/{total}")

    print(
        f"Operational pass rate: {percentage:.1f}%"
    )

    print(
        "Conversation history saved:",
        memory_result["history_saved"],
    )

    print(
        "Follow-up rewrite returned:",
        memory_result["rewrite_returned"],
    )

    print(
        "Rewritten question:",
        memory_result["standalone_question"],
    )


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------

def main():

    print("IRELAND NAVIGATOR AI — PHASE 8")

    # Initialize the shared backend services once.
    services = create_rag_services()

    # Build the original Phase 5 agent.
    agent_graph = build_agent_graph(*services)

    # Build the separate Phase 6 memory agent.
    memory_graph = build_memory_graph(*services)

    # Run classification and evidence checks.
    agent_results = evaluate_agent(agent_graph)

    # Run the two-turn conversation test.
    memory_result = evaluate_memory(
        memory_graph
    )

    # Display combined results.
    print_summary(
        agent_results,
        memory_result,
    )


if __name__ == "__main__":
    main()
