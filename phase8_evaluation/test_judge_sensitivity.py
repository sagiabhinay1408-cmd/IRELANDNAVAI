
"""
Phase 8 — Judge Sensitivity Test

Checks whether the LLM evaluator can distinguish
a correct answer from an intentionally incorrect one.

Does not modify Qdrant or existing evaluation reports.
"""

from phase4_rag.rag_pipeline import (
    create_rag_services,
)

from phase3_vectorstore.vector_store import (
    search_documents,
)

from phase8_evaluation.evaluate_answer_quality import (
    judge_answer,
)

from phase8_evaluation.evaluation_config import (
    evaluate_pass_fail,
)


def main():
    embedding_model, client, llm_client = (
        create_rag_services()
    )

    case = {
        "id": "IRP-SENSITIVITY-001",
        "question": (
            "How do I renew my Irish Residence Permit?"
        ),
        "topic": "immigration",
        "reference_answer": (
            "IRP renewals are completed online. "
            "In-person renewal is only available "
            "when directed by a Registration Officer."
        ),
        "required_facts": [
            "IRP renewals are completed online.",
            "In-person renewal is not normally available.",
        ],
    }

    documents = search_documents(
        question=case["question"],
        embedding_model=embedding_model,
        client=client,
        topic=case["topic"],
        limit=5,
    )

    correct_answer = (
        "Irish Residence Permit renewals must be "
        "completed online. In-person renewal is "
        "normally unavailable unless directed by "
        "a Registration Officer. [Source 2]"
    )

    incorrect_answer = (
        "Irish Residence Permits can only be renewed "
        "by visiting a Garda station in person. "
        "Online renewal is not permitted. [Source 2]"
    )

    for label, answer in (
        ("CORRECT ANSWER", correct_answer),
        ("INCORRECT ANSWER", incorrect_answer),
    ):
        print("\n" + "=" * 60)
        print(label)
        print("=" * 60)

        judgment = judge_answer(
            case=case,
            answer=answer,
            documents=documents,
            llm_client=llm_client,
        )

        result = evaluate_pass_fail(
            scores=judgment["scores"],
            critical_safety_violation=(
                judgment["critical_safety_violation"]
            ),
        )

        for metric, score in judgment["scores"].items():
            print(f"{metric}: {score}/5")

        print("Overall:", result["overall_score"])
        print("Status:", result["status"])

        print("\nExplanations:")
        for metric, explanation in judgment[
            "explanations"
        ].items():
            print(f"{metric}: {explanation}")


if __name__ == "__main__":
    main()
