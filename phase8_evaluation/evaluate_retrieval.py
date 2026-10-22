
"""
Phase 8 — Qdrant Retrieval Evaluation

Evaluates semantic retrieval using manually labelled
relevant Qdrant point IDs.

Metrics:
- Precision@5
- Recall@5
- MRR@5

Reads:
    phase8_evaluation/evaluation_dataset.json

Writes:
    reports/phase8/retrieval_report.json

This script is read-only with respect to Qdrant.
It does not upload, delete, or modify points.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from phase3_vectorstore.vector_store import (
    create_services,
    search_documents,
)

from phase8_evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT
    / "phase8_evaluation"
    / "evaluation_dataset.json"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "phase8"
    / "retrieval_report.json"
)

TOP_K = 5


# --------------------------------------------------
# Load evaluation dataset
# --------------------------------------------------

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: "
            f"{DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(
            "Evaluation dataset must be "
            "a non-empty JSON list."
        )

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        if not isinstance(item, dict):
            raise ValueError(
                f"Test case {index} must be an object."
            )

        question = item.get("question")
        topic = item.get("topic")
        relevant_ids = item.get(
            "relevant_document_ids"
        )

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Test case {index} has an invalid question."
            )

        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(
                f"Test case {index} has an invalid topic."
            )

        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or not all(
                isinstance(doc_id, str) and doc_id.strip()
                for doc_id in relevant_ids
            )
        ):
            raise ValueError(
                f"Test case {index} requires a "
                "non-empty list of relevant document IDs."
            )

    return dataset


# --------------------------------------------------
# Evaluate Qdrant retrieval
# --------------------------------------------------

def evaluate_retrieval():

    dataset = load_dataset()

    embedding_model, client = create_services()

    results = []

    all_retrieved_ids = []
    all_relevant_ids = []

    print("\n" + "=" * 60)
    print("IRELAND NAVIGATOR AI — PHASE 8")
    print("QDRANT RETRIEVAL EVALUATION")
    print("=" * 60)

    for index, test in enumerate(
        dataset,
        start=1,
    ):

        question = test["question"]
        topic = test["topic"]
        relevant_ids = test[
            "relevant_document_ids"
        ]

        print(
            f"\nQuestion {index}: {question}"
        )

        retrieved = search_documents(
            question=question,
            embedding_model=embedding_model,
            client=client,
            topic=topic,
            limit=TOP_K,
        )

        # Use actual point IDs returned by Qdrant.
        retrieved_ids = [
            item["id"]
            for item in retrieved
        ]

        precision = precision_at_k(
            retrieved_ids,
            relevant_ids,
            k=TOP_K,
        )

        recall = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k=TOP_K,
        )

        # Restrict reciprocal rank to top K.
        rr = reciprocal_rank(
            retrieved_ids[:TOP_K],
            relevant_ids,
        )

        result = {
            "question": question,
            "topic": topic,
            "top_k": TOP_K,
            "relevant_document_ids": relevant_ids,
            "retrieved_document_ids": retrieved_ids,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "reciprocal_rank_at_k": rr,
            "retrieved_documents": [
                {
                    "id": item["id"],
                    "score": item["score"],
                    "source": item["metadata"].get(
                        "source",
                        "",
                    ),
                    "chunk_index": item["metadata"].get(
                        "chunk_index"
                    ),
                    "text_preview": item["text"][:200],
                }
                for item in retrieved
            ],
        }

        results.append(result)

        all_retrieved_ids.append(
            retrieved_ids[:TOP_K]
        )

        all_relevant_ids.append(
            relevant_ids
        )

        print(
            f"Precision@{TOP_K}: "
            f"{precision:.3f}"
        )

        print(
            f"Recall@{TOP_K}: "
            f"{recall:.3f}"
        )

        print(
            f"Reciprocal Rank@{TOP_K}: "
            f"{rr:.3f}"
        )

        print("Retrieved document IDs:")

        for rank, item in enumerate(
            retrieved,
            start=1,
        ):
            print(
                f"  {rank}. {item['id']} "
                f"(score={item['score']:.3f})"
            )

    # --------------------------------------------------
    # Aggregate results
    # --------------------------------------------------

    total = len(results)

    avg_precision = sum(
        item["precision_at_k"]
        for item in results
    ) / total

    avg_recall = sum(
        item["recall_at_k"]
        for item in results
    ) / total

    mrr = mean_reciprocal_rank(
        all_retrieved_ids,
        all_relevant_ids,
    )

    summary = {
        "total_questions": total,
        "top_k": TOP_K,
        "average_precision_at_k": avg_precision,
        "average_recall_at_k": avg_recall,
        "mrr_at_k": mrr,
    }

    report = {
        "project": "Ireland Navigator AI",
        "phase": 8,
        "evaluation_type": "retrieval",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "summary": summary,
        "results": results,
        "limitations": [
            "Ground-truth relevance labels must be "
            "manually verified.",
            "Incomplete relevance labels can distort "
            "precision and recall.",
            "Retrieval metrics do not measure answer "
            "faithfulness or citation accuracy.",
        ],
    }

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    print(
        "Total questions:",
        total,
    )

    print(
        f"Average Precision@{TOP_K}: "
        f"{avg_precision:.3f}"
    )

    print(
        f"Average Recall@{TOP_K}: "
        f"{avg_recall:.3f}"
    )

    print(
        f"MRR@{TOP_K}: "
        f"{mrr:.3f}"
    )

    print(
        "\nReport saved to:",
        REPORT_PATH,
    )

    return report


if __name__ == "__main__":
    evaluate_retrieval()
