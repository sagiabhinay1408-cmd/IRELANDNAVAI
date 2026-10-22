
"""
Phase 8 — Retrieval Evaluation Metrics

Metrics:
1. Precision@K
2. Recall@K
3. Reciprocal Rank
4. Mean Reciprocal Rank (MRR)

All functions are pure Python.

Relevance labels must be independently verified.
"""

from collections.abc import Sequence


# --------------------------------------------------
# Precision@K
# --------------------------------------------------

def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int = 5,
) -> float:
    """
    Fraction of the top K retrieved positions
    containing relevant documents.

    Uses K as the denominator.
    """

    if k <= 0:
        raise ValueError(
            "K must be greater than zero."
        )

    relevant = set(relevant_ids)

    top_k = list(
        retrieved_ids[:k]
    )

    matches = sum(
        1
        for doc_id in set(top_k)
        if doc_id in relevant
    )

    return matches / k


# --------------------------------------------------
# Recall@K
# --------------------------------------------------

def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int = 5,
) -> float:
    """
    Fraction of all labelled relevant documents
    retrieved within the top K results.
    """

    if k <= 0:
        raise ValueError(
            "K must be greater than zero."
        )

    relevant = set(
        relevant_ids
    )

    if not relevant:
        return 0.0

    retrieved = set(
        retrieved_ids[:k]
    )

    matches = len(
        retrieved.intersection(
            relevant
        )
    )

    return matches / len(relevant)


# --------------------------------------------------
# Reciprocal Rank
# --------------------------------------------------

def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    """
    Reciprocal of the position of the first
    relevant document.

    Example:
        First relevant document at rank 2
        gives a score of 0.5.
    """

    relevant = set(
        relevant_ids
    )

    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


# --------------------------------------------------
# Mean Reciprocal Rank
# --------------------------------------------------

def mean_reciprocal_rank(
    retrieval_results: Sequence[Sequence[str]],
    relevance_labels: Sequence[Sequence[str]],
) -> float:
    """
    Calculate average reciprocal rank
    across multiple evaluation questions.
    """

    if len(retrieval_results) != len(
        relevance_labels
    ):
        raise ValueError(
            "Retrieval results and relevance labels "
            "must have the same number of queries."
        )

    if not retrieval_results:
        return 0.0

    scores = [
        reciprocal_rank(
            retrieved,
            relevant,
        )
        for retrieved, relevant in zip(
            retrieval_results,
            relevance_labels,
        )
    ]

    return sum(scores) / len(scores)


# --------------------------------------------------
# Local metric verification
# --------------------------------------------------

def main():

    retrieved = [
        "doc1",
        "doc2",
        "doc3",
    ]

    relevant = [
        "doc1",
        "doc3",
    ]

    print("RETRIEVAL METRICS TEST")

    print(
        "Precision@3:",
        precision_at_k(
            retrieved,
            relevant,
            k=3,
        ),
    )

    print(
        "Recall@3:",
        recall_at_k(
            retrieved,
            relevant,
            k=3,
        ),
    )

    print(
        "Reciprocal Rank:",
        reciprocal_rank(
            retrieved,
            relevant,
        ),
    )

    print(
        "MRR:",
        mean_reciprocal_rank(
            [retrieved],
            [relevant],
        ),
    )


if __name__ == "__main__":
    main()
