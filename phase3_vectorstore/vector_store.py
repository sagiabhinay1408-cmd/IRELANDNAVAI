
"""
Phase 3 — Qdrant Vector Database

Responsibilities:
1. Initialize the embedding model.
2. Connect to Qdrant Cloud.
3. Create a vector collection when necessary.
4. Create payload indexes.
5. Upload official document chunks.
6. Search documents by semantic similarity.
7. Return actual Qdrant point IDs for evaluation.

Existing documents are not deleted.
"""

import sys
import uuid
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


# --------------------------------------------------
# Project imports
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    RETRIEVAL_TOP_K,
    validate_configuration,
)

from phase2_ingestion.ingest_documents import prepare_documents


# --------------------------------------------------
# Initialize services
# --------------------------------------------------

def create_services():
    """
    Load the embedding model and connect to Qdrant.

    Returns:
        embedding_model: SentenceTransformer instance
        client: QdrantClient instance
    """

    validate_configuration(require_nvidia=False)

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Connecting to Qdrant Cloud...")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
    )

    return embedding_model, client


# --------------------------------------------------
# Ensure collection exists
# --------------------------------------------------

def ensure_collection(client):
    """
    Create the collection if it does not exist.

    Existing collections are preserved.
    """

    if client.collection_exists(QDRANT_COLLECTION):

        info = client.get_collection(
            QDRANT_COLLECTION
        )

        vectors = info.config.params.vectors

        if not isinstance(vectors, dict):

            if vectors.size != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    "Existing collection has an "
                    "incompatible vector dimension."
                )

        else:
            raise RuntimeError(
                "This application expects an unnamed "
                "single-vector Qdrant collection."
            )

        print("Existing collection found.")
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )

    print("Created Qdrant collection.")


# --------------------------------------------------
# Create payload indexes
# --------------------------------------------------

def create_payload_indexes(client):
    """
    Create indexes used for topic and source filtering.

    Existing indexes are preserved.
    """

    for field_name in (
        "metadata.topic",
        "metadata.source_type",
    ):

        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

        except Exception as error:
            print(
                f"Index notice for {field_name}: "
                f"{error}"
            )


# --------------------------------------------------
# Upload document chunks
# --------------------------------------------------

def upload_documents(
    documents,
    embedding_model,
    client,
):
    """
    Generate embeddings and upload document chunks.

    Deterministic UUIDs prevent duplicate uploads
    when source URLs and chunk indexes are unchanged.
    """

    if not documents:
        raise ValueError(
            "No documents to upload."
        )

    texts = [
        document.page_content
        for document in documents
    ]

    print("Generating embeddings...")

    vectors = embedding_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    points = []

    for document, vector in zip(
        documents,
        vectors,
    ):

        metadata = document.metadata

        identity = (
            f"{metadata['source']}:"
            f"{metadata['chunk_index']}"
        )

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                identity,
            )
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={
                    "page_content": document.page_content,
                    "metadata": metadata,
                },
            )
        )

    batch_size = 32

    for start in range(
        0,
        len(points),
        batch_size,
    ):

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points[
                start:start + batch_size
            ],
            wait=True,
        )

    print(
        f"Uploaded {len(points)} chunks."
    )


# --------------------------------------------------
# Semantic document retrieval
# --------------------------------------------------

def search_documents(
    question,
    embedding_model,
    client,
    topic=None,
    limit=RETRIEVAL_TOP_K,
):
    """
    Search official documents by semantic similarity.

    Args:
        question: User question
        embedding_model: Loaded SentenceTransformer
        client: Connected QdrantClient
        topic: Optional topic filter
        limit: Maximum number of retrieved documents

    Returns:
        List of dictionaries containing:
            id: Actual Qdrant point ID
            text: Retrieved document text
            metadata: Document metadata
            score: Similarity score

    Actual point IDs are returned directly from Qdrant,
    so chunk_index is not required for evaluation.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if limit <= 0:
        raise ValueError(
            "Retrieval limit must be greater than zero."
        )

    query_vector = embedding_model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    conditions = [
        models.FieldCondition(
            key="metadata.source_type",
            match=models.MatchValue(
                value="official_website"
            ),
        )
    ]

    if topic:
        conditions.append(
            models.FieldCondition(
                key="metadata.topic",
                match=models.MatchValue(
                    value=topic
                ),
            )
        )

    query_filter = models.Filter(
        must=conditions
    )

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )

    results = []

    for point in response.points:

        payload = point.payload or {}

        metadata = payload.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        text = payload.get(
            "page_content",
            "",
        )

        if not isinstance(text, str):
            text = ""

        # Skip malformed or empty records.
        if not text.strip():
            continue

        results.append({
            "id": str(point.id),
            "text": text,
            "metadata": metadata,
            "score": float(point.score),
        })

    return results


# --------------------------------------------------
# Standalone Phase 3 execution
# --------------------------------------------------

def main():

    embedding_model, client = create_services()

    ensure_collection(client)

    create_payload_indexes(client)

    documents = prepare_documents()

    upload_documents(
        documents,
        embedding_model,
        client,
    )

    print("\nTesting semantic retrieval...")

    results = search_documents(
        question=(
            "How do I renew my "
            "Irish Residence Permit?"
        ),
        embedding_model=embedding_model,
        client=client,
        topic="immigration",
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nResult {index}: "
            f"score={result['score']:.3f}"
        )

        print(
            "Document ID:",
            result["id"],
        )

        print(
            "Source:",
            result["metadata"].get(
                "source",
                "",
            ),
        )

        print(
            "Text:",
            result["text"][:250],
        )


if __name__ == "__main__":
    main()
