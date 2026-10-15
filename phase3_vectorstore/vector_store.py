
"""
Phase 3 — Qdrant Vector Database

Responsibilities:
1. Initialize the embedding model.
2. Connect to Qdrant Cloud.
3. Create a vector collection when necessary.
4. Upload official document chunks.
5. Search documents by semantic similarity.
"""

import sys
import uuid
from pathlib import Path

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient, models


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

from phase2_ingestion.ingest_documents import (
    prepare_documents,
)


def create_services():
    """
    Initialize the embedding model and Qdrant client.

    SentenceTransformer loads the local embedding model.
    QdrantClient connects to your cloud database.
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


def ensure_collection(client):
    """
    Create the collection only if it does not exist.

    Cosine distance compares vector direction and
    is commonly used for text similarity.
    """

    if client.collection_exists(QDRANT_COLLECTION):

        info = client.get_collection(
            QDRANT_COLLECTION
        )

        # The collection must be compatible with
        # our 384-dimensional embedding model.
        vectors = info.config.params.vectors

        if not isinstance(vectors, dict):

            if vectors.size != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    "Existing collection has an "
                    "incompatible vector dimension."
                )

        else:
            raise RuntimeError(
                "This example expects an unnamed "
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


def create_payload_indexes(client):
    """
    Index metadata fields used for filtering.

    Keyword indexes allow efficient filtering by
    topic and source type.
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
            # Existing indexes may cause a harmless
            # "already exists" response.
            #
            # We report the issue rather than silently
            # hiding unexpected configuration errors.
            print(
                f"Index notice for {field_name}: {error}"
            )


def upload_documents(
    documents,
    embedding_model,
    client,
):
    """
    Embed and upload prepared document chunks.

    Each Qdrant point contains:
        id: unique deterministic chunk ID
        vector: semantic embedding
        payload: document text and metadata
    """

    if not documents:
        raise ValueError("No documents to upload.")

    texts = [
        document.page_content
        for document in documents
    ]

    print("Generating embeddings...")

    # Encode all document chunks in batches.
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

        # A deterministic UUID means re-ingesting the
        # same source and chunk number updates that
        # point instead of always creating a new one.
        #
        # If chunk boundaries change, old points may
        # remain. A full refresh needs separate cleanup.

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

    # Upload in batches to avoid oversized requests.
    batch_size = 32

    for start in range(0, len(points), batch_size):

        batch = points[
            start:start + batch_size
        ]

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch,
            wait=True,
        )

    print(f"Uploaded {len(points)} chunks.")


def search_documents(
    question,
    embedding_model,
    client,
    topic=None,
    limit=RETRIEVAL_TOP_K,
):
    """
    Find chunks semantically related to a question.

    Parameters:
        question: the user's question
        topic: optional metadata topic filter
        limit: maximum returned chunks

    Returns:
        A list of dictionaries containing:
        text, metadata and similarity score
    """

    # Convert the question into a 384D vector.
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

    # Apply topic filtering when requested.
    if topic:
        conditions.append(
            models.FieldCondition(
                key="metadata.topic",
                match=models.MatchValue(value=topic),
            )
        )

    query_filter = models.Filter(
        must=conditions
    )

    # Query Qdrant using its vector search API.
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

        results.append({
            "text": payload.get(
                "page_content",
                ""
            ),
            "metadata": payload.get(
                "metadata",
                {}
            ),
            "score": float(point.score),
        })

    return results


def main():
    """Ingest official webpages into Qdrant."""

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
        "How do I renew my Irish Residence Permit?",
        embedding_model,
        client,
        topic="immigration",
    )

    for index, result in enumerate(results, 1):

        print(
            f"\nResult {index}: "
            f"score={result['score']:.3f}"
        )

        print(
            result["metadata"].get(
                "source",
                ""
            )
        )

        print(result["text"][:250])


if __name__ == "__main__":
    main()
