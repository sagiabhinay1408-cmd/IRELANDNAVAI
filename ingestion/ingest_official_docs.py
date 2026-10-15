
import os
import sys
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sources.official_sources import OFFICIAL_SOURCES

load_dotenv(ROOT / ".env")

COLLECTION_NAME = "ireland_public_services"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 5
MAX_RETRIES = 3

REQUEST_TIMEOUT = 30
QDRANT_TIMEOUT = 120


# ============================================================
# 2. DOWNLOAD AND CLEAN OFFICIAL WEBPAGES
# ============================================================

def download_page(source):
    url = source["url"]

    response = requests.get(
        url,
        headers={
            "User-Agent": "IrelandNavigatorAI/0.1"
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Remove unnecessary HTML elements
    for element in soup.select(
        "script, style, nav, footer, header, "
        "form, aside, noscript"
    ):
        element.decompose()

    # Extract main webpage content
    main = soup.select_one(
        "main, article, [role='main']"
    )

    if main is None:
        raise ValueError(
            f"No main content found on {url}. "
            "Inspect the page before ingestion."
        )

    text = main.get_text(
        separator=" ",
        strip=True
    )

    text = " ".join(text.split())

    if len(text) < 200:
        raise ValueError(
            f"Insufficient readable content at {url}"
        )

    # Create a LangChain document
    return Document(
        page_content=text,
        metadata={
            "source": url,
            "title": source["title"],
            "topic": source["topic"],
            "retrieved_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "source_type": "official_website",
            "verified": False,
        },
    )


# ============================================================
# 3. GENERATE STABLE QDRANT POINT IDS
# ============================================================

def generate_point_id(chunk):
    """
    Generate a deterministic UUID for each document chunk.

    Identical source URL + text produces the same ID.
    This helps avoid duplicate points during retries.
    """

    unique_content = (
        chunk.metadata["source"]
        + "\n"
        + chunk.page_content
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            unique_content
        )
    )


# ============================================================
# 4. UPLOAD DOCUMENTS IN SMALL BATCHES
# ============================================================

def upload_in_batches(vector_store, chunks, ids):
    total_chunks = len(chunks)

    print(
        f"\nStarting upload of {total_chunks} chunks..."
    )

    for start in range(0, total_chunks, BATCH_SIZE):
        end = min(
            start + BATCH_SIZE,
            total_chunks
        )

        batch_chunks = chunks[start:end]
        batch_ids = ids[start:end]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vector_store.add_documents(
                    documents=batch_chunks,
                    ids=batch_ids,
                )

                print(
                    f"Uploaded {end}/{total_chunks} chunks"
                )

                break

            except Exception as error:
                print(
                    f"Batch {start}:{end} failed "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )

                print(f"Error: {error}")

                if attempt == MAX_RETRIES:
                    print(
                        "\nUpload failed after maximum retries."
                    )
                    raise

                wait_time = 3 * attempt

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

    print(
        f"\nSuccessfully uploaded {total_chunks} chunks!"
    )


# ============================================================
# 5. MAIN INGESTION PIPELINE
# ============================================================

def main():

    print("=" * 55)
    print("IRELAND NAVIGATOR AI")
    print("Official Document Ingestion Pipeline")
    print("=" * 55)

    # --------------------------------------------------------
    # Step A: Validate environment variables
    # --------------------------------------------------------

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError(
            "Missing QDRANT_URL or QDRANT_API_KEY. "
            "Check your .env file."
        )

    # --------------------------------------------------------
    # Step B: Download official documents
    # --------------------------------------------------------

    documents = []

    for source in OFFICIAL_SOURCES:

        print(
            f"\nDownloading: {source['title']}"
        )

        try:
            document = download_page(source)

            documents.append(document)

            print(
                f"Loaded {len(document.page_content)} "
                "characters"
            )

        except Exception as error:
            print(
                f"Skipping {source['url']}: {error}"
            )

    if not documents:
        raise RuntimeError(
            "No documents were downloaded successfully."
        )

    print(
        f"\nSuccessfully downloaded "
        f"{len(documents)} documents."
    )

    # --------------------------------------------------------
    # Step C: Split documents into chunks
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = splitter.split_documents(documents)

    print(
        f"\nCreated {len(chunks)} document chunks."
    )

    # --------------------------------------------------------
    # Step D: Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded successfully.")

    # --------------------------------------------------------
    # Step E: Connect to Qdrant Cloud
    # --------------------------------------------------------

    print("\nConnecting to Qdrant Cloud...")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=QDRANT_TIMEOUT,
    )

    if not client.collection_exists(
        COLLECTION_NAME
    ):
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' "
            "does not exist. Run create_collection.py first."
        )

    print("Connected to Qdrant Cloud successfully.")

    # --------------------------------------------------------
    # Step F: Initialize LangChain Qdrant vector store
    # --------------------------------------------------------

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )

    # --------------------------------------------------------
    # Step G: Generate unique document IDs
    # --------------------------------------------------------

    ids = [
        generate_point_id(chunk)
        for chunk in chunks
    ]

    print(
        f"\nGenerated {len(ids)} stable document IDs."
    )

    # --------------------------------------------------------
    # Step H: Upload chunks to Qdrant Cloud
    # --------------------------------------------------------

    upload_in_batches(
        vector_store=vector_store,
        chunks=chunks,
        ids=ids,
    )

    # --------------------------------------------------------
    # Step I: Verify Qdrant collection
    # --------------------------------------------------------

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    print("\n" + "=" * 55)
    print("INGESTION COMPLETED")
    print("=" * 55)

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Documents downloaded: {len(documents)}"
    )

    print(
        f"Chunks processed: {len(chunks)}"
    )

    print(
        f"Total collection points: "
        f"{collection_info.points_count}"
    )

    print(
        f"Collection status: {collection_info.status}"
    )

    print("=" * 55)


# ============================================================
# 6. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
