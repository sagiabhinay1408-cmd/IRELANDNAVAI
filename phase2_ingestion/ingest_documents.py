
"""
Phase 2 — Download and Prepare Official Documents

Workflow:
1. Read the official source registry.
2. Download each webpage.
3. Remove navigation, scripts and unnecessary HTML.
4. Extract readable text.
5. Split the text into overlapping chunks.
6. Attach source metadata.

This phase prepares documents in memory.
Phase 3 uploads them to Qdrant.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


# Add the project root to Python's import path.
#
# This supports direct execution of this file and
# avoids requiring __init__.py files.

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from phase2_ingestion.official_sources import (
    OFFICIAL_SOURCES,
)


def fetch_webpage(url):
    """
    Download a webpage and return its HTML.

    A timeout prevents the program from waiting
    indefinitely when a website is unavailable.
    """

    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "IrelandNavigatorAI/1.0 "
                "(educational public-information project)"
            )
        },
        timeout=25,
    )

    # Raise an exception for HTTP errors such as 404.
    response.raise_for_status()

    return response.text


def clean_html(html):
    """
    Convert HTML into readable webpage text.

    Remove elements that usually contain navigation,
    JavaScript, styling or unrelated interface content.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted HTML elements.
    for element in soup.select(
        "script, style, nav, footer, header, "
        "noscript, form, aside"
    ):
        element.decompose()

    # Prefer the main content area.
    # Fall back to the body if no <main> exists.
    main_content = soup.find("main")

    if main_content is None:
        main_content = soup.body or soup

    # Extract text separated by spaces.
    raw_text = main_content.get_text(
        separator=" ",
        strip=True,
    )

    # Normalize repeated whitespace.
    cleaned_text = " ".join(raw_text.split())

    return cleaned_text


def create_text_splitter():
    """
    Create a reusable document chunker.

    chunk_size:
        Maximum approximate character length.

    chunk_overlap:
        Repeats some text between neighboring chunks
        to reduce context loss at boundaries.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def prepare_documents():
    """
    Download, clean and chunk all official sources.

    Returns:
        A list of LangChain Document objects.

    Each document contains:
        page_content: the text chunk
        metadata: URL, topic, title and timestamps
    """

    splitter = create_text_splitter()

    all_documents = []

    for source in OFFICIAL_SOURCES:

        url = source["url"]
        title = source["title"]
        topic = source["topic"]

        print(f"\nDownloading: {title}")

        try:
            # Download HTML from the official website.
            html = fetch_webpage(url)

            # Extract meaningful text.
            text = clean_html(html)

            # Ignore pages without enough useful text.
            if len(text) < 100:
                print("Skipped: insufficient text")
                continue

            # Record the time at which we collected it.
            retrieved_at = datetime.now(
                timezone.utc
            ).isoformat()

            # Store the full page as a Document first.
            document = Document(
                page_content=text,
                metadata={
                    "source": url,
                    "title": title,
                    "topic": topic,
                    "retrieved_at": retrieved_at,
                    "source_type": "official_website",
                    "verified": False,
                },
            )

            # Divide the page into smaller chunks.
            chunks = splitter.split_documents(
                [document]
            )

            # Add chunk numbers for traceability.
            for index, chunk in enumerate(chunks):

                chunk.metadata["chunk_index"] = index

                all_documents.append(chunk)

            print(f"Prepared {len(chunks)} chunks")

        except requests.RequestException as error:
            # A failed download should not stop every
            # other source from being processed.
            print(f"Download failed: {error}")

        except Exception as error:
            print(f"Processing failed: {error}")

    print(
        f"\nTotal prepared chunks: {len(all_documents)}"
    )

    return all_documents


def main():
    """Run the document preparation phase."""

    documents = prepare_documents()

    if not documents:
        print("No documents were prepared.")
        return

    # Display a short preview, not the entire dataset.
    first_document = documents[0]

    print("\nFIRST DOCUMENT METADATA")
    print(first_document.metadata)

    print("\nTEXT PREVIEW")
    print(first_document.page_content[:350])


if __name__ == "__main__":
    main()
