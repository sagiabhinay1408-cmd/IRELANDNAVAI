
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# Project configuration
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

COLLECTION_NAME = "ireland_public_services"


def main():
    print("Connecting to Qdrant Cloud...")

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=120,
    )

    print("Creating payload index...")

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.source_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )

    print("Payload index created successfully!")

    collection = client.get_collection(
        COLLECTION_NAME
    )

    print("Collection status:", collection.status)
    print("Payload schema:", collection.payload_schema)


if __name__ == "__main__":
    main()
