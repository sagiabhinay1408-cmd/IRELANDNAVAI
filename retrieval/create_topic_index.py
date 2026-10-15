
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

COLLECTION_NAME = "ireland_public_services"


def main():

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=120,
    )

    field_name = "metadata.topic"

    print(f"Creating index: {field_name}")

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field_name,
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )

    print("Topic index created successfully!")


if __name__ == "__main__":
    main()

