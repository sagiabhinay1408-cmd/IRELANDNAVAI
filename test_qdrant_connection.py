
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load API credentials from .env
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Check that credentials exist
if not qdrant_url or not qdrant_api_key:
    raise ValueError(
        "Qdrant credentials are missing. Check your .env file."
    )

# Connect to Qdrant Cloud
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout=30,
)

# Test the connection
try:
    collections = client.get_collections()

    print("SUCCESS: Connected to Qdrant Cloud!")
    print("Available collections:", collections)

except Exception as error:
    print("ERROR: Could not connect to Qdrant Cloud.")
    print("Details:", error)
