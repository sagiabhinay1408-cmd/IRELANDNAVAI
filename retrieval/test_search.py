
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")

question = "Where can I find information about renewing my IRP?"

query_vector = model.encode(question).tolist()

results = client.query_points(
    collection_name="ireland_public_services",
    query=query_vector,
    limit=3,
    with_payload=True,
).points

for result in results:
    print("\nSimilarity score:", round(result.score, 4))
    print("Topic:", result.payload.get("topic"))
    print("Text:", result.payload.get("text"))
    print("Source:", result.payload.get("source"))
