
import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION_NAME = "ireland_public_services"

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    {
        "text": (
            "Irish Residence Permit information includes "
            "registration and renewal procedures. "
            "Consult Immigration Service Delivery for current guidance."
        ),
        "source": "https://www.irishimmigration.ie/",
        "topic": "immigration",
        "verified": False,
    },
    {
        "text": (
            "Leap Card is a public transport payment card in Ireland. "
            "Consult the official TFI website for fares and eligibility."
        ),
        "source": "https://www.transportforireland.ie/",
        "topic": "transport",
        "verified": False,
    },
    {
        "text": (
            "Information about Irish driving licences is available "
            "through the National Driver Licence Service."
        ),
        "source": "https://www.ndls.ie/",
        "topic": "driving",
        "verified": False,
    },
]

texts = [doc["text"] for doc in documents]
vectors = embedding_model.encode(texts).tolist()

points = []

for doc, vector in zip(documents, vectors):
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=doc,
    )
    points.append(point)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True,
)

print(f"Successfully uploaded {len(points)} sample documents!")
