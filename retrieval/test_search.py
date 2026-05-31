import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import warnings

# Suppress some noisy PyTorch warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "indian_legal_rag"

print("Loading embedding model (this takes a few seconds)...")
# Must use the EXACT same model used for ingestion!
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_URL)

# ─────────────────────────────────────────────────────────
# SEARCH FUNCTION
# ─────────────────────────────────────────────────────────


def search_legal_db(query: str, limit: int = 3):
    print(f"\n🔍 Searching for: '{query}'...")

    # 1. Convert the user's text query into a 384-dimension vector
    query_vector = model.encode(query).tolist()

    # 2. Search Qdrant for the closest matching vectors
    try:
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
    except Exception as e:
        print(f"Database error: {e}")
        return

    # 3. Print the results nicely
    print(f"\n--- TOP {limit} RESULTS ---")
    for i, hit in enumerate(results, 1):
        payload = hit.payload
        doc_type = payload.get('doc_type', 'unknown')
        score = hit.score

        # A score closer to 1.0 means a better match
        print(f"\n[{i}] Match Score: {score:.4f}")

        if doc_type == "judgment":
            print(
                f"Source: Judgment - {payload.get('court')} ({payload.get('year')})")
        elif doc_type == "act":
            print(
                f"Source: Act - {payload.get('act_name')} (Section {payload.get('section_number')})")
        elif doc_type == "cad":
            print(
                f"Source: Constituent Assembly Debate (Vol {payload.get('volume')})")

        # Print a snippet of the actual text
        text = payload.get('text', '')
        print(f"Text snippet: {text[:400]}...")


# ─────────────────────────────────────────────────────────
# INTERACTIVE TERMINAL
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n✅ Legal RAG Search is ready!")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        user_query = input("\n⚖️  Enter a legal query: ")

        if user_query.strip().lower() in ['quit', 'exit', 'q']:
            print("Exiting search...")
            break

        if user_query.strip():
            search_legal_db(user_query)
