"""
embeddings/batch_ingest.py
Uses local GPU (NVIDIA 3050 6GB) for embeddings — faster than API, free, no rate limits.
Model: all-MiniLM-L6-v2 — tiny, fast, fits easily in 6GB VRAM
"""

import os
import json
import uuid
import glob
import time
import gc
import re
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType
)
from sentence_transformers import SentenceTransformer
import torch

load_dotenv()

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "indian_legal_rag"
VECTOR_SIZE = 384    # all-MiniLM-L6-v2 dimension
BATCH_SIZE = 256     # large batches — GPU handles this easily
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

DATA_DIRS = {
    "judgment": os.path.join(ROOT, "data", "raw", "judgments"),
    # Fixed: Pointing to cleaned folder
    "act":      os.path.join(ROOT, "data", "cleaned", "acts"),
    "cad":      os.path.join(ROOT, "data", "raw", "cad"),
}

PROGRESS_FILE = os.path.join(ROOT, "embeddings", "ingest_progress.json")

# ─────────────────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────────────────


def setup_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU found: {gpu_name} ({vram:.1f} GB VRAM)")
        return "cuda"
    else:
        print("⚠ No GPU found — falling back to CPU")
        return "cpu"


DEVICE = setup_gpu()

# ─────────────────────────────────────────────────────────
# LOAD MODEL ON GPU
# ─────────────────────────────────────────────────────────

print("\nLoading embedding model onto GPU...")
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)
print(f"✓ Model loaded on {DEVICE.upper()}")

qdrant = QdrantClient(url=QDRANT_URL)

# ─────────────────────────────────────────────────────────
# TEST CONNECTIONS
# ─────────────────────────────────────────────────────────


def test_connections() -> bool:
    print("\nTesting connections...")
    print("  Testing GPU embedding...", end=" ", flush=True)
    try:
        test_emb = model.encode(["test"], normalize_embeddings=True)
        print(f"✓ OK (dim: {len(test_emb[0])})")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("  Testing Qdrant...", end=" ", flush=True)
    try:
        qdrant.get_collections()
        print("✓ Connected")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    return True

# ─────────────────────────────────────────────────────────
# QDRANT SETUP
# ─────────────────────────────────────────────────────────


def setup_collection():
    existing = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME in existing:
        count = qdrant.count(COLLECTION_NAME).count
        print(f"✓ Collection '{COLLECTION_NAME}' exists ({count} vectors)")
        ans = input("  Re-create from scratch? (y/N): ").strip().lower()
        if ans != "y":
            print("  Keeping existing — will add new vectors only")
            return
        qdrant.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

    for field, schema in [
        ("doc_type", PayloadSchemaType.KEYWORD),
        ("year",     PayloadSchemaType.INTEGER),
        ("court",    PayloadSchemaType.KEYWORD),
        ("source",   PayloadSchemaType.KEYWORD),
        ("volume",   PayloadSchemaType.INTEGER),
        ("act",      PayloadSchemaType.KEYWORD),
    ]:
        qdrant.create_payload_index(COLLECTION_NAME, field, schema)

    print(f"✓ Collection '{COLLECTION_NAME}' created")

# ─────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────


def chunk_text(text: str) -> list:
    if not text or len(text.strip()) < 100:
        return []

    # Clean up spaces but keep newlines
    text = re.sub(r'[ \t]+', ' ', text)
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        if end >= len(text):
            chunk = text[start:].strip()
            if len(chunk) > 100:
                chunks.append(chunk)
            break

        break_at = text.rfind("\n\n", start, end)
        if break_at <= start:
            break_at = text.rfind(". ", start, end)
        if break_at <= start:
            break_at = end

        chunk = text[start:break_at].strip()
        if len(chunk) > 100:
            chunks.append(chunk)

        # Fail-safe to prevent infinite loops
        next_start = break_at - CHUNK_OVERLAP
        if next_start <= start:
            start += (CHUNK_SIZE - CHUNK_OVERLAP)
        else:
            start = next_start

    return chunks


def build_chunks_from_file(filepath: str) -> list:
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    # FIX: Normalize data so it's ALWAYS a list
    if isinstance(data, dict):
        documents = [data]
    elif isinstance(data, list):
        documents = data
    else:
        return []

    all_chunks = []

    # Loop through every document (works for 1 judgment or 500 act sections)
    for doc in documents:
        text = doc.get("text", "")
        doc_type = doc.get("doc_type", "unknown")

        if not text or len(text.strip()) < 100:
            continue

        raw_chunks = chunk_text(text)

        if not raw_chunks:
            continue

        for i, chunk_text_val in enumerate(raw_chunks):
            if doc_type == "judgment":
                payload = {
                    "doc_type": "judgment", "text": chunk_text_val, "chunk_idx": i,
                    "title":    doc.get("title", ""),
                    "year":     doc.get("year") or 0,
                    "date":     doc.get("date", ""),
                    "court":    doc.get("court", "Supreme Court of India"),
                    "source":   doc.get("source", ""),
                    "filename": doc.get("filename", os.path.basename(filepath)),
                }
            elif doc_type == "act":
                payload = {
                    "doc_type":       "act", "text": chunk_text_val, "chunk_idx": i,
                    "act":            doc.get("act", ""),
                    "act_name":       doc.get("act_name", ""),
                    "section_number": doc.get("section_number", 0),
                    "section_title":  doc.get("title", ""),
                    "source":         doc.get("source", ""),
                    "filename":       os.path.basename(filepath),
                }
            elif doc_type == "cad":
                payload = {
                    "doc_type": "cad", "text": chunk_text_val, "chunk_idx": i,
                    "date":     doc.get("date", ""),
                    "volume":   doc.get("volume", 0),
                    "speaker":  doc.get("speaker", ""),
                    "source":   doc.get("source", "constitutionofindia.net"),
                    "filename": os.path.basename(filepath),
                }
            else:
                payload = {
                    "doc_type": doc_type, "text": chunk_text_val, "chunk_idx": i,
                    "source":   doc.get("source", ""),
                    "filename": os.path.basename(filepath),
                }
            all_chunks.append(payload)

    return all_chunks

# ─────────────────────────────────────────────────────────
# GPU EMBEDDING + UPLOAD
# ─────────────────────────────────────────────────────────


def embed_and_upload(chunks: list, file_label: str = ""):
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"    [X-RAY] Sending data to RTX 3050 GPU...")

    for i in range(0, len(texts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch_texts = texts[i: i + BATCH_SIZE]
        batch_payloads = chunks[i: i + BATCH_SIZE]

        print(f"    [{file_label}] batch {batch_num}/{n_batches} "
              f"({len(batch_texts)} chunks)...", end=" ", flush=True)

        # Embed on GPU
        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
            device=DEVICE
        )

        points = [
            PointStruct(id=str(uuid.uuid4()),
                        vector=emb.tolist(), payload=payload)
            for emb, payload in zip(embeddings, batch_payloads)
        ]

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print("✓")

        # Free GPU memory
        del embeddings, points
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

# ─────────────────────────────────────────────────────────
# PROGRESS
# ─────────────────────────────────────────────────────────


def load_progress() -> set:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_progress(done: set):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(done), f)

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────


def run_ingest():
    if not test_connections():
        return

    print("\n" + "="*55)
    print("STEP 1: Qdrant collection")
    print("="*55)
    setup_collection()

    print("\n" + "="*55)
    print("STEP 2: Collecting JSON files")
    print("="*55)
    print(f"Root: {ROOT}")

    all_files = []
    for doc_type, data_dir in DATA_DIRS.items():
        if not os.path.exists(data_dir):
            print(f"  ⚠ Not found: {data_dir}")
            continue
        files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
        files = [f for f in files
                 if "manifest" not in os.path.basename(f)
                 and "progress" not in os.path.basename(f)]
        print(f"  {doc_type:12s}: {len(files)} files")
        all_files.extend(files)

    print(f"\n  Total: {len(all_files)} files")

    if not all_files:
        print("  ✗ No files found.")
        return

    done_files = load_progress()
    remaining = [f for f in all_files if f not in done_files]
    print(f"  Already ingested: {len(done_files)}")
    print(f"  To process:       {len(remaining)}")

    if not remaining:
        count = qdrant.count(COLLECTION_NAME).count
        print(f"\n✅ All done! Qdrant has {count} vectors.")
        return

    print("\n" + "="*55)
    print(f"STEP 3: Chunking → Embedding ({DEVICE.upper()}) → Qdrant")
    print("="*55)

    total_chunks = 0
    total_files = 0
    failed = []
    start_time = time.time()

    for idx, filepath in enumerate(remaining, 1):
        label = f"{idx}/{len(remaining)}"
        print(f"\n  File {label}: {os.path.basename(filepath)}")

        try:
            chunks = build_chunks_from_file(filepath)

            if not chunks:
                print(f"    Skipped — no usable text")
                done_files.add(filepath)
                continue

            embed_and_upload(chunks, file_label=label)

            total_chunks += len(chunks)
            total_files += 1
            done_files.add(filepath)

            if total_files % 20 == 0:
                save_progress(done_files)
                elapsed = time.time() - start_time
                per_file = elapsed / total_files
                remaining_files = len(remaining) - idx
                eta = remaining_files * per_file
                count = qdrant.count(COLLECTION_NAME).count
                print(f"\n  ── Progress ──────────────────────")
                print(f"  Vectors in Qdrant : {count}")
                print(f"  Speed             : {per_file:.1f}s/file")
                print(f"  ETA               : {eta/60:.1f} mins")
                print(f"  ──────────────────────────────────\n")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed.append(filepath)
            continue

    save_progress(done_files)
    final_count = qdrant.count(COLLECTION_NAME).count
    elapsed = time.time() - start_time

    print("\n" + "="*55)
    print("COMPLETE")
    print("="*55)
    print(f"✓ Files processed:   {total_files}")
    print(f"✓ Chunks created:    {total_chunks}")
    print(f"✓ Vectors in Qdrant: {final_count}")
    print(f"✓ Total time:        {elapsed/60:.1f} mins")
    if failed:
        print(f"✗ Failed:            {len(failed)}")
    print(f"\nNext step: python api/main.py")


if __name__ == "__main__":
    run_ingest()
