FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only PyTorch keeps the image small; use a GPU base image if you have one.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Bake the embedding model into the image so cold starts don't download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY api ./api
COPY retrieval ./retrieval
COPY frontend ./frontend

ENV PORT=8000 RELOAD=0
EXPOSE 8000
CMD ["sh", "-c", "python api/main.py"]
