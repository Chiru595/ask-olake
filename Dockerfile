FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HOME=/home/user FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed HF_HOME=/home/user/.cache/huggingface \
    WORKDIR=/home/user/work PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER user
# bake the BM25 model and tokenizer into the image so startup never downloads them
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25'); import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY --chown=user app ./app
COPY --chown=user widget ./widget
RUN mkdir -p /home/user/work

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s CMD curl -fsS http://localhost:${PORT}/health || exit 1
# one worker: the poller and the in-memory cache must live in a single process
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
