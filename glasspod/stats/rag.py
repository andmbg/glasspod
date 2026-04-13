"""rag.py - Retrieval-Augmented Generation over podcast transcript chunks."""

import os

from loguru import logger
from sentence_transformers import SentenceTransformer

from config import EMBEDDER_ARGS
from glasspod.search.elasticsearch import CHUNK_INDEX_NAME

_model = SentenceTransformer(EMBEDDER_ARGS["model"])


def retrieve_chunks(question: str, es_client=None, top_k: int = 8) -> list[dict]:
    """
    Run kNN search across all episodes in the chunk index and return the
    top_k most relevant chunks with their source metadata.

    es_client is optional: if None, a new client is created (needed for
    background callbacks which run in a separate process).
    """
    from glasspod.search.elasticsearch import get_es_client

    query_vector = _model.encode(question, normalize_embeddings=True).tolist()

    if es_client is None:
        es_client = get_es_client()

    knn_response = es_client.search(
        index=CHUNK_INDEX_NAME,
        body={
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k * 3,
                "num_candidates": top_k * 50,
            },
            "size": top_k * 3,
            "_source": ["text", "title", "pub_date", "start"],
        },
    )

    bm25_response = es_client.search(
        index=CHUNK_INDEX_NAME,
        body={
            "query": {"match": {"text": question}},
            "size": top_k * 3,
            "_source": ["text", "title", "pub_date", "start"],
        },
    )

    # Manual RRF: score = sum of 1/(rank + 60) across both result lists
    rrf_scores: dict[str, float] = {}
    sources: dict[str, dict] = {}

    for rank, hit in enumerate(knn_response["hits"]["hits"]):
        doc_id = hit["_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 60)
        sources[doc_id] = hit["_source"]

    for rank, hit in enumerate(bm25_response["hits"]["hits"]):
        doc_id = hit["_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 60)
        sources[doc_id] = hit["_source"]

    top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

    chunks = []
    for doc_id in top_ids:
        src = sources[doc_id]
        try:
            start_sec = float(src.get("start", 0) or 0)
            timestamp = f"{int(start_sec // 60)}:{int(start_sec % 60):02d}"
        except (ValueError, TypeError):
            timestamp = "?"
        chunks.append(
            {
                "text": src.get("text", ""),
                "title": src.get("title", ""),
                "pub_date": (src.get("pub_date", "") or "")[:10],
                "timestamp": timestamp,
                "score": rrf_scores[doc_id],
            }
        )
    return chunks


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Assemble a prompt from retrieved chunks and call the Claude API to generate
    an answer grounded in the podcast transcripts.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "_Error: `ANTHROPIC_API_KEY` environment variable is not set._"

    context_parts = []
    for chunk in chunks:
        header = f'**"{chunk["title"]}"** ({chunk["pub_date"]}, ~{chunk["timestamp"]})'
        context_parts.append(f"{header}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "The following are excerpts from podcast transcripts. "
        "Use them to answer the question. "
        "For each statement, make reference to the episode title. "
        "If the excerpts don't contain sufficient information, say so clearly.\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    for i, chunk in enumerate(chunks):
        logger.debug(
            f"RAG chunk {i+1}/{len(chunks)} | score={chunk['score']:.3f} | "
            f"{chunk['title']!r} @{chunk['timestamp']}\n{chunk['text']}"
        )
    logger.debug(f"Sending prompt of length {len(prompt)} to Anthropic")
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    logger.debug("Response received")
    logger.info(f"RAG answer generated for question: {question[:60]!r}")
    return message.content[0].text
