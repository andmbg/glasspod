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

    response = es_client.search(
        index=CHUNK_INDEX_NAME,
        body={
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 15,
            },
            "size": top_k,
            "_source": ["text", "title", "pub_date", "start"],
        },
    )

    chunks = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        try:
            start_sec = float(src.get("start", 0) or 0)
            minutes = int(start_sec // 60)
            seconds = int(start_sec % 60)
            timestamp = f"{minutes}:{seconds:02d}"
        except (ValueError, TypeError):
            timestamp = "?"
        chunks.append(
            {
                "text": src.get("text", ""),
                "title": src.get("title", ""),
                "pub_date": (src.get("pub_date", "") or "")[:10],
                "timestamp": timestamp,
                "score": hit["_score"],
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
    logger.debug(f"Sending prompt of length {len(prompt)} to Anthropic")
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    logger.debug("Response received")
    logger.info(f"RAG answer generated for question: {question[:60]!r}")
    return message.content[0].text
