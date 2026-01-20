from typing import List
import sqlite3

import pandas as pd
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from podology.search.elasticsearch import (
    get_es_client,
    TRANSCRIPT_INDEX_NAME,
    EPISODE_INDEX_NAME,
)
from podology.search.utils import normalize_column
from config import DB_PATH


def get_term_frequencies(terms: list, es_client: Elasticsearch) -> pd.DataFrame:
    """
    For each episode and term, compute normalized frequency (per 1000 words) from
    Elasticsearch index search.

    Args:
        terms (list): list of search terms
        es_client: Elasticsearch client connected to podology's
            Elasticsearch index

    Returns:
        df: DataFrame with columns [eid, term, count, total_words, freq_per_1k]
    """
    assert len(terms) > 0

    # Get total word counts for all episodes from database
    # (we need all episodes to show zero counts where no hits are found)
    conn = sqlite3.connect(DB_PATH)
    word_counts = pd.read_sql("SELECT eid, count as total_words FROM word_count", conn)
    conn.close()

    all_eids = word_counts["eid"].tolist()

    # Search for each term in Elasticsearch
    term_counts = []

    for term in terms:
        response = es_client.search(
            index=TRANSCRIPT_INDEX_NAME,
            body={
                "query": {"match_phrase": {"text": term}},
                "size": 0,  # Don't return docs, just aggregations
                "aggs": {
                    "by_episode": {
                        "terms": {"field": "eid", "size": 10000}  # Get all episodes
                    }
                },
            },
        )

        # Create a dict for quick lookup of counts per episode
        episode_counts = {}
        for bucket in response["aggregations"]["by_episode"]["buckets"]:
            eid = bucket["key"]
            count = bucket["doc_count"]
            episode_counts[eid] = count

        # Add all episodes for this term, with 0 for those without hits
        for eid in all_eids:
            term_counts.append({
                "eid": eid,
                "term": term,
                "count": episode_counts.get(eid, 0)
            })

    # Convert to DataFrame
    df = pd.DataFrame(term_counts)

    # Merge with word counts
    df = df.merge(word_counts, on="eid", how="left")

    # Calculate frequency (per 1000 words)
    df["freq1k"] = (df["count"] / df["total_words"]) * 1000

    # Fill NAs with zeroes
    df[["count", "freq1k"]] = df[["count", "freq1k"]].fillna(0)

    # Calculate normalized frequency (0..1)
    df["norm"] = normalize_column(df.freq1k, min=0, max=1)

    # add episode metadata like title and pub_date
    conn = sqlite3.connect(DB_PATH)
    unique_eids = df.eid.unique().tolist()
    placeholders = ",".join("?" * len(unique_eids))
    query = f"SELECT eid, title, pub_date FROM episodes WHERE eid IN ({placeholders})"
    ep_metadata = pd.read_sql(query, conn, params=unique_eids)
    ep_metadata.pub_date = ep_metadata.pub_date.apply(pd.Timestamp)
    conn.close()

    df = df.merge(ep_metadata, on="eid", how="left").sort_values("pub_date")

    return df


def get_concept_relevances(concepts, es_client):
    """
    For each episode, compute mean relevance of each concept from Elasticsearch
    chunk embeddings index search.

    Args:
        concepts (list): list of prompts or concepts
        es_client: Elasticsearch client connected to podology's
            Elasticsearch index

    Returns:
        df: DataFrame with columns [eid, concept, avg_relevance]
    """
    assert len(concepts) > 0

    from sentence_transformers import SentenceTransformer
    from config import EMBEDDER_ARGS

    model = SentenceTransformer(EMBEDDER_ARGS["model"])

    concept_relevances = []

    for concept in concepts:
        # Encode concept
        concept_vector = model.encode(concept, normalize_embeddings=True).tolist()

        # Search episode index (FAST - only ~1000 vectors!)
        response = es_client.search(
            index=EPISODE_INDEX_NAME,
            body={
                "knn": {
                    "field": "episode_vector",
                    "query_vector": concept_vector,
                    "k": 1000,  # Get all episodes
                    "num_candidates": 1000,
                },
                "size": 1000,
                "_source": ["eid", "title"],
            },
        )

        # Extract results
        for hit in response["hits"]["hits"]:
            concept_relevances.append(
                {
                    "eid": hit["_source"]["eid"],
                    "concept": concept,
                    "avg_relevance": hit["_score"],
                }
            )

    df = pd.DataFrame(concept_relevances)
    df["norm"] = normalize_column(df.avg_relevance)

    # add episode metadata
    conn = sqlite3.connect(DB_PATH)
    unique_eids = df.eid.unique().tolist()
    placeholders = ",".join("?" * len(unique_eids))
    query = f"SELECT eid, title, pub_date FROM episodes WHERE eid IN ({placeholders})"
    ep_metadata = pd.read_sql(query, conn, params=unique_eids)
    conn.close()

    df = df.merge(ep_metadata, on="eid", how="left").sort_values(
        ["pub_date", "concept"]
    )

    return df
