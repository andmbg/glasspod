"""
Plotting functions
"""

import os
from typing import List
import sqlite3

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from loguru import logger

from podology.search.elasticsearch import TRANSCRIPT_INDEX_NAME, CHUNK_INDEX_NAME
from podology.data.EpisodeStore import EpisodeStore
from podology.data.Episode import Episode
from podology.data.Transcript import Transcript
from podology.search.search_classes import ResultSet
from podology.stats.preparation import DB_PATH
from podology.frontend.utils import colorway, empty_term_hit_fig
from podology.stats.counting import get_term_frequencies, get_concept_relevances
from config import HITS_PLOT_BINS, EMBEDDER_ARGS


model = SentenceTransformer(EMBEDDER_ARGS["model"])
episode_store = EpisodeStore()
colordict = {i[0]: i[1] for i in colorway}
_transcript_cache = {}


def plot_word_freq(
    key_colid_tuples: List[tuple], es_client: Elasticsearch, template: str = "plotly"
) -> go.Figure:
    """Time series plot of both word frequencies and concept relevances
    in the Across Episodes tab.

    To make them comparable, we standardize both measures to 0..1 and put
    them on the same y axis.
    """
    fig = go.Figure()
    yaxis_range = [0, 1]

    # Part 1: Term frequencies
    term_colid_tuples = [i for i in key_colid_tuples if i[2] == "term"]
    if len(term_colid_tuples) > 0:

        term_colid_dict = {i[0]: i[1] for i in term_colid_tuples}
        dft = get_term_frequencies(list(term_colid_dict), es_client=es_client)
        yaxis_range = [0, dft["freq1k"].max() * 1.1]
        xaxis_range = [dft.pub_date.min(), dft.pub_date.max()]

        for term, grp in dft.groupby("term"):

            fig.add_trace(
                go.Scatter(
                    x=grp["pub_date"],
                    y=grp["freq1k"],
                    yaxis="y",
                    mode="lines+markers",
                    line=dict(
                        color=colordict[term_colid_dict[term]],
                        width=0.5,
                    ),
                    marker=dict(
                        color=colordict[term_colid_dict[term]],
                        size=4,
                    ),
                    name=term,
                    showlegend=True,
                    customdata=grp[["title", "term", "count", "total_words"]],
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "%{y:.2f} words/1000<br>"
                        "%{customdata[2]} occurrences<br>"
                        "%{customdata[3]} total<br><br>"
                        "<i>%{customdata[0]}</i><extra>Term</extra>"
                    ),
                )
            )

    # Part 2: Semantic relevances
    concept_colid_tuples = [i for i in key_colid_tuples if i[2] == "semantic"]
    if len(concept_colid_tuples) > 0:

        concept_colid_dict = {i[0]: i[1] for i in concept_colid_tuples}
        dfs = get_concept_relevances(list(concept_colid_dict), es_client=es_client)
        xaxis_range = [dfs.pub_date.min(), dfs.pub_date.max()]

        for concept, grp in dfs.groupby("concept"):

            fig.add_trace(
                go.Scatter(
                    x=grp["pub_date"],
                    y=grp["norm"],
                    yaxis="y2",
                    mode="lines",
                    line=dict(
                        color=colordict[concept_colid_dict[concept]],
                        width=0.5,
                    ),
                    name=concept,
                    showlegend=True,
                    customdata=grp[["title", "concept"]],
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "%{y:.3f} standardized relevance<br><br>"
                        "<i>%{customdata[0]}</i><extra>Concept</extra>"
                    ),
                )
            )

    fig.update_layout(
        template=template,
        font=dict(size=14),
        plot_bgcolor="rgba(0,0,0, .0)",
        paper_bgcolor="rgba(255,255,255, .0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            y=-0.05,
            yanchor="top",
            orientation="h",
        ),
        yaxis2=dict(
            title=dict(
                text="Standardized Relevance",
                font=dict(color="rgba(128,128,128, .6)", size=22),
            ),
            overlaying="y",
            side="right",
            showgrid=False,  # or True if you want gridlines
            range=[0, 1],
        ),
        yaxis=dict(
            gridcolor=(
                "rgba(255,255,255, .2)"
                if template == "plotly_dark"
                else "rgba(0,0,0, .3)"
            ),
            gridwidth=1,
            griddash="2, 5, 2, 5",
            title=dict(
                text="Occurrences per 1000",
                font=dict(
                    color="rgba(128,128,128, .6)",
                    size=22,
                ),
            ),
            zerolinecolor="rgba(128,128,128, .5)",
            range=yaxis_range,
        ),
        xaxis=dict(
            showgrid=False,
            range=xaxis_range,
        ),
    )

    return fig


def _get_all_episode_term_counts(
    term_colid_tuples: List[tuple], es_client: Elasticsearch
) -> tuple[pd.DataFrame, dict]:
    """For each term from the terms store, count occurrences in transcripts.

    Return a df with a row for each episode (including zero hits). For use
    by the plotting function below.

    Returns:
        pd.DataFrame with cols [
            term, eid, pub_date, title, count, colorid, total, freq1k
        ]
    """
    # Filter out semantic search prompts:
    term_colid_tuples = [i for i in term_colid_tuples if i[2] == "term"]

    term_colid_dict = {i[0]: i[1] for i in term_colid_tuples}
    terms: list[str] = list(term_colid_dict.keys())
    eids = [ep.eid for ep in episode_store if ep.transcript.status]

    # Span all episodes & dates for every term:
    df = pd.MultiIndex.from_product([terms, eids], names=["term", "eid"]).to_frame(
        index=False
    )
    df["pub_date"] = df.eid.apply(lambda x: episode_store[x].pub_date)
    df["title"] = df.eid.apply(lambda x: episode_store[x].title)
    df["count"] = 0
    df["colorid"] = pd.NA
    df.set_index(["term", "eid"], inplace=True)

    # Insert counts of hits into the dataframe:
    result_set = ResultSet(
        es_client=es_client,
        index_name=TRANSCRIPT_INDEX_NAME,
        term_colorids=term_colid_tuples,
    )
    for term, hits in result_set.term_hits.items():
        df.loc[(term, slice(None)), "colorid"] = term_colid_dict[term]

        for hit in hits:
            eid = hit["_source"]["eid"]
            df.loc[(term, eid), "count"] += 1

    df.reset_index(inplace=True)

    # Add total word counts of each episode:
    unique_eps = tuple(df.eid.unique())
    unique_eps_query = ",".join(["?"] * len(unique_eps))
    query = (
        f"select eid, count as total from word_count where eid in ({unique_eps_query})"
    )
    word_counts = pd.read_sql(
        query,
        sqlite3.connect(DB_PATH),
        params=unique_eps,
    )

    df = pd.merge(df, word_counts, on="eid", how="left")
    df["freq1k"] = df["count"] / df["total"] * 1000

    df.sort_values("pub_date", inplace=True)

    return df, term_colid_dict


def plot_transcript_hits_es(
    term_colid_tuples: List[list],
    eid: str,
    es_client: Elasticsearch,
    nbins: int = HITS_PLOT_BINS,
) -> go.Figure:
    """Plot the vertical column plot for transcript hits.

    Use elastic to find terms, glean each hit's timing from a transcript word list
    method usage, cached upon first use.
    """
    if not term_colid_tuples:
        return empty_term_hit_fig

    # Get episode duration for binning
    episode = EpisodeStore()[eid]
    duration = episode.duration

    # Create time bins
    bin_edges = np.linspace(0, duration, nbins + 1)
    all_bins = np.arange(nbins)
    allbins_df = pd.DataFrame({"bin": all_bins})

    # Search each term in Elasticsearch, index and search method depending on term_or_prompt
    # Target shape per term:
    # list(time_1, time_2, ...)
    for term, colorid, term_or_semantic in term_colid_tuples:

        # Textual search terms:
        if term_or_semantic == "term":
            hit_positions = _search_term_positions(
                es_client, eid, term, term_or_semantic
            )

            # Bin the hit positions
            if hit_positions:
                hit_bins = pd.cut(
                    hit_positions, bins=bin_edges, labels=False, include_lowest=True
                )
                term_counts = (
                    pd.Series(hit_bins).value_counts().reindex(all_bins, fill_value=0)
                )

            else:
                term_counts = pd.Series(0, index=all_bins)

            allbins_df[term] = term_counts

        elif term_or_semantic == "semantic":
            relevances = get_chunk_similarities(es_client, episode, term)
            allbins_df[term] = relevances["similarity"].values

    allbins_df.set_index("bin", inplace=True)

    # Create plot (same as before)
    return _create_term_hits_plot(allbins_df, term_colid_tuples)


def _create_term_hits_plot(
    allbins_df: pd.DataFrame, term_colid_tuples: list[list]
) -> go.Figure:
    """
    Create a bar plot for the hit counts of each term.

    Args:
        allbins_df: DataFrame with hit counts for each term and bin
        term_colid_dict: Dictionary mapping term names to color IDs

    Returns:
        Plotly Figure object
    """

    semantic_cols = [term for term, _, type in term_colid_tuples if type == "semantic"]
    term_cols = [term for term, _, type in term_colid_tuples if type == "term"]

    maxrange = allbins_df[term_cols].apply(sum, axis=1).max()
    max_similarity = allbins_df[semantic_cols].max().max() if semantic_cols else 0
    min_similarity = allbins_df[semantic_cols].min().min() if semantic_cols else 0

    # allbins_df[semantic_cols] = allbins_df[semantic_cols].apply(np.exp)

    fig = go.Figure()

    # col is at the same time part of the tuples (we surmise):
    for i, col in enumerate(allbins_df.columns):
        term, colid, term_or_prompt = term_colid_tuples[i]

        if term_or_prompt == "term":
            fig.add_trace(
                go.Bar(
                    y=-allbins_df.index,
                    x=allbins_df[col],
                    orientation="h",
                    marker=dict(
                        line_width=0,
                        color=(colordict[colid] if term else "grey"),
                    ),
                    xaxis="x",
                ),
            )

        elif term_or_prompt == "semantic":
            # max_similarity = max(max_similarity, allbins_df[col].max())
            # min_similarity = max(min_similarity, allbins_df[col].min())
            fig.add_trace(
                go.Scatter(
                    y=-allbins_df.index,
                    x=allbins_df[col],
                    mode="lines",
                    line=dict(
                        width=1,
                        color=(colordict[colid] if term else "grey"),
                    ),
                    opacity=0.8,
                    xaxis="x2",
                ),
            )

    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        plot_bgcolor="rgba(100,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            title=None,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-allbins_df.index.max(), 0],
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, maxrange],
            domain=[0, 1],
        ),
        xaxis2=dict(
            title=None,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[
                min_similarity,
                max_similarity * 1.01 if max_similarity > 0 else 1.0,
            ],
            overlaying="x",  # Overlay on the primary x-axis
            domain=[0, 1],
        ),
        barmode="stack",
        bargap=0.02,
    )

    return fig


def _search_term_positions(
    es_client: Elasticsearch, eid: str, term: str, term_or_prompt: str
) -> List[float]:
    """
    Search for term positions using Elasticsearch highlight with positions.
    """
    query = {
        "query": {
            "bool": {
                "must": [{"match": {"eid": eid}}, {"match_phrase": {"text": term}}]
            }
        },
        "highlight": {
            "fields": {
                "text": {
                    "fragment_size": 0,
                    "number_of_fragments": 0,
                    "pre_tags": ["<START>"],
                    "post_tags": ["<END>"],
                }
            }
        },
        "_source": ["start", "text"],
        "size": 1000,
    }
    transcript = _get_transcript_with_elastic_ids(eid)

    try:
        response = es_client.search(index=TRANSCRIPT_INDEX_NAME, body=query)

        positions = []
        for hit in response["hits"]["hits"]:
            if "highlight" in hit:
                hit_elastic_id = hit["_id"]
                hit_transcript_seg = transcript.loc[
                    transcript.elastic_id == hit_elastic_id
                ]
                hit_transcript_seg_len = hit_transcript_seg.shape[0]

                hl_text_words = hit["highlight"]["text"][0].split()
                hit_indices = [
                    min(i, hit_transcript_seg_len - 1)
                    for i, word in enumerate(hl_text_words)
                    if "START" in word
                ]
                hit_times = hit_transcript_seg.iloc[hit_indices].start.values.tolist()
                positions.extend(hit_times)

        return positions

    except Exception as e:
        logger.error(
            f"Error searching term positions for '{term}' in episode {eid}: {e}"
        )
        return []


def get_chunk_similarities(
    es_client: Elasticsearch,
    episode: Episode,
    term: str,
) -> pd.DataFrame:
    """
    Get relevance scores for a term or prompt using Elasticsearch.
    """
    vector_query = {
        "query": {"bool": {"must": [{"match": {"eid": episode.eid}}]}},
        "knn": {
            "field": "embedding",
            "query_vector": _get_embedding(term),
            "k": 1000,
            "num_candidates": 1000,
            "filter": {"term": {"eid": episode.eid}},
        },
        # "_source": ["eid", "text", "start", "end", "title"]
        "size": 1000,
    }

    response = es_client.search(index=CHUNK_INDEX_NAME, body=vector_query)

    chunk_similarities = [
        {
            "start": hit["_source"]["start"],
            "end": hit["_source"]["end"],
            "similarity_score": hit["_score"],
        }
        for hit in response["hits"]["hits"]
    ]
    relevance_df = pd.DataFrame(chunk_similarities).sort_values("start")
    binned_relevance = bin_relevance_scores(
        relevance_df, ep_duration=episode.duration, n_bins=HITS_PLOT_BINS
    )

    return binned_relevance


def _get_embedding(term: str) -> List[float]:
    """Return an L2-normalized query vector (matches indexed vectors)."""
    try:
        vec = model.encode(term, normalize_embeddings=True)
    except TypeError:
        vec = model.encode(term)
        a = np.asarray(vec, dtype=float)
        n = np.linalg.norm(a) + 1e-12
        vec = a / n
    # ensure list
    return list(vec)


def _get_transcript_with_elastic_ids(eid: str) -> pd.DataFrame:
    """Prep df in which to find the timing for search terms.

    Cache transcript data with elastic segment IDs to avoid recomputation.
    """
    if eid in _transcript_cache:
        return _transcript_cache[eid]

    transcript = Transcript(EpisodeStore()[eid]).words(
        word_attr=["word", "start"],
        seg_attr=["seg_start", "seg_end"],
    )

    # More efficient string concatenation
    transcript["elastic_id"] = (
        eid
        + "_"
        + transcript["seg_start"].astype(str)
        + "_"
        + transcript["seg_end"].astype(str)
    )

    # Create lookup dict for faster access
    _transcript_cache[eid] = transcript
    return transcript


def bin_relevance_scores(relevance_df, ep_duration, n_bins=500):
    """
    Bin relevance scores into time-based bins, averaging overlapping chunks.
    """
    if relevance_df.empty:
        return pd.DataFrame({"bin_start": [], "bin_end": [], "avg_similarity": []})

    # Get the total time span
    min_time = 0
    max_time = ep_duration

    # Create bin edges
    bin_edges = np.linspace(min_time, max_time, n_bins + 1)

    # Calculate bin centers and create result dataframe

    binned_scores = []

    for i in range(n_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        # Find chunks that overlap with this bin
        overlapping_chunks = relevance_df[
            (relevance_df["start"] < bin_end) & (relevance_df["end"] > bin_start)
        ]

        if len(overlapping_chunks) > 0:
            # Calculate overlap weights for averaging
            weighted_scores = []
            total_weight = 0

            for _, chunk in overlapping_chunks.iterrows():
                # Calculate overlap duration
                overlap_start = max(chunk["start"], bin_start)
                overlap_end = min(chunk["end"], bin_end)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > 0:
                    # Weight by overlap duration
                    weighted_scores.append(chunk["similarity_score"] * overlap_duration)
                    total_weight += overlap_duration

            if total_weight > 0:
                avg_score = sum(weighted_scores) / total_weight
            else:
                avg_score = overlapping_chunks["similarity_score"].mean()
        else:
            # No chunks in this bin
            avg_score = np.nan

        binned_scores.append(avg_score)

    return pd.DataFrame(
        {
            "similarity": binned_scores,
        }
    )


def get_relevance_bars(term_colid_tuples, es_client) -> go.Figure:
    """Plot sorted frequency or relevance bars.

    Args:
        terms (list): list of terms
        es_client (_type_): _description_

    Returns:
        go.Figure: _description_
    """
