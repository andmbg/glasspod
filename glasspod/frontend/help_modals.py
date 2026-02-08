import dash_mantine_components as dmc
from dash import dcc

MODAL_FONT_STYLE = {
    "font-family": (
        "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "
        "sans-serif, Apple Color Emoji, Segoe UI Emoji"
    )
}

info_help_modal = dmc.Modal(
    title="Under the Hood",
    id="info-help-modal",
    size="lg",
    children=[
        dcc.Markdown(
            """
### How It Works

Glasspod processes podcast episodes through a multi-stage pipeline. Starting from an
RSS feed, episodes are downloaded, transcribed with speaker diarization, chunked, and
embedded into a vector space for semantic search.

**Transcription** is handled by a custom API
([fluesterx](https://github.com/andmbg/fluesterx)) built on WhisperX. It can run
locally or on a cloud GPU instance. An RTX 4090, for example, transcribes at roughly
15--23x realtime speed depending on the language model, at about $0.30/hour
(as of late 2025). That puts the cost of transcribing 100 one-hour episodes at around
$1.50 -- well below commercial transcription services.

**Search** comes in two flavors. Literal search uses Elasticsearch's full-text engine
to find exact term matches across all transcripts. Semantic search encodes your prompt
with a sentence transformer model and compares it against precomputed episode-level
embeddings via kNN, surfacing episodes by topical relevance rather than exact wording.

**Indexing and embeddings** are computed once per episode. Transcripts are split into
overlapping chunks of 100--150 words, each embedded independently. Episode-level vectors
are then derived by mean-pooling all chunk vectors and L2-normalizing the result.

### Source Code

Find the full project on [GitHub](https://github.com/andmbg/glasspod).
            """,
            style=MODAL_FONT_STYLE,
        )
    ],
)

episodes_help_modal = dmc.Modal(
    title="Episode List",
    id="episodes-help-modal",
    size="lg",
    children=[
        dcc.Markdown(
            """
### Episode List

This table is your directory of all episodes in the feed. Each row shows the episode's
publication date, title, duration, and processing status.

**Exploring episodes:** Hover over a title to see the show notes in a tooltip. If a
word cloud has been generated for the episode, it appears there as well.

**Opening an episode:** Click on the title of any fully processed episode to jump
straight into its transcript in the *Within Episode* tab.

![](assets/images/scrshot_episodelist.png)

**Sorting and filtering:** The table supports sorting by any column (click the header)
and text filtering. Use the date or title filters to narrow things down quickly.

**Triggering processing** (local instances only): In a local development setup, you
can click the *Status* column of an unprocessed episode to queue it for transcription.
The status indicator updates in real time as the job progresses.
            """,
            style=MODAL_FONT_STYLE,
        )
    ],
)

within_help_modal = dmc.Modal(
    title="Within Episode",
    id="within-help-modal",
    size="lg",
    children=[
        dcc.Markdown(
            """
### Within Episode

This tab lets you dive into a single episode's content. It is divided into three areas
that work together:

**The transcript** (center) is the full diarized text of the episode. Speaker turns are
visually separated, and timestamps let you orient yourself within the recording.

- **Playback:** Click anywhere in the transcript to start audio playback from that point.
- **Search highlighting:** Any literal search terms you've entered are highlighted
  in-place, so you can scan for where and how often a term appears.

![](assets/images/scrshot_transcript.png)

**The word ticker** (left) is an animated display of named entities -- people, places,
organizations -- extracted from the transcript. As you scroll, it updates to reflect
the entities near your current reading position, giving you a sense of who and what
is being discussed at any given point.

**The hit strip** (right) is a bird's-eye view of the entire episode. For literal
search terms, colored markers show exactly where each occurrence falls in the episode
timeline. For semantic prompts, it becomes a continuous relevance gradient -- a heatmap
of how topically relevant each section is to your query. A small overlay tracks your
current scroll position within this view.

            """,
            style=MODAL_FONT_STYLE,
        )
    ],
)

across_help_modal = dmc.Modal(
    title="Across Episodes",
    id="across-help-modal",
    size="lg",
    children=[
        dcc.Markdown(
            """
### Across Episodes

This tab reveals patterns across the entire podcast catalogue. It responds to the
search terms and semantic prompts you've entered at the top of the page.

**Time series** (top): Each search term or prompt is plotted as a line over time,
one data point per episode. For literal terms, the y-axis shows normalized frequency
(occurrences per 1,000 words). For semantic prompts, it shows the similarity score
between your prompt and each episode's embedding. This makes it easy to spot trends,
recurring topics, or sudden spikes in attention.

![](assets/images/scrshot_timeseries.png)

**Ranked relevance** (bottom): All episodes are ranked by how relevant they are to
your search terms. The ranking order is determined by the first tag in your search
list, which is why the arrow button on each tag matters -- it moves a tag to the
front, re-sorting the bar chart by that term. Bars are color-coded to match their
tags, and hovering shows exact counts and scores.

For both plots, **clicking on an episode** takes you directly to it in the
*Within Episode* tab.

**Combining literal and semantic search** is where this tab shines. Try pairing
an exact term (like a person's name) with a broader semantic prompt (like a topic)
to see how they correlate over time and which episodes sit at their intersection.
            """,
            style=MODAL_FONT_STYLE,
        )
    ],
)
