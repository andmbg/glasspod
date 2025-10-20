import dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash import html, dcc


def create_step_flowchart(image_folder="assets/images"):
    """Create the vertical flowchart on the info tab."""

    steps = [
        {
            "image": "fc_rss.png",
            "text": "User supplies the podcast's RSS feed URL, which we download and parse to populate the Episode List.",
        },
        {
            "image": "fc_ar_0.png",
            "text": "",
        },
        {
            "image": "fc_audio.png",
            "text": "Given that the podology transcriber API is running somewhere (e.g. vast.ai is cheap) User selects episodes and they are downloaded and sent to the API for transcription.",
        },
        {
            "image": "fc_ar_1.png",
            "text": "",
        },
        {
            "image": "fc_diarized.png",
            "text": "We end up with a diarized transcript for each episode, which we store in our database.",
        },
        {
            "image": "fc_ar_2.png",
            "text": "",
        },
        {
            "image": "fc_diarized_html.png",
            "text": "This transcript is displayed in the Within Episode tab.",
        },
        {
            "image": "fc_ar_3.png",
            "text": "",
        },
        {
            "image": "fc_cloud.png",
            "text": "We identify Named Entities and create a word cloud that helps scan long episodes for interesting terms.",
        },
        {
            "image": "fc_ar_4.png",
            "text": "",
        },
        {
            "image": "fc_elastic1.png",
            "text": "The whole transcript is indexed in an Elasticsearch instance to make it searchable across episodes.",
        },
        {
            "image": "fc_ar_5.png",
            "text": "",
        },
        {
            "image": "fc_search.png",
            "text": "User provides search keywords or phrases to find within and across episodes.",
        },
        {
            "image": "fc_ar_6.png",
            "text": "",
        },
        {
            "image": "fc_search_display.png",
            "text": "The scroll bar shows concentrations of search terms within the episode transcript.",
        },
        {
            "image": "fc_ar_7.png",
            "text": "",
        },
        {
            "image": "fc_searchplot.png",
            "text": "The time series plot shows the relative frequency of search terms across all episodes.",
        },
        {
            "image": "fc_ar_8.png",
            "text": "",
        },
        {
            "image": "fc_vectors.png",
            "text": "Transcripts get chunked and embedded into vector representations for semantic search and downstream LLM applications.",
        },
        {
            "image": "fc_ar_9.png",
            "text": "",
        },
        {
            "image": "fc_elastic2.png",
            "text": "Those vectors are also stored in Elasticsearch to enable hybrid search capabilities.",
        },
        {
            "image": "fc_ar_a.png",
            "text": "",
        },
        {
            "image": "fc_prompt.png",
            "text": "Switching the search input to Prompt Mode allows semantic search, i.e. search by semantic similarity.",
        },
        {
            "image": "fc_ar_b.png",
            "text": "",
        },
        {
            "image": "fc_prompt_display.png",
            "text": "Similarity or relevance is displayed also next to the transcript, along with text search hits.",
        },
    ]

    flowchart_items = []

    for i, step in enumerate(steps):
        step_item = dmc.Grid(
            [
                # Left column: Image
                dmc.GridCol(
                    html.Img(
                        src=f"/{image_folder}/{step['image']}",
                        style={
                            "maxWidth": "100%",
                            "display": "block",
                        },
                    ),
                    span=3,
                    style={"padding": "0"},
                ),
                # Right column: Text content
                dmc.GridCol(
                    dmc.Text(step["text"], c="dimmed", size="sm"),
                    span=9,
                    style={
                        "paddingLeft": "20px",
                        "display": "flex",
                        "alignItems": "center",
                    },
                ),
            ],
            gutter=0,
            style={"margin": "0", "rowGap": "0"},
        )

        flowchart_items.append(step_item)

    return dmc.Stack(flowchart_items, style={"rowGap": "0"})


info_text = dcc.Markdown(
    """
    ## Podology: Podcast Content Analysis and Search Tool

    This application enables users to analyze and search podcast episodes.
    Run locally, it implements the full backend pipeline from an RSS location to searchable transcripts, analyses and LLM applications.

    This flowchart illustrates the step-by-step process Podology uses so far.
    Each step represents a key stage in the data processing pipeline, from fetching RSS feeds to displaying visualizations.

    ## Data Pipeline Sketch
    """,
    style={
        "font-family": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji",
    },
)
