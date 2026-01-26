from dash import html, dcc


def create_step_flowchart(image_folder="assets/images"):
    """Display the pipeline flowchart SVG."""
    return html.Img(
        src=f"{image_folder}/flowchart.svg",
        style={"width": "100%", "maxWidth": "900px"},
    )


info_text = dcc.Markdown(
    """
    ## Glasspod: Podcast Search and Content Analysis

    This application enables users to analyze and search the episodes of a podcast all
    at once. It implements the full backend pipeline from an RSS location down to
    transcripts searchable both literally and semantically, time series analyses and
    (potentially in the future) LLM applications.

    This flowchart illustrates the step-by-step process Glasspod uses so far.
    Each step represents a key stage in the data processing pipeline, from fetching an
    RSS feeds to displaying visualizations.

    ## Data Pipeline Sketch
    """,
    style={
        "font-family": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji",
    },
)
