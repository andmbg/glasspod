import dash_mantine_components as dmc
from dash import dcc

info_help_modal = dmc.Modal(
    title="More on the Backend",
    id="info-help-modal",
    children=[
        dcc.Markdown(
            """
### More on the Backend

This app connects to a customized transcription API
([fluesterx](https://github.com/andmbg/fluesterx)) that can be run locally or on a
cloud instance, typically with GPU access. For instance an RTX 4090 instance leads to
ca. 15-23x transcription speed depending on the language model used. Cost is ca. $0.30
per hour (Oct 2025). Transcribing 100 one-hour episodes at this rate costs about $1.50,
which is well below commercial transcription services.

### github

Find this project at [github](https://github.com/andmbg/glasspod).
            """,
            style={
                "font-family": (
                    "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "
                    "sans-serif, Apple Color Emoji, Segoe UI Emoji"
                )
            },
        )
    ],
)

episodes_help_modal = dmc.Modal(
    title="The Episodes Tab",
    id="episodes-help-modal",
    children=[
        dcc.Markdown(
            """
### The Episodes Tab

This tab can be a starting point to explore. Where data are present, inspect titles, look at
show notes and word clouds. Click on titles to open that episode in the Within Episode tab.

In a productive instance, this tab works as a directory. In a local instance, it also
allows to trigger processing of yet unprocessed episodes by clicking on the Status column of a
given episode.
            """,
            style={
                "font-family": (
                    "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "
                    "sans-serif, Apple Color Emoji, Segoe UI Emoji"
                )
            },
        )
    ],
)

within_help_modal = dmc.Modal(
    title="Within Episode",
    id="within-help-modal",
    children=[
        dcc.Markdown(
            """
### The Within Episode Tab

In this tab, you can analyze the content of a single episode. The main element is the transcript
on the right.

- Click on the text to start the playback from there.
- Search terms entered in the search box are highlighted in the transcript.

The animated word cloud to the left shows the Named Entities that are close to the current scroll position.

Next to the transcript is the 10,000 foot view:

- Search terms are shown at their occurrence positions.
- Prompts or similarity searches turn into a relative relevance graph across the episode duration.
            """,
            style={
                "font-family": (
                    "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "
                    "sans-serif, Apple Color Emoji, Segoe UI Emoji"
                )
            },
        )
    ],
)

across_help_modal = dmc.Modal(
    title="Across Episodes",
    id="across-help-modal",
    children=[
        dcc.Markdown(
            """
### The Across Episodes Tab

This tab gives insights to developments in time and about relationships among search
terms and semantic prompts.

The **time series** plot at the top shows the frequency of search terms and relevance of
semantic prompts and their importance across time.

The **ranked relevance barplot** ranks episode as to the frequency of literal search
terms or relevance of semantic prompts. Here, the order of terms/prompts matters, and
this is why the arrow button on the tags at the top reorders them. 

Clicking on a given episode takes you to it in the Within Episode tab.
            """,
            style={
                "font-family": (
                    "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "
                    "sans-serif, Apple Color Emoji, Segoe UI Emoji"
                )
            },
        )
    ],
)
