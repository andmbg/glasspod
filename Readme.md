# ![](icon.png) Glasspod

A research tool that provides a web interface to search, analyze, and visualize podcast contents.

This project was born from the question, "In which episode did they have that nice comparison...?", followed up by "Actually, how many more times did they talk about this?",
and ultimately, "Was this theme always important over time?"

When faced with a back catalog of hundreds of hour-long episodes of speech, finding quotes is hard, and so is preparing time series analyses or extracting information.

![demo gif](demo_readme.gif)

## Features

- Fast and flexible search of transcripts using
  [Elasticsearch](https://github.com/elastic/elasticsearch).
- Interactive time series visualization for use of terminology, names, or prevalence of
  topics.
- RSS connector, along with an extensible architecture to include other forms of getting
  podcast data (e.g. you may have your own pod locally)


## Built with

- Python
- Pandas
- FastAPI
- Elasticsearch
- wordcloud
- nltk
- Plotly/Dash
- Bootstrap
- Redis

## Getting Started

### Installation

```bash
git clone https://github.com/andmbg/glasspod.git
cd glasspod
make install
```

### Setup

Rename `.env.example` to `.env` and edit:

- set an `API_TOKEN`. This is to protect the `fluesterx` transcription API from access
  by others than yourself. For a secure token, you can use something like

  ```bash
  openssl rand -base64 24
  ```

  ...on the command line and paste it both here and in your API `.env` file.

- in `.env`, set the URL and port of your transcriber API and renderer API

  ```conf
  API_TOKEN=<yourtoken>
  TRANSCRIBER_URL_PORT=http://...:...
  ```

In `config.py`,

- set the project name

  ```py
  PROJECT_NAME = "<name of your project, e.g., the pod's name>"
  ```

- enter the URL of the RSS resource at which the podcast resides.

  ```python
  CONNECTOR_ARGS = {
      "remote_resource": "<RSS URL>"
  }
  ```

### Run

Run the docker compose file, build at first run, and follow the log:

```bash
docker compose up --build -d && docker compose logs -f
```

### Develop

For shorter dev cycles, omit the initial steps where embeddings are created from 
transcripts and both are indexed. For this, thes steps must have been gone through once.

---

**Image credits**: *"professional-retro-microphone--dj-headphones-"* by www.ilmicrofono.it is licensed under CC BY 2.0.