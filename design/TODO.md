# Features to do

- [#B] Writeback: When looking at a transcript, I want to be able to
  - re-label speakers
  - correct mistakes
  - ideally, after correction, aggregate a dataset to fine-tune the Whisper model

## Meta/Download tab:

- [#B] separate cols for each status
- BUG: min size of tooltip is too big, short notes leave empty space
  Ideal: Size of wordcloud + Title + 2-5 lines of notes
- make click on wordcloud select episode & transcript tab

## Within Episode tab:

- more speaker display classes

## Across Episodes tab:

- Timeseries: switch between occurrences per 1000 and absolute count on y-axis

---

# Features NOT to do

- After a startup that creates the search index but doesn't fill it for some reason, currently, a startup sees the 
  index and doesn't question its completeness. We need to implement a check that verifies the index's completeness 
  and triggers a reindex if necessary.  
  - > Too much work for a very edge case. We can live with that.
  
# IMMINENT

- We can also store frequencies/relevances in a Store on the client side, so we don't have to get relevance (2 secs runtime in dev) for every key everytime a literal search term changes or so.
- At some point throughout the transcript, the highlight position and playback position get out of sync for most (not all) episodes.


# Bugs

- process depiction on page 1 sensitive against screen size
- Search for term "Jones." does not highlight "Jones." in transcript (but in the hits column).
- Search for "Pamporio" highlights "Pamporio's" inline but doesn't show a hit. It requires searching "Pamporio's".
- if one search term contains another, highlights in the transcript get messed up.
- When database-writing operations (like transcript post-processing) are done wholesale with concurrent processing, we get sqlite3 lock errors due to the db being locked on file-level. Writes may have to be queued.
- API has alignment model hardcoded even though some code suggests parameterization.
