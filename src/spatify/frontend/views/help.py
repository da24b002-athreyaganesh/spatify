"""Help: the in-app user manual, written for someone who has never used Spatify."""

import streamlit as st

from spatify.frontend import ui

ui.page_view("help")

st.title("Help")
st.write("Spatify suggests songs that sound like the ones you love. No account, no history needed.")

st.header("Get recommendations in three steps")
st.markdown(
    """
1. **Start from a song (optional).** On **Discover**, type part of a title or an artist,
   then pick the song from the list.
2. **Describe the feel (optional).** Switch on any slider that matters to you, such as
   *Energy* or *Mood*, and move it. Anything left off stays open.
3. **Press "Get recommendations".** You need a song, a slider, or both.

Then press **Like** or **Skip** on each suggestion. Your feedback teaches Spatify what
listeners enjoy, and the next version of the model learns from it.
"""
)

st.header("What the sliders mean")
rows = "\n".join(f"| {s.label} | {s.hint} |" for s in ui.SLIDERS + ui.MORE_SLIDERS)
st.markdown(
    "| Slider | From → to |\n|---|---|\n" + rows + "\n| Key | minor (darker) or major (brighter) |"
)
st.caption(
    "With a song and sliders together, the sliders change that song's character: "
    "for example, the same song but calmer."
)

st.header("Reading a recommendation")
st.markdown(
    """
- **Similarity** runs from -1 to 1; the closer to 1, the closer the song is to what you asked for.
- **Why** names what the suggestion shares with your song or your sliders, strongest first:
  for example *energetic, danceable*. These are the features that add most to its similarity.
- A **wildcard** is a song picked at random, about one in ten. Liking or skipping it teaches
  Spatify about songs it wouldn't have chosen for you.
- Each list says which **model version** chose it, so the team can see which version you liked.
"""
)

st.header("If something goes wrong")
st.markdown(
    """
| What you see | What to do |
|---|---|
| "Spatify can't reach its server right now" | It's starting up. Try again in a few seconds. |
| "No songs match that" | Check the spelling, try the artist's name, or use only the sliders. |
| "That track isn't in the current model's catalogue" | The model was just updated. Search again. |
| The button is greyed out | Pick a song or switch on at least one slider first. |
"""
)

st.header("For operators")
st.markdown(
    """
The **Operator console** shows:

- **Pipeline:** the DVC pipeline as a diagram, coloured by each stage's last result, with the
  throughput of the data stages and the history of pipeline runs. *Run the pipeline now*
  starts a run.
- **Models:** every registered model version, which one is serving (the *champion*), its
  like rate, and the data and code it came from. *Roll back* serves an earlier version again;
  listeners get it within about 15 seconds.
- **Data drift:** how far new data has moved from the training data, and *Retrain now*.
- **Tools:** links to MLflow, Grafana, Prometheus, the Ray Dashboard and the API docs.

Rolling back and starting runs need the **admin key**: type it into the console's sidebar.
It's `SPATIFY_API_KEY` in the project's `.env` file.
"""
)
