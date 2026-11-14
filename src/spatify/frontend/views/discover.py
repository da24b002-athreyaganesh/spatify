"""Discover: recommendations from a seed song, the feel you describe, or both; then like or
skip each one. Designed to need no instructions: a first-time listener should get songs in
about 30 seconds (the problem statement's usability target)."""

import streamlit as st

from spatify.frontend import api, ui
from spatify.frontend.metrics import FEEDBACK_CLICKS

ui.page_view("discover")
st.session_state.setdefault("feedback", {})


@st.cache_data(ttl=60, show_spinner=False)
def search(query: str) -> dict:
    return api.search(query, limit=10)


def send_feedback(request_id: str, track_id: str, action: str) -> None:
    """Button callback: runs before the page redraws, so the buttons show the new state."""
    try:
        api.feedback(request_id, track_id, action)
    except api.GatewayError as err:
        st.session_state.feedback_error = err
        return
    st.session_state.feedback[track_id] = action
    st.session_state.thanks = True
    FEEDBACK_CLICKS.labels(action).inc()


st.title("Discover music")
st.write(
    "Start from a song you like, describe the feel you want, or both. "
    "Then tell Spatify what you think of each suggestion."
)

# 1. A seed song (optional)
st.subheader("1. Start from a song", help="Optional. Spatify finds songs that sound like it.")
query = st.text_input(
    "Search by title or artist",
    max_chars=100,
    placeholder="For example: Unholy, or Bad Bunny",
    key="search",
).strip()
seed = None
if len(query) == 1:
    st.caption("Type at least two letters.")
elif query:
    try:
        found = search(query)
    except api.GatewayError as err:
        ui.show_error(err, "search")
        found = None
    if found is not None and not found["tracks"]:
        st.info("No songs match that. Check the spelling, or skip this step and use the sliders.")
    elif found is not None:
        seed = st.selectbox(
            "Pick the song",
            found["tracks"],
            index=None,
            format_func=ui.track_label,
            placeholder=f"{len(found['tracks'])} matches: pick one",
            key=f"seed-{query.lower()}",  # a new list for each search
        )

# 2. The feel (optional)
st.subheader(
    "2. Describe the feel",
    help="Optional. Switch on only what matters to you; anything left off stays open.",
)
sliders = {}
for spec in ui.SLIDERS:
    value = ui.slider(spec)
    if value is not None:
        sliders[spec.key] = value
with st.expander("More options"):
    for spec in ui.MORE_SLIDERS:
        value = ui.slider(spec)
        if value is not None:
            sliders[spec.key] = value
    mode = ui.key_choice()
    if mode is not None:
        sliders["mode"] = mode

k = st.slider("How many songs", min_value=1, max_value=50, value=10, key="k")
ready = seed is not None or bool(sliders)
if not ready:
    st.caption("Pick a song, or switch on at least one slider, to get recommendations.")
if st.button(
    "Get recommendations",
    type="primary",
    icon=":material/queue_music:",
    disabled=not ready,
    key="go",
):
    with st.spinner("Finding songs for you…"):
        try:
            st.session_state.result = api.recommend(seed["track_id"] if seed else None, sliders, k)
            st.session_state.feedback = {}
        except api.GatewayError as err:
            ui.show_error(err, "recommend")

# 3. The recommendations
result = st.session_state.get("result")
if "feedback_error" in st.session_state:
    ui.show_error(st.session_state.pop("feedback_error"), "feedback")
if st.session_state.pop("thanks", False):
    st.toast("Thanks! Your likes and skips train the next version.", icon=":material/favorite:")
if result:
    st.divider()
    st.subheader("Your recommendations")
    version = result.get("model_version")
    st.caption(
        (f"Chosen by model version {version}. " if version else "")
        + "Like or skip each song: your feedback teaches Spatify what you enjoy."
    )
    if not result["recommendations"]:
        st.info("Nothing came back. Try another song, or switch off a slider or two.")
    request_id = result["request_id"]
    for rec in result["recommendations"]:
        track_id = rec["track_id"]
        sent = st.session_state.feedback.get(track_id)
        with st.container(border=True):
            text, buttons = st.columns([5, 2], vertical_alignment="center")
            with text:
                st.markdown(
                    f"**{rec['rank']}. {ui.md(rec['track_name'] or 'Untitled')}**  \n"
                    f"{ui.md(rec['artists'] or 'Unknown artist')}"
                    + (f" · *{ui.md(rec['album_name'])}*" if rec.get("album_name") else "")
                )
                genres = ", ".join(rec["genres"])
                st.caption((f"{genres} · " if genres else "") + f"similarity {rec['score']:.2f}")
                if rec.get("why"):
                    st.markdown("**Why:** " + ", ".join(rec["why"]))
            with buttons:
                like, skip = st.columns(2)
                like.button(
                    "Like",
                    icon=":material/thumb_up:",
                    key=f"like-{request_id}-{track_id}",
                    disabled=sent == "like",
                    on_click=send_feedback,
                    args=(request_id, track_id, "like"),
                    width="stretch",
                )
                skip.button(
                    "Skip",
                    icon=":material/thumb_down:",
                    key=f"skip-{request_id}-{track_id}",
                    disabled=sent == "skip",
                    on_click=send_feedback,
                    args=(request_id, track_id, "skip"),
                    width="stretch",
                )
                if sent:
                    st.caption("You liked this." if sent == "like" else "You skipped this.")
