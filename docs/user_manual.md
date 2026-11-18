# Spatify user manual

Spatify suggests songs that sound like the ones you love. You don't need an account or a listening history: pick a song you like, or describe the kind of music you're in the mood for, and Spatify finds songs that match. Tell it which suggestions you like, and it gets better for everyone.

This manual is for anyone using Spatify; you don't need to know anything about computers or music theory. Sections 1–7 cover finding music. Section 8 is for the person who looks after Spatify. Appendix A explains how to start it.

**Contents**
1. [Opening Spatify](#1-opening-spatify)
2. [Finding music from a song you like](#2-finding-music-from-a-song-you-like)
3. [Finding music by describing the feel](#3-finding-music-by-describing-the-feel)
4. [Reading your recommendations](#4-reading-your-recommendations)
5. [Liking and skipping](#5-liking-and-skipping)
6. [Ideas to try](#6-ideas-to-try)
7. [If something goes wrong](#7-if-something-goes-wrong)
8. [For the person looking after Spatify](#8-for-the-person-looking-after-spatify)
9. [What Spatify keeps](#9-what-spatify-keeps)
10. [Words used in Spatify](#10-words-used-in-spatify)
- [Appendix A: Starting and stopping Spatify](#appendix-a-starting-and-stopping-spatify)

---

## 1. Opening Spatify

1. Open a web browser (Chrome, Firefox, Edge or Safari) on the computer Spatify runs on.
2. Go to **http://localhost:8501**.
3. The **Discover** page opens. The menu on the left also has **Operator console** and **Help**.

![The Discover page as it first opens](images/manual_01_discover.png)

*Figure 1. The Discover page: start from a song (step 1), describe the feel (step 2), or both.*

If the page doesn't open, Spatify isn't running yet: ask the person who looks after it, or see [Appendix A](#appendix-a-starting-and-stopping-spatify).

## 2. Finding music from a song you like

1. Under **1. Start from a song**, type part of the song's title or the artist's name: for example *Unholy* or *Bad Bunny*. Type at least two letters.
2. A list of matching songs appears. Click the one you mean.
3. Press **Get recommendations**.

![Searching for a song](images/manual_02_search.png)

*Figure 2. Searching for "unholy": pick the song you mean from the list.*

Within a second or two, **Your recommendations** appear below: songs that sound like the one you picked.

![The recommendations](images/manual_03_results.png)

*Figure 3. Recommendations for the song picked in Figure 2. The line under the heading says which version of Spatify's model chose them (here, version 8).*

**How many songs?** The **How many songs** slider sets the length of the list, from 1 to 50 (10 to start with). Change it before pressing the button.

## 3. Finding music by describing the feel

You don't need a song to start from. Under **2. Describe the feel**, each quality has a switch:

1. Switch on a quality that matters to you, for example **Energy**.
2. Move its slider: left for less, right for more.
3. Switch on as many as you like; anything left off stays open.
4. Press **Get recommendations**.

![Recommendations from sliders alone](images/manual_04_sliders.png)

*Figure 4. Recommendations from sliders alone: Energy at 1.0, Danceability at 0 and Mood at 0.15 asked for intense, dark songs that are hard to dance to. Each "Why" names the qualities the song matches. (The switched-on sliders are visible in Figure 8.)*

| Quality | From → to |
|---|---|
| Energy | calm → intense |
| Danceability | hard to dance to → made for dancing |
| Mood | sad → happy |
| Acoustic | electric → acoustic |
| Tempo | slower → faster (beats per minute) |
| Instrumental | with vocals → no vocals |

**More options** opens three more qualities (*Spoken words*, *Live feel*, *Loudness*) and **Key** (minor sounds darker, major brighter).

**A song and the feel together.** Pick a song *and* switch on a quality to get "songs like this one, but…": for example, your song but calmer (Energy switched on and moved left).

The **Get recommendations** button stays grey until you've picked a song or switched on at least one quality.

## 4. Reading your recommendations

![Recommendations in detail](images/manual_05_card.png)

*Figure 5. Each recommendation shows its genres and similarity, why it was chosen, and the Like and Skip buttons. The third song here has just been liked.*

Each suggestion shows:
- **Its place in the list** (1 is the closest match), the **title**, the **artist** and the **album**.
- **Its genres** and its **similarity**: how close it is to what you asked for, from -1 to 1. The closer to 1, the closer the match; most suggestions are above 0.9.
- **Why:** what it shares with your song or your sliders, strongest first, for example *energetic, danceable*.
- **Like** and **Skip** buttons (section 5).

**Wildcards.** In a list of 10 or more, the last song is a **wildcard**: a song picked at random, marked "A wildcard: something different". It isn't meant to match. Liking or skipping it teaches Spatify about songs it wouldn't otherwise have suggested.

![A wildcard](images/manual_06_wildcard.png)

*Figure 6. A wildcard at the end of the list. Its similarity is low (-0.38) because it was picked at random.*

## 5. Liking and skipping

Press **Like** on songs you enjoy and **Skip** on songs you don't.

- A small "Thanks!" message appears, the button you pressed greys out, and the song says "You liked this." or "You skipped this."
- You can change your mind: press the other button, and your new choice replaces the first.
- You don't have to rate every song. Every rating helps.

![After pressing Like](images/manual_07_liked.png)

*Figure 7. After pressing Like on song 9: the thank-you message, "You liked this.", and Like greyed out. Song 10 is the wildcard.*

**What your ratings do.** Spatify remembers which songs were liked and which skipped, and which version of its model suggested them. After every 2,000 ratings it trains a new version that has learnt what listeners like, and switches to it only if the new version is better. A new search gets your list from the latest version.

## 6. Ideas to try

| You want | Do this |
|---|---|
| More songs like a favourite | Search for it, pick it, press Get recommendations |
| Something to dance to | Switch on Danceability and Energy; move both right |
| Something calm to study to | Switch on Energy (left), Acoustic (right) and Instrumental (right) |
| A favourite, but happier | Pick the song, then switch on Mood and move it right |
| Songs without singing | Switch on Instrumental and move it right |

## 7. If something goes wrong

Spatify always tells you in plain words what happened and what to do.

| What you see | What it means | What to do |
|---|---|---|
| "Spatify can't reach its server right now" | It's starting, or switching to a new version | Wait a few seconds and try again |
| "The recommender is starting or briefly unavailable" | A new version is being switched in (about 5 seconds) | Try again in a few seconds |
| "No songs match that" | No title or artist contains what you typed | Check the spelling, try the artist's name, or use the sliders instead |
| "That track isn't in the current model's catalogue" | The model was just updated | Search for the song again |
| "Nothing came back" | The combination of qualities is too unusual | Switch off a slider or two |
| The **Get recommendations** button is grey | Nothing to search with yet | Pick a song or switch on at least one quality |
| The page stops responding | The browser lost its connection | Reload the page (F5) |

![A friendly error message](images/manual_08_error.png)

*Figure 8. What you see when Spatify's server is unavailable: no technical detail, just what to do. Behind the message, three qualities are switched on and set.*

The **Help** page in the menu has a short version of this manual.

![The Help page](images/manual_09_help.png)

*Figure 9. The in-app Help page.*

## 8. For the person looking after Spatify

The **Operator console** (in the menu) shows how Spatify is doing and lets you undo a bad update. At the top, three tiles show whether the **Database**, the **Model server** and the **Registry** are working.

**The admin key.** Two actions change Spatify: rolling back to an earlier version, and starting a retrain. They need the admin key. Type it into **Admin key** in the left sidebar; it's `SPATIFY_API_KEY` in the project's `.env` file. It stays in your browser tab only.

### Pipeline

How Spatify's data and models are built, step by step.

![The Pipeline tab](images/manual_10_console_pipeline.png)

*Figure 10. The Pipeline tab: each step coloured by how it last went (here the train step shows as failed, after a deliberate failure test), how fast the data steps run, and their last results.*

- **The diagram:** each step in the order it runs, coloured by its last result: succeeded, failed, running, or not run yet.
- **Throughput:** how many rows of song data each data step handled per second.
- **Run history:** every run, what started it (new songs, a change in the data, enough new ratings, or a person), whether it succeeded, and a one-line explanation. Open *Stages of run …* to see each step's result.
- **Run the pipeline now** (needs the admin key): retrains straight away.

### Models

Every version of Spatify's model, and which one listeners get.

![The Models tab](images/manual_11_console_models.png)

*Figure 11. The Models tab: version 8 is being served; versions 9 to 11 were made by later runs but weren't better, so they weren't promoted. Each row shows where the version came from and its likes and skips.*

- **Serving now:** the version listeners get (the *champion*).
- **The table:**
  - each version, when it was made and what started it;
  - its likes, skips and like rate;
  - the data and code it came from;
  - why it was or wasn't promoted.
- **The chart:** the like rate of each version. A big drop after a change is a warning sign.

**To undo a bad update (roll back):**
1. Enter the admin key in the sidebar.
2. Under **Roll back**, choose the version to bring back. The previous one is chosen for you.
3. Tick the box that confirms it.
4. Press **Roll back**.

Within about 15 seconds listeners get that version again. Rolling back again undoes the rollback.

![Rolling back](images/manual_12_rollback.png)

*Figure 12. Rolling back: version 1, the previous version, is chosen and the confirmation ticked; the Roll back button is ready. Above it, the like rate of each version that has been rated.*

### Data drift

Whether new songs, or listeners' reactions, have changed.

![The Data drift tab](images/manual_13_drift.png)

*Figure 13. The Data drift tab: each batch of new songs and what became of it (bad_batch.csv was rejected), and listeners' like rate: 40% for the latest 200 ratings against 41% before, so no feedback drift. A chart of how far each quality moved appears while a batch's results are available.*

- **Batches:** each file of new songs Spatify has received, whether it was accepted, and whether it **drifted** (differs from the songs Spatify learnt from). A drifted batch makes Spatify retrain by itself.
- **The chart:** how far each quality of the chosen batch moved. Blue is stable, orange moderate, red a big change.
- **Listener feedback:** the like rate of the latest 200 ratings against the 1,000 before. A warning appears if it dropped 15 points or more. If that follows an update, consider rolling back.
- **Retrain now** (needs the admin key).

### Monitoring

![The Monitoring tab](images/manual_14_monitoring.png)

*Figure 14. The Monitoring tab: no alerts firing, 14 past alerts in the history, and live charts of Spatify's speed and use.*

- **Alerts:** anything that needs attention right now (a part of Spatify down, slow answers, a failed run, new songs that differ, a falling like rate). "No alerts are firing" means all is well. *Alert history* lists past alerts and when they cleared.
- **Live charts:** how fast Spatify answers, how much it's used, like rates per version, and how far new songs moved. They update every 10 seconds.

### Tools

Links to the specialist tools behind Spatify:

| Tool | What it's for |
|---|---|
| MLflow | Every training run, its results, and every model version |
| Grafana | Four dashboards of live charts |
| Prometheus targets and alerts | Which parts are being watched, and each alert's state |
| Alertmanager | Alerts being sent to the console |
| Ray Dashboard | The data steps' work, while a Ray cluster is running |
| Gateway API docs | Every request Spatify's server accepts, with a form to try each |

### Adding new songs

Copy a file of new songs (a CSV in the dataset's format) into the project's `data/incoming` folder. Within about 15 seconds Spatify checks it:
- **A file with too many bad rows** is set aside in `data/quarantine`, and Spatify keeps serving as before.
- **A file whose songs differ** from what Spatify learnt from makes it retrain.

The Pipeline and Data drift tabs show what happened.

## 9. What Spatify keeps

- **What it stores:** every list it suggested, and every like and skip, with the time and the model version. That's all: no names, accounts or personal details, and nothing leaves the computer it runs on.
- **Most ratings so far are simulated:** a test program plays the part of many listeners, so there's enough feedback to learn from.

## 10. Words used in Spatify

| Word | Meaning |
|---|---|
| **Recommendation** | A song Spatify suggests |
| **Similarity** | How close a song is to what you asked for, from -1 to 1 |
| **Wildcard** | A random song in the last place of a list, to learn about songs Spatify wouldn't suggest |
| **Like rate** | Of the songs people rated, the share they liked |
| **Model version** | One version of Spatify's song-matching model; each new one is numbered |
| **Champion** | The model version listeners get now |
| **Roll back** | Go back to an earlier model version |
| **Drift** | New songs, or listeners' reactions, differing from what Spatify learnt from |
| **Pipeline** | The steps that check new songs and build a new model version |
| **Admin key** | The password for actions that change Spatify |

---

## Appendix A: Starting and stopping Spatify

For whoever runs Spatify on their computer. It needs Docker, and the project folder set up as in the project's README.

Open a terminal in the project's `spatify` folder, then:

| To | Type |
|---|---|
| Start Spatify | `docker compose up -d` |
| Check every part is running | `docker compose ps` (each should say "healthy" or "Up" within about a minute) |
| Stop Spatify (everything is kept) | `docker compose stop` |
| Start it again after stopping | `docker compose start` |

Then open **http://localhost:8501**. The live charts are at **http://localhost:3000**.
