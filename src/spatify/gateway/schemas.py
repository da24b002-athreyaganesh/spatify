"""Request and response schemas: the gateway's REST contract (shown at /docs).

Validation happens here, before anything reaches the model: a bad request gets a 422
naming the field, not an error from deeper down.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

UNIT = {"ge": 0.0, "le": 1.0}


class Sliders(BaseModel):
    """Audio features a listener can ask for directly, in the dataset's units. All optional."""

    model_config = ConfigDict(extra="forbid")

    danceability: float | None = Field(None, **UNIT, description="0 (not) to 1 (very)")
    energy: float | None = Field(None, **UNIT, description="0 (calm) to 1 (intense)")
    loudness: float | None = Field(None, ge=-60.0, le=5.0, description="Average loudness, dB")
    speechiness: float | None = Field(None, **UNIT, description="0 (music) to 1 (spoken word)")
    acousticness: float | None = Field(None, **UNIT, description="0 (electric) to 1 (acoustic)")
    instrumentalness: float | None = Field(None, **UNIT, description="1: no vocals")
    liveness: float | None = Field(None, **UNIT, description="1: recorded live")
    valence: float | None = Field(None, **UNIT, description="0 (sad) to 1 (happy)")
    tempo: float | None = Field(None, ge=0.0, le=250.0, description="Beats per minute")
    mode: int | None = Field(None, ge=0, le=1, description="0 minor, 1 major")


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str | None = Field(
        None, min_length=1, max_length=64, description="Seed track, from /tracks/search"
    )
    sliders: Sliders = Field(default_factory=Sliders)
    k: int = Field(10, ge=1, le=50, description="Number of recommendations")

    @model_validator(mode="after")
    def needs_a_seed_or_a_slider(self) -> RecommendRequest:
        if self.track_id is None and not self.sliders.model_dump(exclude_none=True):
            raise ValueError("give a seed track_id, at least one slider, or both")
        return self


class Track(BaseModel):
    track_id: str = Field(description="Spotify track ID")
    track_name: str | None = Field(description="Title")
    artists: str | None = Field(description="Artists, separated by ';'")
    album_name: str | None = Field(description="Album")
    genres: list[str] = Field(description="Every genre the track is listed under")
    popularity: int = Field(description="Spotify popularity, 0-100")


class Recommendation(Track):
    rank: int = Field(description="1 is the closest match")
    score: float = Field(description="Cosine similarity to the query, -1 to 1")
    why: list[str] = Field(
        default_factory=list,
        description="Traits shared with the query that add most to the score, strongest first",
    )
    explored: bool = Field(
        False, description="A random pick in place of the lowest-ranked slot, to learn from"
    )


class RecommendResponse(BaseModel):
    request_id: UUID = Field(description="Send this back with feedback on these tracks")
    model_version: int | None = Field(description="Registry version that answered")
    model_run_id: str = Field(description="MLflow run that trained the answering model")
    recommendations: list[Recommendation] = Field(description="Best match first")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(description="From the /recommend answer the track was in")
    track_id: str = Field(min_length=1, max_length=64, description="A track in that list")
    action: Literal["like", "skip"] = Field(description="What the listener thought of it")


class FeedbackResponse(BaseModel):
    status: Literal["recorded"] = "recorded"
    model_version: int | None = Field(description="The version that made the list")


class SearchResponse(BaseModel):
    query: str = Field(description="The text searched for")
    model_version: int = Field(description="The version whose catalogue was searched")
    tracks: list[Track] = Field(description="Title matches first, then artist matches")


class ModelVersion(BaseModel):
    version: int = Field(description="Registry version number")
    aliases: list[str] = Field(
        description="champion (served), previous (rollback target), challenger"
    )
    run_id: str = Field(description="MLflow run that trained it")
    created_at: datetime = Field(description="When it was registered")
    description: str | None = Field(description="The promotion gate's verdict")
    variant: str | None = Field(description="Feature set and weighting, e.g. all-feedback")
    dataset_version: str | None = Field(description="DVC hash of the data it was trained on")
    git_commit: str | None = Field(description="Commit of the code that trained it")
    trigger: str | None = Field(
        description="What started its run: manual, new_data, drift, feedback, ci, forced_bad"
    )
    likes: int = Field(0, description="Likes on lists it served")
    skips: int = Field(0, description="Skips on lists it served")
    like_rate: float | None = Field(None, description="likes / (likes + skips)")


class ModelsResponse(BaseModel):
    name: str = Field(description="Registered model name")
    champion: int | None = Field(description="The version listeners get")
    previous: int | None = Field(description="The rollback target")
    challenger: int | None = Field(description="The latest version the gate didn't promote")
    versions: list[ModelVersion] = Field(description="Newest first")


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int | None = Field(None, ge=1, description="Default: the `previous` alias")


class RollbackResponse(BaseModel):
    champion: int = Field(description="The version now served")
    previous: int = Field(description="The version it replaced (the next rollback's default)")
    detail: str = Field(description="What happens next, in words")


class PipelineRun(BaseModel):
    id: int = Field(description="Run number")
    requested_at: datetime = Field(description="When it was queued or the worker started it")
    started_at: datetime | None = Field(description="Empty until the worker picks it up")
    finished_at: datetime | None = Field(description="Empty while it runs")
    trigger: str = Field(description="manual, new_data, drift or feedback")
    status: str = Field(description="requested, running, succeeded or failed")
    stages: dict = Field(description="Per stage: its result and seconds")
    model_version: int | None = Field(description="The version this run registered, if any")
    promoted: bool | None = Field(description="Whether that version became champion")
    message: str | None = Field(description="One-line explanation of what happened")


class PipelineStatus(BaseModel):
    latest: PipelineRun | None = Field(description="The newest run")
    counts: dict[str, int] = Field(description="Runs per status")


class StageView(BaseModel):
    name: str = Field(description="Stage name in dvc.yaml")
    status: str = Field(description="ok, failed, running or not run: the stage's last result")
    command: str = Field(description="The stage's command")
    metrics: dict[str, float] = Field(description="Rows, seconds, rows per second, ...")


class PipelineDag(BaseModel):
    stages: list[StageView] = Field(description="In dvc.yaml order")
    inputs: list[str] = Field(description="Data the pipeline reads but no stage produces")
    edges: list[tuple[str, str]] = Field(description="(from, to): stage or input to stage")


class DriftFeature(BaseModel):
    feature: str = Field(description="Audio feature")
    psi: float = Field(description="Population Stability Index against the training data")
    ks_stat: float = Field(description="Kolmogorov-Smirnov statistic")
    ks_pvalue: float = Field(description="KS test p-value (reported; PSI decides)")
    level: str = Field(description="stable (< 0.1), moderate (0.1-0.2) or major (>= 0.2)")


class DriftBatch(BaseModel):
    name: str = Field(description="File name in data/incoming")
    status: str = Field(description="new, processed, quarantined, failed, removed or unknown")
    first_seen: datetime | None = Field(description="When the worker first saw it")
    message: str | None = Field(description="What became of it")
    rows: int | None = Field(description="Valid rows checked; none if validation rejected it")
    drifted: bool | None = Field(description="Whether it drifted; empty if it wasn't checked")
    features: list[DriftFeature] = Field(
        description="One row per audio feature; empty if not checked"
    )


class FeedbackDrift(BaseModel):
    recent_events: int = Field(description="Ratings in the recent window")
    recent_like_rate: float | None = Field(description="Like rate of the latest ratings")
    baseline_events: int = Field(description="Ratings in the window before")
    baseline_like_rate: float | None = Field(description="Like rate of the ratings before")
    drop: float | None = Field(description="baseline like rate minus recent like rate")
    threshold: float = Field(description="The drop that counts as drift")
    enough_feedback: bool = Field(description="Whether both windows have enough ratings to judge")
    drifted: bool = Field(description="True when the drop reaches the threshold")


class DriftReport(BaseModel):
    reference_rows: int | None = Field(description="Training rows the batches are compared with")
    thresholds: dict[str, float] = Field(description="PSI levels: moderate and major")
    batches: list[DriftBatch] = Field(description="Newest first")
    feedback: FeedbackDrift | None = Field(None, description="The like rate: recent vs before")


class AlertmanagerAlert(BaseModel):
    """One alert in Alertmanager's webhook payload (the fields Spatify keeps)."""

    status: Literal["firing", "resolved"]
    labels: dict[str, str]
    annotations: dict[str, str] = Field(default_factory=dict)
    startsAt: datetime  # noqa: N815 (Alertmanager's field names)
    endsAt: datetime | None = None  # noqa: N815
    fingerprint: str = Field(min_length=1, max_length=64)


class AlertmanagerWebhook(BaseModel):
    """What Alertmanager posts to a webhook receiver; other fields are ignored."""

    alerts: list[AlertmanagerAlert] = Field(max_length=500)


class AlertRecord(BaseModel):
    fingerprint: str = Field(description="Alertmanager's ID for the alert")
    alertname: str = Field(description="The rule's name, e.g. DataDrift")
    status: Literal["firing", "resolved"] = Field(description="firing or resolved")
    severity: str | None = Field(description="warning or critical")
    summary: str | None = Field(description="One line")
    description: str | None = Field(description="What it means and what to do")
    starts_at: datetime = Field(description="When it started firing")
    ends_at: datetime | None = Field(description="Empty while firing")
    received_at: datetime = Field(description="When the latest notice arrived")


class TriggerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger: Literal["manual"] = "manual"


class Problem(BaseModel):
    """The body of every error answer (4xx, 5xx); 422 adds the field that failed."""

    detail: str = Field(description="What went wrong, in words a person can act on")
    request_id: str | None = Field(None, description="Only on 500: quote it when reporting")


class Health(BaseModel):
    status: Literal["ok", "unavailable"] = Field(
        description="ok only when recommendations can be served"
    )
    checks: dict[str, str] = Field(
        description="postgres, model_server and registry, each ok or what's wrong"
    )
