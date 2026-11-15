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
    track_id: str
    track_name: str | None
    artists: str | None
    album_name: str | None
    genres: list[str]
    popularity: int


class Recommendation(Track):
    rank: int
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
    model_run_id: str
    recommendations: list[Recommendation]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    track_id: str = Field(min_length=1, max_length=64)
    action: Literal["like", "skip"]


class FeedbackResponse(BaseModel):
    status: Literal["recorded"] = "recorded"
    model_version: int | None


class SearchResponse(BaseModel):
    query: str
    model_version: int
    tracks: list[Track]


class ModelVersion(BaseModel):
    version: int
    aliases: list[str]
    run_id: str
    created_at: datetime
    description: str | None
    variant: str | None
    dataset_version: str | None
    git_commit: str | None
    trigger: str | None
    likes: int = 0
    skips: int = 0
    like_rate: float | None = Field(None, description="likes / (likes + skips)")


class ModelsResponse(BaseModel):
    name: str
    champion: int | None
    previous: int | None
    challenger: int | None
    versions: list[ModelVersion]


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int | None = Field(None, ge=1, description="Default: the `previous` alias")


class RollbackResponse(BaseModel):
    champion: int
    previous: int
    detail: str


class PipelineRun(BaseModel):
    id: int
    requested_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    trigger: str
    status: str
    stages: dict
    model_version: int | None
    promoted: bool | None
    message: str | None


class PipelineStatus(BaseModel):
    latest: PipelineRun | None
    counts: dict[str, int] = Field(description="Runs per status")


class StageView(BaseModel):
    name: str
    status: str = Field(description="ok, failed, running or not run: the stage's last result")
    command: str
    metrics: dict[str, float] = Field(description="Rows, seconds, rows per second, ...")


class PipelineDag(BaseModel):
    stages: list[StageView] = Field(description="In dvc.yaml order")
    inputs: list[str] = Field(description="Data the pipeline reads but no stage produces")
    edges: list[tuple[str, str]] = Field(description="(from, to): stage or input to stage")


class DriftFeature(BaseModel):
    feature: str
    psi: float = Field(description="Population Stability Index against the training data")
    ks_stat: float = Field(description="Kolmogorov-Smirnov statistic")
    ks_pvalue: float
    level: str = Field(description="stable (< 0.1), moderate (0.1-0.2) or major (>= 0.2)")


class DriftBatch(BaseModel):
    name: str
    status: str = Field(description="new, processed, quarantined, failed, removed or unknown")
    first_seen: datetime | None
    message: str | None
    rows: int | None = Field(description="Valid rows checked; none if validation rejected it")
    drifted: bool | None
    features: list[DriftFeature]


class FeedbackDrift(BaseModel):
    recent_events: int
    recent_like_rate: float | None
    baseline_events: int
    baseline_like_rate: float | None
    drop: float | None = Field(description="baseline like rate minus recent like rate")
    threshold: float
    enough_feedback: bool
    drifted: bool


class DriftReport(BaseModel):
    reference_rows: int | None = Field(description="Training rows the batches are compared with")
    thresholds: dict[str, float]
    batches: list[DriftBatch] = Field(description="Newest first")
    feedback: FeedbackDrift | None = Field(None, description="The like rate: recent vs before")


class TriggerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger: Literal["manual"] = "manual"


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    checks: dict[str, str]
