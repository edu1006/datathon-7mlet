"""API de recomendação de canal. Não usa `duration`. Não inventa cliente."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data import label_arm
from src.policies import EpsilonGreedy
from src.state import STATE_PATH, ensure_state

app = FastAPI(
    title="Datathon 7MLET",
    description="Recomenda canal de contato: celular (móvel) ou telefone fixo. Códigos UCI: cellular / telephone.",
    version="0.1.0",
)


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int | None = None
    job: str | None = None
    marital: str | None = None
    education: str | None = None
    default: str | None = None
    housing: str | None = None
    loan: str | None = None
    contact: str | None = Field(default=None, description="Canal logado na base; a política não é obrigada a repetir.")
    month: str | None = None
    day_of_week: str | None = None
    campaign: int | None = None
    pdays: int | None = None
    previous: int | None = None
    poutcome: str | None = None
    seed: int | None = Field(default=None, description="Opcional; reproduz exploração.")

    @model_validator(mode="before")
    @classmethod
    def reject_duration(cls, data: Any) -> Any:
        if isinstance(data, dict) and "duration" in data:
            raise ValueError("duration é vazamento temporal e não é aceito")
        return data


class RecommendResponse(BaseModel):
    arm: Literal["cellular", "telephone"]
    canal: str
    mode: Literal["exploit", "explore"]
    epsilon: float
    means: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    state_path: str
    means: dict[str, float]


def _policy() -> EpsilonGreedy:
    policy, _metrics = ensure_state()
    return policy


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    policy = _policy()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        state_path=str(STATE_PATH),
        means=policy.means(),
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest) -> RecommendResponse:
    policy = _policy()
    rng = np.random.default_rng(body.seed if body.seed is not None else 42)
    decision = policy.choose(rng)
    if decision.arm not in ("cellular", "telephone"):
        raise HTTPException(status_code=500, detail="braço inválido")
    return RecommendResponse(
        arm=decision.arm,  # type: ignore[arg-type]
        canal=label_arm(decision.arm),
        mode=decision.mode,  # type: ignore[arg-type]
        epsilon=policy.epsilon,
        means=policy.means(),
    )
