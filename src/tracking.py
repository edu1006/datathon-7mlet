"""Registro MLflow das métricas do replay (params reais, sem y inventado)."""

from __future__ import annotations

import mlflow

from src.replay import ReplayResult


def log_replay(base: ReplayResult, eg: ReplayResult, epsilon: float, seed: int) -> str:
    mlflow.set_experiment("datathon-7mlet")
    with mlflow.start_run(run_name="epsilon-greedy-replay") as run:
        mlflow.log_param("epsilon", epsilon)
        mlflow.log_param("seed", seed)
        mlflow.log_param("baseline", "always_telephone")
        mlflow.log_metric("baseline_conversion", base.conversion)
        mlflow.log_metric("baseline_n_accepted", base.n_accepted)
        mlflow.log_metric("eg_conversion", eg.conversion)
        mlflow.log_metric("eg_n_accepted", eg.n_accepted)
        mlflow.log_metric("eg_explore_fraction", eg.explore_fraction)
        return run.info.run_id
