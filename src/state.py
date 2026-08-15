"""Estado persistido da política após o replay (médias reais por braço)."""

from __future__ import annotations

import json
from pathlib import Path

from src.data import FIXTURE_CSV, ROOT, load_raw, prepare
from src.policies import EpsilonGreedy
from src.replay import run_comparison

ARTIFACTS = ROOT / "artifacts"
STATE_PATH = ARTIFACTS / "policy_state.json"


def save_state(policy: EpsilonGreedy, metrics: dict, path: Path = STATE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = policy.to_dict()
    payload["metrics"] = metrics
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_state(path: Path = STATE_PATH) -> EpsilonGreedy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EpsilonGreedy.from_dict(payload)


def ensure_state(csv_path: Path | None = None, seed: int = 42) -> tuple[EpsilonGreedy, dict]:
    """Gera o estado a partir da fixture (ou CSV dado) se ainda não existir."""
    if STATE_PATH.exists():
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return EpsilonGreedy.from_dict(payload), payload.get("metrics", {})
    df = prepare(load_raw(csv_path or FIXTURE_CSV))
    base, eg_result, eg = run_comparison(df, seed=seed)
    metrics = {
        "baseline": base.as_metrics(),
        "epsilon_greedy": eg_result.as_metrics(),
    }
    save_state(eg, metrics)
    return eg, metrics
