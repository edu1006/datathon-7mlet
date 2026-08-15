"""Replay offline: recompensa só quando o braço escolhido = contact logado (y real)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data import CONTACT_COL, TARGET_COL
from src.policies import BaselineAlwaysTelephone, Decision, EpsilonGreedy


@dataclass
class ReplayResult:
    policy_name: str
    n_rows: int
    n_accepted: int
    conversion: float
    explore_fraction: float
    rewards: list[int] = field(default_factory=list)
    accepted_index: list[int] = field(default_factory=list)
    accepted_y_from_csv: list[int] = field(default_factory=list)

    def as_metrics(self) -> dict:
        return {
            "policy": self.policy_name,
            "n_rows": self.n_rows,
            "n_accepted": self.n_accepted,
            "conversion": self.conversion,
            "explore_fraction": self.explore_fraction,
        }


def replay(policy, df: pd.DataFrame, seed: int = 42) -> ReplayResult:
    """Percorre as linhas em ordem. Sem Bernoulli. Sem y inventado."""
    rng = np.random.default_rng(seed)
    accepted_y: list[int] = []
    accepted_idx: list[int] = []
    n_explore = 0
    for idx, row in df.iterrows():
        decision: Decision = policy.choose(rng)
        if decision.arm != row[CONTACT_COL]:
            continue
        y = int(row[TARGET_COL])
        policy.update(decision.arm, y)
        accepted_y.append(y)
        accepted_idx.append(int(idx))
        if decision.mode == "explore":
            n_explore += 1
    n_acc = len(accepted_y)
    conversion = float(np.mean(accepted_y)) if n_acc else 0.0
    explore_fraction = (n_explore / n_acc) if n_acc else 0.0
    name = type(policy).__name__
    return ReplayResult(
        policy_name=name,
        n_rows=len(df),
        n_accepted=n_acc,
        conversion=conversion,
        explore_fraction=explore_fraction,
        rewards=list(accepted_y),
        accepted_index=accepted_idx,
        accepted_y_from_csv=list(accepted_y),
    )


def run_comparison(df: pd.DataFrame, epsilon: float = 0.1, seed: int = 42) -> tuple[ReplayResult, ReplayResult, EpsilonGreedy]:
    baseline = BaselineAlwaysTelephone()
    eg = EpsilonGreedy(epsilon=epsilon)
    base_result = replay(baseline, df, seed=seed)
    eg_result = replay(eg, df, seed=seed)
    return base_result, eg_result, eg
