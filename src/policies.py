"""Políticas: regra fixa (sempre telephone) e Epsilon-Greedy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data import ARMS


@dataclass
class Decision:
    arm: str
    mode: str  # exploit | explore


@dataclass
class BaselineAlwaysTelephone:
    """Regra de negócio: sempre o canal legado (telefone fixo)."""

    def choose(self, rng: np.random.Generator | None = None) -> Decision:
        return Decision(arm="telephone", mode="exploit")

    def update(self, arm: str, reward: int) -> None:
        return None


@dataclass
class EpsilonGreedy:
    epsilon: float = 0.1
    arms: tuple[str, ...] = ARMS
    counts: dict[str, int] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for arm in self.arms:
            self.counts.setdefault(arm, 0)
            self.rewards.setdefault(arm, 0.0)

    def means(self) -> dict[str, float]:
        return {
            arm: (self.rewards[arm] / self.counts[arm] if self.counts[arm] else 0.0)
            for arm in self.arms
        }

    def choose(self, rng: np.random.Generator) -> Decision:
        if rng.random() < self.epsilon:
            arm = str(rng.choice(self.arms))
            return Decision(arm=arm, mode="explore")
        means = self.means()
        best = max(means.values())
        candidates = [arm for arm, value in means.items() if value == best]
        arm = str(candidates[int(rng.integers(0, len(candidates)))])
        return Decision(arm=arm, mode="exploit")

    def greedy_arm(self) -> str:
        means = self.means()
        best = max(means.values())
        for arm in self.arms:
            if means[arm] == best:
                return arm
        return self.arms[0]

    def update(self, arm: str, reward: int) -> None:
        self.counts[arm] = self.counts.get(arm, 0) + 1
        self.rewards[arm] = self.rewards.get(arm, 0.0) + float(reward)

    def to_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "arms": list(self.arms),
            "counts": dict(self.counts),
            "rewards": dict(self.rewards),
            "means": self.means(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EpsilonGreedy":
        policy = cls(
            epsilon=float(payload["epsilon"]),
            arms=tuple(payload.get("arms", ARMS)),
        )
        policy.counts = {k: int(v) for k, v in payload["counts"].items()}
        policy.rewards = {k: float(v) for k, v in payload["rewards"].items()}
        return policy
