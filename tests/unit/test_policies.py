import numpy as np
import pytest

from src.policies import BaselineAlwaysTelephone, EpsilonGreedy

pytestmark = pytest.mark.unit


def test_baseline_only_telephone():
    policy = BaselineAlwaysTelephone()
    rng = np.random.default_rng(0)
    for _ in range(20):
        decision = policy.choose(rng)
        assert decision.arm == "telephone"
        assert decision.mode == "exploit"


def test_epsilon_zero_is_greedy():
    policy = EpsilonGreedy(epsilon=0.0)
    policy.update("cellular", 1)
    policy.update("telephone", 0)
    rng = np.random.default_rng(1)
    for _ in range(30):
        assert policy.choose(rng).arm == "cellular"
        assert policy.choose(rng).mode == "exploit"


def test_epsilon_one_only_explores():
    policy = EpsilonGreedy(epsilon=1.0)
    rng = np.random.default_rng(2)
    modes = {policy.choose(rng).mode for _ in range(20)}
    assert modes == {"explore"}


def test_seed_reproduces_sequence():
    def sequence(seed: int) -> list[str]:
        policy = EpsilonGreedy(epsilon=0.5)
        rng = np.random.default_rng(seed)
        return [policy.choose(rng).arm for _ in range(15)]

    assert sequence(7) == sequence(7)
    assert sequence(7) != sequence(8)
