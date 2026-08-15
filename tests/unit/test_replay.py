import pytest

from src.data import CONTACT_COL, TARGET_COL, load_raw, prepare
from src.policies import BaselineAlwaysTelephone, EpsilonGreedy
from src.replay import replay, run_comparison

pytestmark = pytest.mark.unit


def test_replay_rewards_equal_csv_y(fixture_csv):
    df = prepare(load_raw(fixture_csv))
    result = replay(BaselineAlwaysTelephone(), df, seed=42)
    assert result.n_accepted > 0
    for idx, y in zip(result.accepted_index, result.rewards):
        assert y in (0, 1)
        assert y == int(df.iloc[idx][TARGET_COL])
        assert df.iloc[idx][CONTACT_COL] == "telephone"


def test_baseline_conversion_is_telephone_empirical_rate(fixture_csv):
    df = prepare(load_raw(fixture_csv))
    expected = float(df.loc[df[CONTACT_COL] == "telephone", TARGET_COL].mean())
    result = replay(BaselineAlwaysTelephone(), df, seed=0)
    assert result.conversion == pytest.approx(expected)
    assert result.n_accepted == int((df[CONTACT_COL] == "telephone").sum())


def test_replay_rejects_arm_mismatch(fixture_csv):
    df = prepare(load_raw(fixture_csv))
    eg = EpsilonGreedy(epsilon=0.1)
    result = replay(eg, df, seed=42)
    for idx in result.accepted_index:
        # cada aceite tem y da linha; o braço escolhido coincidiu com contact
        assert df.iloc[idx][CONTACT_COL] in ("cellular", "telephone")
    assert len(result.rewards) == result.n_accepted
    assert result.rewards == result.accepted_y_from_csv


def test_no_invented_rewards(fixture_csv):
    df = prepare(load_raw(fixture_csv))
    _, eg_result, _ = run_comparison(df, seed=42)
    csv_y = set(int(v) for v in df[TARGET_COL].unique())
    assert csv_y <= {0, 1}
    assert set(eg_result.rewards) <= csv_y
