import pytest

from src.data import DURATION_COL, GOLDEN_INDICES, TARGET_COL, arm_rates, golden_rows, label_arm, load_raw, prepare

pytestmark = pytest.mark.unit


def test_fixture_is_real_uci_shape(fixture_csv):
    raw = load_raw(fixture_csv)
    assert len(raw) == 1200
    assert DURATION_COL in raw.columns
    assert set(raw["contact"].unique()) <= {"cellular", "telephone"}
    assert set(raw["y"].unique()) <= {"yes", "no"}


def test_prepare_drops_duration(fixture_csv):
    prepared = prepare(load_raw(fixture_csv))
    assert DURATION_COL not in prepared.columns
    assert set(prepared[TARGET_COL].unique()) <= {0, 1}


def test_arm_rates_match_fixture_means(fixture_csv):
    raw = load_raw(fixture_csv)
    prepared = prepare(raw)
    rates = arm_rates(prepared).set_index("contact")
    for arm in prepared["contact"].unique():
        expected = float(prepared.loc[prepared["contact"] == arm, TARGET_COL].mean())
        assert rates.loc[arm, "conversao"] == pytest.approx(expected)
        assert int(rates.loc[arm, "n"]) == int((prepared["contact"] == arm).sum())


def test_arm_labels_pt_are_distinct():
    assert label_arm("cellular") == "Celular (móvel)"
    assert label_arm("telephone") == "Telefone fixo"
    assert label_arm("cellular") != label_arm("telephone")


def test_golden_rows_are_fixture_indices(fixture_csv):
    prepared = prepare(load_raw(fixture_csv))
    golden = golden_rows(prepared)
    assert list(golden["golden_index"]) == list(GOLDEN_INDICES)
    for i, idx in enumerate(GOLDEN_INDICES):
        assert int(golden.loc[i, TARGET_COL]) == int(prepared.iloc[idx][TARGET_COL])
        assert golden.loc[i, "contact"] == prepared.iloc[idx]["contact"]
