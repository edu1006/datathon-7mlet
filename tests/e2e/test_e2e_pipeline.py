import json

import pytest
from fastapi.testclient import TestClient

from src.data import CONTACT_COL, GOLDEN_INDICES, TARGET_COL, golden_rows, load_raw, prepare
from src.replay import run_comparison
from src.serving import app
import src.serving as serving
import src.state as state_mod

pytestmark = pytest.mark.e2e


@pytest.fixture
def client(tmp_path, fixture_csv, monkeypatch):
    state_file = tmp_path / "policy_state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_file)
    monkeypatch.setattr(serving, "STATE_PATH", state_file)
    df = prepare(load_raw(fixture_csv))
    base, eg_result, eg = run_comparison(df, seed=42)
    state_mod.save_state(
        eg,
        {"baseline": base.as_metrics(), "epsilon_greedy": eg_result.as_metrics()},
        path=state_file,
    )
    with TestClient(app) as test_client:
        yield test_client, df, base, eg_result


def test_e2e_pipeline_real_y_and_api(client):
    test_client, df, base, eg_result = client

    assert base.n_accepted > 0
    assert eg_result.n_accepted > 0
    expected_tel = float(df.loc[df[CONTACT_COL] == "telephone", TARGET_COL].mean())
    assert base.conversion == pytest.approx(expected_tel)
    for idx, y in zip(eg_result.accepted_index, eg_result.rewards):
        assert y == int(df.iloc[idx][TARGET_COL])
        assert y in (0, 1)

    health = test_client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert set(body["means"].keys()) <= {"cellular", "telephone"}

    rec = test_client.post("/recommend", json={"contact": "cellular", "campaign": 1, "seed": 42})
    assert rec.status_code == 200
    payload = rec.json()
    assert payload["arm"] in {"cellular", "telephone"}
    assert payload["canal"] == (
        "Celular (móvel)" if payload["arm"] == "cellular" else "Telefone fixo"
    )
    assert payload["mode"] in {"exploit", "explore"}
    assert "epsilon" in payload


def test_e2e_rejects_duration_leakage(client):
    test_client, *_ = client
    response = test_client.post("/recommend", json={"duration": 180, "contact": "cellular"})
    assert response.status_code == 422


def test_e2e_golden_set_real_rows(client, fixture_csv):
    test_client, df, *_ = client
    golden = golden_rows(df)
    assert len(golden) == 5
    assert list(golden["golden_index"]) == list(GOLDEN_INDICES)
    for _, row in golden.iterrows():
        response = test_client.post(
            "/recommend",
            json={
                "contact": row["contact"],
                "campaign": int(row["campaign"]),
                "age": int(row["age"]),
                "job": str(row["job"]),
                "month": str(row["month"]),
                "seed": 0,
            },
        )
        assert response.status_code == 200
        assert response.json()["arm"] in {"cellular", "telephone"}


def test_e2e_records_both_rates_without_forcing_winner(client, tmp_path):
    _, _df, base, eg_result = client
    report = {
        "baseline_conversion": base.conversion,
        "eg_conversion": eg_result.conversion,
        "baseline_n": base.n_accepted,
        "eg_n": eg_result.n_accepted,
    }
    out = tmp_path / "replay_rates.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["baseline_n"] > 0
    assert loaded["eg_n"] > 0
    # identidade: números persistidos = resultado do replay (não um chute)
    assert loaded["baseline_conversion"] == pytest.approx(base.conversion)
    assert loaded["eg_conversion"] == pytest.approx(eg_result.conversion)
