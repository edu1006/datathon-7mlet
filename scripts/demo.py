#!/usr/bin/env python3
"""Demo de apresentação: taxas do replay (y real) + Golden Set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import FIXTURE_CSV, TARGET_COL, arm_rates, golden_rows, label_arm, load_raw, prepare
from src.replay import run_comparison
from src.state import save_state


def main() -> None:
    df = prepare(load_raw(FIXTURE_CSV))
    rates = arm_rates(df)
    base, eg_result, eg = run_comparison(df, seed=42)
    save_state(
        eg,
        {"baseline": base.as_metrics(), "epsilon_greedy": eg_result.as_metrics()},
    )

    print("Caso de uso: escolher canal de contato — celular (móvel) vs telefone fixo — para depósito a prazo.")
    print("UCI: contact=cellular → celular; contact=telephone → telefone fixo. Não são o mesmo canal.")
    print("Fonte: recorte real UCI/Kaggle bank-additional-full, linhas 11600:12800 (n=1200). Sem y inventado.\n")
    print("Taxas empíricas da tabela (P(y=1 | contact)):")
    print(rates.to_string(index=False))
    print()
    print("Replay (recompensa = y da linha, só se o braço escolhido = contact logado):")
    print(f"  Baseline sempre telefone fixo: conversao={base.conversion:.4f}  n_aceito={base.n_accepted}")
    print(f"  Epsilon-Greedy ε=0.1:      conversao={eg_result.conversion:.4f}  n_aceito={eg_result.n_accepted}  explore={eg_result.explore_fraction:.4f}")
    print(f"  Médias aprendidas: {json.dumps(eg.means())}")
    print()
    print("Golden Set (5 linhas reais da fixture):")
    golden = golden_rows(df)
    import numpy as np

    from src.policies import EpsilonGreedy

    print(f"{'idx':>5} {'canal_logado':>18} {'y':>3} {'campaign':>8} {'oferta':>18} {'mode':>8}")
    for _, row in golden.iterrows():
        rng = np.random.default_rng(0)
        frozen = EpsilonGreedy.from_dict(eg.to_dict())
        decision = frozen.choose(rng)
        print(
            f"{int(row['golden_index']):5d} {label_arm(row['contact']):>18} {int(row[TARGET_COL]):3d} "
            f"{int(row['campaign']):8d} {label_arm(decision.arm):>18} {decision.mode:>8}"
        )
    print("\nAPI: uvicorn src.serving:app --port 8000  → http://localhost:8000/docs")
    print("UI:  streamlit run demo/app.py")


if __name__ == "__main__":
    main()
