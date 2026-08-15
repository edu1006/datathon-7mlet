"""Tela de apresentação — só linhas reais da fixture, sem cliente inventado."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.data import CONTACT_COL, FIXTURE_CSV, TARGET_COL, arm_rates, golden_rows, label_arm, load_raw, prepare
from src.policies import EpsilonGreedy
from src.replay import run_comparison
from src.state import save_state

st.set_page_config(page_title="Datathon 7MLET", layout="wide")


@st.cache_data
def _load() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    df = prepare(load_raw(FIXTURE_CSV))
    rates = arm_rates(df)
    base, eg_result, eg = run_comparison(df, seed=42)
    save_state(eg, {"baseline": base.as_metrics(), "epsilon_greedy": eg_result.as_metrics()})
    return df, rates, base.as_metrics(), eg_result.as_metrics(), eg.to_dict()


df, rates, base_metrics, eg_metrics, eg_state = _load()
policy = EpsilonGreedy.from_dict(eg_state)

st.title("Datathon 7MLET — canal de contato")
st.write(
    "Mesa de campanha escolhe **como ligar** (celular ou telefone fixo) para oferecer "
    "depósito a prazo. Os números vêm do recorte real da base UCI/Kaggle "
    "(`bank-additional-full`, linhas 11600 a 12800). Não há cliente sintético nem `y` simulado."
)

c1, c2, c3 = st.columns(3)
c1.metric("Baseline (sempre telefone fixo)", f"{base_metrics['conversion']:.2%}", help="Conversão no replay = taxa empírica de telephone na fixture")
c2.metric("Epsilon-Greedy ε=0.1", f"{eg_metrics['conversion']:.2%}")
c3.metric("Linhas na fixture", f"{len(df)}")

st.caption(
    "Na base UCI, `cellular` = **celular (móvel)** e `telephone` = **telefone fixo**. "
    "Não são o mesmo canal: custo, alcance e conversão diferem (na base full, 14,7% vs 5,2%)."
)
st.subheader("Taxas empíricas da tabela")
st.dataframe(rates, hide_index=True, width="stretch")

st.subheader("Golden Set — 5 ligações que existem na base")
golden = golden_rows(df)
show = golden[["golden_index", "age", "job", CONTACT_COL, "month", "campaign", TARGET_COL]].copy()
show["canal_logado"] = show[CONTACT_COL].map(label_arm)
show = show.rename(columns={TARGET_COL: "y_historico"}).drop(columns=[CONTACT_COL])
st.dataframe(show, hide_index=True, width="stretch")

options = {
    f"idx {int(row['golden_index'])} · {label_arm(row['contact'])} · y={int(row[TARGET_COL])} · {row['job']}": int(row["golden_index"])
    for _, row in golden.iterrows()
}
choice = st.selectbox("Escolha uma linha real", list(options.keys()))
idx = options[choice]
row = golden.loc[golden["golden_index"] == idx].iloc[0]

if st.button("Recomendar canal"):
    decision = policy.choose(np.random.default_rng(0))
    st.write(
        f"Política: **{label_arm(decision.arm)}** ({decision.mode}). "
        f"Na base esta ligação foi **{label_arm(row['contact'])}** com y={int(row[TARGET_COL])}. "
        f"Médias aprendidas no replay: "
        + ", ".join(f"{label_arm(k)}={v:.3f}" for k, v in policy.means().items())
    )

st.caption(
    "Epsilon-Greedy neste MVP não usa features do cliente na escolha (não é contextual). "
    "A linha serve para explicar o caso; a decisão usa só as médias dos braços e ε. "
    "`duration` foi descartado (vazamento)."
)
