"""Carga e preparação da base Bank Marketing (UCI / Kaggle henriqueyamahata)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ARMS = ("cellular", "telephone")
# Valores da base UCI (inglês). Em pt-BR não são sinônimos: móvel vs linha fixa.
ARM_LABELS_PT = {
    "cellular": "Celular (móvel)",
    "telephone": "Telefone fixo",
}
DURATION_COL = "duration"
TARGET_COL = "y"
CONTACT_COL = "contact"


def label_arm(arm: str) -> str:
    return ARM_LABELS_PT.get(arm, arm)

KAGGLE_URL = "https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/222/bank+marketing"
UCI_ZIP = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FULL_CSV = DATA_DIR / "bank-additional-full.csv"
FIXTURE_CSV = ROOT / "tests" / "fixtures" / "bank_sample.csv"

# Recorte cronológico real do full (linhas 11600:12800), usado nos testes e na demo.
GOLDEN_INDICES = (0, 90, 784, 757, 766)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Lê o CSV UCI (separador `;`). Não inventa linhas."""
    csv_path = Path(path) if path else _resolve_csv()
    df = pd.read_csv(csv_path, sep=";")
    if CONTACT_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"CSV inesperado em {csv_path}: faltam contact/y")
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Remove `duration` (vazamento), filtra braços conhecidos, mapeia y para 0/1."""
    out = df.copy()
    if DURATION_COL in out.columns:
        out = out.drop(columns=[DURATION_COL])
    out = out[out[CONTACT_COL].isin(ARMS)].copy()
    if out[TARGET_COL].dtype == object:
        out[TARGET_COL] = (out[TARGET_COL].astype(str).str.lower() == "yes").astype(int)
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    out = out.reset_index(drop=True)
    return out


def arm_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Taxa empírica de conversão por `contact` — só números da tabela."""
    prepared = df if TARGET_COL in df.columns and df[TARGET_COL].dtype != object else prepare(df)
    stats = (
        prepared.groupby(CONTACT_COL, sort=True)[TARGET_COL]
        .agg(conversao="mean", n="count", conversoes="sum")
        .reset_index()
    )
    stats.insert(1, "canal", stats[CONTACT_COL].map(label_arm))
    return stats


def download_full(dest: Path = FULL_CSV) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(UCI_ZIP, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        with zf.open("bank-additional/bank-additional-full.csv") as src:
            dest.write_bytes(src.read())
    return dest


def _resolve_csv() -> Path:
    if FULL_CSV.exists():
        return FULL_CSV
    if FIXTURE_CSV.exists():
        return FIXTURE_CSV
    return download_full()


def golden_rows(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cinco linhas reais da fixture (índices fixos no recorte)."""
    prepared = df if df is not None else prepare(load_raw(FIXTURE_CSV))
    missing = [i for i in GOLDEN_INDICES if i >= len(prepared)]
    if missing:
        raise IndexError(f"Golden Set fora da base: {missing}")
    out = prepared.iloc[list(GOLDEN_INDICES)].copy()
    out.insert(0, "golden_index", list(GOLDEN_INDICES))
    return out.reset_index(drop=True)
