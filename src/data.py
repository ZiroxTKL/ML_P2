"""
Carga y preparación del dataset eco-acústico (Proyecto 2, CS3061).

Funciones puras que devuelven matrices NumPy listas para el pipeline.
El escalado se aplica en cada script de análisis (StandardScaler ajustado
sobre el conjunto de entrenamiento) para evitar fuga de información.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los CSV de entrenamiento y prueba."""
    df_train = pd.read_csv(config.TRAIN_CSV)
    df_test = pd.read_csv(config.TEST_CSV)
    return df_train, df_test


def features(df: pd.DataFrame) -> np.ndarray:
    """Matriz de características X ∈ ℝ^(N×64) (coeficientes MFCC)."""
    return df[config.FEATURE_COLS].to_numpy(dtype=float)


def labels_idx(df: pd.DataFrame) -> np.ndarray:
    """species_id remapeado a índices contiguos 0..4 (para el MLP/loss)."""
    return df[config.TARGET].map(config.ID_TO_IDX).to_numpy()


def labels_id(df: pd.DataFrame) -> np.ndarray:
    """species_id original (10,12,17,18,23)."""
    return df[config.TARGET].to_numpy()


def is_tp_mask(df: pd.DataFrame) -> np.ndarray:
    """Máscara booleana de registros validados como True Positive."""
    return df["is_tp"].to_numpy().astype(bool)
