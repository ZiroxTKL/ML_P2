"""
Configuración compartida del pipeline de clasificación eco-acústica.
Proyecto 2 - CS3061 Machine Learning (UTEC).

Centraliza rutas, el esquema del dataset (X in R^64), el mapeo de clases y el
estilo de figuras (fuente >= 14 para cumplir la rúbrica y evitar la penalización
de -3.0 puntos por tamaño de fuente < 14 en ejes/leyendas).
"""
from __future__ import annotations

from pathlib import Path

# --- Rutas ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
TRAIN_CSV = DATA_DIR / "eco_acoustic_train.csv"
TEST_CSV = DATA_DIR / "eco_acoustic_test.csv"

for _d in (FIG_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Esquema del dataset ---
ID_COL = "recording_id"
TARGET = "species_id"
META_COLS = ["recording_id", "species_id", "songtype_id", "is_tp"]
FEATURE_COLS = [f"mel_{i}" for i in range(64)]   # X in R^64 (coeficientes MFCC)
N_FEATURES = 64

# species_id NO es contiguo (10,12,17,18,23) -> se remapea a índices 0..4
# para alimentar el MLP / la función de pérdida categórica.
SPECIES_IDS = [10, 12, 17, 18, 23]
ID_TO_IDX = {sid: i for i, sid in enumerate(SPECIES_IDS)}
IDX_TO_ID = {i: sid for sid, i in ID_TO_IDX.items()}
N_CLASSES = len(SPECIES_IDS)

# Catálogo taxonómico (de Documentacion_de_dataset.pdf)
SPECIES_NAME = {
    10: "Leptodactylus discodactylus",
    12: "Osteocephalus taurinus",
    17: "Chiroxiphia lineata",
    18: "Saltator grossus",
    23: "Pheucticus chrysopeplus",
}
SPECIES_KIND = {10: "Anfibio", 12: "Anfibio", 17: "Ave", 18: "Ave", 23: "Ave"}


def short_label(sid: int) -> str:
    """Etiqueta compacta 'Nombre científico (id)' para ejes y leyendas."""
    return f"{SPECIES_NAME[sid]} ({sid})"


# --- Umbrales de moderación (Sec 3.5) ---
THRESH_CONFIDENCE = 0.85   # P >= 85%  -> clasificación automática (verde)
THRESH_REJECT = 0.40       # P <  40%  -> descarte (rojo); intermedio -> auditoría (amarillo)

RANDOM_STATE = 42


# --- Estilo de figuras (rúbrica: fuente >= 14; penalización -3.0 si < 14) ---
def apply_style() -> None:
    """Aplica un estilo consistente con todas las fuentes >= 14."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 15,
        "figure.titlesize": 18,
    })
