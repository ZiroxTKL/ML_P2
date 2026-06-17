"""
Orquestador del pipeline completo (Proyecto 2, CS3061).

Ejecuta cada etapa en orden, cada una en su propio proceso para aislar el estado
global (TensorFlow, semillas). Reproduce todas las figuras y tablas del informe.

Uso:
    python src/run_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# El orden importa: clustering lee los embeddings de la reducción; las políticas
# de umbral leen las predicciones de la clasificación.
STAGES = [
    ("EDA y espacio vectorial (3.1)", "eda.py"),
    ("Reducción de dimensionalidad (3.2)", "dimreduction.py"),
    ("Clustering (3.3)", "clustering.py"),
    ("Clasificación MLP vs ensambles (3.4)", "classification.py"),
    ("Políticas de umbral y trade-offs (3.5)", "thresholds.py"),
    ("Diagrama de arquitectura (3.1)", "diagram.py"),
]


def main() -> int:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               MPLBACKEND="Agg", TF_CPP_MIN_LOG_LEVEL="2")
    for title, script in STAGES:
        path = os.path.join(HERE, script)
        if not os.path.exists(path):
            print(f"[skip] {title}: {script} no existe todavía")
            continue
        print(f"\n{'=' * 70}\n[run] {title}  ({script})\n{'=' * 70}")
        t0 = time.perf_counter()
        result = subprocess.run([sys.executable, path], env=env)
        if result.returncode != 0:
            print(f"[ERROR] {script} terminó con código {result.returncode}")
            return result.returncode
        print(f"[ok] {script} en {time.perf_counter() - t0:.1f}s")
    print("\nPipeline completo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
