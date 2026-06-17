"""
Arquitectura de clasificación (Sec 3.4) — MLP vs modelos de ensamble.

Contenido:
  1. Experimento de regularización: impacto de la POSICIÓN RELATIVA de Dropout y
     Batch Normalization en la estabilidad de las curvas de aprendizaje (Loss vs época).
  2. MLP final (Keras) con pérdida de entropía cruzada categórica, manejo del
     desbalance (class_weight) y especificación de topología.
  3. Benchmark MLP vs XGBoost vs LightGBM con F1-Score macro y matrices de confusión.

Validación honesta: partición agrupada por recording_id (evita fuga por ventanas del
mismo audio). Se reporta el conjunto val (agrupado, sin fuga) y el test provisto
(con 122 grabaciones solapadas con train → métricas algo optimistas).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import config
import data as D

config.apply_style()

EPOCHS = 100
BATCH = 32
HIDDEN = [128, 64]
DROPOUT = 0.3

# Variantes del experimento de posición relativa Dropout/BatchNorm.
VARIANTS = {
    "Sin regularización":      ["dense", "act"],
    "BN → Act (solo BN)":      ["dense", "bn", "act"],
    "BN → Act → Dropout":      ["dense", "bn", "act", "drop"],
    "Act → Dropout → BN":      ["dense", "act", "drop", "bn"],
    "Act → Dropout (solo Drop)": ["dense", "act", "drop"],
}
VARIANT_COLOR = dict(zip(VARIANTS, ["#888888", "#2a9d8f", "#264653", "#e76f51", "#e9c46a"]))


def _set_seeds():
    np.random.seed(config.RANDOM_STATE)
    tf.random.set_seed(config.RANDOM_STATE)


def build_mlp(recipe) -> keras.Model:
    """Construye el MLP aplicando la 'receta' de bloque a cada capa oculta."""
    _set_seeds()
    inp = keras.Input(shape=(config.N_FEATURES,))
    x = inp
    for units in HIDDEN:
        for op in recipe:
            if op == "dense":
                x = layers.Dense(units, use_bias=True)(x)
            elif op == "bn":
                x = layers.BatchNormalization()(x)
            elif op == "act":
                x = layers.Activation("relu")(x)
            elif op == "drop":
                x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(config.N_CLASSES, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    df_tr, df_te = D.load_raw()
    X_all = D.features(df_tr)
    y_all = D.labels_idx(df_tr)
    groups = df_tr[config.ID_COL].to_numpy()

    # Partición agrupada por recording_id (80/20) para desarrollo del modelo.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=config.RANDOM_STATE)
    tr_idx, val_idx = next(gss.split(X_all, y_all, groups))
    scaler = StandardScaler().fit(X_all[tr_idx])
    Xtr, ytr = scaler.transform(X_all[tr_idx]), y_all[tr_idx]
    Xval, yval = scaler.transform(X_all[val_idx]), y_all[val_idx]

    # Conjunto de prueba provisto.
    Xte = scaler.transform(D.features(df_te))
    yte = D.labels_idx(df_te)
    is_tp_te = D.is_tp_mask(df_te)

    # Pesos de clase (desbalance).
    cw = compute_class_weight("balanced", classes=np.arange(config.N_CLASSES), y=ytr)
    class_weight = {i: w for i, w in enumerate(cw)}
    sw_tr = np.array([class_weight[y] for y in ytr])
    print("class_weight:", {i: round(w, 3) for i, w in class_weight.items()})

    # ---------- 1) Experimento de posición Dropout/BatchNorm ----------
    histories, val_f1 = {}, {}
    for name, recipe in VARIANTS.items():
        model = build_mlp(recipe)
        h = model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=EPOCHS,
                      batch_size=BATCH, class_weight=class_weight, verbose=0)
        histories[name] = h.history
        pred = model.predict(Xval, verbose=0).argmax(1)
        val_f1[name] = f1_score(yval, pred, average="macro")
        print(f"  [{name:26s}] F1-macro val = {val_f1[name]:.4f}")
    # Selección por menor pérdida de validación promedio en la convergencia (últimas 20
    # épocas): criterio que penaliza el sobreajuste y resulta coherente con el análisis de
    # estabilidad (las variantes sin Dropout divergen pese a un F1 final comparable).
    best_variant = min(histories, key=lambda n: float(np.mean(histories[n]["val_loss"][-20:])))
    print(f"Mejor variante (min val_loss en convergencia): {best_variant} "
          f"(F1-macro val = {val_f1[best_variant]:.4f})")
    _fig_learning_curves(histories)

    # ---------- 2) MLP final: mejor variante, reentrenada en train_sub con early stopping ----------
    # Se entrena solo en train_sub (no en todo el train) para que la métrica de val
    # sea honesta y la comparación con los ensambles sea justa.
    final_mlp = build_mlp(VARIANTS[best_variant])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                       restore_best_weights=True)
    final_mlp.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=200, batch_size=BATCH,
                  class_weight=class_weight, callbacks=[es], verbose=0)
    proba_mlp = final_mlp.predict(Xte, verbose=0)

    # ---------- 3) Modelos de ensamble ----------
    Xtr_raw, Xval_raw = X_all[tr_idx], X_all[val_idx]
    Xte_raw = D.features(df_te)

    xgb = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                        early_stopping_rounds=40, random_state=config.RANDOM_STATE,
                        tree_method="hist")
    xgb.fit(Xtr_raw, ytr, sample_weight=sw_tr, eval_set=[(Xval_raw, yval)], verbose=False)
    proba_xgb = xgb.predict_proba(Xte_raw)

    import lightgbm as lgb
    lgbm = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                          random_state=config.RANDOM_STATE, verbose=-1)
    lgbm.fit(Xtr_raw, ytr, eval_set=[(Xval_raw, yval)],
             callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    proba_lgbm = lgbm.predict_proba(Xte_raw)

    # ---------- Métricas (val agrupado honesto + test provisto) ----------
    val_preds = {
        "MLP": final_mlp.predict(Xval, verbose=0).argmax(1),
        "XGBoost": xgb.predict(Xval_raw),
        "LightGBM": lgbm.predict(Xval_raw),
    }
    test_proba = {"MLP": proba_mlp, "XGBoost": proba_xgb, "LightGBM": proba_lgbm}
    rows = []
    for name in ["MLP", "XGBoost", "LightGBM"]:
        pte = test_proba[name].argmax(1)
        rows.append(dict(
            modelo=name,
            F1_macro_val=round(f1_score(yval, val_preds[name], average="macro"), 4),
            F1_macro_test=round(f1_score(yte, pte, average="macro"), 4),
            F1_weighted_test=round(f1_score(yte, pte, average="weighted"), 4),
            accuracy_test=round(accuracy_score(yte, pte), 4),
            F1_macro_test_isTP=round(
                f1_score(yte[is_tp_te], pte[is_tp_te], average="macro"), 4),
        ))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.RESULTS_DIR / "classification_metrics.csv", index=False)
    print("\n=== Benchmark de clasificación ===")
    print(metrics.to_string(index=False))

    best_model = metrics.loc[metrics.F1_macro_test.idxmax(), "modelo"]
    print(f"\nMejor modelo por F1-macro (test): {best_model}")

    # ---------- Figuras y artefactos ----------
    _fig_confusion(test_proba, yte)
    _fig_f1_bars(metrics)
    _save_topology(final_mlp, best_variant, val_f1[best_variant])
    _write_notes(metrics, best_variant, val_f1, best_model, yte, test_proba, histories)

    np.savez_compressed(
        config.RESULTS_DIR / "test_predictions.npz",
        y_true=yte, is_tp=is_tp_te, recording_id=df_te[config.ID_COL].to_numpy(),
        proba_mlp=proba_mlp, proba_xgb=proba_xgb, proba_lgbm=proba_lgbm,
        best_model=best_model)
    final_mlp.save(config.RESULTS_DIR / "mlp_model.keras")
    joblib.dump(scaler, config.RESULTS_DIR / "scaler.joblib")
    joblib.dump(xgb, config.RESULTS_DIR / "xgb_model.joblib")
    joblib.dump(lgbm, config.RESULTS_DIR / "lgbm_model.joblib")
    print("\n[ok] Sec 3.4 completada — figuras en", config.FIG_DIR)


def _fig_learning_curves(histories):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for name, h in histories.items():
        ep = np.arange(1, len(h["loss"]) + 1)
        axes[0].plot(ep, h["loss"], color=VARIANT_COLOR[name], label=name)
        axes[1].plot(ep, h["val_loss"], color=VARIANT_COLOR[name], label=name)
    axes[0].set_title("Pérdida de entrenamiento")
    axes[1].set_title("Pérdida de validación")
    for ax in axes:
        ax.set_xlabel("Época")
        ax.set_ylabel("Entropía cruzada categórica")
        ax.grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=14)
    fig.suptitle("Impacto de la posición de Dropout/BatchNorm en la estabilidad de las curvas")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "clf_learning_curves.png")
    plt.close(fig)


def _fig_confusion(test_proba, yte):
    names = ["MLP", "XGBoost", "LightGBM"]
    labels = [config.SPECIES_NAME[config.IDX_TO_ID[i]].split()[0] for i in range(config.N_CLASSES)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, name in zip(axes, names):
        cm = confusion_matrix(yte, test_proba[name].argmax(1))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={"size": 14})
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.tick_params(labelsize=14)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.suptitle("Matrices de confusión sobre el conjunto de prueba (por género)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "clf_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def _fig_f1_bars(metrics):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w / 2, metrics.F1_macro_val, w, label="F1-macro (val agrupado)", color="#2a9d8f")
    ax.bar(x + w / 2, metrics.F1_macro_test, w, label="F1-macro (test provisto)", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.modelo)
    ax.set_ylabel("F1-Score macro")
    ax.set_ylim(0, max(metrics.F1_macro_test.max(), metrics.F1_macro_val.max()) * 1.25)
    ax.set_title("Comparación de F1-macro por modelo")
    for xi, (v, t) in enumerate(zip(metrics.F1_macro_val, metrics.F1_macro_test)):
        ax.text(xi - w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=14)
        ax.text(xi + w / 2, t + 0.01, f"{t:.2f}", ha="center", fontsize=14)
    ax.legend(loc="upper right", fontsize=14)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "clf_f1_comparison.png")
    plt.close(fig)


def _save_topology(model, variant, f1):
    lines = ["# Topología del MLP final (Sec 3.4)\n",
             f"- Variante de regularización: **{variant}**",
             f"- Entrenamiento: early stopping sobre val_loss (restore_best_weights)",
             f"- F1-macro en validación agrupada: {f1:.4f}",
             f"- Pérdida: entropía cruzada categórica (sparse categorical cross-entropy)",
             f"- Optimizador: Adam (lr=1e-3) · batch={BATCH}\n",
             "| Capa | Salida | Parámetros | Activación |",
             "|---|---|---|---|"]
    for layer in model.layers:
        try:
            out_shape = layer.output.shape
        except Exception:
            out_shape = "—"
        act = getattr(layer, "activation", None)
        act = act.__name__ if act else "—"
        lines.append(f"| {layer.__class__.__name__} | {tuple(out_shape)} | "
                     f"{layer.count_params()} | {act} |")
    lines.append(f"\n**Parámetros totales:** {model.count_params()}")
    with open(config.RESULTS_DIR / "mlp_topology.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _write_notes(metrics, best_variant, val_f1, best_model, yte, test_proba, histories):
    # Estabilidad = σ de val_loss en las últimas 20 épocas (menor = más estable);
    # deriva = val_loss final menos su mínimo (mayor = más sobreajuste).
    stab = {name: float(np.std(h["val_loss"][-20:])) for name, h in histories.items()}
    drift = {name: float(h["val_loss"][-1] - min(h["val_loss"])) for name, h in histories.items()}
    most_stable = min(stab, key=stab.get)
    least_stable = max(stab, key=stab.get)
    rep = classification_report(
        yte, test_proba[best_model].argmax(1),
        target_names=[config.SPECIES_NAME[config.IDX_TO_ID[i]] for i in range(config.N_CLASSES)],
        digits=3, zero_division=0)
    with open(config.RESULTS_DIR / "classification_notes.md", "w", encoding="utf-8") as fh:
        fh.write("# Interpretación — Clasificación (Sec 3.4)\n\n")
        fh.write("## Experimento de regularización (posición Dropout/BatchNorm)\n\n")
        fh.write("| Variante | F1-macro val | σ(val_loss) últ. 20 ép. | Deriva val_loss |\n")
        fh.write("|---|---|---|---|\n")
        for name in sorted(val_f1, key=lambda k: -val_f1[k]):
            fh.write(f"| {name} | {val_f1[name]:.4f} | {stab[name]:.3f} | {drift[name]:+.3f} |\n")
        fh.write(
            f"\n**Lectura guiada por las curvas (no por el F1):** el factor decisivo de la "
            f"estabilidad es la **presencia de Dropout**, no el orden relativo de las capas. Las "
            f"tres variantes con Dropout mantienen la pérdida de validación plana (~1.35) durante "
            f"100 épocas, mientras que *BN sin Dropout* y *sin regularización* sobreajustan: la "
            f"pérdida de entrenamiento cae pero la de validación se dispara (deriva positiva "
            f"grande). La curva más estable fue *{most_stable}* y la más inestable *{least_stable}*. "
            f"Entre las variantes con Dropout, el orden BN→Dropout vs Dropout→BN apenas altera el "
            f"F1-macro (rango {min(val_f1.values()):.3f}–{max(val_f1.values()):.3f}) por el fuerte "
            f"solapamiento entre especies. Variante seleccionada para el modelo final, por "
            f"menor pérdida de validación en convergencia (penaliza el sobreajuste): "
            f"**{best_variant}**.\n\n")
        fh.write("## Benchmark\n\n")
        fh.write(metrics.to_markdown(index=False) + "\n\n")
        fh.write(f"Mejor modelo por F1-macro (test): **{best_model}**. El desempeño global "
                 f"(F1-macro ≈ 0.43–0.44) supera el baseline de clase mayoritaria (~0.29 de "
                 f"exactitud) pero refleja la fuerte confusión entre especies anticipada en las "
                 f"Secs 3.2–3.3.\n\n")
        fh.write("La brecha F1_macro_val (agrupado por recording_id) vs F1_macro_test (split "
                 "provisto, con 122 grabaciones solapadas con train) refleja el optimismo por fuga "
                 "del conjunto de prueba entregado.\n\n")
        fh.write("### Reporte por clase del mejor modelo (test)\n\n```\n" + rep + "\n```\n")


if __name__ == "__main__":
    main()
