# ==============================================================================
# ¿QUÉ FUENTES MEJORAN CADA RAMA? (PLAN B)
# Proyecto: Immuno-XAI - UTEM
#
# Para cada rama (células / citoquinas) se mide el error de la rama sola:
#   - sin fuentes extra (solo los pacientes pareados)
#   - + cada fuente por separado
#   - + todas las fuentes de esa rama
# Evaluación: CV agrupada por paciente 5x5 sobre los 473 pareados; las fuentes
# solo se agregan al entrenamiento y se excluyen las personas de validación.
# Modelo de cada rama: LightGBM afinado con tu train (rápido, para comparar fuentes).
# ==============================================================================

import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

warnings.filterwarnings("ignore")
RANDOM_STATE, REPETICIONES = 42, 5
MIN_CELULAS_FLUJO = 6
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
carpeta = ruta("resultados_dos_ramas")
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:5.1f} min] {m}", flush=True)

par = pd.read_csv(ruta("pareados.csv"))
y = par["age"].values.astype(float)
grupos = par["subject_accession"].values
es_train = (par["conjunto"] == "train").values
cols_cel = ["B_cells", "CD16_neg_monocytes", "CD16_pos_monocytes", "CD4_T_cells", "CD8_T_cells",
            "Central_Memory_CD4_T_cells", "Central_Memory_CD8_T_cells", "Effector_CD4_T_cells", "Effector_CD8_T_cells",
            "Effector_Memory_CD4_T_cells", "Effector_Memory_CD8_T_cells", "Gamma_Delta_T_cells", "Lymphocytes",
            "Memory_B_cells", "Monocytes", "Naive_B_cells", "Naive_CD4_T_cells", "Naive_CD8_T_cells", "NK_cells",
            "NKT_cells", "Plasmablasts", "T_cells", "Transitional_B_cells", "Tregs"]
todas_cit = [c for c in par.columns if c not in cols_cel + ["subject_accession", "gender", "age", "conjunto"]]
cob = par.loc[es_train, todas_cit].notna().mean()
cols_cit = [c for c in todas_cit if cob[c] >= 0.8]

def razones(df):
    e = 0.01
    g = lambda c: df[c].astype(float)
    return pd.DataFrame({
        "r_CD4_CD8": g("CD4_T_cells") / (g("CD8_T_cells") + e),
        "r_naiveCD8_CD8": g("Naive_CD8_T_cells") / (g("CD8_T_cells") + e),
        "r_naiveCD4_CD4": g("Naive_CD4_T_cells") / (g("CD4_T_cells") + e),
        "r_naive_mem_CD8": g("Naive_CD8_T_cells") / (g("Central_Memory_CD8_T_cells") + g("Effector_Memory_CD8_T_cells") + e),
        "r_naive_mem_CD4": g("Naive_CD4_T_cells") / (g("Central_Memory_CD4_T_cells") + g("Effector_Memory_CD4_T_cells") + e),
        "r_EM_CD8": g("Effector_Memory_CD8_T_cells") / (g("CD8_T_cells") + e),
        "r_Eff_CD8": g("Effector_CD8_T_cells") / (g("CD8_T_cells") + e),
        "r_Bnaive_Bmem": g("Naive_B_cells") / (g("Memory_B_cells") + e),
        "r_Treg_CD4": g("Tregs") / (g("CD4_T_cells") + e),
        "r_mono_lymph": g("Monocytes") / (g("Lymphocytes") + e),
        "r_naiveCD8_T": g("Naive_CD8_T_cells") / (g("T_cells") + e),
    }, index=df.index)

sexo = lambda df: (df["gender"] == "Male").astype(float).values.reshape(-1, 1)
RAMA = {
    "Células": {"X": lambda d: np.hstack([d[cols_cel].astype(float).values, razones(d).values, sexo(d)]),
                "cols": cols_cel, "carpeta": "fuentes_celulas"},
    "Citoquinas": {"X": lambda d: np.hstack([d[cols_cit].astype(float).values, sexo(d)]),
                   "cols": cols_cit, "carpeta": "fuentes_citoquinas"},
}

def leer(carpeta_f, cols):
    fuentes = {}
    for p in sorted(glob.glob(ruta(carpeta_f, "*.csv"))):
        d = pd.read_csv(p)
        for c in cols:
            if c not in d.columns:
                d[c] = np.nan
        if "flujo" in os.path.basename(p):
            d = d[d[cols].notna().sum(axis=1) >= MIN_CELULAS_FLUJO]
        fuentes[os.path.basename(p).replace(".csv", "")] = d.reset_index(drop=True)
    return fuentes

GRILLA_LGB = {"n_estimators": [200, 400, 600, 800, 1200], "learning_rate": [0.005, 0.01, 0.02, 0.04],
              "max_depth": [3, 4, 5, 6, -1], "num_leaves": [7, 15, 31], "min_child_samples": [5, 10, 20, 30],
              "subsample": [0.7, 0.8, 1.0], "subsample_freq": [1], "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
              "reg_alpha": [0.0, 0.1, 1.0], "reg_lambda": [0.0, 1.0, 5.0]}
SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REPETICIONES)]
idx_tr, idx_te = np.where(es_train)[0], np.where(~es_train)[0]

def ajustar(modelo, X, tr, va, Xe, ye, ge):
    if Xe is not None and len(Xe):
        ok = ~np.isin(ge, list(set(grupos[va])))
        Xt, yt = np.vstack([X[tr], Xe[ok]]), np.concatenate([y[tr], ye[ok]])
    else:
        Xt, yt = X[tr], y[tr]
    return clone(modelo).fit(Xt, yt).predict(X[va])

filas = []
for rama, cfg in RAMA.items():
    X = cfg["X"](par)
    s = RandomizedSearchCV(lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1), GRILLA_LGB, n_iter=40,
                           scoring="neg_mean_absolute_error", cv=GroupKFold(n_splits=5, shuffle=True, random_state=100),
                           random_state=RANDOM_STATE, n_jobs=-1, refit=False)
    s.fit(X[es_train], y[es_train], groups=grupos[es_train])
    modelo = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **s.best_params_)
    fuentes = leer(cfg["carpeta"], cfg["cols"])
    variantes = [("sin fuentes extra", [])] + [(f"+ {k}", [k]) for k in fuentes] + [("+ todas", list(fuentes))]
    oof_base = None
    for nombre, usar in variantes:
        if usar:
            d = pd.concat([fuentes[k] for k in usar], ignore_index=True)
            Xe, ye, ge = cfg["X"](d), d["age"].values.astype(float), d["subject_accession"].values
        else:
            Xe = ye = ge = None
        tareas = [(r, tr, va) for r in range(REPETICIONES) for tr, va in SPLITS[r]]
        preds = Parallel(n_jobs=-1)(delayed(ajustar)(modelo, X, tr, va, Xe, ye, ge) for _, tr, va in tareas)
        oof = np.zeros((REPETICIONES, len(y)))
        for (r, _, va), p in zip(tareas, preds):
            oof[r, va] = p
        p_te = ajustar(modelo, X, idx_tr, idx_te, Xe, ye, ge)
        fila = {"rama": rama, "variante": nombre, "filas_extra": 0 if Xe is None else len(Xe),
                "MAE_CV": np.abs(oof - y).mean(axis=1).mean(),
                "R2_CV": np.mean([r2_score(y, oof[r]) for r in range(REPETICIONES)]),
                "MAE_test": np.abs(y[idx_te] - p_te).mean(), "R2_test": r2_score(y[idx_te], p_te)}
        if oof_base is None:
            oof_base = oof
        else:
            d_i = np.abs(oof - y).mean(axis=0) - np.abs(oof_base - y).mean(axis=0)
            db = d_i[np.random.default_rng(0).integers(0, len(y), size=(5000, len(y)))].mean(axis=1)
            fila.update({"dif_vs_sin_extra": d_i.mean(), "IC95_bajo": np.percentile(db, 2.5),
                         "IC95_alto": np.percentile(db, 97.5), "P_mejora": (db < 0).mean() * 100})
        filas.append(fila)
        extra = f" | dif {fila['dif_vs_sin_extra']:+.2f} [{fila['IC95_bajo']:+.2f}, {fila['IC95_alto']:+.2f}]" if "dif_vs_sin_extra" in fila else ""
        log(f"{rama:<10s} | {nombre:<28s} ({fila['filas_extra']:4d} filas) | MAE CV {fila['MAE_CV']:.2f} "
            f"R² {fila['R2_CV']:.3f} | test {fila['MAE_test']:.2f} R² {fila['R2_test']:.3f}{extra}")

df = pd.DataFrame(filas).round(3)
df.to_csv(os.path.join(carpeta, "fuentes_por_rama.csv"), index=False)
pd.set_option("display.width", 250)
print("\n" + df.to_string(index=False))
log("Listo.")
