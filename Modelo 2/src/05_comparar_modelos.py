# ==============================================================================
# PASO 05: COMPARACIÓN DE MODELOS CON LA MISMA EVALUACIÓN
# Proyecto: Immuno-XAI - UTEM
#
# Entrena varios modelos con los mismos pacientes y las mismas particiones, y
# los evalúa igual que al modelo final (paso 06). Así se ve, en una sola tabla,
# por qué se eligió TabPFN.
#
# Datos  : Stanford 10K Immunomes, 473 pacientes con células y citoquinas
#          (train 354 / test 119), igual que el paso 06.
# Modelos:
#   - 5 modelos clásicos (los mismos del inicio del proyecto y de Modelo 1):
#     ElasticNet, SVR (RBF), Random Forest, LightGBM y XGBoost.
#     Variables: los 45 biomarcadores del k45 + sexo (las mismas de TabPFN).
#     Hiperparámetros elegidos SOLO con los 354 de train (búsqueda aleatoria,
#     CV agrupada por paciente) y luego fijos en todas las evaluaciones.
#   - Dos ramas + fusión (modelo final anterior). Usa su propio diseño: 24
#     poblaciones + 11 razones en la rama de células (con el CyTOF extra del 10K)
#     y 46 citoquinas en la otra. Se reutiliza el código de
#     experimentos/05_modelo_dos_ramas/modelo_final_dos_ramas.py.
#   - TabPFN v2 (modelo final), con sus valores por defecto.
#
# Evaluaciones (idénticas para todos):
#   1. CV agrupada por paciente 5x5 sobre los 473 pacientes
#   2. Test de 119 pacientes (modelo entrenado solo con los 354 de train)
#   3. Estudios no vistos (se deja un estudio completo fuera, 5 estudios)
# Además, la diferencia de MAE en el test frente a TabPFN, con IC95 por bootstrap.
#
# Requiere: Datasets/pareados.csv (paso 04).
# Uso:  python -X utf8 src/05_comparar_modelos.py      (~15 min en CPU)
# Salida: resultados/comparacion_modelos/
# ==============================================================================

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import torch
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
from config import (
    ARCHIVO_ELISA_10K,
    ARCHIVO_PAREADOS,
    ARCHIVO_SCRIPT_DOS_RAMAS,
    ARCHIVO_TEST_K45,
    CARPETA_RES_COMPARACION,
)  # rutas del proyecto: src/config.py
carpeta = CARPETA_RES_COMPARACION
os.makedirs(carpeta, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:6.1f} min] {m}", flush=True)
RANDOM_STATE, REPETICIONES, N_ITER = 42, 5, 30
LOSO_ESTUDIOS = ["SDY420", "SDY515", "SDY519", "SDY113", "SDY311"]

def metricas(yv, p):
    return {"MAE": mean_absolute_error(yv, p), "RMSE": np.sqrt(mean_squared_error(yv, p)), "R2": r2_score(yv, p),
            "r": np.corrcoef(yv, p)[0, 1], "acierto_5a_%": np.mean(np.abs(yv - p) <= 5) * 100}

# ------------------------------------------------------------------------------
# Datos (los mismos del paso 06)
# ------------------------------------------------------------------------------
par = pd.read_csv(ARCHIVO_PAREADOS)
el = pd.read_csv(ARCHIVO_ELISA_10K)
k = el.assign(edad=el.age.round(2))[["subject_accession", "edad", "study_accession"]].drop_duplicates(["subject_accession", "edad"])
par = par.assign(edad=par.age.round(2)).merge(k, on=["subject_accession", "edad"], how="left")
y = par["age"].values.astype(float)
grupos = par["subject_accession"].values
estudios = par["study_accession"].fillna("otro").values
idx_tr = np.where(par["conjunto"] == "train")[0]
idx_te = np.where(par["conjunto"] == "test")[0]

test_k45 = pd.read_csv(ARCHIVO_TEST_K45)
VARIABLES = [c for c in test_k45.columns if c not in ["age", "gender", "source_type", "subject_accession", "race"]]
X = np.hstack([par[VARIABLES].astype(float).values,
               (par["gender"] == "Male").astype(float).values.reshape(-1, 1)]).astype(np.float32)
log(f"Pacientes {len(y)} (train {len(idx_tr)}, test {len(idx_te)}) | variables {X.shape[1]}")

# Mismas particiones que el paso 06
SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REPETICIONES)]
TAREAS = ([("cv", r, tr, va, None) for r in range(REPETICIONES) for tr, va in SPLITS[r]]
          + [("test", None, idx_tr, idx_te, None)]
          + [("loso", e, np.where(estudios != e)[0], np.where(estudios == e)[0], e) for e in LOSO_ESTUDIOS])

def ordenar(salidas):
    """Pasa las predicciones de TAREAS a {oof (rep x pacientes), test, loso {estudio: pred}}."""
    d = {"oof": np.zeros((REPETICIONES, len(y))), "test": None, "loso": {}}
    for (tipo, r, _, va, _), p in zip(TAREAS, salidas):
        if tipo == "cv":
            d["oof"][r, va] = p
        elif tipo == "test":
            d["test"] = p
        else:
            d["loso"][r] = p
    return d

RES, HIPER = {}, []

# ------------------------------------------------------------------------------
# 1. Modelos clásicos (45 biomarcadores + sexo)
# ------------------------------------------------------------------------------
def log1p_positivo(Z):
    return np.log1p(np.clip(Z, 0, None))

def escalado(m):
    """Imputación + log + estandarización (las citoquinas tienen colas largas)."""
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("log", FunctionTransformer(log1p_positivo)),
                     ("esc", StandardScaler()), ("m", m)])

CLASICOS = {
    "ElasticNet": (escalado(ElasticNet(random_state=RANDOM_STATE, max_iter=10000, tol=1e-3)),
                   {"m__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}),
    "SVR (RBF)": (escalado(SVR()),
                  {"m__C": [1, 5, 10, 20, 50, 100, 200], "m__gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
                   "m__epsilon": [0.1, 0.5, 1, 2, 5]}),
    "Random Forest": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("m", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))]),
                      {"m__n_estimators": [200, 300, 500], "m__max_depth": [4, 6, 8, 12, None],
                       "m__min_samples_leaf": [1, 2, 4, 8], "m__max_features": ["sqrt", "log2", 0.5, 0.8]}),
    "LightGBM": (lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
                 {"n_estimators": [200, 400, 600, 800, 1200], "learning_rate": [0.005, 0.01, 0.02, 0.04],
                  "max_depth": [3, 4, 5, 6, -1], "num_leaves": [7, 15, 31], "min_child_samples": [5, 10, 20, 30],
                  "subsample": [0.7, 0.8, 1.0], "subsample_freq": [1], "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
                  "reg_alpha": [0.0, 0.1, 1.0], "reg_lambda": [0.0, 1.0, 5.0]}),
    "XGBoost": (xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1),
                {"n_estimators": [200, 400, 800, 1200], "learning_rate": [0.005, 0.01, 0.02, 0.04],
                 "max_depth": [2, 3, 4, 5], "min_child_weight": [1, 3, 5, 10], "subsample": [0.7, 0.8, 1.0],
                 "colsample_bytree": [0.3, 0.5, 0.7, 0.9], "reg_alpha": [0, 0.1, 1], "reg_lambda": [0.5, 1, 5]}),
}

def afinar(est, grilla):
    """Hiperparámetros elegidos solo con los 354 pacientes de train."""
    s = RandomizedSearchCV(est, grilla, n_iter=N_ITER, scoring="neg_mean_absolute_error",
                           cv=GroupKFold(n_splits=5, shuffle=True, random_state=100),
                           random_state=RANDOM_STATE, n_jobs=-1, refit=False)
    s.fit(X[idx_tr], y[idx_tr], groups=grupos[idx_tr])
    return clone(est).set_params(**s.best_params_), s.best_params_

def ajustar_predecir(modelo, tr, va):
    return clone(modelo).fit(X[tr], y[tr]).predict(X[va])

for nombre, (est, grilla) in CLASICOS.items():
    t = time.time()
    modelo, mejores = afinar(est, grilla)
    HIPER += [{"modelo": nombre, "parametro": p.replace("m__", ""), "valor": v} for p, v in mejores.items()]
    salidas = Parallel(n_jobs=-1)(delayed(ajustar_predecir)(modelo, tr, va) for _, _, tr, va, _ in TAREAS)
    RES[nombre] = ordenar(salidas)
    log(f"{nombre:<14s} | MAE CV {np.abs(RES[nombre]['oof'] - y).mean():.2f} | "
        f"test {mean_absolute_error(y[idx_te], RES[nombre]['test']):.2f} | {(time.time()-t)/60:.1f} min")

# ------------------------------------------------------------------------------
# 2. Dos ramas + fusión (modelo final anterior)
# Se ejecuta modelo_final_dos_ramas.py hasta antes de su evaluación: carga sus
# datos, afina sus dos ramas con el train y define dos_ramas(tr, va, estudio).
# ------------------------------------------------------------------------------
t = time.time()
log("Dos ramas + fusión: afinando sus ramas (código de experimentos/05_modelo_dos_ramas)...")
argv_original, sys.argv = sys.argv, [sys.argv[0]]  # versión por defecto (sin ELISA extra ni ImmPort)
codigo = open(ARCHIVO_SCRIPT_DOS_RAMAS, encoding="utf-8").read()
corte = codigo.index("# ------------------------------------------------------------------------------\n# 1. CV agrupada")
ns = {"__file__": str(ARCHIVO_SCRIPT_DOS_RAMAS), "__name__": "dos_ramas"}
exec(compile(codigo[:corte], ARCHIVO_SCRIPT_DOS_RAMAS.name, "exec"), ns)
sys.argv = argv_original
assert np.array_equal(ns["grupos"], grupos) and np.allclose(ns["y"], y), "Los pacientes no coinciden con el paso 06"
salidas = Parallel(n_jobs=-1)(delayed(ns["dos_ramas"])(tr, va, e) for _, _, tr, va, e in TAREAS)
RES["Dos ramas + fusión"] = ordenar([s[0] for s in salidas])
HIPER.append({"modelo": "Dos ramas + fusión", "parametro": "(ver script)",
              "valor": "afinados en experimentos/05_modelo_dos_ramas/modelo_final_dos_ramas.py"})
log(f"Dos ramas + fusión | MAE CV {np.abs(RES['Dos ramas + fusión']['oof'] - y).mean():.2f} | "
    f"test {mean_absolute_error(y[idx_te], RES['Dos ramas + fusión']['test']):.2f} | {(time.time()-t)/60:.1f} min")

# ------------------------------------------------------------------------------
# 3. TabPFN v2 (modelo final; mismo código que el paso 06)
# ------------------------------------------------------------------------------
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
torch.set_num_threads(os.cpu_count())
t = time.time()
salidas = []
for i, (_, _, tr, va, _) in enumerate(TAREAS):
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)
    salidas.append(m.fit(X[tr], y[tr]).predict(X[va]))
    if (i + 1) % 5 == 0:
        log(f"TabPFN v2: {i+1}/{len(TAREAS)} ajustes")
RES["TabPFN v2"] = ordenar(salidas)
HIPER.append({"modelo": "TabPFN v2", "parametro": "(todos)", "valor": "valores por defecto (ModelVersion.V2)"})
log(f"TabPFN v2       | MAE CV {np.abs(RES['TabPFN v2']['oof'] - y).mean():.2f} | "
    f"test {mean_absolute_error(y[idx_te], RES['TabPFN v2']['test']):.2f} | {(time.time()-t)/60:.1f} min")

# ------------------------------------------------------------------------------
# Tablas
# ------------------------------------------------------------------------------
filas, filas_l, resumen = [], [], []
y_te = y[idx_te]
e_tabpfn = np.abs(RES["TabPFN v2"]["test"] - y_te)
for nombre, d in RES.items():
    m_cv = [metricas(y, d["oof"][r]) for r in range(REPETICIONES)]
    fila = {"modelo": nombre, "evaluacion": "CV 5x5 (473 pacientes)"}
    for k_ in m_cv[0]:
        vals = [m[k_] for m in m_cv]
        fila[k_], fila[f"{k_}_sd"] = np.mean(vals), np.std(vals)
    filas.append(fila)
    m_te = metricas(y_te, d["test"])
    filas.append({"modelo": nombre, "evaluacion": "Test (119 pacientes)", **m_te})
    loso = []
    for e in LOSO_ESTUDIOS:
        yv = y[estudios == e]
        loso.append({"modelo": nombre, "estudio_no_visto": e, "n": len(yv), "edades": f"{yv.min():.0f}-{yv.max():.0f}",
                     **metricas(yv, d["loso"][e])})
    filas_l += loso
    m_lu = {k_: np.mean([l[k_] for l in loso]) for k_ in ["MAE", "RMSE", "R2", "r", "acierto_5a_%"]}
    filas.append({"modelo": nombre, "evaluacion": "Estudios no vistos (promedio de 5)", **m_lu})

    # Diferencia de MAE en el test frente a TabPFN (bootstrap pareado, igual que el paso 06)
    if nombre != "TabPFN v2":
        e_m = np.abs(d["test"] - y_te)
        rng = np.random.default_rng(0)
        dif = np.array([(e_m[i] - e_tabpfn[i]).mean() for i in (rng.integers(0, len(y_te), len(y_te)) for _ in range(10000))])
        comp = {"tabpfn_mejor_por_anos": dif.mean(), "IC95_bajo": np.percentile(dif, 2.5),
                "IC95_alto": np.percentile(dif, 97.5), "prob_tabpfn_mejor_%": np.mean(dif > 0) * 100}
    else:
        comp = {"tabpfn_mejor_por_anos": np.nan, "IC95_bajo": np.nan, "IC95_alto": np.nan, "prob_tabpfn_mejor_%": np.nan}
    resumen.append({"modelo": nombre, "MAE_CV": fila["MAE"], "MAE_CV_sd": fila["MAE_sd"], "R2_CV": fila["R2"],
                    "MAE_test": m_te["MAE"], "R2_test": m_te["R2"], "acierto_5a_test_%": m_te["acierto_5a_%"],
                    "MAE_estudios_no_vistos": m_lu["MAE"], "r_estudios_no_vistos": m_lu["r"], **comp})

df = pd.DataFrame(filas).round(3)
df_l = pd.DataFrame(filas_l).round(3)
df_r = pd.DataFrame(resumen).sort_values("MAE_CV").round(3)
df.to_csv(os.path.join(carpeta, "metricas_comparacion.csv"), index=False)
df_l.to_csv(os.path.join(carpeta, "metricas_estudios_no_vistos.csv"), index=False)
df_r.to_csv(os.path.join(carpeta, "resumen_comparacion.csv"), index=False)
pd.DataFrame(HIPER).to_csv(os.path.join(carpeta, "hiperparametros.csv"), index=False)
pd.DataFrame({"subject_accession": grupos[idx_te], "edad_real": y_te,
              **{n: d["test"] for n, d in RES.items()}}).round(2).to_csv(
    os.path.join(carpeta, "predicciones_test.csv"), index=False)
np.savez(os.path.join(carpeta, "predicciones_cv.npz"), y=y, grupos=grupos,
         **{n: d["oof"] for n, d in RES.items()})

# ------------------------------------------------------------------------------
# Gráfico: MAE de cada modelo en las tres evaluaciones (mismo orden en los tres)
# ------------------------------------------------------------------------------
AZUL, GRIS, TINTA, TINTA2, FONDO = "#2a78d6", "#b9b8b1", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "figure.facecolor": FONDO, "axes.facecolor": FONDO,
                     "axes.spines.top": False, "axes.spines.right": False})
orden = df_r["modelo"].tolist()[::-1]  # el mejor en CV arriba
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
for ax, (col, titulo) in zip(axes, [("MAE_CV", "Validación cruzada 5×5 (473)"),
                                    ("MAE_test", "Test (119 pacientes)"),
                                    ("MAE_estudios_no_vistos", "Estudios no vistos (promedio de 5)")]):
    vals = df_r.set_index("modelo").loc[orden, col].values
    colores = [AZUL if n == "TabPFN v2" else GRIS for n in orden]
    ax.barh(orden, vals, color=colores, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, i, f"{v:.2f}", va="center", color=TINTA, fontsize=9.5,
                fontweight="bold" if orden[i] == "TabPFN v2" else "normal")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("MAE (años, menor es mejor)")
    ax.set_title(titulo, color=TINTA, loc="left", fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
fig.suptitle("Comparación de modelos con la misma evaluación", color=TINTA, x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(carpeta, "comparacion_modelos.png"), dpi=200)
plt.close(fig)

pd.set_option("display.width", 250)
print("\n" + df_r.to_string(index=False))
log("Listo.")
